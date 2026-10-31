"""Cascaded trajectory smoothing, run after the track refinement when post_ema_smoothing.mode = tracked.

--part depth  (in24d_wan):  Depth-Anything-V2 relative depth of every frame of the tracked views.
--part smooth (in24d_4dgs): in sliding windows of `window` frames (step `stride`), CoTracker3 tracks on the front video
                            (views: all = also the five Era3D view videos),
                            depth aligned to the model's rendered depth, tracks lifted to 3D with the camera intrinsics;
                            tracks visible in >= min_visible of the window select their `knn` nearest deformed Gaussians.
                            Rendering then applies slerp/EMA on rotation and scale to exactly those Gaussians.

Outputs: 4d/smoothing/{depth/frag_XX_vV.npy, frag_XX.pt {mask [T, N] bool}, log.json}
"""
import numpy as np
import torch

from core.cameras import ERA3D_VIEWS
from core.io import read_image, read_json, write_json
from core.stage import StageContext, stage_parser

SUBDIR = "4d/smoothing"
V_FRONT = ERA3D_VIEWS.index("front")


def tracked_views(c):
    return [V_FRONT] + ([v for v in range(len(ERA3D_VIEWS)) if v != V_FRONT] if c.views == "all" else [])


def load_masks(dirs, seq) -> dict:
    masks = {}
    for k, f in enumerate(seq.frags):
        mask = torch.load(dirs.root / SUBDIR / f"frag_{f['id']:02d}.pt")["mask"]
        assert mask.shape[1] == seq.models[k].g.get_xyz.shape[0], "smoothing masks are from another checkpoint"
        masks[k] = mask.cuda()
    return masks


def windows(T: int, w: int, stride: int):
    if T <= w:
        return [(0, T)]
    starts = list(range(0, T - w + 1, stride))
    if starts[-1] + w < T:
        starts.append(T - w)
    return [(a, a + w) for a in starts]


@torch.no_grad()
def depth_part(ctx):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    c = ctx.cfg.post_ema_smoothing
    processor = AutoImageProcessor.from_pretrained(c.depth_model)
    model = AutoModelForDepthEstimation.from_pretrained(c.depth_model).cuda().eval()
    jobs = [(f, v) for f in read_json(ctx.dirs.fragments_json)["fragments"] for v in tracked_views(c)]
    for f, v in jobs:
        path = ctx.out / "depth" / f"frag_{f['id']:02d}_v{v}.npy"
        if path.exists() and not ctx.args.force:
            continue
        images = ctx.dirs.frag_dir(6, f["id"]) / "data" / "images"
        out = []
        for t in range(f["num_frames"]):
            rgba = read_image(images / f"t{t:03d}_v{v}.png", "RGBA").astype(np.float32) / 255.0
            rgb = (rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])) * 255.0
            inputs = processor(images=rgb.round().astype(np.uint8), return_tensors="pt").to("cuda")
            pred = model(**inputs).predicted_depth[:, None]
            pred = torch.nn.functional.interpolate(pred, size=rgba.shape[:2], mode="bicubic", align_corners=False)
            out.append(pred[0, 0].float().cpu().numpy())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.stack(out).astype(np.float32))
    return {}


class AlignedDepth:
    """Depth-Anything relative depth of frame t mapped to the model's camera-space depth by a per-frame affine fit
    on the foreground (the rasterizer's depth divided by its accumulated alpha)."""

    def __init__(self, model, data, rel_depth: torch.Tensor, view: int):
        self.model, self.data, self.rel, self.view = model, data, rel_depth, view
        self.cache, self.residuals = {}, []

    def __call__(self, t: int) -> torch.Tensor:
        if t not in self.cache:
            m, cam = self.model, self.data.cam(self.view, t)
            st = m.state(t)
            rendered = m.render(cam, st)["depth"][0]
            alpha = m.render_alpha(cam, st)[0]
            fg = (alpha > 0.5) & (self.data.alpha[t][self.view][0] > 0.5)
            z, d = rendered[fg] / alpha[fg], self.rel[t][fg]
            A = torch.stack([d, torch.ones_like(d)], -1)
            sol = torch.linalg.lstsq(A, z[:, None]).solution[:, 0]
            self.residuals.append(float((A @ sol - z).abs().mean()))
            self.cache[t] = sol[0] * self.rel[t] + sol[1]
        return self.cache[t]


def sample(depth: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    H, W = depth.shape
    u = px[:, 0].round().long().clamp(0, W - 1)
    v = px[:, 1].round().long().clamp(0, H - 1)
    return depth[v, u]


@torch.no_grad()
def smooth_fragment(cotracker, model, data, rel_depths: dict, T: int, c):
    N = model.g.get_xyz.shape[0]
    mask = torch.zeros(T, N, dtype=torch.bool, device="cuda")
    depths = {v: AlignedDepth(model, data, d, v) for v, d in rel_depths.items()}
    stats = {"windows": [], "lift_dist": []}
    for a, b in windows(T, c.window, c.stride):
        selected = torch.zeros(N, dtype=torch.bool, device="cuda")
        info = {"frames": [a, b], "tracks": 0, "stable": 0}
        for v, depth in depths.items():
            if any(data.rgb[t][v] is None for t in range(a, b)):
                continue
            n_tracks, n_stable = select_window(cotracker, model, data, depth, v, a, b, c, selected, stats)
            info["tracks"] += n_tracks
            info["stable"] += n_stable
        mask[a:b] |= selected
        stats["windows"].append({**info, "gaussians": int(selected.sum())})
    stats["depth_fit_residual"] = float(np.mean([r for d in depths.values() for r in d.residuals]))
    stats["lift_dist_median"] = float(np.median(stats.pop("lift_dist")))
    stats["smoothed_fraction"] = float(mask.float().mean())
    return mask.cpu(), stats


def select_window(cotracker, model, data, depth, view, a, b, c, selected, stats):
    """Tracks of one view in frames [a, b): mark the Gaussians near tracks that stay visible in min_visible."""
    from core.cameras import unproject_points
    video = torch.stack([data.rgb[t][view] for t in range(a, b)])[None] * 255.0
    H, W = video.shape[-2:]
    alpha0 = data.alpha[a][view][0]
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, c.grid_size, device="cuda"),
                            torch.linspace(0, W - 1, c.grid_size, device="cuda"), indexing="ij")
    keep = alpha0[ys.round().long(), xs.round().long()] > 0.5
    queries = torch.stack([torch.zeros_like(xs[keep]), xs[keep], ys[keep]], -1)
    tracks, vis = cotracker(video, queries=queries[None])
    tracks, vis = tracks[0], vis[0].bool()
    stable = vis.float().mean(0) >= c.min_visible
    for i, t in enumerate(range(a, b)):
        ok = stable & vis[i]
        if ok.sum() == 0:
            continue
        px = tracks[i][ok]
        pts = unproject_points(px, sample(depth(t), px), data.cam(view, t))
        xyz = model.state(t)["xyz"]
        dist, idx = torch.cdist(pts, xyz).topk(c.knn, dim=1, largest=False)
        selected[idx[dist < c.max_dist]] = True
        stats["lift_dist"].append(float(dist[:, 0].median()))
    return int(keep.sum()), int(stable.sum())


def smooth_part(ctx):
    from core.gs.sequence import Sequence
    seq = Sequence(ctx.dirs, ctx.cfg)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
    log = {}
    for k, f in enumerate(seq.frags):
        c = ctx.cfg.post_ema_smoothing
        rel = {v: torch.from_numpy(np.load(ctx.out / "depth" / f"frag_{f['id']:02d}_v{v}.npy")).cuda()
               for v in tracked_views(c)}
        mask, stats = smooth_fragment(cotracker, seq.models[k], seq.data[k], rel, f["num_frames"], c)
        torch.save({"mask": mask, "source": seq.stages[k]}, ctx.out / f"frag_{f['id']:02d}.pt")
        log[f"frag_{f['id']:02d}"] = stats
        print(f"[smoothing] frag {f['id']:02d}: {stats['smoothed_fraction']:.3f} of Gaussian-frames, "
              f"lift dist {stats['lift_dist_median']:.4f}, depth residual {stats['depth_fit_residual']:.4f}")
    return log


def main():
    p = stage_parser(__doc__)
    p.add_argument("--part", choices=["depth", "smooth"], default="smooth")
    args = p.parse_args()
    ctx = StageContext(args, 8, subdir=SUBDIR)
    c = ctx.cfg.post_ema_smoothing
    if not c.enabled or c.mode != "tracked":
        print("[smoothing] post_ema_smoothing.enabled/mode=tracked not set, skipping")
        return
    done = ctx.out / f"{args.part}.done"
    if ctx.done(done):
        return
    log = depth_part(ctx) if args.part == "depth" else smooth_part(ctx)
    if log:
        write_json(log, ctx.out / "log.json")
    done.write_text("ok\n")


if __name__ == "__main__":
    main()
