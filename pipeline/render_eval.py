"""Stage 9 — renders and evaluation.

--part render  (in24d_4dgs): 4d/renders/era3d/vV/%04d.png (every global frame, 6 Era3D views),
               4d/renders/eval/vV/%04d.png (64 normalized times, 5 views at 0 deg elevation,
               eval.radius / eval.fov_deg), 4d/pointclouds/%04d.npy, metrics vs eval.gt_dir if given.
--part analyze (in24d_wan):  motion-progress linearity of the final front video; cross-view temporal LPIPS of Stage 5
               with injection on vs off (needs eval.injection_ablation in Stage 5).
Both parts merge their numbers into 4d/metrics.json.
"""
import math
from pathlib import Path

import numpy as np
import torch

from core.cameras import ERA3D_VIEWS, GSCamera, look_at_c2w
from core.io import read_frames, read_image, read_json, write_image, write_json
from core.stage import StageContext, stage_parser


def _save(img: torch.Tensor, path):
    write_image((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8), path)


class EmaSmoother:
    """Optional smoothing at render time: EMA on scales, slerp on rotations (alpha = 0.6).
    With `masks` ({fragment: [T, N] bool} from pipeline.trajectory_smoothing) only the Gaussians tracked at that
    frame are smoothed; without, all of them."""

    def __init__(self, alpha, masks=None):
        self.alpha, self.prev, self.masks = alpha, {}, masks

    def __call__(self, k, state, local=0.0):
        raw = (state["rotations"], state["scales"])
        if k in self.prev:
            q0, s0 = self.prev[k]
            q1 = torch.nn.functional.normalize(state["rotations"], dim=-1)
            dot = (q0 * q1).sum(-1, keepdim=True)
            q1 = torch.where(dot < 0, -q1, q1)
            theta = torch.acos(dot.abs().clamp(max=1 - 1e-6))
            a = self.alpha
            w0, w1 = torch.sin((1 - a) * theta) / torch.sin(theta), torch.sin(a * theta) / torch.sin(theta)
            state["rotations"] = torch.nn.functional.normalize(w0 * q0 + w1 * q1, dim=-1)
            state["scales"] = (1 - a) * s0 + a * state["scales"]
            if self.masks is not None:
                mask = self.masks[k][min(int(round(local)), self.masks[k].shape[0] - 1)][:, None]
                state["rotations"] = torch.where(mask, state["rotations"], raw[0])
                state["scales"] = torch.where(mask, state["scales"], raw[1])
        self.prev[k] = (torch.nn.functional.normalize(state["rotations"], dim=-1), state["scales"])
        return state


class SequenceRenderer:
    """Renders the trained sequence at any (fractional) global frame from any camera: fragment hand-over,
    optional boundary cross-fade and render-time EMA smoothing. Shared by Stage 9 and render_video.py."""

    def __init__(self, dirs, cfg, load_data=True):
        from core.gs.sequence import Sequence
        self.cfg, self.dirs = cfg, dirs
        self.seq = Sequence(dirs, cfg, load_data=load_data)
        self.white = torch.ones(3, device="cuda")
        self.smooth = None

    def reset_smoothing(self):
        """Call before each pass over time: the EMA state carries from frame to frame."""
        c = self.cfg.post_ema_smoothing
        masks = None
        if c.enabled and c.get("mode", "all") == "tracked":
            from pipeline.trajectory_smoothing import load_masks
            masks = load_masks(self.dirs, self.seq)
        self.smooth = EmaSmoother(c.alpha, masks) if c.enabled else None

    def state_at(self, p):
        """One deformation per global frame p, shared by all views."""
        seq, xfade = self.seq, self.cfg.merge.boundary_crossfade_frames
        k, local = seq.locate(p)
        m = seq.models[k]
        st = m.state(local)
        if self.smooth:
            st = self.smooth(k, st, local)
        blend, start = None, seq.frags[k]["global_start"]
        if xfade > 0 and k > 0 and p - start < xfade:
            prev = seq.models[k - 1]
            t_prev = min(seq.frags[k - 1]["num_frames"] - 1 + (p - start), prev.T_train - 1)
            blend = (prev, prev.state(t_prev), (p - start) / xfade)
        return k, local, m, st, blend

    def draw(self, cam, m, st, blend):
        from core.gs.render import rasterize_sets
        if blend is None:
            return m.render(cam, st)["render"]
        prev, pst, w = blend
        return rasterize_sets(cam, [(prev.g, pst, 1 - w), (m.g, st, w)], m.pipe, self.white)

    def orbit_camera(self, m, local, azimuth, elevation, radius, fov_deg, size):
        return GSCamera(look_at_c2w(azimuth, elevation, radius), math.radians(fov_deg), size, size, m.time(local),
                        znear=0.01, zfar=100.0)


@torch.no_grad()
def render_part(ctx):
    cfg, ev = ctx.cfg, ctx.cfg.eval
    r = SequenceRenderer(ctx.dirs, cfg)
    seq, state_at, draw = r.seq, r.state_at, r.draw

    r.reset_smoothing()
    for p in range(seq.total):
        k, local, m, st, blend = state_at(p)
        for v in range(len(ERA3D_VIEWS)):
            img = draw(seq.data[k].cams[v].at_time(m.time(local)), m, st, blend)
            _save(img, ctx.out / "renders" / "era3d" / f"v{v}" / f"{p:04d}.png")

    (ctx.out / "pointclouds").mkdir(parents=True, exist_ok=True)
    r.reset_smoothing()
    for i in range(ev.num_frames):
        p = i / (ev.num_frames - 1) * (seq.total - 1)
        k, local, m, st, blend = state_at(p)
        for v, az in enumerate(ev.views_azimuth):
            cam = r.orbit_camera(m, local, az, ev.elevation, ev.radius, ev.fov_deg, ev.img_size)
            _save(draw(cam, m, st, blend), ctx.out / "renders" / "eval" / f"v{v}" / f"{i:04d}.png")
        opac = m.g.opacity_activation(st["opacity"])[:, 0]
        np.save(ctx.out / "pointclouds" / f"{i:04d}.npy", st["xyz"][opac > ev.pc_opacity].cpu().numpy())
    return {"render_sources": dict(zip([f["id"] for f in seq.frags], seq.stages)), **gt_metrics(ctx)}


def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    d = torch.cdist(a, b)
    return float(d.min(1).values.mean() + d.min(0).values.mean())


def _normalize(pc: torch.Tensor) -> torch.Tensor:
    pc = pc - pc.mean(0)
    return pc / pc.norm(dim=1).max().clamp(min=1e-8)


@torch.no_grad()
def gt_metrics(ctx) -> dict:
    """LPIPS / CLIP vs ground-truth eval renders, CD / SI-CD vs ground-truth point clouds (gt_dir layout:
    vV/%04d.png and pointclouds/%04d.npy). FVD needs an I3D model and is not computed here."""
    ev = ctx.cfg.eval
    if not ev.gt_dir:
        return {"gt_metrics": "skipped (eval.gt_dir not set)"}
    from core.metrics import lpips_model
    gt = Path(ev.gt_dir)
    to_t = lambda path: torch.from_numpy(read_image(path, "RGB").astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    lp, clip_s, cd, sicd = [], [], [], []
    clip = _clip_model()
    for v in range(len(ev.views_azimuth)):
        for i in range(ev.num_frames):
            pr, gtp = ctx.out / "renders" / "eval" / f"v{v}" / f"{i:04d}.png", gt / f"v{v}" / f"{i:04d}.png"
            if gtp.exists():
                a, b = to_t(pr), to_t(gtp)
                lp.append(lpips_model("cuda")(a, b, normalize=True).item())
                if clip:
                    clip_s.append(clip(a, b))
    for i in range(ev.num_frames):
        gtp = gt / "pointclouds" / f"{i:04d}.npy"
        if gtp.exists():
            a = torch.from_numpy(np.load(ctx.out / "pointclouds" / f"{i:04d}.npy")).float().cuda()
            b = torch.from_numpy(np.load(gtp)).float().cuda()
            cd.append(chamfer(a, b))
            sicd.append(chamfer(_normalize(a), _normalize(b)))
    mean = lambda x: float(np.mean(x)) if x else None
    return {"LPIPS": mean(lp), "CLIP": mean(clip_s), "CD": mean(cd), "SI-CD": mean(sicd), "FVD": None}


def _clip_model():
    try:
        import open_clip
    except ImportError:
        return None
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.cuda().eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda")[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda")[:, None, None]

    def score(a, b):
        f = lambda x: torch.nn.functional.normalize(model.encode_image(
            (torch.nn.functional.interpolate(x, (224, 224), mode="bicubic") - mean) / std), dim=-1)
        return float((f(a) * f(b)).sum())
    return score


def analyze_part(ctx):
    from core.matting import matte_white_threshold
    from core.metrics import masked_lpips
    from pipeline.motion_progress import ProgressAnalyzer

    cfg = ctx.cfg
    front = read_frames(ctx.out / "renders" / "era3d" / "v0")
    alphas = np.stack([matte_white_threshold(f, cfg.matting.white_threshold) for f in front])
    res = ProgressAnalyzer(cfg).analyze(front, alphas)
    out = {"final_front_progress_linearity": res["lin"], "final_front_progress_r2": res["r2"]}

    frags = read_json(ctx.dirs.fragments_json)["fragments"]
    for name in ("mv", "mv_noinj"):
        vals = []
        for f in frags:
            d = ctx.dirs.frag_dir(5, f["id"]) / name
            if not d.exists():
                continue
            for v in range(1, len(ERA3D_VIEWS)):
                for t in range(1, f["num_frames"]):
                    a, b = read_image(d / f"t{t - 1:03d}_v{v}.png", "RGB"), read_image(d / f"t{t:03d}_v{v}.png", "RGB")
                    m = np.maximum(read_image(d / f"t{t - 1:03d}_v{v}_alpha.png", "L"),
                                   read_image(d / f"t{t:03d}_v{v}_alpha.png", "L")) / 255.0
                    vals.append(masked_lpips(a, b, m))
        out[f"stage5_temporal_lpips_{name}"] = float(np.mean(vals)) if vals else None
    return out


def main():
    p = stage_parser(__doc__)
    p.add_argument("--part", choices=["render", "analyze"], default="render")
    ctx = StageContext(p.parse_args(), 9)
    done = ctx.out / f"{ctx.args.part}.done"
    if ctx.done(done):
        return
    metrics_path = ctx.out / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    metrics.update(render_part(ctx) if ctx.args.part == "render" else analyze_part(ctx))
    write_json(metrics, metrics_path)
    done.write_text("ok\n")
    print(f"[s9] {metrics}")


if __name__ == "__main__":
    main()
