"""Stage 8 — point-tracking-loss global fine-tune.

Outputs: 4d/refinement/{tracks/frag_XX_vV.pt {tracks [T, N, 2] px, visibility [T, N]}, refined/frag_XX.pth, log.json}
"""
import random

import torch
from tqdm import tqdm

from core.cameras import ERA3D_VIEWS
from core.io import write_json
from core.stage import StageContext, stage_parser
from core.gs.backbone import spatial_extent
from core.gs.sequence import Sequence
from core.gs.train import Objective, deformation_param_groups, freeze_canonical
from patches.point_tracking_loss import TrackingLoss


@torch.no_grad()
def cotrack(cotracker, data, view: int, T: int, grid_size: int):
    """CoTracker3 offline on one view of a fragment, queried on a foreground grid of frame 0."""
    video = torch.stack([data.rgb[t][view] for t in range(T)])[None] * 255.0
    H, W = video.shape[-2:]
    alpha0 = data.alpha[0][view][0]
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, grid_size, device=alpha0.device),
                            torch.linspace(0, W - 1, grid_size, device=alpha0.device), indexing="ij")
    keep = alpha0[ys.round().long(), xs.round().long()] > 0.5
    queries = torch.stack([torch.zeros_like(xs[keep]), xs[keep], ys[keep]], -1).cuda()
    tracks, vis = cotracker(video, queries=queries[None])
    return {"tracks": tracks[0].cpu(), "visibility": vis[0].cpu(), "view": view}


def load_tracks(ctx, seq):
    views = [0] + (list(range(1, len(ERA3D_VIEWS))) if ctx.cfg.track.track_mv_views else [])
    out, cotracker = {}, None
    for k, f in enumerate(seq.frags):
        for v in views:
            path = ctx.out / "tracks" / f"frag_{f['id']:02d}_v{v}.pt"
            if not path.exists() or ctx.args.force:
                if cotracker is None:
                    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cotrack(cotracker, seq.data[k], v, f["num_frames"], ctx.cfg.track.grid_size), path)
            out[(k, v)] = torch.load(path)
    return out


def tracking_term(loss_fn, model, data, tracks, view, lj, T, window, rng, target_state):
    """Tracking loss for target frame lj with a source frame in [lj - w, lj + w] of the same fragment."""
    cands = [i for i in range(max(0, lj - window), min(T - 1, lj + window) + 1) if i != lj]
    if not cands:
        return None
    li = rng.choice(cands)
    tr, vis = tracks["tracks"].cuda(), tracks["visibility"].cuda().bool()
    both = vis[li] & vis[lj]
    if both.sum() < 4:
        return None
    with torch.no_grad():
        src = model.state(li)
    g = model.g
    return loss_fn(data.cam(view, li), data.cam(view, lj), src["xyz"], target_state["xyz"],
                   g.opacity_activation(src["opacity"]), g.scaling_activation(src["scales"]),
                   tr[li][both], tr[lj][both])


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 8)
    cfg, tc = ctx.cfg, ctx.cfg.track
    out = ctx.out / "refined"
    if ctx.done(out / "done.json"):
        return
    seq = Sequence(ctx.dirs, cfg, stage="s7")
    tracks = load_tracks(ctx, seq)

    optims = []
    for m in seq.models:
        freeze_canonical(m.g)
        optims.append(torch.optim.Adam(deformation_param_groups(m.g, tc.lr_mult, m.opt, spatial_extent(cfg)),
                                       lr=0.0, eps=1e-15))
    objective = Objective(cfg, cfg.seed)
    loss_fn = TrackingLoss(K=tc.knn, r=tc.radius_px)
    rng = random.Random(cfg.seed)
    bar = tqdm(range(tc.iters), desc="track-refine")
    for _ in bar:
        k, lj = seq.locate(rng.randrange(seq.total))
        lj = int(lj)
        model, data, T = seq.models[k], seq.data[k], seq.frags[k]["num_frames"]
        state = model.state(lj)
        loss, logs, _ = objective(model, data, lj, state)
        views = [(0, 1.0)]
        if tc.track_mv_views:
            views.append((rng.randrange(1, len(ERA3D_VIEWS)), tc.mv_weight))
        for v, w in views:
            lt = tracking_term(loss_fn, model, data, tracks[(k, v)], v, lj, T, tc.window, rng, state)
            if lt is not None:
                loss = loss + tc.weight * w * lt
                logs[f"track_v{v}"] = lt.item()
        optims[k].zero_grad(set_to_none=True)
        loss.backward()
        optims[k].step()
        bar.set_postfix({n: f"{x:.4f}" for n, x in logs.items()})

    for f, m in zip(seq.frags, seq.models):
        m.save(out / f"frag_{f['id']:02d}.pth")
    write_json({"iters": tc.iters, "fragments": len(seq.frags)}, out / "done.json")


if __name__ == "__main__":
    main()
