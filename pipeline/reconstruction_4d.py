"""Stage 6 — per-fragment canonical 3DGS + deformation field.

Outputs: 4d/reconstruction/frag_XX/{data/ (Blender-style dataset), ckpt/{static,dynamic}.pth, meta.json}
"""
import math

import numpy as np
import torch
from tqdm import tqdm

from core.cameras import ERA3D_VIEWS, project_points
from core.io import read_json, write_json
from core.metrics import mask_iou
from core.stage import StageContext, stage_parser
from core.gs.data import FragmentData, write_dataset
from core.gs.backbone import gs_hparams, new_gaussians, spatial_extent, sphere_points
from core.gs.model import FragmentModel
from core.gs.train import Objective, densify_step, regulation

V_FRONT, V_RIGHT, V_LEFT = ERA3D_VIEWS.index("front"), ERA3D_VIEWS.index("right"), ERA3D_VIEWS.index("left")


def visual_hull(data: FragmentData, views, n=200_000, seed=0) -> np.ndarray:
    """Points in [-1, 1]^3 whose projection lies inside the t=0 alpha of every given view."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pts = torch.rand(n, 3, device="cuda", generator=g) * 2 - 1
    keep = torch.ones(n, dtype=torch.bool, device="cuda")
    for v in views:
        px, _ = project_points(pts, data.cam(v, 0))
        a = data.alpha[0][v][0]
        H, W = a.shape
        u, w = px[:, 0].round().long(), px[:, 1].round().long()
        inside = (u >= 0) & (u < W) & (w >= 0) & (w < H)
        keep &= inside & (a[w.clamp(0, H - 1), u.clamp(0, W - 1)] > 0.5)
    return pts[keep].cpu().numpy()


def camera_test(data, cfg, check, frag_id):
    """hull from front+left; front render must match the input alpha, right must not be mirrored."""
    hyper, _, pipe = gs_hparams(cfg, data.T)
    n = 2_000_000
    pts = visual_hull(data, [V_FRONT, V_LEFT], n=n)
    if len(pts) < 100:
        return check(f"camera_test_frag{frag_id:02d}", False, reason="visual hull empty: inconsistent cameras")
    g = new_gaussians(hyper, pts, np.full_like(pts, 0.5), cfg)
    with torch.no_grad():
        g._opacity.fill_(4.0)
        g._scaling.fill_(math.log(0.25 * (8 / n) ** (1 / 3)))
        model = FragmentModel(g, hyper, None, pipe, data.T)
        st = model.state(0, stage="coarse")
        iou = {name: mask_iou(model.render_alpha(data.cam(v, 0), st)[0].cpu().numpy(), data.alpha[0][v][0].cpu().numpy())
               for name, v in (("front", V_FRONT), ("right", V_RIGHT))}
    return check(f"camera_test_frag{frag_id:02d}",
                 iou["front"] > cfg.gs.camera_test_iou_front and iou["right"] > cfg.gs.camera_test_iou_right, **iou)


def train_fragment(data: FragmentData, cfg, out_dir, seed: int):
    hyper, opt, pipe = gs_hparams(cfg, data.T)
    extent = spatial_extent(cfg)
    if cfg.gs.init == "visual_hull":
        pts = visual_hull(data, range(len(ERA3D_VIEWS)), n=cfg.gs.hull_samples, seed=seed)
        cols = np.full_like(pts, 0.5)
    else:
        pts, cols = sphere_points(cfg.gs.init_points, cfg.gs.init_radius, seed)
    g = new_gaussians(hyper, pts, cols, cfg)
    model = FragmentModel(g, hyper, opt, pipe, data.T)
    objective = Objective(cfg, seed)
    log = {}

    for stage, iters in (("coarse", cfg.gs.static_iters or opt.coarse_iterations),
                         ("fine", cfg.gs.dynamic_iters or opt.iterations)):
        opt.position_lr_max_steps = iters
        # densify (and reset opacity every opacity_reset_interval) only in the first part of each stage, so the last
        # reset has time to recover: a reset on the final iteration saves every Gaussian at opacity <= 0.01
        opt.densify_until_iter = int(cfg.gs.densify_until_frac * iters)
        if not cfg.gs.opacity_reset:
            opt.opacity_reset_interval = iters + 1
        g.training_setup(opt)
        k = 1 if stage == "coarse" else min(cfg.gs.frames_per_step, data.T)
        bar = tqdm(range(1, iters + 1), desc=f"{out_dir.name}-{stage}")
        for it in bar:
            g.update_learning_rate(it)
            if it % 1000 == 0:
                g.oneupSHdegree()
            ts = [0] if stage == "coarse" else objective.rng.sample(range(data.T), k)
            loss, logs, renders = 0.0, {}, []
            for t in ts:
                loss_t, logs_t, renders_t = objective(model, data, t, model.state(t, stage), use_rigid=(stage == "fine"))
                loss = loss + loss_t / k
                renders += renders_t
                for name, v in logs_t.items():
                    logs[name] = logs.get(name, 0.0) + v / k
            if stage == "fine":
                loss = loss + regulation(g, hyper)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at {stage} iteration {it}")
            loss.backward()
            densify_step(g, opt, renders, it, stage, extent)
            late = cfg.gs.late_prune_interval
            if stage == "fine" and late and it > opt.densify_until_iter and it % late == 0 and it < iters:
                with torch.no_grad():
                    small = g.get_opacity[:, 0] < 0.01
                    huge = g.get_scaling.max(dim=1).values > 0.1 * extent
                g.prune_points(small | huge)
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
            if it % 50 == 0:
                bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()} | {"pts": g.get_xyz.shape[0]})
        log[stage] = logs
        model.save(out_dir / "ckpt" / ("static.pth" if stage == "coarse" else "dynamic.pth"))
    return log


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 6)
    frags = read_json(ctx.dirs.fragments_json)["fragments"]
    for k, f in enumerate(frags):
        out = ctx.dirs.frag_dir(6, f["id"])
        if not ctx.mine(f["id"]) or ctx.done(out / "meta.json"):
            continue
        info = write_dataset(ctx.dirs, frags, k, ctx.cfg, out / "data")
        data = FragmentData(out / "data", ctx.cfg)
        camera_test(data, ctx.cfg, ctx.check, f["id"])
        log = train_fragment(data, ctx.cfg, out, seed=ctx.cfg.seed + f["id"])
        write_json({**info, "final_losses": log, "fragment": f}, out / "meta.json")


if __name__ == "__main__":
    main()
