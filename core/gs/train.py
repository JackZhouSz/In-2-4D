"""Training steps shared by Stages 6-8: optimizer/densification loop around the reconstruction objective."""
import random

import torch

from .losses import RigidLoss, l_mask, l_mv, l_ref, low_res


class Objective:
    """L = w_ref L_ref + w_mask L_mask + w_rigid L_rigid (+ w_mv L_mv) at one frame t."""

    def __init__(self, cfg, seed: int):
        self.cfg, self.rng = cfg, random.Random(seed)
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.rigid = RigidLoss(cfg.rigid.knn, cfg.rigid.anchors)

    def __call__(self, model, data, t: int, state: dict, use_rigid=True, use_mv=True):
        """Returns (loss, logs, renders); renders carry viewspace points for densification."""
        c, w = self.cfg, self.cfg.loss
        views = data.views_at(t)
        renders = []

        bg = None
        if c.gs.random_bg_prob > 0 and self.rng.random() < c.gs.random_bg_prob:
            bg = torch.zeros(3, device="cuda")

        def target(v):
            gt = data.rgb[t][v]
            return gt if bg is None else gt - (1.0 - data.alpha[t][v])

        front = model.render(data.cam(0, t), state, bg)
        renders.append(front)
        loss_ref = l_ref(front["render"], target(0))

        v_mask = self.rng.choice(views)
        loss_mask = l_mask(model.render_alpha(data.cam(v_mask, t), state), data.alpha[t][v_mask], c.loss.mask_type)

        loss = w.ref * loss_ref + w.mask * loss_mask
        logs = {"ref": loss_ref.item(), "mask": loss_mask.item()}
        if use_rigid:
            loss_rigid = self.rigid(model.g.get_xyz, state["xyz"], self.gen)
            loss = loss + w.rigid * loss_rigid
            logs["rigid"] = loss_rigid.item()

        others = [v for v in views if v != 0]
        if use_mv and c.use_mv_rgb_loss and others:
            v = self.rng.choice(others)
            cam, gt = data.cam(v, t), target(v)
            if c.gs.low_res_aug:
                cam, gt = low_res(cam, gt, self.rng)
            out = model.render(cam, state, bg)
            renders.append(out)
            loss_mv = l_mv(out["render"], gt, w.mv_lpips)
            loss = loss + w.mv * loss_mv
            logs["mv"] = loss_mv.item()
        return loss, logs, renders


def densify_step(g, opt, renders, iteration: int, stage: str, extent: float, white_background=True):
    """Densification/pruning for a list of renders of one step."""
    if iteration >= opt.densify_until_iter:
        return
    with torch.no_grad():
        grad = sum(r["viewspace_points"].grad for r in renders if r["viewspace_points"].grad is not None)
        radii = torch.stack([r["radii"] for r in renders]).max(0).values
        vis = torch.stack([r["visibility_filter"] for r in renders]).any(0)
        g.max_radii2D[vis] = torch.max(g.max_radii2D[vis], radii[vis])
        g.add_densification_stats(grad, vis)
        if stage == "coarse":
            opacity_th, densify_th = opt.opacity_threshold_coarse, opt.densify_grad_threshold_coarse
        else:
            frac = iteration / opt.densify_until_iter
            opacity_th = opt.opacity_threshold_fine_init - frac * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)
            densify_th = opt.densify_grad_threshold_fine_init - frac * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)
        size_th = 20 if iteration > opt.opacity_reset_interval else None
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            g.densify(densify_th, opacity_th, extent, size_th)
        if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
            g.prune(densify_th, opacity_th, extent, size_th)
        if iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter):
            g.reset_opacity()


def regulation(g, hyper):
    """Deformation-grid regularizers (time smoothness, plane TV, L1 on time planes)."""
    if hyper.time_smoothness_weight == 0:
        return 0.0
    return g.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes)


def deformation_param_groups(g, lr_mult: float, opt, extent: float, color_head=True):
    """Only the deformation network (MLP incl. color head, and grids) at lr_mult x Stage 6 lr."""
    net = g._deformation
    mlp = [p for n, p in net.named_parameters() if "grid" not in n and (color_head or "color_deform" not in n)]
    return [{"params": mlp, "lr": lr_mult * opt.deformation_lr_init * extent, "name": "deformation"},
            {"params": list(net.get_grid_parameters()), "lr": lr_mult * opt.grid_lr_init * extent, "name": "grid"}]


def freeze_canonical(g):
    for p in (g._xyz, g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation):
        p.requires_grad_(False)
