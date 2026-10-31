"""Stage 7 — cascaded motion merging.

For each boundary (cumulative: (1,2), then (merged,3), ...), on the overlap frames of fragment j:
    xyz = x_c + lam * dPhi_i + (1 - lam) * dPhi*_j
dPhi_i (fragment i's frozen motion) is carried onto j's Gaussians by nearest-neighbour correspondence with
i's deformed state at the shared keyframe; dPhi*_j is a learnable copy of j's deformation, initialized
from (frozen) dPhi_j. Only dPhi*_j is optimized, with the reconstruction objective on the overlap frames.

Outputs: 4d/merge/merged/frag_XX.pth, 4d/merge/merge_log.json
"""
import copy
import random

import torch
from tqdm import tqdm

from core.io import read_json, write_json
from core.metrics import lpips_model
from core.stage import StageContext, stage_parser
from core.gs.data import FragmentData
from core.gs.backbone import freeze, spatial_extent
from core.gs.model import FragmentModel, MergeInfo
from core.gs.render import deform
from core.gs.sequence import ckpt_path
from core.gs.train import Objective, deformation_param_groups, freeze_canonical


@torch.no_grad()
def carried_motion(model_i: FragmentModel, T_i: int, model_j: FragmentModel, overlap: int) -> torch.Tensor:
    """dPhi_i on j's Gaussians: [overlap, N_j, 3] = y_i(m)[nn] - x_j, nn matched at the shared keyframe."""
    x_j = model_j.g.get_xyz
    y0 = model_i.state(T_i - 1)["xyz"]
    nn = torch.cat([torch.cdist(chunk, y0).argmin(1) for chunk in torch.split(x_j, 1024)])
    return torch.stack([model_i.state(T_i - 1 + m)["xyz"][nn] - x_j for m in range(overlap)])


def merge_pair(model_i, T_i, model_j, data_j: FragmentData, cfg, seed):
    mc = cfg.merge
    model_j.merge = MergeInfo(mc["lambda"], carried_motion(model_i, T_i, model_j, mc.overlap_frames))
    frozen_j = freeze(copy.deepcopy(model_j.g._deformation))
    freeze_canonical(model_j.g)
    optim = torch.optim.Adam(deformation_param_groups(model_j.g, mc.lr_mult, model_j.opt, spatial_extent(cfg)),
                             lr=0.0, eps=1e-15)
    objective = Objective(cfg, seed)
    rng = random.Random(seed)
    non_overlap = list(range(mc.overlap_frames, data_j.T))
    for _ in tqdm(range(mc.iters), desc="merge"):
        m = rng.randrange(mc.overlap_frames)
        loss, _, _ = objective(model_j, data_j, m, model_j.state(m), use_rigid=True, use_mv=False)
        if mc.anchor_weight > 0 and non_overlap:
            tt = model_j.time(rng.choice(non_overlap))
            loss = loss + mc.anchor_weight * (deform(model_j.g, tt)["xyz"] -
                                              deform(model_j.g, tt, deformation=frozen_j)["xyz"]).abs().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()


@torch.no_grad()
def boundary_lpips(model_i, T_i, model_j, data_j) -> float:
    """front render of the boundary frame from fragment k and from fragment k+1."""
    cam_i = data_j.cams[0].at_time(model_i.time(T_i - 1))
    a = model_i.render(cam_i, model_i.state(T_i - 1))["render"].clamp(0, 1)
    b = model_j.render(data_j.cam(0, 0), model_j.state(0))["render"].clamp(0, 1)
    return float(lpips_model("cuda")(a[None], b[None], normalize=True).item())


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 7)
    cfg = ctx.cfg
    frags = read_json(ctx.dirs.fragments_json)["fragments"]
    out = ctx.out / "merged"
    log_path = ctx.out / "merge_log.json"
    log = read_json(log_path) if log_path.exists() and not ctx.args.force else {}

    prev, _ = FragmentModel.load(ckpt_path(ctx.dirs, "s6", frags[0]["id"]), cfg)
    if not ctx.done(out / "frag_00.pth"):
        prev.save(out / "frag_00.pth")
    for k in range(1, len(frags)):
        path_j = out / f"frag_{frags[k]['id']:02d}.pth"
        T_i = frags[k - 1]["num_frames"]
        if ctx.done(path_j):
            prev, _ = FragmentModel.load(path_j, cfg)
            continue
        model_j, _ = FragmentModel.load(ckpt_path(ctx.dirs, "s6", frags[k]["id"]), cfg)
        data_j = FragmentData(ctx.dirs.frag_dir(6, frags[k]["id"]) / "data", cfg)
        before = boundary_lpips(prev, T_i, model_j, data_j)
        merge_pair(prev, T_i, model_j, data_j, cfg, seed=cfg.seed + k)
        after = boundary_lpips(prev, T_i, model_j, data_j)
        model_j.save(path_j)
        log[f"boundary_{k - 1}_{k}"] = {"lpips_before": before, "lpips_after": after}
        write_json(log, log_path)
        ctx.check(f"merge_continuity_{k - 1}_{k}", after < cfg.merge.continuity_lpips, lpips=after, before=before)
        prev = model_j
    print(f"[s7] {log}")


if __name__ == "__main__":
    main()
