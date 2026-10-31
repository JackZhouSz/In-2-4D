"""Bridge to the 4DGS code in third_party: hyperparameters and GaussianModel construction."""
import copy
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from core.paths import GS4D, use_repo

use_repo(GS4D)

from arguments import ModelHiddenParams, OptimizationParams, PipelineParams  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.graphics_utils import BasicPointCloud  # noqa: E402
from utils.sh_utils import SH2RGB  # noqa: E402

def gs_hparams(cfg, time_resolution: int):
    """(hyper, opt, pipe): argparse defaults, then gs.hparams, then the time resolution of this fragment."""
    parser = ArgumentParser()
    groups = (ModelHiddenParams(parser), OptimizationParams(parser), PipelineParams(parser))
    args = parser.parse_args([])
    for k, v in (cfg.gs.get("hparams") or {}).items():
        assert hasattr(args, k), f"gs.hparams: unknown parameter {k}"
        setattr(args, k, v)
    args.kplanes_config = copy.deepcopy(args.kplanes_config)
    args.kplanes_config["resolution"] = list(args.kplanes_config["resolution"][:3]) + [int(time_resolution)]
    args.predict_color_deform = True
    args.render_ref_time = 0.0
    return tuple(g.extract(args) for g in groups)


def spatial_extent(cfg) -> float:
    """Position/deformation lrs scale with the camera extent (1.1 x max camera distance). The cameras sit at
    D=200 only to fake an ortho projection, so use the physical orbit radius instead."""
    return cfg.gs.orbit_radius * 1.1


def new_gaussians(hyper, points: np.ndarray, colors: np.ndarray, cfg) -> GaussianModel:
    g = GaussianModel(cfg.gs.sh_degree, hyper)
    a = float(cfg.gs.aabb)
    g._deformation.deformation_net.grid.set_aabb([a, a, a], [-a, -a, -a])   # before training_setup (new Parameter)
    g.create_from_pcd(BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points)),
                      spatial_extent(cfg), 0)
    return g


def sphere_points(n: int, radius: float, seed: int):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    pts = d / np.linalg.norm(d, axis=1, keepdims=True) * radius * rng.random((n, 1)) ** (1 / 3)
    return pts.astype(np.float32), SH2RGB(rng.random((n, 3)) / 255.0).astype(np.float32)


def empty_gaussians(hyper, sh_degree: int) -> GaussianModel:
    g = GaussianModel(sh_degree, hyper)
    g._deformation = g._deformation.to("cuda")
    return g


def as_namespace(group) -> Namespace:
    return Namespace(**vars(group))


def freeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module
