"""Deformation + rasterization, split in two (`deform`, `rasterize`) so Stages 7-8 can edit the deformed state
and reuse one deformation for several views. The rasterizer is diff_gaussian_rasterization, unchanged.
"""
import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from . import backbone  # noqa: F401  (puts the 4DGS code on sys.path)
from utils.sh_utils import eval_sh  # noqa: E402


def deform(pc, time: float, stage: str = "fine", deformation=None) -> dict:
    """Deformed (pre-activation) Gaussian attributes at normalized time `time`.
    `deformation` overrides pc._deformation (used by the Stage 7 learnable copy)."""
    means3D, scales, rotations, opacity = pc.get_xyz, pc._scaling, pc._rotation, pc._opacity
    if stage == "coarse":
        return {"xyz": means3D, "scales": scales, "rotations": rotations, "opacity": opacity, "color_deform": None}
    net = deformation if deformation is not None else pc._deformation
    t = torch.full((means3D.shape[0], 1), float(time), device=means3D.device)
    sel = pc._deformation_table
    xyz_d, sc_d, rot_d, op_d, _, _, _, color_d = net(means3D[sel], scales[sel], rotations[sel], opacity[sel], t[sel])
    out = {"xyz": means3D.clone(), "scales": scales.clone(), "rotations": rotations.clone(),
           "opacity": opacity.clone(), "color_deform": None}
    out["xyz"][sel], out["scales"][sel], out["rotations"][sel], out["opacity"][sel] = xyz_d, sc_d, rot_d, op_d
    if color_d is not None:
        out["color_deform"] = torch.zeros(means3D.shape[0], color_d.shape[1], device=means3D.device)
        out["color_deform"][sel] = color_d
    return out


def rasterize(cam, pc, pipe, bg: torch.Tensor, state: dict, override_color=None, scaling_modifier=1.0) -> dict:
    screenspace_points = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:  # noqa: BLE001
        pass
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5), bg=bg,
        scale_modifier=scaling_modifier, viewmatrix=cam.world_view_transform.cuda(),
        projmatrix=cam.full_proj_transform.cuda(), sh_degree=pc.active_sh_degree,
        campos=cam.camera_center.cuda(), prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scales = pc.scaling_activation(state["scales"])
    rotations = pc.rotation_activation(state["rotations"])
    opacity = pc.opacity_activation(state["opacity"])
    shs, colors_precomp = None, None
    if override_color is not None:
        colors_precomp = override_color
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - cam.camera_center.cuda().repeat(pc.get_features.shape[0], 1)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp / dir_pp.norm(dim=1, keepdim=True))
        if state["color_deform"] is not None:
            w_dim = pc._deformation.deformation_net.color_deform_w[-2].out_features
            sh2rgb = sh2rgb * (state["color_deform"][:, :w_dim] + 1) + state["color_deform"][:, w_dim:]
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

    image, radii, depth = rasterizer(means3D=state["xyz"], means2D=screenspace_points, shs=shs,
                                     colors_precomp=colors_precomp, opacities=opacity, scales=scales,
                                     rotations=rotations, cov3D_precomp=None)
    return {"render": image, "viewspace_points": screenspace_points, "visibility_filter": radii > 0,
            "radii": radii, "depth": depth}


def render_alpha(cam, pc, pipe, state: dict) -> torch.Tensor:
    """Accumulated opacity [1, H, W]: white override color on a black background."""
    ones = torch.ones_like(pc.get_xyz)
    out = rasterize(cam, pc, pipe, torch.zeros(3, device="cuda"), state, override_color=ones)
    return out["render"][:1]


def rasterize_sets(cam, sets, pipe, bg: torch.Tensor) -> torch.Tensor:
    """One rasterization of several Gaussian sets [(pc, state, opacity_weight)] (boundary opacity cross-fade)."""
    xyz, sc, rot, op, col = [], [], [], [], []
    for pc, state, weight in sets:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - cam.camera_center.cuda()
        rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp / dir_pp.norm(dim=1, keepdim=True))
        if state["color_deform"] is not None:
            w_dim = pc._deformation.deformation_net.color_deform_w[-2].out_features
            rgb = rgb * (state["color_deform"][:, :w_dim] + 1) + state["color_deform"][:, w_dim:]
        xyz.append(state["xyz"])
        sc.append(pc.scaling_activation(state["scales"]))
        rot.append(pc.rotation_activation(state["rotations"]))
        op.append(pc.opacity_activation(state["opacity"]) * weight)
        col.append(torch.clamp_min(rgb + 0.5, 0.0))
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5), bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform.cuda(), projmatrix=cam.full_proj_transform.cuda(),
        sh_degree=0, campos=cam.camera_center.cuda(), prefiltered=False, debug=pipe.debug)
    xyz = torch.cat(xyz)
    image, _, _ = GaussianRasterizer(raster_settings=raster_settings)(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=None, colors_precomp=torch.cat(col),
        opacities=torch.cat(op), scales=torch.cat(sc), rotations=torch.cat(rot), cov3D_precomp=None)
    return image
