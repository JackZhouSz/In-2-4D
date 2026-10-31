"""Reconstruction objective: reference, mask, rigidity and multi-view RGB terms."""
import random

import torch
import torch.nn.functional as F

from core.metrics import lpips_model


def l_ref(img, gt):
    """MSE between the front-view render and the Wan frame."""
    return F.mse_loss(img, gt)


def l_mask(alpha, gt_alpha, kind="l1"):
    if kind == "bce":
        return F.binary_cross_entropy(alpha.clamp(1e-5, 1 - 1e-5), gt_alpha)
    return F.l1_loss(alpha, gt_alpha)


def l_mv(img, gt, w_lpips):
    """RGB L1 + w * LPIPS on a non-reference Era3D view."""
    loss = F.l1_loss(img, gt)
    if w_lpips > 0:
        loss = loss + w_lpips * lpips_model("cuda", "vgg")(img[None], gt[None], normalize=True).mean()
    return loss


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """[N, 3] -> [N, k] nearest neighbours (self excluded)."""
    try:
        from pytorch3d.ops import knn_points
        return knn_points(xyz[None], xyz[None], K=k + 1)[1][0, :, 1:]
    except ImportError:
        out = []
        for chunk in torch.split(xyz, 4096):
            out.append(torch.cdist(chunk, xyz).topk(k + 1, largest=False).indices[:, 1:])
        return torch.cat(out)


class RigidLoss:
    """Rigidity: || d(mu_i^c, mu_j^c) - d(mu_i^t, mu_j^t) ||_1 over K canonical nearest neighbours.

    Neighbourhoods are recomputed whenever densification/pruning changes the Gaussian count."""

    def __init__(self, k=16, anchors=4096):
        self.k, self.anchors = k, anchors
        self.n, self.knn = -1, None

    def __call__(self, canonical_xyz: torch.Tensor, deformed_xyz: torch.Tensor, gen: torch.Generator = None):
        xyz_c = canonical_xyz.detach()
        if xyz_c.shape[0] != self.n:
            self.n, self.knn = xyz_c.shape[0], knn_indices(xyz_c, self.k)
        idx = torch.randperm(self.n, device=xyz_c.device, generator=gen)[:self.anchors]
        nb = self.knn[idx]
        d_c = (xyz_c[idx, None] - xyz_c[nb]).norm(dim=-1)
        d_t = (deformed_xyz[idx, None] - deformed_xyz[nb]).norm(dim=-1)
        return (d_c - d_t).abs().mean()


def low_res(cam, gt, rng: random.Random):
    """Multiscale downsample augmentation."""
    r = rng.choice([1, 2, 3, 4])
    if r == 1:
        return cam, gt
    h, w = cam.image_height // r, cam.image_width // r
    return cam.resized(w, h), F.interpolate(gt[None], (h, w), mode="bilinear", align_corners=False)[0]
