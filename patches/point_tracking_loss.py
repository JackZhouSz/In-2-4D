"""2D point-tracking loss on projected Gaussians, for (near-)orthographic cameras.

Details:
  * D_i = D_j = 1: under our (near-)orthographic cameras pixel distance is already metric, so the
    depth normalization is dropped; distances are normalized by image width instead.
  * per_segment=False (single object).
  * KNN: frnn if installed, else pytorch3d knn_points, else torch.cdist; neighbours beyond `r` px are dropped.
  * torch_scatter.scatter_sum -> index_add_ (no extra dependency).
Influence weights come from accumulated opacity with an isotropic conical opacity.
"""
import torch
import torch.nn as nn

from core.cameras import project_points


def knn_2d(query: torch.Tensor, points: torch.Tensor, K: int, r: float) -> torch.Tensor:
    """[M, 2], [N, 2] -> [M, K] indices of the K nearest points within radius r (-1 = none)."""
    K = min(K, points.shape[0])
    try:
        import frnn
        lengths1 = torch.tensor([query.shape[0]], device=query.device)
        lengths2 = torch.tensor([points.shape[0]], device=query.device)
        _, idx, _, _ = frnn.frnn_grid_points(query[None], points[None], lengths1, lengths2, K, r,
                                             grid=None, return_nn=False, return_sorted=True)
        return idx[0]
    except ImportError:
        pass
    try:
        from pytorch3d.ops import knn_points
        d2, idx, _ = knn_points(query[None].float(), points[None].float(), K=K)
        d2, idx = d2[0], idx[0]
    except ImportError:
        d2_all = torch.cdist(query.float(), points.float()) ** 2
        d2, idx = torch.topk(d2_all, K, dim=1, largest=False)
    idx = idx.clone()
    idx[d2 > r * r] = -1
    return idx


def conical_opacity(depths, opacity, scales, fx, fy):
    """Isotropic 2D conic from scales[:, 0]."""
    cov_1d = scales[:, 0] ** 2
    conical_x = cov_1d * (fx / (depths + 0.0001)) ** 2 + 0.3
    conical_y = cov_1d * (fy / (depths + 0.0001)) ** 2 + 0.3
    det = conical_x * conical_y
    conical_x, conical_y = conical_x / det, conical_y / det
    return torch.stack([conical_x, torch.zeros_like(conical_x), conical_y, opacity.flatten()], dim=-1)


class TrackingLoss(nn.Module):
    def __init__(self, K=32, r=8.0):
        super().__init__()
        self.K, self.r = K, r

    def forward(self, src_camera, target_camera, src_means3D, target_means3D, opacity, scales,
                src_particles, target_particles):
        """src/target_means3D: [N, 3] Gaussian centers at times i and j (source side is detached).
        opacity [N, 1] and scales [N, 3] are activated values. particles: [M, 2] pixel tracks visible at i and j."""
        W = float(target_camera.image_width)
        means2D_target, _ = project_points(target_means3D, target_camera)
        with torch.no_grad():
            means2D_src, depths_src = project_points(src_means3D.detach(), src_camera)
            conic = conical_opacity(depths_src, opacity.detach(), scales.detach(), src_camera.fx, src_camera.fy)

        knn = knn_2d(src_particles, means2D_src, self.K, self.r)
        particles_idxs, knn_idxs = self._gather_knn(knn, depths_src)
        if knn_idxs.numel() == 0:
            return means2D_target.sum() * 0.0

        dists = means2D_src[knn_idxs] - src_particles[particles_idxs]
        influences, dists = self._accumulate_opacity_into_influence(particles_idxs, dists, conic[knn_idxs])
        dists_source = torch.norm(dists, dim=-1)
        dists_target = torch.norm(means2D_target[knn_idxs] - target_particles[particles_idxs], dim=-1)

        errors = torch.abs(dists_target - dists_source) * influences / W
        per_particle = torch.zeros(src_particles.shape[0], device=errors.device, dtype=errors.dtype)
        per_particle = per_particle.index_add(0, particles_idxs, errors)
        return per_particle[torch.unique(particles_idxs)].mean()

    @staticmethod
    def _gather_knn(knn: torch.Tensor, depths_src: torch.Tensor):
        """Depth-order (front to back) each particle's valid neighbours; flatten particle-major."""
        valid = knn >= 0
        d = torch.where(valid, depths_src[knn.clamp(min=0)], torch.full_like(knn, float("inf"), dtype=depths_src.dtype))
        order = torch.argsort(d, dim=1)
        knn, valid = torch.gather(knn, 1, order), torch.gather(valid, 1, order)
        valid = valid & (depths_src[knn.clamp(min=0)] > 0.01)
        M, K = knn.shape
        particles_idxs = torch.arange(M, device=knn.device).repeat_interleave(K)[valid.flatten()]
        return particles_idxs, knn[valid]

    @staticmethod
    def _accumulate_opacity_into_influence(batch, dists, conical_opacity):
        power = -0.5 * (conical_opacity[:, 0] + dists[:, 0] ** 2 + conical_opacity[:, 2] * dists[:, 1] ** 2)
        power = power - conical_opacity[:, 1] * dists[:, 0] * dists[:, 1]
        power = torch.clip(power, max=0)
        alpha = torch.clip(conical_opacity[:, 3] * torch.exp(power), 0, 0.99)

        _, track_idxs_local = torch.unique(batch, return_inverse=True)
        ptr = torch.where((track_idxs_local[1:] - track_idxs_local[:-1]) > 0)[0] + 1
        log_transparencies = torch.log2(1 - alpha)
        log_agg_transparencies = torch.cumsum(log_transparencies, dim=0)
        diff_corrections = torch.cat([torch.zeros(1).to(log_transparencies), log_agg_transparencies[ptr - 1]])
        diff_corrections = diff_corrections[track_idxs_local]
        log_agg_transparencies = log_agg_transparencies - diff_corrections
        agg_transparencies = torch.exp2(log_agg_transparencies)

        ptr_wzero = torch.cat([torch.zeros(1).to(ptr), ptr])
        init_transparencies = agg_transparencies[ptr_wzero]
        init_transparencies = init_transparencies[track_idxs_local]
        agg_transparencies = agg_transparencies / init_transparencies

        influences = alpha * agg_transparencies
        influences = influences.detach().clone()
        return influences, dists
