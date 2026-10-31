"""RoPE retiming for Wan2.2.

Wan's temporal RoPE uses 22 complex bands with theta_b = 10000^(-2b/44): band 0 is the HIGHEST frequency.
The schedule warps low frequencies strongly, so alpha is indexed by frequency rank r_b = 21 - b.

`install()` replaces `rope_apply` in both wan.modules.model and wan.distributed.sequence_parallel
(module-global lookup inside the attention forward makes attribute assignment sufficient). With
RETIME.enabled False the originals are called unchanged.
"""
import math

import numpy as np
import torch

N_BANDS = 22


def gamma(t_norm: float) -> float:
    """Timestep modulation, t_norm in [0, 1] with 1 = max noise."""
    return (math.exp(3 * t_norm) - 1) / (math.exp(3) - 1)


def band_alphas(alpha_low: float, alpha_high: float, kappa: float, n_bands: int = N_BANDS) -> np.ndarray:
    b = np.arange(n_bands)
    r = (n_bands - 1) - b
    return alpha_high + (alpha_low - alpha_high) * np.exp(-kappa * r / (n_bands - 1))


def latent_centers(n_latents: int) -> np.ndarray:
    """Frame center of each latent (0-indexed): latent 0 <-> frame 0, latent i>=1 <-> 4i - 1.5."""
    c = 4.0 * np.arange(n_latents) - 1.5
    c[0] = 0.0
    return c


def _interp_extrap(x, xp, fp):
    """np.interp with linear extrapolation on both sides."""
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


class RetimeState:
    """Per-band frame-unit positions Pf [T, 22] plus the current diffusion time."""

    def __init__(self):
        self.enabled = False
        self.Pf = None
        self.t_norm = 1.0

    def reset(self, num_frames: int, n_bands: int = N_BANDS):
        self.Pf = np.repeat(np.arange(num_frames, dtype=np.float64)[:, None], n_bands, 1)

    def set_timestep(self, t_norm: float):
        self.t_norm = float(t_norm)

    def update(self, delta: np.ndarray, alphas: np.ndarray):
        """Pf[:, b] += alpha_b * delta."""
        self.Pf = self.Pf + delta[:, None] * alphas[None, :]

    def latent_positions(self, n_latents: int) -> torch.Tensor:
        """[n_latents, 22] float64 latent-unit positions for the current timestep."""
        T, n_bands = self.Pf.shape
        assert (T - 1) // 4 + 1 == n_latents, f"Pf has {T} frames but the latent grid has {n_latents}"
        k = np.arange(T, dtype=np.float64)
        eff = k[:, None] + gamma(self.t_norm) * (self.Pf - k[:, None])
        c = latent_centers(n_latents)
        lat = np.arange(n_latents, dtype=np.float64)
        out = np.empty((n_latents, n_bands))
        for b in range(n_bands):
            sampled = np.interp(c, k, eff[:, b])
            out[:, b] = _interp_extrap(sampled, c, lat)
        return torch.from_numpy(out)


RETIME = RetimeState()
_ORIG = {}


def _temporal_freqs(x, f, ct):
    theta = 1.0 / torch.pow(10000, torch.arange(0, 2 * ct, 2, dtype=torch.float64, device=x.device) / (2 * ct))
    P = RETIME.latent_positions(f).to(x.device)
    assert P.shape[1] == ct, f"RETIME has {P.shape[1]} bands, the model's temporal RoPE has {ct}"
    return torch.polar(torch.ones_like(P), P * theta[None, :])


def _freqs_i(x, f, h, w, freqs):
    c = x.size(3) // 2
    ct = c - 2 * (c // 3)
    _, f_h, f_w = freqs.split([ct, c // 3, c // 3], dim=1)
    f_t = _temporal_freqs(x, f, ct)
    return torch.cat([f_t.view(f, 1, 1, -1).expand(f, h, w, -1),
                      f_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                      f_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_retimed(x, grid_sizes, freqs):
    """Drop-in for wan.modules.model.rope_apply; only the temporal factor changes."""
    if not RETIME.enabled:
        return _ORIG["model"](x, grid_sizes, freqs)
    n = x.size(2)
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        x_i = torch.view_as_real(x_i * _freqs_i(x, f, h, w, freqs)).flatten(2)
        out.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(out).float()


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_retimed_sp(x, grid_sizes, freqs):
    """Drop-in for wan.distributed.sequence_parallel.rope_apply (pads freqs and slices this rank's chunk)."""
    if not RETIME.enabled:
        return _ORIG["sp"](x, grid_sizes, freqs)
    from wan.distributed.sequence_parallel import pad_freqs
    from wan.distributed.util import get_rank, get_world_size

    s, n = x.size(1), x.size(2)
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = pad_freqs(_freqs_i(x, f, h, w, freqs), s * get_world_size())
        rank = get_rank()
        x_i = torch.view_as_real(x_i * freqs_i[rank * s:(rank + 1) * s]).flatten(2)
        out.append(torch.cat([x_i, x[i, s:]]))
    return torch.stack(out).float()


def install():
    import wan.distributed.sequence_parallel as sp
    import wan.modules.model as model

    if "model" not in _ORIG:
        _ORIG["model"], _ORIG["sp"] = model.rope_apply, sp.rope_apply
    model.rope_apply = rope_apply_retimed
    sp.rope_apply = rope_apply_retimed_sp
