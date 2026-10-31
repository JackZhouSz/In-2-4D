"""Motion progress curve of a clip from frame-to-frame feature distances.

Pure numpy/torch, no model code.
"""
import numpy as np
import torch
import torch.nn.functional as F



def foreground_tokens(feat: torch.Tensor, alpha: np.ndarray, thresh=0.5) -> torch.Tensor:
    """feat [C, h, w] + frame alpha [H, W] -> L2-normalized foreground tokens [n, C].

    The alpha is area-downsampled to the feature grid; white background tokens would dominate similarity.
    """
    C, h, w = feat.shape
    a = torch.from_numpy(alpha.astype(np.float32))[None, None]
    a = F.interpolate(a, size=(h, w), mode="area")[0, 0].to(feat.device)
    tokens = feat.reshape(C, -1).T[a.flatten() > thresh]
    if tokens.shape[0] == 0:
        tokens = feat.reshape(C, -1).T
    return F.normalize(tokens.float(), dim=-1)


def heat_similarity(Fi: torch.Tensor, Fj: torch.Tensor) -> float:
    """s_ij = 0.5 (H_ij + H_ji), H_ij = mean_p max_q <F_i[p], F_j[q]> (symmetrized)."""
    sim = Fi @ Fj.T
    return 0.5 * float(sim.max(1).values.mean() + sim.max(0).values.mean())


def frame_pairs(T: int, window: int) -> np.ndarray:
    """All (i, j) with 0 < i - j <= window."""
    return np.array([(i, j) for i in range(T) for j in range(max(0, i - window), i)], dtype=np.int64)


def pair_distances(tokens, window: int, p: float, global_emb=None, w_local=1.0):
    """Returns pairs (M, 2), similarities (M,), distances (M,).

    d = arccos(clip(s))**p; with `global_emb` (L2-normalized SigLIP [T, D]) and w_local < 1 the
    distance is fused: w_local * d_local + (1 - w_local) * arccos(z_i . z_j).
    """
    pairs = frame_pairs(len(tokens), window)
    sims = np.array([heat_similarity(tokens[i], tokens[j]) for i, j in pairs])
    d = np.arccos(np.clip(sims, -1, 1)) ** p
    if global_emb is not None and w_local < 1.0:
        z = np.asarray(global_emb, dtype=np.float64)
        d_sig = np.arccos(np.clip((z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1), -1, 1))
        d = w_local * d + (1 - w_local) * d_sig
    return pairs, sims, d



def solve(T: int, pairs: np.ndarray, d: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    """Closed-form weighted ridge least squares S_i - S_j ~ d_ij, min-max normalized."""
    M = len(pairs)
    A = np.zeros((M, T))
    A[np.arange(M), pairs[:, 0]] = 1.0
    A[np.arange(M), pairs[:, 1]] = -1.0
    gap = (pairs[:, 0] - pairs[:, 1]).astype(np.float64)
    w = np.exp(-gap ** 2 / (2 * sigma ** 2))
    S = np.linalg.solve(A.T @ (w[:, None] * A) + lam * np.eye(T), A.T @ (w * d))
    rng = S.max() - S.min()
    return (S - S.min()) / rng if rng > 0 else np.linspace(0, 1, T)


def monotonize(S: np.ndarray) -> np.ndarray:
    """Required before any inversion: cumulative max, tie-break ramp, endpoints pinned to 0 and 1."""
    S = np.maximum.accumulate(np.asarray(S, dtype=np.float64)) + 1e-6 * np.arange(len(S))
    return (S - S[0]) / (S[-1] - S[0])


def inverse(S_mono: np.ndarray, y) -> np.ndarray:
    """Piecewise-linear inverse of a strictly increasing S: frame position k with S(k) = y."""
    return np.interp(y, S_mono, np.arange(len(S_mono), dtype=np.float64))


def linearity(S: np.ndarray) -> float:
    """lin = 1 - max_k |S[k] - k/(T-1)|."""
    ramp = np.linspace(0, 1, len(S))
    return float(1.0 - np.abs(S - ramp).max())


def r2_vs_ramp(S: np.ndarray) -> float:
    ramp = np.linspace(0, 1, len(S))
    ss_tot = ((S - S.mean()) ** 2).sum()
    return float(1.0 - ((S - ramp) ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0



def _segment_sse_table(y: np.ndarray):
    """Returns sse(i, j) for the least-squares line through points i..j (inclusive), O(1) via prefix sums."""
    x = np.arange(len(y), dtype=np.float64)
    pre = [np.concatenate([[0.0], np.cumsum(v)]) for v in (np.ones_like(x), x, y, x * x, x * y, y * y)]

    def sse(i, j):
        n, sx, sy, sxx, sxy, syy = (p[j + 1] - p[i] for p in pre)
        vx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        res = syy - sy * sy / n - np.where(vx > 0, cxy * cxy / np.where(vx > 0, vx, 1), 0)
        return np.maximum(res, 0.0)

    return sse


def segmented_least_squares(S: np.ndarray, c_pen: float):
    """Optimal piecewise-linear fit with shared endpoints.

    OPT[0] = 0;  OPT[j] = min_{i<j} OPT[i] + SSE(i..j) + C,  C = c_pen / T.
    Returns boundaries 0 = b_0 < b_1 < ... < b_K = T-1.
    """
    y = np.asarray(S, dtype=np.float64)
    T = len(y)
    C = c_pen / T
    sse = _segment_sse_table(y)
    opt = np.zeros(T)
    arg = np.zeros(T, dtype=np.int64)
    for j in range(1, T):
        i = np.arange(j)
        cost = opt[i] + sse(i, np.full_like(i, j)) + C
        arg[j] = int(np.argmin(cost))
        opt[j] = cost[arg[j]]
    bounds = [T - 1]
    while bounds[-1] > 0:
        bounds.append(int(arg[bounds[-1]]))
    return bounds[::-1]


def segment_stats(S: np.ndarray, boundaries):
    slopes, dS = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        x = np.arange(a, b + 1)
        slopes.append(float(np.polyfit(x, S[a:b + 1], 1)[0]))
        dS.append(float(S[b] - S[a]))
    return slopes, dS


def round_to_4n1(x: float) -> int:
    return int(4 * round((x - 1) / 4) + 1)
