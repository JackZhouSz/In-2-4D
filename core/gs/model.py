"""One fragment's 4D Gaussians: canonical 3DGS + deformation field (+ the Stage 7 merge on overlap frames)."""
import torch

from .backbone import gs_hparams, empty_gaussians
from .render import deform, rasterize, render_alpha


class MergeInfo:
    """Motion merge on the overlap: xyz = x_c + lam * dPhi_i + (1 - lam) * dPhi*_j.

    dPhi_i is fragment i's frozen motion carried onto fragment j's Gaussians through nearest-neighbour
    correspondence at the shared keyframe ([overlap, N, 3], relative to j's canonical centers)."""

    def __init__(self, lam: float, delta_i: torch.Tensor):
        self.lam, self.delta_i = lam, delta_i

    def to_dict(self):
        return {"lam": self.lam, "delta_i": self.delta_i.cpu()}

    @classmethod
    def from_dict(cls, d):
        return None if d is None else cls(d["lam"], d["delta_i"].cuda())


class FragmentModel:
    def __init__(self, gaussians, hyper, opt, pipe, T_train: int, merge: MergeInfo = None):
        self.g, self.hyper, self.opt, self.pipe = gaussians, hyper, opt, pipe
        self.T_train, self.merge = T_train, merge
        self.white = torch.ones(3, device="cuda")

    def time(self, t: float) -> float:
        return t / (self.T_train - 1)

    def state(self, t: float, stage="fine", deformation=None) -> dict:
        st = deform(self.g, self.time(t), stage, deformation)
        m = self.merge
        if m is not None and stage == "fine" and float(t).is_integer() and int(t) < m.delta_i.shape[0]:
            x_c = self.g.get_xyz
            st["xyz"] = x_c + m.lam * m.delta_i[int(t)] + (1 - m.lam) * (st["xyz"] - x_c)
        return st

    def render(self, cam, state, bg=None) -> dict:
        return rasterize(cam, self.g, self.pipe, self.white if bg is None else bg, state)

    def render_alpha(self, cam, state) -> torch.Tensor:
        return render_alpha(cam, self.g, self.pipe, state)

    def save(self, path, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture": self.g.capture(), "T_train": self.T_train,
                    "merge": None if self.merge is None else self.merge.to_dict(), **extra}, path)

    @classmethod
    def load(cls, path, cfg):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        hyper, opt, pipe = gs_hparams(cfg, ck["T_train"])
        g = empty_gaussians(hyper, round((ck["capture"][5].shape[1] + 1) ** 0.5) - 1)
        a = float(cfg.gs.aabb)
        g._deformation.deformation_net.grid.set_aabb([a, a, a], [-a, -a, -a])
        g._deformation = g._deformation.to("cuda")
        g.restore(ck["capture"], opt)
        g.active_sh_degree = ck["capture"][0]
        return cls(g, hyper, opt, pipe, ck["T_train"], MergeInfo.from_dict(ck["merge"])), ck

