"""The whole case as a sequence of fragment models (fragments.json is the single source of truth)."""
from core.io import read_json

from .data import FragmentData
from .model import FragmentModel


def ckpt_path(dirs, stage: str, frag_id: int):
    return {"s6": dirs.frag_dir(6, frag_id) / "ckpt" / "dynamic.pth",
            "s7": dirs.s7 / "merged" / f"frag_{frag_id:02d}.pth",
            "s8": dirs.s8 / "refined" / f"frag_{frag_id:02d}.pth"}[stage]


def latest_stage(dirs, frag_id: int) -> str:
    for stage in ("s8", "s7", "s6"):
        if ckpt_path(dirs, stage, frag_id).exists():
            return stage
    raise FileNotFoundError(f"no checkpoint for fragment {frag_id}")


class Sequence:
    def __init__(self, dirs, cfg, stage: str = None, load_data=True):
        self.meta = read_json(dirs.fragments_json)
        self.frags = self.meta["fragments"]
        self.total = self.meta["total_frames"]
        self.models, self.data, self.stages = [], [], []
        for f in self.frags:
            st = stage or latest_stage(dirs, f["id"])
            self.models.append(FragmentModel.load(ckpt_path(dirs, st, f["id"]), cfg)[0])
            self.data.append(FragmentData(dirs.frag_dir(6, f["id"]) / "data", cfg) if load_data else None)
            self.stages.append(st)

    def locate(self, p: float):
        """Global (possibly fractional) frame -> (fragment index, local frame). Hard switch at boundaries:
        fragment k renders [start_k, end_k), the last fragment also renders its end frame."""
        for k, f in enumerate(self.frags):
            if p < f["global_end"] or k == len(self.frags) - 1:
                return k, max(0.0, p - f["global_start"])
        raise ValueError(p)
