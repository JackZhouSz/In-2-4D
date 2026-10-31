"""Locations of the third-party submodules and of each case's on-disk artifacts."""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
THIRD_PARTY = PKG / "third_party"

WAN = THIRD_PARTY / "Wan2.2"
ERA3D = THIRD_PARTY / "Era3D"
GS4D = THIRD_PARTY / "EG4D"
LTX = THIRD_PARTY / "LTX-Video"

ERA3D_PROMPT_EMBEDS = ERA3D / "mvdiffusion" / "data" / "fixed_prompt_embeds_6view"


def use_repo(repo: Path):
    """Make a third-party submodule importable without editing it."""
    p = str(repo)
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve(path) -> Path:
    """Paths in the config are relative to the repo root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else PKG / p


STEP_DIRS = ("video/inputs", "video/caption", "video/generated", "video/motion_progress", "video/clips",
             "multiview", "4d/reconstruction", "4d/merge", "4d/refinement", "4d")


class CaseDirs:
    """All artifact paths for one case: outputs/<case>/{video,multiview,4d,novel_view_videos}/..."""

    def __init__(self, outputs_root, case: str):
        self.root = resolve(outputs_root) / case
        for i, d in enumerate(STEP_DIRS):
            setattr(self, f"s{i}", self.root / d)
        self.videos = self.root / "novel_view_videos"

    @property
    def framing_json(self):
        return self.s0 / "framing.json"

    @property
    def fragments_json(self):
        return self.s4 / "fragments.json"

    def frag_dir(self, stage: int, frag_id: int) -> Path:
        return getattr(self, f"s{stage}") / f"frag_{frag_id:02d}"
