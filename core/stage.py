"""Shared CLI and resumability for `python -m pipeline.<stage> --cfg <yaml> --case <name>`."""
import argparse
import random
from pathlib import Path

import numpy as np

from .checks import CheckLog
from .config import load_cfg, save_cfg
from .paths import CaseDirs


def stage_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cfg", required=True, help="config yaml")
    p.add_argument("--case", required=True, help="case name (outputs/<case>/)")
    p.add_argument("--force", action="store_true", help="recompute even if the output artifact exists")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--shard", type=int, default=0, help="this process handles fragments with id %% num_shards == shard")
    p.add_argument("--num_shards", type=int, default=1)
    return p


class StageContext:
    def __init__(self, args, stage: int, subdir: str = None):
        self.args = args
        self.cfg = load_cfg(args.cfg, args.set)
        self.dirs = CaseDirs(self.cfg.paths.outputs, args.case)
        self.out = getattr(self.dirs, f"s{stage}") if subdir is None else self.dirs.root / subdir
        self.out.mkdir(parents=True, exist_ok=True)
        self.shard, self.num_shards = args.shard, args.num_shards
        if self.shard == 0:
            save_cfg(self.cfg, self.out / "config_used.yaml")
        checks = "checks.json" if self.num_shards == 1 else f"checks_shard{self.shard}.json"
        self.check = CheckLog(self.out, strict=self.cfg.checks.strict, filename=checks)
        seed_everything(self.cfg.seed)
        use_aux_device(self.cfg.aux_device)

    def mine(self, frag_id: int) -> bool:
        return frag_id % self.num_shards == self.shard

    def done(self, artifact: Path) -> bool:
        """True (and logs a skip) when the stage output exists and --force is not set."""
        if Path(artifact).exists() and not self.args.force:
            print(f"[skip] {artifact} exists (use --force to recompute)")
            return True
        return False


def use_aux_device(index):
    """Make plain "cuda" (diffusion features, LPIPS, SAM2, ...) mean cuda:<index> when Wan spans several GPUs.
    Ignored when that GPU is not visible (e.g. a sharded process that sees a single GPU)."""
    if index is None:
        return
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > index:
        torch.cuda.set_device(index)


def seed_everything(seed: int):
    """Global seeding; stages additionally create explicit generators where the seed is part of the method."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
