"""YAML config with attribute access and `key.sub=value` overrides."""
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access; nested dicts become Cfg too."""

    def __init__(self, data=None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Cfg) else v for k, v in self.items()}


def apply_override(cfg: Cfg, assignment: str):
    """Apply one `a.b.c=value` override; the value is parsed as YAML (so 3, 1e-3, [1,2], true work)."""
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"override must look like key=value, got '{assignment}'")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        if p not in node:
            raise KeyError(f"unknown config section '{p}' in override '{assignment}'")
        node = node[p]
    if leaf not in node:
        raise KeyError(f"unknown config key '{key}' in override '{assignment}'")
    value = yaml.safe_load(raw)
    if isinstance(value, str):          # YAML 1.1 reads "1e-4" (no dot) as a string
        try:
            value = float(value)
        except ValueError:
            pass
    node[leaf] = Cfg(value) if isinstance(value, dict) else value


def _merge(base: dict, over: dict, where=""):
    """Deep-merge a profile onto its base; keys must already exist in the base (catches typos)."""
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"profile sets unknown config key '{where}{k}'")
        if isinstance(v, dict) and isinstance(base[k], dict):
            _merge(base[k], v, f"{where}{k}.")
        else:
            base[k] = v
    return base


def _read(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    base = data.pop("_base_", None)
    return _merge(_read(path.parent / base), data) if base else data


def load_cfg(path, overrides=()) -> Cfg:
    """Load a yaml config; a `_base_: other.yaml` key makes it a profile layered on that file."""
    cfg = Cfg(_read(Path(path)))
    for o in overrides:
        apply_override(cfg, o)
    return cfg


def save_cfg(cfg: Cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
