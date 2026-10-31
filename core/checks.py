"""Required checks. A failing check stops the stage unless `checks.strict` is false."""
import json
from pathlib import Path


class CheckFailed(RuntimeError):
    pass


class CheckLog:
    """Records every check of a stage to <stage_dir>/checks.json."""

    def __init__(self, stage_dir, strict=True, filename="checks.json"):
        self.path = Path(stage_dir) / filename
        self.strict = strict
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __call__(self, name: str, passed: bool, **details):
        self.records[name] = {"passed": bool(passed), **{k: _jsonable(v) for k, v in details.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2))
        status = "PASS" if passed else "FAIL"
        print(f"[check] {status} {name} {details}")
        if not passed and self.strict:
            raise CheckFailed(f"check '{name}' failed: {details} (see {self.path}; set checks.strict=false to continue)")
        return passed


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return float(v) if hasattr(v, "__float__") else str(v)
