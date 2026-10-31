"""Orchestrator (pure stdlib, any env): runs each stage in its conda env; stages talk only through files.

    python main.py --case NAME --start I_s.png --end I_e.png [--stages s0-s9] [--force] [--set k=v ...]
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent

STAGES = [
    ("s0", "wan", "preprocess", []),
    ("s1", "wan", "caption", []),
    ("s2", "wan", "video_generation", []),
    ("s3", "wan", "motion_progress", []),
    ("s4", "wan", "clip_hierarchy", []),
    ("s5", "era3d", "multiview_generation", []),
    ("s6", "4dgs", "reconstruction_4d", []),
    ("s7", "4dgs", "motion_merge", []),
    ("s8", "4dgs", "track_refinement", []),
    ("s8", "wan", "trajectory_smoothing", ["--part", "depth"]),
    ("s8", "4dgs", "trajectory_smoothing", ["--part", "smooth"]),
    ("s9", "4dgs", "render_eval", ["--part", "render"]),
    ("s9", "wan", "render_eval", ["--part", "analyze"]),
]
DEFAULT_ENVS = {"wan": "in24d_wan", "era3d": "in24d_era3d", "4dgs": "in24d_4dgs"}
SHARDED = {"s5", "s6"}


def select(spec: str):
    """'s0-s9', 's2,s5', 's6-' or 's4' -> ordered stage ids."""
    ids = [f"s{i}" for i in range(10)]
    chosen = []
    for part in spec.split(","):
        a, sep, b = part.strip().partition("-")
        lo = ids.index(a)
        hi = ids.index(b) if b else (len(ids) - 1 if sep else lo)
        chosen += ids[lo:hi + 1]
    return [s for s in ids if s in chosen]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--cfg", default=str(PKG / "configs" / "default.yaml"))
    p.add_argument("--start", help="start state image (Stage 0)")
    p.add_argument("--end", help="end state image (Stage 0)")
    p.add_argument("--prompt", help="motion prompt override (Stage 1)")
    p.add_argument("--stages", default="s0-s9")
    p.add_argument("--force", action="store_true", help="recompute selected stages even if outputs exist")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--no_conda", action="store_true", help="run every stage with the current python")
    p.add_argument("--gpus", default=os.environ.get("IN24D_GPUS"),
                   help="e.g. 0,1,2,3: run s5/s6 as one process per GPU (fragment id %% n)")
    p.add_argument("--dry_run", action="store_true")
    for key, name in DEFAULT_ENVS.items():
        p.add_argument(f"--env_{key}", default=os.environ.get(f"IN24D_ENV_{key.upper()}", name))
    args = p.parse_args()

    wanted = select(args.stages)
    if "s0" in wanted and not (args.start and args.end):
        p.error("--start and --end are required when running s0")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PKG), os.environ.get("PYTHONPATH")]))}

    for sid, env_key, module, extra in STAGES:
        if sid not in wanted:
            continue
        cmd = ["python", "-m", f"pipeline.{module}", "--cfg", args.cfg, "--case", args.case, *extra]
        if sid == "s0":
            cmd += ["--start", args.start, "--end", args.end]
        if sid == "s1" and args.prompt:
            cmd += ["--prompt", args.prompt]
        if args.force:
            cmd.append("--force")
        if args.set:
            cmd += ["--set", *args.set]
        if not args.no_conda:
            cmd = ["conda", "run", "--no-capture-output", "-n", getattr(args, f"env_{env_key}"), *cmd]
        gpus = args.gpus.split(",") if args.gpus and sid in SHARDED else [None]
        procs = []
        for i, gpu in enumerate(gpus):
            c, e = list(cmd), dict(env)
            if gpu is not None:
                c += ["--shard", str(i), "--num_shards", str(len(gpus))]
                e["CUDA_VISIBLE_DEVICES"] = gpu
            gpu_note = f" [GPU {gpu}]" if gpu is not None else ""
            print(f"\n=== {sid} [{env_key}]{gpu_note} {' '.join(shlex.quote(x) for x in c)}", flush=True)
            if not args.dry_run:
                procs.append(subprocess.Popen(c, cwd=PKG, env=e))
        code = next((c for c in [p.wait() for p in procs] if c != 0), 0)
        if code != 0:
            print(f"=== {sid} failed with exit code {code}; fix and re-run (finished stages are skipped)")
            sys.exit(code)
    print("\n=== done")


if __name__ == "__main__":
    main()
