#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export IN24D_GPUS="${IN24D_GPUS-0,1,2,3}"
CFG="${IN24D_CFG:-configs/t4x4.yaml}"
cd "$HERE"

usage() {
  cat <<'EOF'
Inbetweening from two single-view images to 4D.

  ./run.sh setup [wan|era3d|4dgs]                create the conda envs (once)
  ./run.sh CASE START.png END.png [options]      full pipeline, steps s0-s9
  ./run.sh resume CASE [options]                 continue a case (finished steps are skipped)
  ./run.sh video CASE [options]                  novel-view videos of a trained case (render_video.py --help)

Options are passed to main.py, e.g.
  --stages s4-s6            run a subset            --force          recompute selected steps
  --prompt "a flower ..."   custom motion prompt    --set wan.steps=30 retime.enabled=true
  --no_conda                use the current python  --dry_run        print the commands only

Machine profile: IN24D_CFG (default configs/t4x4.yaml = 4x T4) and IN24D_GPUS (default 0,1,2,3).
On an 80 GB GPU use: IN24D_CFG=configs/default.yaml IN24D_GPUS= ./run.sh ...
Outputs: outputs/CASE/{video,multiview,4d,novel_view_videos} (see readme.md).
EOF
}

case "${1:-}" in
  ""|-h|--help) usage ;;
  setup)  shift; bash "$HERE/setup_envs.sh" "$@" ;;
  resume) shift; case_name="$1"; shift; python "$HERE/main.py" --cfg "$CFG" --case "$case_name" --stages s1-s9 "$@" ;;
  video)  shift; case_name="$1"; shift
          conda run --no-capture-output -n in24d_4dgs python render_video.py --cfg "$CFG" --case "$case_name" "$@" ;;
  *)
    [[ $# -ge 3 ]] || { usage; exit 1; }
    case_name="$1"; start="$2"; end="$3"; shift 3
    python "$HERE/main.py" --cfg "$CFG" --case "$case_name" --start "$start" --end "$end" "$@"
    ;;
esac
