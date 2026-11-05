#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Export an original successful Task and browse it without changing its split.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: run_timing_data_preview.sh ORIGINAL_TASK_DIR NEW_RUN_DIR" >&2
  exit 2
fi
cd "${PAOS_ROOT}"
PAOS_TIMING_TASK="$1"
PAOS_TIMING_OUT="$2"
mkdir -- "$PAOS_TIMING_OUT"
mkdir -p -- "$PAOS_TIMING_OUT/data/final"
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"${PIPER_SIM_ROOT}/.venv/bin/python" \
  scripts/prepare_joint_dataset.py --task-dir "$PAOS_TIMING_TASK" \
  --out "$PAOS_TIMING_OUT/data/final/prepared"
"${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" \
  scripts/export_joint_lerobot.py --prepared "$PAOS_TIMING_OUT/data/final/prepared" \
  --out "$PAOS_TIMING_OUT/data/final/lerobot-0.4.4" --repo-id local/phyagentos-timing-review
"${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" \
  scripts/prepare_sim_browser_preview.py --run "$PAOS_TIMING_OUT" \
  --out "$PAOS_TIMING_OUT/data/browser-preview"
