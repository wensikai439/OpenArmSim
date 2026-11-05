#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Offline data export only. Supply original succeeded Task directory and a new output directory.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: run_binary_data_view.sh ORIGINAL_TASK_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi
cd "${PAOS_ROOT}"
PAOS_DATA_TASK="$1"
PAOS_DATA_OUT="$2"
mkdir -- "$PAOS_DATA_OUT"
MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "${PIPER_SIM_ROOT}/.venv/bin/python" \
  scripts/prepare_joint_dataset.py --task-dir "$PAOS_DATA_TASK" --out "$PAOS_DATA_OUT/prepared"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" \
  scripts/export_joint_lerobot.py --prepared "$PAOS_DATA_OUT/prepared" \
  --out "$PAOS_DATA_OUT/lerobot-0.4.4" --repo-id local/phyagentos-piper-binary-executed-v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" \
  scripts/align_act_domains.py --sim-root "$PAOS_DATA_OUT/lerobot-0.4.4" \
  --kinematics-audit runs/domain-alignment-20261011T075240Z/kinematics-audit.json \
  --out "$PAOS_DATA_OUT/view-20hz"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" \
  scripts/prepare_binary_gripper_view.py --view-root "$PAOS_DATA_OUT/view-20hz" \
  --sim-root "$PAOS_DATA_OUT/lerobot-0.4.4" --out "$PAOS_DATA_OUT/binary-view"
