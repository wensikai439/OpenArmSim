#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Export a formally successful 30 Hz Task; source data and old exports stay in place.
set -euo pipefail
cd "${PAOS_ROOT}"
task_dir="${1:?Provide a succeeded Task output directory}"
out_run="${2:?Provide a new output run directory}"
repo_id="${3:-local/phyagentos_wrist_top30_20261015}"
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PAOS_CONTROL_HZ=30
export PAOS_FROZEN_SIM_SOURCE_LOCK="${PAOS_ROOT}/assets/piper_dual_wrist_top_30hz_v1/lock.json"
"${PIPER_SIM_ROOT}/.venv/bin/python" scripts/prepare_joint_dataset.py \
  --task-dir "$task_dir" --out "$out_run/data/final/prepared"
"${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" scripts/export_joint_lerobot.py \
  --prepared "$out_run/data/final/prepared" --out "$out_run/data/final/lerobot-0.4.4" \
  --repo-id "$repo_id"
"${PIPER_TRAINING_ROOT}/.venvs/act/cpu/bin/python" scripts/prepare_sim_browser_preview.py \
  --run "$out_run" --out "$out_run/data/browser-preview"
