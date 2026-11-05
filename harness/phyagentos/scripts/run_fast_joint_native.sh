#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Independent native diagnostics; official Task verification is a separate entry.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: run_fast_joint_native.sh NEW_OUTPUT_DIR" >&2
  exit 2
fi
cd "${PAOS_ROOT}"
PAOS_TIMING_RUN="${PAOS_ROOT}/runs/trajectory-timing-20261012T113912Z"
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PAOS_CANCEL_REQUEST="$1/cancel-request.json"
export PAOS_FROZEN_SIM_SOURCE_LOCK="$PAOS_TIMING_RUN/asset-source-v2/lock.json"
export PAOS_MOTION_TIMING_PROFILE=feedback_joint_v1
exec "${PIPER_SIM_ROOT}/.venv/bin/python" \
  scripts/run_joint_expert_native.py --out "$1" \
  --program "$PAOS_TIMING_RUN/expert-v13/expert.py" --max-proposals 240 \
  --binary-gripper --scene-config "$PAOS_TIMING_RUN/scene-train-prepared-p040.json" --render
