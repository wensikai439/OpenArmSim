#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: run_fast26_joint_expert.sh NEW_OUTPUT_DIR CODEX_REVIEW_URL" >&2
  exit 2
fi
cd "${PAOS_ROOT}"
PAOS_TIMING_RUN="$PAOS_ROOT/runs/trajectory-timing-20261012T113912Z"
exec "$PAOS_ROOT/.venv/bin/python" "$PAOS_ROOT/scripts/run_sim_expert.py" \
  --program "$PAOS_TIMING_RUN/expert-v16/expert.py" --out "$1" --max-actions 240 \
  --require-joint-targets --require-binary-gripper --verification-mode enforce \
  --reviewer codex --codex-review-url "$2" \
  --scene-config "$PAOS_TIMING_RUN/followup-release/scenarios/train-fast-p040.json" --run
