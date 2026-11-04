#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Derived from the successful official-01 command; requires an explicit fresh scene.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: run_yaw26_joint_expert.sh NEW_OUTPUT_DIR CODEX_REVIEW_URL" >&2
  exit 2
fi
cd "${PAOS_ROOT}"
PAOS_YAW_RUN="$PAOS_ROOT/runs/trajectory-repair-20261012T164835Z"
exec "$PAOS_ROOT/.venv/bin/python" "$PAOS_ROOT/scripts/run_sim_expert.py" \
  --program "$PAOS_YAW_RUN/expert-v22/expert.py" --out "$1" --max-actions 240 \
  --require-joint-targets --require-binary-gripper --verification-mode enforce \
  --reviewer codex --codex-review-url "$2" \
  --scene-config "$PAOS_YAW_RUN/scenarios/train-yaw060-p040.json" --run
