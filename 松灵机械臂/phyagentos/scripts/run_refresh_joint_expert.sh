#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Thin replay of this refresh's successful reset + official enforce command.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: run_refresh_joint_expert.sh SCENE_JSON NEW_OUTPUT_DIR CODEX_REVIEW_URL" >&2
  exit 2
fi
cd "${PAOS_ROOT}"
.venv/bin/python scripts/reset_sim_scene.py \
  --scene-config "$1" --audit-dir "$2/lifecycle"
exec .venv/bin/python scripts/run_sim_expert.py \
  --program "$PAOS_ROOT/runs/trajectory-repair-20261012T164835Z/expert-v22/expert.py" \
  --out "$2/formal" --max-actions 240 --require-joint-targets --require-binary-gripper \
  --verification-mode enforce --reviewer codex --codex-review-url "$3" \
  --scene-config "$1" --run
