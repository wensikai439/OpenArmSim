#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -e
cd "${PAOS_ROOT}"
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: run_binary_expert.sh NEW_OUTPUT_DIR [ALREADY_ACTIVE_SCENE_CONFIG]" >&2
  exit 2
fi
scene_args=()
if [ "$#" -eq 2 ]; then scene_args=(--scene-config "$2"); fi
exec "$PAOS_ROOT/.venv/bin/python" "$PAOS_ROOT/scripts/run_sim_expert.py" --program "${PAOS_ROOT}/runs/joint-expert-20261011T014718Z/joint-guided-expert/expert.py" --out "$1" --max-actions 180 --require-joint-targets --require-binary-gripper --verification-mode enforce --run "${scene_args[@]}"
