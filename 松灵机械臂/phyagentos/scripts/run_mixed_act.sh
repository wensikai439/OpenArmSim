#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s FRESH_OUTPUT_DIR [STEPS=2000]\n' "$0" >&2
  exit 2
fi
root="${PAOS_ROOT}"
cd "$root"
exec "${PIPER_TRAINING_ROOT}/.venvs/act/cu128/bin/python" -u \
  "$root/scripts/train_binary_mixed_act.py" \
  --data-run "$root/runs/binary-data-contract-20261011T145308Z" \
  --out "$1" --steps "${2:-2000}" --batch-size 16 --seed 42
