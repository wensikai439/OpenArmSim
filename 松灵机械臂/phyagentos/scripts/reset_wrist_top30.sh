#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Reset only the PhyAgentOS-owned simulator to the verified upper-wrist 30 Hz scene.
set -euo pipefail
cd "${PAOS_ROOT}"
exec .venv/bin/python scripts/reset_sim_scene.py \
  --scene-config "${PAOS_ROOT}/runs/wrist-top-30hz-20261015T123011Z/scenarios/wrist-top30-01.json" \
  --audit-dir "${1:?Provide a new audit directory}"
