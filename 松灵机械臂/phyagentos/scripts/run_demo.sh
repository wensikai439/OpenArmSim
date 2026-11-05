#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -euo pipefail
cd "${PAOS_ROOT}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
# Run after all task/invocation states are terminal; do not submit from the console concurrently.
paos skill stop rpent-piper-sim
paos skill start rpent-piper-sim --profile rpent
paos agent --session "cli:rpent-$stamp" --logs -m "$(cat runs/rpent-forge/task-enforce.txt)" 2>&1 | tee "runs/rpent-forge/replay-$stamp.log"
