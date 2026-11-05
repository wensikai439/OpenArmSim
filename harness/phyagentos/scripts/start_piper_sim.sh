#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Start only the PhyAgentOS-owned shared PiPER simulation adapter.
set -euo pipefail
cd "${PAOS_ROOT}"
exec paos skill start piper-dual-sim --profile "${1:-joint}"
