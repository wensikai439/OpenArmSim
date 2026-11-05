#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -euo pipefail
cd "${PAOS_ROOT}"
test -f "$PAOS_CONFIG" || { echo "Configure $PAOS_CONFIG first; see README.md" >&2; exit 2; }
exec "$PAOS_PYTHON" -m uvicorn console.server:app --host 127.0.0.1 --port "${PAOS_CONSOLE_PORT:-19081}" --no-access-log --timeout-graceful-shutdown 5
