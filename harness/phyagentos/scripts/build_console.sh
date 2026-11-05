#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -euo pipefail
cd "${PAOS_ROOT}"
export PATH="$PWD/toolchains/node-v22.23.3-linux-x64/bin:$PATH"
cd console/web
npm ci --cache ../../.cache/npm --no-audit --no-fund
npm run build
