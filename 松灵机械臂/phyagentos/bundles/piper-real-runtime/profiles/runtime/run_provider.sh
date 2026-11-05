#!/usr/bin/env bash
set -euo pipefail
: "${PAOS_ROOT:?Source scripts/env.sh before starting a Skill}"
exec "${PAOS_PYTHON:-$PAOS_ROOT/.venv/bin/python}" -B "$(dirname "$0")/provider.py"
