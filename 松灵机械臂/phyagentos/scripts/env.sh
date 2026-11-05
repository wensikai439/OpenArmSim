#!/usr/bin/env bash
# Source this before CLI/console use. Optional host paths live in ignored config/local.env.
export PAOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PAOS_ROOT/config/local.env" ]]; then
  source "$PAOS_ROOT/config/local.env"
fi
export PAOS_CONFIG="${PAOS_CONFIG:-$PAOS_ROOT/.runtime/config.json}"
export PAOS_PYTHON="${PAOS_PYTHON:-$PAOS_ROOT/.venv/bin/python}"
export PAOS_PROJECTS_ROOT="${PAOS_PROJECTS_ROOT:-$PAOS_ROOT/external/projects}"
export PIPER_CONTROL_ROOT="${PIPER_CONTROL_ROOT:-$PAOS_ROOT/external/robot-control}"
export PIPER_SIM_ROOT="${PIPER_SIM_ROOT:-$PAOS_ROOT/external/piper_robocasa}"
export PIPER_TRAINING_ROOT="${PIPER_TRAINING_ROOT:-$PAOS_ROOT/external/piper_training_project}"
export RPENT_ROOT="${RPENT_ROOT:-$PAOS_ROOT/external/rpent}"
export PAOS_SCENE_SELECTION_FILE="${PAOS_SCENE_SELECTION_FILE:-$PAOS_ROOT/config/sim_scene_selection.json}"
export PATH="$(dirname -- "$PAOS_PYTHON"):$PAOS_ROOT/toolchains/bin:$PAOS_ROOT/toolchains/node-v22.23.3-linux-x64/bin:$PAOS_ROOT/toolchains/dora-0.4.1/dora-cli-x86_64-unknown-linux-gnu:$PATH"
export PYTHONPATH="$PAOS_ROOT/PhyAgentOS-core:$PAOS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
