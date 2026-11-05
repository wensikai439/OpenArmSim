"""Tests for per-node environment injection in rendered Skill dataflows."""

from __future__ import annotations

import pytest
import yaml

from PhyAgentOS.skill_runtime.installer import InstallerError, _inject_node_environment

NL = chr(10)

DATAFLOW = """nodes:
  - id: gateway
    path: /bin/gateway
    args: --config gateway.yaml
    env:
      MUJOCO_GL: egl
      PAOS_SKILL_VERSION: "0.0.0"
    outputs:
      - tool_response
  - id: worker
    path: /bin/worker
"""


def _nodes(rendered: str) -> dict:
    return {node["id"]: node for node in yaml.safe_load(rendered)["nodes"]}


def test_injects_profile_and_runtime_environment_into_every_node() -> None:
    rendered = _inject_node_environment(
        DATAFLOW,
        {"MUJOCO_GL": "osmesa", "HF_HUB_OFFLINE": "1"},
        {"HOME": "/tmp/runtime-home", "PAOS_SKILL_VERSION": "0.1.1"},
    )
    nodes = _nodes(rendered)
    gateway_env = nodes["gateway"]["env"]
    worker_env = nodes["worker"]["env"]
    # Node-declared values win over the profile-level environment.
    assert gateway_env["MUJOCO_GL"] == "egl"
    # Profile-level values reach nodes that never declared them.
    assert worker_env["HF_HUB_OFFLINE"] == "1"
    assert worker_env["MUJOCO_GL"] == "osmesa"
    # Runtime-derived identity wins over both, so a shared Dora daemon cannot
    # leak a stale HOME or PAOS_SKILL_VERSION into the node.
    assert gateway_env["PAOS_SKILL_VERSION"] == "0.1.1"
    assert worker_env["HOME"] == "/tmp/runtime-home"
    assert worker_env["PAOS_SKILL_VERSION"] == "0.1.1"


def test_preserves_unrelated_node_fields() -> None:
    rendered = _inject_node_environment(DATAFLOW, {}, {})
    gateway = _nodes(rendered)["gateway"]
    assert gateway["path"] == "/bin/gateway"
    assert gateway["args"] == "--config gateway.yaml"
    assert gateway["outputs"] == ["tool_response"]


def test_rejects_dataflow_without_nodes() -> None:
    with pytest.raises(InstallerError, match="nodes list"):
        _inject_node_environment("path: /bin/gateway" + NL, {}, {})


def test_rejects_invalid_yaml() -> None:
    with pytest.raises(InstallerError, match="not valid YAML"):
        _inject_node_environment("nodes: [", {}, {})


def test_rejects_non_mapping_node() -> None:
    with pytest.raises(InstallerError, match="node must be a mapping"):
        _inject_node_environment("nodes:" + NL + "  - 1" + NL, {}, {})


def test_rejects_non_mapping_node_environment() -> None:
    with pytest.raises(InstallerError, match="node env must be a mapping"):
        _inject_node_environment(
            "nodes:" + NL + "  - id: a" + NL + "    env: 1" + NL, {}, {}
        )
