from __future__ import annotations

from pathlib import Path

import pytest

from PhyAgentOS.skill_runtime.manager import RuntimeManager
from PhyAgentOS.skill_runtime.manifest import ManifestError, RuntimeProfile


def _manager() -> RuntimeManager:
    manager = object.__new__(RuntimeManager)
    manager.health_timeout_s = 30.0
    return manager


def test_startup_timeout_defaults_to_health_timeout() -> None:
    profile = RuntimeProfile(dataflow=Path("dataflow.yaml"))
    assert _manager()._startup_timeout_s(profile) == 30.0


def test_startup_timeout_uses_profile_override() -> None:
    profile = RuntimeProfile.from_dict(
        {"dataflow": "dataflow.yaml", "startup_timeout_s": 90}, "profiles.demo"
    )
    assert _manager()._startup_timeout_s(profile) == 90.0


@pytest.mark.parametrize("value", [0, -1, False, True, "90"])
def test_startup_timeout_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ManifestError):
        RuntimeProfile.from_dict(
            {"dataflow": "dataflow.yaml", "startup_timeout_s": value}, "profiles.demo"
        )
