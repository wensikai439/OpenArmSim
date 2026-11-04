from __future__ import annotations

from PhyAgentOS.skill_runtime.manager import RuntimeManager

FLOW = "paos-libero-lingbot_va"

ACTIVE = (
    '{"uuid":"a1","name":"paos-libero-lingbot_va","status":"Running","nodes":3}\n'
    '{"uuid":"a2","name":"paos-libero-lingbot_va","status":"Failed","nodes":0}\n'
    '{"uuid":"a3","name":"paos-libero-lingbot_va","status":"Finished","nodes":0}\n'
    '{"uuid":"a4","name":"paos-libero-lingbot_va","status":"Succeeded","nodes":0}\n'
)

HISTORICAL = (
    '{"name":"paos-libero-lingbot_va","status":"Finished","nodes":0}\n'
    '{"name":"paos-libero-lingbot_va","status":"Failed","nodes":0}\n'
    '{"name":"paos-libero-lingbot_va","status":"Succeeded","nodes":0}\n'
)


def test_parse_ndjson() -> None:
    flows = RuntimeManager._parse_flow_list(ACTIVE)
    assert len(flows) == 4
    assert flows[0]["status"] == "Running"


def test_parse_single_json_array() -> None:
    flows = RuntimeManager._parse_flow_list(
        '[{"name":"paos-libero-lingbot_va","status":"Running","nodes":3}]'
    )
    assert len(flows) == 1
    assert RuntimeManager._has_active_flow(flows, FLOW) is True


def test_active_detected_over_historical() -> None:
    flows = RuntimeManager._parse_flow_list(ACTIVE)
    assert RuntimeManager._has_active_flow(flows, FLOW) is True


def test_only_historical_is_not_active() -> None:
    flows = RuntimeManager._parse_flow_list(HISTORICAL)
    assert RuntimeManager._has_active_flow(flows, FLOW) is False


def test_wrong_name_is_not_active() -> None:
    flows = RuntimeManager._parse_flow_list(
        '{"name":"other_flow","status":"Running","nodes":3}'
    )
    assert RuntimeManager._has_active_flow(flows, FLOW) is False


def test_empty_and_garbage() -> None:
    assert RuntimeManager._has_active_flow(RuntimeManager._parse_flow_list(""), FLOW) is False
    assert RuntimeManager._has_active_flow(RuntimeManager._parse_flow_list("not json"), FLOW) is False
