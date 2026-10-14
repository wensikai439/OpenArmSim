"""Runtime skill registry derived from the tools actually exposed to the agent.

Tool JSON schemas remain the execution contract. This module adds explicit
scope/evidence/lifecycle metadata and reconciles the UI catalogue against that
same live tool list, including skills installed by optional adapters.
"""
from __future__ import annotations

import copy


_MOTION = {
    'move_to_pose', 'move_relative', 'set_gripper', 'pick', 'handover', 'place',
    'relocate', 'run_policy',
}

_META = {
    'list_skills': {
        'kind': 'capability_query', 'preconditions': [], 'effects': ['returns_current_capability_catalogue'],
        'observation_source': 'runtime_configuration', 'evidence_level': 'runtime_contract',
        'cancellation': 'not_applicable', 'deadline': 'request',
    },
    'inspect_manipulation': {
        'preconditions': ['fresh_observation_id'], 'effects': ['returns_registered_object_state'],
        'observation_source': 'simulator_oracle', 'evidence_level': 'oracle_read_only',
        'cancellation': 'not_applicable', 'deadline': 'request',
    },
    'move_to_pose': {
        'preconditions': ['fresh_observation_id', 'valid_world_pose'],
        'effects': ['bounded_cartesian_motion', 'native_path_recheck'],
        'observation_source': 'robot_proprioception_plus_simulator_geometry',
        'evidence_level': 'simulator_oracle', 'cancellation': 'atomic_skill_boundary',
        'deadline': 'task_budget',
    },
    'move_relative': {
        'preconditions': ['fresh_observation_id', 'delta_within_15cm'],
        'effects': ['bounded_cartesian_motion', 'native_path_recheck'],
        'observation_source': 'robot_proprioception_plus_simulator_geometry',
        'evidence_level': 'simulator_oracle', 'cancellation': 'atomic_skill_boundary',
        'deadline': 'task_budget',
    },
    'set_gripper': {
        'preconditions': ['fresh_observation_id', 'selected_arm'],
        'effects': ['gripper_encoder_reaches_command'],
        'observation_source': 'simulated_gripper_encoder', 'evidence_level': 'command_only',
        'cancellation': 'atomic_skill_boundary', 'deadline': 'task_budget',
    },
    'pick': {
        'preconditions': ['fresh_observation_id', 'supported_object_geometry', 'empty_selected_arm'],
        'effects': ['native_contact_and_lift_verification'],
        'observation_source': 'simulator_oracle', 'evidence_level': 'simulator_oracle',
        'cancellation': 'atomic_skill_boundary', 'deadline': 'task_budget',
    },
    'handover': {
        'preconditions': ['fresh_observation_id', 'donor_holds_object', 'receiver_is_empty'],
        'effects': ['receiver_contact_before_donor_release', 'receiver_only_hold_verification'],
        'observation_source': 'simulator_oracle', 'evidence_level': 'simulator_oracle',
        'cancellation': 'atomic_skill_boundary', 'deadline': 'task_budget',
    },
    'place': {
        'preconditions': ['fresh_observation_id', 'selected_arm_holds_object', 'locked_placement_goal'],
        'effects': ['support_release_retreat_and_stability_verification'],
        'observation_source': 'simulator_oracle', 'evidence_level': 'simulator_oracle',
        'cancellation': 'atomic_skill_boundary', 'deadline': 'task_budget',
    },
    'relocate': {
        'preconditions': ['fresh_observation_id', 'locked_placement_goal', 'supported_object_geometry'],
        'effects': ['composite_pick_transport_place_and_verification'],
        'observation_source': 'simulator_oracle', 'evidence_level': 'simulator_oracle',
        'cancellation': 'atomic_skill_boundary', 'deadline': 'task_budget',
    },
    'run_policy': {
        'preconditions': ['fresh_observation_id', 'compatible_checkpoint_and_io_contract'],
        'effects': ['policy_action_chunk_with_native_guards'],
        'observation_source': 'configured_policy_cameras_and_proprioception',
        'evidence_level': 'learned_policy_task_scope', 'cancellation': 'policy_session_boundary',
        'deadline': 'fixed_policy_budget',
    },
    'inspect_scene': {
        'kind': 'scene_context', 'preconditions': ['fresh_scene'],
        'effects': ['returns_declared_scene_geometry'], 'observation_source': 'simulator_oracle',
        'evidence_level': 'oracle_read_only', 'cancellation': 'not_applicable', 'deadline': 'request',
    },
    'resolve_scene_target': {
        'kind': 'scene_context', 'preconditions': ['fresh_scene', 'known_reference_id'],
        'effects': ['returns_bound_pose_without_motion'], 'observation_source': 'simulator_oracle',
        'evidence_level': 'oracle_read_only', 'cancellation': 'not_applicable', 'deadline': 'request',
    },
    'resolve_placement_transit': {
        'kind': 'scene_context', 'preconditions': ['fresh_hold_state', 'locked_placement_goal'],
        'effects': ['returns_transport_hypothesis_without_motion'], 'observation_source': 'simulator_oracle',
        'evidence_level': 'oracle_read_only', 'cancellation': 'not_applicable', 'deadline': 'request',
    },
    'select_manipulation_object': {
        'kind': 'scene_context', 'preconditions': ['fresh_observation_id', 'no_motion_started'],
        'effects': ['binds_registered_object_without_physics_change'], 'observation_source': 'scene_registry',
        'evidence_level': 'runtime_contract', 'cancellation': 'not_applicable', 'deadline': 'request',
    },
}


def _scope(parameters: dict) -> dict:
    props = parameters.get('properties', {})
    scope = {}
    for key, label in (('object_id', 'object_ids'), ('arm', 'arms'),
                       ('from_arm', 'source_arms'), ('to_arm', 'target_arms'),
                       ('backend', 'policy_backends'), ('skill_id', 'policy_skills')):
        value = props.get(key, {})
        if value.get('enum'):
            scope[label] = list(value['enum'])
    scope['frames'] = list(props.get('frame', {}).get('enum', []))
    if 'position_m' in props:
        scope['coordinate_frame'] = 'world'
        scope['units'] = 'metres; quaternion_xyzw in radians convention'
    return scope


def _provider_status(backends: list, adapters: list, snapshot: dict) -> list[dict]:
    records = {item.get('id'): item for item in adapters}
    policy = snapshot.get('policy_status', {})
    rows = []
    for name in backends:
        status = records.get(name, {})
        connected = status.get('connection_ready')
        if name in {'act', 'pi05'} and connected is None:
            connected = bool(snapshot.get('policy_type') == name and policy.get('connection_ready'))
        if name == 'tamp':
            callable_now = bool(status.get('ready'))
        elif name == 'curobo':
            callable_now = bool(status.get('ready') and status.get('connection_ready', True))
        else:
            callable_now = bool(status.get('ready') and connected)
        row = {
            'id': name,
            'callable': callable_now,
            'connection_ready': bool(connected) if name in {'act', 'pi05'} else bool(status.get('connection_ready', status.get('ready'))),
            'performance_verified': bool(status.get('performance_verified', False)),
        }
        if not callable_now:
            row['reason'] = status.get('reason') or ('backend is not connected or compatible in this runtime')
        rows.append(row)
    return rows


def build_specs(tools: list[dict], catalogue: dict, snapshot: dict) -> list[dict]:
    """Build immutable-by-convention SkillSpec JSON from the live tool schemas."""
    current = {item.get('id'): item for item in catalogue.get('skills', []) if item.get('id')}
    adapters = catalogue.get('adapters', [])
    result = []
    for tool in tools:
        function = tool.get('function', {})
        name = function.get('name')
        parameters = function.get('parameters')
        if not name or not isinstance(parameters, dict):
            continue
        visible = current.get(name, {})
        declared_backends = visible.get('backends')
        if name == 'run_policy':
            backend_enum = parameters.get('properties', {}).get('backend', {}).get('enum')
            if backend_enum:
                declared_backends = list(backend_enum)
        if not declared_backends:
            if name == 'run_policy':
                declared_backends = ['pi05', 'act']
            elif name in {'move_to_pose', 'move_relative'}:
                declared_backends = ['tamp', 'curobo']
            else:
                declared_backends = ['tamp']
        metadata = _META.get(name, {})
        motion = name in _MOTION
        providers = _provider_status(declared_backends, adapters, snapshot)
        spec = {key: copy.deepcopy(value) for key, value in visible.items()
                if key not in {'id', 'description', 'parameters', 'backends'}}
        spec.update({
            'id': name,
            'version': '1.0',
            'description': function.get('description', visible.get('description', '')),
            'parameters': copy.deepcopy(parameters),
            'backends': list(declared_backends),
            'providers': providers,
            'available': any(row['callable'] for row in providers),
            'kind': metadata.get('kind', visible.get('kind', 'motion' if motion else 'scene_context')),
            'starts_motion': motion,
            'scope': _scope(parameters),
            'preconditions': list(metadata.get('preconditions', ['fresh_observation_id'])),
            'effects': list(metadata.get('effects', ['see_tool_description'])),
            'observation_source': metadata.get('observation_source', 'declared_tool_contract'),
            'evidence_level': metadata.get('evidence_level', 'contract_scope_unverified'),
            'deadline': metadata.get('deadline', 'task_budget' if motion else 'request'),
            'cancellation': metadata.get('cancellation', 'atomic_skill_boundary' if motion else 'not_applicable'),
        })
        result.append(spec)
    # Do not let a catalogue advertise a tool that is absent from the live model
    # allow-list. Preserve diagnostics at top level; drop stale skill rows here.
    return result


def reconcile_catalogue(catalogue: dict, tools: list[dict], snapshot: dict) -> dict:
    value = copy.deepcopy(catalogue)
    value['skills'] = build_specs(tools, value, snapshot)
    callable_by_provider = {}
    for skill in value['skills']:
        for provider in skill['providers']:
            callable_by_provider[provider['id']] = (callable_by_provider.get(provider['id'], False)
                or provider['callable'])
    for adapter in value.get('adapters', []):
        adapter['configured_ready'] = bool(adapter.get('ready'))
        adapter['callable'] = callable_by_provider.get(adapter.get('id'), False)
        adapter['ready'] = adapter['callable']
        provider_rows = [provider for skill in value['skills'] for provider in skill['providers']
                         if provider['id'] == adapter.get('id')]
        adapter['connection_ready'] = any(provider['connection_ready'] for provider in provider_rows)
        adapter['performance_verified'] = any(provider['performance_verified'] for provider in provider_rows)
        if adapter['configured_ready'] and not adapter['callable']:
            adapter['reason'] = adapter.get('reason') or 'configured, but no compatible connected skill is callable'
    value['skill_registry'] = {'schema_version': '1', 'source': 'live_model_tool_allowlist',
        'skill_count': len(value['skills']), 'motion_skill_count': sum(s['starts_motion'] for s in value['skills']),
        'unavailable': [s['id'] for s in value['skills'] if not s['available']]}
    return value
