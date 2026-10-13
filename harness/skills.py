"""Typed skill contracts. Planning and learned control share one simulation adapter.

An unavailable backend is never substituted by a scripted controller. Object
IDs are declared scene assets, not claims that a perception model found them.
"""
from __future__ import annotations

import math


MOTION_TOOLS = {'move_to_pose', 'move_relative', 'pick', 'handover', 'place', 'relocate', 'run_policy','set_gripper'}
LEGACY_MOTION_TOOLS = {'run_tamp', 'run_pi05_vla'}


def function(name, description, properties=None, required=None):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties or {},
                       'required': required or [], 'additionalProperties': False}}}


TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 2000}
OBS = {'type': 'string', 'description': 'Exact fresh observation_id returned by observe.'}
ARM = {'type': 'string', 'enum': ['left', 'right']}
VECTOR = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
QUAT = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}
XY = {'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}
SPEED = {'type': 'number', 'minimum': 0.05, 'maximum': 1.0}
PLANNER = {'type':'string','enum':['tamp','curobo'],'description':'Default tamp. curobo supports one selected arm empty or alone holding the declared bottle; other-arm/dual holds unsupported. GPU attachment collision plus mandatory native mesh check; no backend substitution.'}

SKILL_TOOLS = [
    function('list_skills', 'Read supported skills, preconditions, coordinate conventions and REAL backend readiness. Declared assets are not visual detections.'),
    function('inspect_manipulation', 'Read current declared bottle hold/upright state, recent object-in-hand stability and table/basket bounds before a TAMP object goal. Stable_recent_window covers only the measured recent 2 seconds; contact/lift is not stable or long-duration hold. Explicit simulator oracle, NOT camera perception; forbids a learned-policy run in this epoch until reset. Never substitute an old memory pose for this observation.',
             {'observation_id':OBS},['observation_id']),
    function('move_to_pose', 'Move one gripper TCP to a world-frame 6-D pose using bounded IK/collision planning and native actuator execution. Other arm and gripper commands are preserved. Planning geometry uses simulator truth; oracle baseline. Goal reached is checked from robot FK, not object truth.',
             {'observation_id': OBS, 'arm': ARM, 'position_m': VECTOR, 'quaternion_xyzw': QUAT,
              'frame': {'type': 'string', 'enum': ['world']}, 'speed_scale': SPEED, 'planner':PLANNER},
             ['observation_id', 'arm', 'position_m', 'quaternion_xyzw', 'frame']),
    function('move_relative', 'Move a gripper TCP by at most 15 cm from the freshly measured robot pose, keeping its orientation. Delta may be in world or current tool axes. Collision checks include carried bottle; no direct joint commands.',
             {'observation_id': OBS, 'arm': ARM, 'delta_m': VECTOR,
              'frame': {'type': 'string', 'enum': ['world', 'tool']}, 'speed_scale': SPEED, 'planner':PLANNER},
             ['observation_id', 'arm', 'delta_m', 'frame']),
    function('set_gripper','Independently open/close one simulated gripper, preserving both arm targets and the other gripper. opening_fraction: 1 fully open, 0 close command, intermediate values partial opening. Encoder settling verifies command execution only; CLOSED DOES NOT MEAN HOLDING. Opening a held object can drop it; use place for supported release. Simulation only, candidate transmission, no hardware control.',
             {'observation_id':OBS,'arm':ARM,'opening_fraction':{'type':'number','minimum':0,'maximum':1}},
             ['observation_id','arm','opening_fraction']),
    function('pick', 'Pick and lift the declared upright bottle with the selected arm. Search grasp orientation/height from current simulated geometry; unreachable or colliding poses are rejected. No existing hold. Oracle geometry, not arbitrary-object visual grasping. Wait for NEW feedback before the next skill.',
             {'observation_id': OBS, 'object_id': {'type': 'string', 'enum': ['bottle']},
              'arm': ARM}, ['observation_id', 'object_id', 'arm']),
    function('handover', 'Transfer the bottle between distinct selected arms in EITHER direction. Requires donor hold and lift. Search a shared reachable collision-free region, verify receiver contact before donor release, then confirm receiver-only hold. One bounded skill, not the complete task.',
             {'observation_id': OBS, 'object_id': {'type': 'string', 'enum': ['bottle']},
              'from_arm': ARM, 'to_arm': ARM},
             ['observation_id', 'object_id', 'from_arm', 'to_arm']),
    function('place', 'Place a held bottle on a table target OR inside basket using EITHER arm. Requires selected arm alone holding; otherwise pick first. LLM tasks must anchor placement_goal in set_plan first: omit position/offset here to use that goal. Any supplied offset is relative to the goal START, never the moved bottle. Direct manual tools without a model plan need target_xy_m or offset_xy_m for table. World +X robot-forward, +Y robot-left. Transport, descend, release, retreat, verify support/no hand contact/target footprint/stability for 1 s; table additionally requires upright bottle. Oracle geometry, not visual grasping. Empty TCP motion cannot satisfy object goals.',
             {'observation_id': OBS, 'object_id': {'type': 'string', 'enum': ['bottle']},
              'receptacle_id': {'type': 'string', 'enum': ['table','basket']}, 'arm': ARM,
              'target_xy_m': XY, 'offset_xy_m':XY, 'speed_scale':SPEED},
             ['observation_id', 'object_id', 'receptacle_id', 'arm']),
    function('relocate','Relocate the declared upright bottle to the LOCKED placement_goal in one bounded native operation: grasp if empty, optionally handover via_arm, optional explicit TCP transport, then place/release/retreat and verify support/stability. Each subskill re-grounds from current native feedback; no API wait while carrying. Oracle geometry only, not arbitrary-object grasping or real transfer. Same collision/hold/goal gates as individual skills. transport_planner defaults tamp; curobo requires transport_delta_m and REAL GPU execution, never substituted. allow_handover_recovery permits ONE alternate-hand route after a verified reachability failure; no reset/goal change/regrasp on lost contact. LLM must anchor placement_goal first; omit coordinates to use it. A successful placement does NOT prove sustained grip during long model waits.',
             {'observation_id':OBS,'object_id':{'type':'string','enum':['bottle']},'arm':ARM,
              'receptacle_id':{'type':'string','enum':['table','basket']},'target_xy_m':XY,'offset_xy_m':XY,
              'via_arm':ARM,'transport_planner':{'type':'string','enum':['tamp','curobo']},
              'transport_delta_m':VECTOR,'speed_scale':SPEED,'allow_handover_recovery':{'type':'boolean'}},
             ['observation_id','object_id','arm','receptacle_id']),
    function('run_policy', 'Run an actually configured learned policy skill with verified OpenArm 16-D I/O. Current capability is atomic_right_to_left_handover from reset only, not arbitrary subgoal language. Unavailable ACT/pi05 is an error, never replaced by TAMP.',
             {'observation_id': OBS, 'backend': {'type': 'string', 'enum': ['pi05', 'act']},
              'skill_id': {'type': 'string', 'enum': ['atomic_right_to_left_handover']}},
             ['observation_id', 'backend', 'skill_id']),
]


def validate_arguments(spec, value, path='arguments'):
    """Validate nested JSON without accepting bool as a number or NaN/Inf."""
    kind = spec.get('type')
    valid = {'object': lambda: isinstance(value, dict), 'array': lambda: isinstance(value, list),
             'string': lambda: isinstance(value, str), 'boolean': lambda: type(value) is bool,
             'number': lambda: type(value) in (int, float) and math.isfinite(value),
             'integer': lambda: type(value) is int}.get(kind, lambda: True)
    if not valid():
        raise ValueError(f'{path}: expected {kind}')
    if 'enum' in spec and value not in spec['enum']:
        raise ValueError(f'{path}: unsupported value')
    if kind == 'object':
        props = spec.get('properties', {})
        if set(spec.get('required', [])) - set(value):
            raise ValueError(f'{path}: missing required arguments')
        if spec.get('additionalProperties') is False and set(value) - set(props):
            raise ValueError(f'{path}: unknown arguments')
        for key, item in value.items():
            if key in props:
                validate_arguments(props[key], item, f'{path}.{key}')
    elif kind == 'array':
        if not spec.get('minItems', 0) <= len(value) <= spec.get('maxItems', 1000):
            raise ValueError(f'{path}: invalid array length')
        for index, item in enumerate(value):
            validate_arguments(spec.get('items', {}), item, f'{path}[{index}]')
    elif kind in {'number', 'integer'}:
        if not spec.get('minimum', -math.inf) <= value <= spec.get('maximum', math.inf):
            raise ValueError(f'{path}: value outside bounds')
    elif kind == 'string':
        if not spec.get('minLength', 1) <= len(value) <= spec.get('maxLength', 2000):
            raise ValueError(f'{path}: invalid string length')


def request(name, args):
    tasks = {'pick': 'pick_bottle', 'handover': 'handover_bottle', 'place': 'place_bottle','relocate':'relocate_object'}
    if name in tasks:
        return {'task': tasks[name], 'skill': name, 'parameters': {k: v for k, v in args.items() if k != 'observation_id'}}
    if name in {'move_to_pose', 'move_relative','set_gripper'}:
        return {'task': name, 'skill': name, 'parameters': {k: v for k, v in args.items() if k != 'observation_id'}}
    raise ValueError('not a TAMP skill')


def catalogue(snapshot, policy_type='pi05'):
    configured = bool(snapshot.get('policy_configured'))
    return {'environment': 'native-mujoco', 'hardware_access': False, 'transfer_ready': False,
        'coordinate_contract': {'frame': 'world', 'units': 'metres, radians',
            'axes':'+X forward away from robot, +Y robot-left, +Z up; operator camera may face the opposite direction',
            'quaternion': 'xyzw', 'tcp_local_offset_m': [0, 0.006, 0.075],
            'tcp_status': 'candidate simulated TCP; physical calibration pending'},
        'declared_assets': [{'id': 'bottle', 'type': 'rigid graspable'}, {'id': 'basket', 'type': 'container'}],
        'asset_source': 'scene configuration, NOT camera detections',
        'adapters': [
            {'id': 'tamp', 'kind': 'planner', 'ready': True, 'engine': 'MuJoCo Jacobian IK + collision search', 'evidence': 'simulator_oracle'},
            {'id': 'curobo', 'kind': 'planner', **snapshot.get('curobo',{'ready':False,'reason':'local cuRobo worker unavailable'}), 'engine':'official cuRobo V2 GPU planning + native MuJoCo path check'},
            {'id': 'pi05', 'kind': 'policy', 'ready': configured and policy_type == 'pi05', 'reason': None if configured and policy_type == 'pi05' else 'compatible OpenArm checkpoint/endpoint required'},
            {'id': 'act', 'kind': 'policy', 'ready': configured and policy_type == 'act', 'reason': None if configured and policy_type == 'act' else snapshot.get('policy_status',{}).get('reason','compatible OpenArm checkpoint/endpoint required'),
             'validation_note':'真机权重已接入；当前仿真任务尚未通过，夹爪越界时停止执行。'}],
        'skills': [{'id': t['function']['name'], 'description': t['function']['description'],
                    'parameters': t['function']['parameters'],
                    'backends': ['tamp','curobo'] if t['function']['name'] in {'move_to_pose','move_relative'} else ['pi05','act'] if t['function']['name']=='run_policy' else ['tamp']}
                   for t in SKILL_TOOLS if t['function']['name'] != 'list_skills']}
