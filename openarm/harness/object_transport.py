"""Resolve a held-object goal into a fresh, bound TCP transport hypothesis.

No robot command or historical pose is emitted automatically. The model must
explicitly execute the returned move_to_pose request, then use place to verify
the locked object goal. This adapter is only installed in the object cell.
"""
import copy

import mujoco
import numpy as np

from .object_handover import ObjectRelayHarness
from .skills import OBS, ARM, function, validate_arguments
from .objects import IDS
from .goals import same_xy
from sim.scene import target, quaternion

TOOL = function('resolve_placement_transit',
    'Compute an UNEXECUTED TCP pose to carry this currently held registered object above the LOCKED original table goal. '
    'Uses current native object pose and measured TCP/grasp relation, upright object yaw and a matching handover placement lookahead when available. '
    'Does not move, establish reachability or prove placement. Explicit simulator oracle; not visual localization or physical calibration. '
    'Call observe/inspect first. Pass the exact returned motion_request to move_to_pose, wait for actual feedback, then place at the original goal. '
    'Do not reuse a proposal after object/hold/goal drift. No backend fallback.',
    {'observation_id': OBS, 'object_id': {'type': 'string', 'enum': IDS}, 'arm': ARM,
        'clearance_m': {'type': 'number', 'minimum': .05, 'maximum': .30},
        'planner': {'type': 'string', 'enum': ['tamp', 'curobo']}},
    ['observation_id', 'object_id', 'arm', 'planner'])


def rotation_from_tcp(tcp):
    quat = np.asarray(tcp['quaternion_xyzw'], dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1.) > .02:
        raise ValueError('invalid current native TCP orientation')
    rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, np.r_[quat[3], quat[:3]] / np.linalg.norm(quat))
    return rotation.reshape(3, 3)


def pose_for_object(tcp, object_row, target_origin, yaw_offset=0.):
    """Rigid grasp equation about the TCP, not the hand-frame origin."""
    hand = rotation_from_tcp(tcp); obj = np.asarray(object_row['rotation_matrix'], dtype=float)
    point = np.asarray(object_row['origin_m'], dtype=float); current_tcp = np.asarray(tcp['position_m'], dtype=float)
    goal = np.asarray(target_origin, dtype=float)
    if (obj.shape != (3, 3) or point.shape != (3,) or current_tcp.shape != (3,) or goal.shape != (3,)
        or not all(np.isfinite(value).all() for value in (obj, point, current_tcp, goal))
        or not np.isfinite(yaw_offset) or abs(yaw_offset) > np.pi): raise ValueError('invalid carried-object pose')
    if not np.allclose(obj.T @ obj, np.eye(3), atol=1e-5) or abs(np.linalg.det(obj) - 1.) > 1e-5:
        raise ValueError('invalid current native object rotation')
    relative_position = hand.T @ (point - current_tcp); relative_rotation = hand.T @ obj
    yaw = np.arctan2(obj[1, 0], obj[0, 0]) + yaw_offset
    desired_object_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
    desired_hand = desired_object_rotation @ relative_rotation.T
    return {'position_m': (goal - desired_hand @ relative_position).tolist(),
        'quaternion_xyzw': quaternion(desired_hand), 'object_target_origin_m': goal.tolist(),
        'object_target_rotation_matrix': desired_object_rotation.tolist(),
        'grasp_transform_source': 'fresh native object pose relative to current robot TCP; no historical transform reuse'}


class ObjectTransportHarness(ObjectRelayHarness):
    def _lookahead(self, snapshot, goal, arm):
        plan = self.workspace.active_plan or {}
        for execution in reversed(plan.get('execution_tools', [])):
            if execution['skill'] != 'handover': continue
            try:
                receipt = self.workspace.execution_receipt('execute', execution['id'], context=snapshot['memory_context'], include_oracle=True)
            except ValueError:
                continue  # An active execution has no terminal receipt yet.
            feedback = receipt['feedback']; lookahead = feedback.get('future_place', {})
            after = receipt.get('evidence', {}).get('after', {}).get('observation_id', '')
            if (after.split(':')[0] == snapshot['observation_id'].split(':')[0]
                and feedback.get('primitive_success') and feedback.get('held_by') == arm
                and feedback.get('execution_binding', {}).get('valid') is True
                and feedback.get('object_id') == goal['object_id'] and lookahead.get('surface') == goal['surface']
                and same_xy(lookahead.get('target_xy_m', []), goal['target_xy_m'])):
                return float(lookahead['yaw_offset_rad']), execution['id']
        return 0., None

    def _invoke(self, name, args, role, context=None):
        if name == 'resolve_placement_transit':
            validate_arguments(TOOL['function']['parameters'], args)
            if role != 'execute': raise ValueError('held-object transport belongs to the execution agent')
            # Common freshness and oracle/VLA boundary; no model command starts.
            super()._invoke('inspect_manipulation', {'observation_id': args['observation_id']}, role, context)
            snapshot = self.service.snapshot(); root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
            plan = self.workspace.active_plan or {}; goal = plan.get('placement_goal')
            if (not goal or goal['surface'] != 'table' or plan.get('turn_id') != root
                or plan.get('epoch') != snapshot['observation_id'].split(':')[0]):
                raise ValueError('declare this task original table placement_goal before resolving object transport')
            scene = snapshot['diagnostics']['manipulation_scene']; arm = args['arm']
            held = snapshot['diagnostics']['evaluation']['held']
            if args['object_id'] != scene['object_id'] or args['object_id'] != goal['object_id']:
                raise ValueError('transport object differs from selected/locked object identity')
            if held != {'left': arm == 'left', 'right': arm == 'right'} or scene['held_by'] != [arm]:
                raise ValueError('selected arm must alone hold the object; no empty or dual-hold transport')
            if snapshot['diagnostics']['evaluation'].get('invalid') or snapshot.get('job'):
                raise ValueError('resolve transport only in a valid idle native scene')
            directory = snapshot['diagnostics']['scene_directory']
            if directory['observation_id'] != args['observation_id']: raise ValueError('scene directory is stale')
            obj = next(row for row in directory['objects'] if row['object_id'] == args['object_id'])
            table = next(row for row in directory['objects'] if row['object_id'] == 'table')
            clearance = args.get('clearance_m', .20); base = np.r_[goal['target_xy_m'], scene['surfaces']['table']['top_z'] + clearance]
            angle, receipt_id = self._lookahead(snapshot, goal, arm)
            pose = pose_for_object(snapshot['observation']['tcp'][arm], obj, base, angle)
            resolved = target(directory, {'observation_id': args['observation_id'], 'arm': arm, 'object_id': 'table',
                'reference_point': 'origin', 'offset_frame': 'world',
                'offset_m': (np.asarray(pose['position_m']) - table['origin_m']).tolist(),
                'quaternion_xyzw': pose['quaternion_xyzw'], 'planner': args['planner']}, snapshot['observation']['tcp'][arm])
            resolved.update(carried_reference=copy.deepcopy(obj), locked_placement_goal=copy.deepcopy(goal),
                current_tcp=copy.deepcopy(snapshot['observation']['tcp'][arm]), object_transport=pose,
                lookahead_execution_id=receipt_id, object_id=args['object_id'],
                scope='unexecuted TCP hypothesis for current registered held object and original table goal; native plan and placement gates mandatory')
            bank = self._targets(context)
            if len(bank) >= 32: raise ValueError('scene target budget exhausted')
            bank[resolved['target_id']] = copy.deepcopy(resolved)
            return resolved
        if name == 'move_to_pose' and args.get('scene_target_id'):
            proposal = self._targets(context).get(args['scene_target_id'], {})
            carried = proposal.get('carried_reference')
            if carried:
                snapshot = self.service.snapshot(); goal = (self.workspace.active_plan or {}).get('placement_goal')
                if goal != proposal['locked_placement_goal']: raise ValueError('locked object goal changed since transport resolution')
                scene = snapshot['diagnostics']['manipulation_scene']; arm = args['arm']
                held = snapshot['diagnostics']['evaluation']['held']
                if scene['object_id'] != carried['object_id'] or held != {'left': arm == 'left', 'right': arm == 'right'} or scene['held_by'] != [arm]:
                    raise ValueError('object or hold changed since transport resolution')
                current = next(row for row in snapshot['diagnostics']['scene_directory']['objects'] if row['object_id'] == carried['object_id'])
                drift = np.linalg.norm(np.subtract(current['origin_m'], carried['origin_m']))
                turn = np.arccos(np.clip((np.trace(np.asarray(current['rotation_matrix']) @ np.asarray(carried['rotation_matrix']).T) - 1) / 2, -1., 1.))
                if drift > .003 or turn > .05: raise ValueError('object transport reference drifted; resolve a fresh hypothesis')
                tcp = snapshot['observation']['tcp'][arm]; old = proposal['current_tcp']
                tcp_drift = np.linalg.norm(np.subtract(tcp['position_m'], old['position_m']))
                tcp_turn = np.arccos(np.clip((np.trace(rotation_from_tcp(tcp) @ rotation_from_tcp(old).T) - 1) / 2, -1., 1.))
                if tcp_drift > .005 or tcp_turn > .05: raise ValueError('grasp TCP changed since transport resolution; resolve a fresh hypothesis')
        result = super()._invoke(name, args, role, context)
        if name == 'select_manipulation_object':
            self.workspace.index_object_id = args['object_id']
        if name == 'observe':
            result['active_scene_contract']['object_transport'] = ('For a held object and locked original table goal, '
                'resolve_placement_transit derives a fresh TCP hypothesis; pass its exact motion_request to move_to_pose, '
                'wait for native GPU feedback, then place. No guessed coordinate transforms or historical target reuse.')
        return result


def install_isolated():
    import harness.agent as agent
    previous = agent.catalogue
    agent.SKILL_TOOLS = agent.SKILL_TOOLS + [copy.deepcopy(TOOL)]
    def catalogue(snapshot, policy_type='pi05'):
        value = previous(snapshot, policy_type)
        value['skills'].append({'id': TOOL['function']['name'], 'description': TOOL['function']['description'],
            'parameters': TOOL['function']['parameters'], 'backends': ['tamp', 'curobo'], 'kind': 'scene_context', 'starts_motion': False})
        return value
    agent.catalogue = catalogue
