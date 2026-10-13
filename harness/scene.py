"""Scene references for the isolated candidate's existing native pose tools."""
import copy

import numpy as np

from .candidate import CandidateHarness
from .skills import OBS, ARM, VECTOR, QUAT, PLANNER, SPEED, function, validate_arguments
from sim.scene import target, CONTRACT

TOOLS = [function('inspect_scene', 'Read the current named scene directory, candidate primitive geometry and source labels. Explicit simulator oracle, not camera detections or verified graspability; blocks VLA in this epoch. Coordinates are current only for this observation.',
    {'observation_id': OBS}, ['observation_id']),
    function('resolve_scene_target', 'Compute a world TCP pose relative to a current named object/table/basket, offset length at most 0.5m. origin=body/geom origin; bounds_center=world primitive bounds center; top_center=top of those bounds (basket RIM, not cavity floor). offset_frame=object rotates the offset with the referenced body/geom; world uses world axes. Default orientation keeps this arm current TCP orientation. This tool does NOT move or prove reachability/grasp. Pass its exact motion_request to move_to_pose including scene_target_id, then verify motion_reached. Unknown IDs/bounds are errors; inspect_scene first. No pose reuse from old epochs.',
    {'observation_id': OBS, 'arm': ARM, 'object_id': {'type': 'string', 'minLength': 1, 'maxLength': 64},
        'reference_point': {'type': 'string', 'enum': ['origin', 'bounds_center', 'top_center']},
        'offset_frame': {'type': 'string', 'enum': ['world', 'object']}, 'offset_m': VECTOR,
        'quaternion_xyzw': QUAT, 'planner': PLANNER, 'speed_scale': SPEED},
    ['observation_id', 'arm', 'object_id', 'reference_point', 'offset_frame', 'offset_m'])]


class SceneHarness(CandidateHarness):
    def _task_progress(self, snapshot, context=None):
        """Compact current-task state, separate from durable method memory."""
        plan = self.workspace.plan_status(snapshot)
        root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id', getattr(self, 'active_session', None)))
        if not plan or plan['turn_id'] != root: return None
        steps = [{key: step[key] for key in ('id', 'title', 'checkpoint', 'skill', 'status') if key in step}
            for step in plan['steps']]
        motions = [step for step in steps if step.get('skill')]
        value = {'source': 'current task stage bindings; historical completion, not current object perception',
            'turn_id': root, 'steps': steps, 'completed_execution_ids': plan.get('stage_bindings', {}),
            'pending_robot_stages': [step['id'] for step in motions if step['status'] != 'done'],
            'all_planned_robot_stages_done': bool(motions) and all(step['status'] == 'done' for step in motions),
            'next_action': 'Review fresh observations and finish; do not repeat completed stages.'
                if motions and all(step['status'] == 'done' for step in motions)
                else 'Continue the next incomplete stage; never replay a successful stage to recover filtered evidence.',
            'contains_privileged_object_state': False, 'receipts': []}
        if self.privileged_epoch == snapshot['observation_id'].split(':')[0]:
            for execution in plan.get('execution_tools', [])[-8:]:
                try:
                    receipt = self.workspace.execution_receipt('execute', execution['id'],
                        context=snapshot.get('memory_context'), include_oracle=True)
                except ValueError:
                    # An active execution may not have its terminal receipt yet.
                    continue
                feedback = receipt['feedback']
                value['receipts'].append({'execution_id': execution['id'], 'skill': execution['skill'],
                    **{key: feedback[key] for key in ('primitive_success', 'verification_source', 'planner',
                        'position_error_m', 'rotation_error_rad', 'failure_code') if key in feedback}})
            value['contains_privileged_object_state'] = bool(value['receipts'])
            value['receipt_scope'] = 'explicit TAMP oracle receipts from this task; not calibrated real-world proof'
        return value

    def _targets(self, context):
        root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id', 'manual'))
        if not hasattr(self, 'scene_targets'): self.scene_targets = {}
        if root not in self.scene_targets:
            if len(self.scene_targets) >= 64: self.scene_targets.pop(next(iter(self.scene_targets)))
            self.scene_targets[root] = {}
        return self.scene_targets[root]

    def _invoke(self, name, args, role, context=None):
        if name in {'inspect_scene', 'resolve_scene_target'}:
            if role != 'execute': raise ValueError('scene references belong to the execution agent')
            spec = next(tool['function'] for tool in TOOLS if tool['function']['name'] == name)
            validate_arguments(spec['parameters'], args)
            # Reuse the standard freshness, oracle labels and VLA-taint boundary.
            receipt = super()._invoke('inspect_manipulation', {'observation_id': args['observation_id']}, role, context)
            snapshot = self.service.snapshot()
            directory = snapshot['diagnostics'].get('scene_directory')
            if not directory or directory['observation_id'] != receipt['observation_id']:
                raise ValueError('scene directory and inspection receipt are not synchronized')
            if name == 'inspect_scene': return copy.deepcopy(directory)
            result = target(directory, args, snapshot['observation']['tcp'][args['arm']])
            bank = self._targets(context)
            if len(bank) >= 32: raise ValueError('scene target budget exhausted; no motion dispatched')
            bank[result['target_id']] = copy.deepcopy(result)
            return result
        if name == 'move_to_pose' and 'scene_target_id' in args:
            from .agent import tools_for
            schema = next(tool['function']['parameters'] for tool in tools_for(role)
                if tool['function']['name'] == name)
            validate_arguments(schema, args)
            bank = self._targets(context)
            selected = bank.get(args['scene_target_id'])
            if selected is None: raise ValueError('unknown scene target; resolve_scene_target first')
            expected = selected['motion_request']
            if args['arm'] != expected['arm'] or args['frame'] != expected['frame']:
                raise ValueError('arm/frame does not match the resolved scene target')
            if not np.allclose(args['position_m'], expected['position_m'], atol=1e-9, rtol=0):
                raise ValueError('position changed after scene target resolution; no motion dispatched')
            rotation = np.asarray(args['quaternion_xyzw']); original = np.asarray(expected['quaternion_xyzw'])
            if min(np.linalg.norm(rotation - original), np.linalg.norm(rotation + original)) > 1e-6:
                raise ValueError('orientation changed after scene target resolution')
            if args.get('planner', 'tamp') != expected.get('planner', 'tamp'):
                raise ValueError('planner changed after scene target resolution')
            snapshot = self.service.snapshot()
            if context and context.get('type') == 'llm' and hasattr(self, 'workspace'):
                progress = self._task_progress(snapshot, context)
                if progress and progress['all_planned_robot_stages_done']:
                    raise ValueError('本任务所有已计划的机器人阶段已有成功回执；禁止重复运动。'
                        '先 get_run_status 查看 task_progress 和绑定回执，再用新观测 finish。'
                        '只有实际失败需要恢复时才修改计划；相机不确定不能作为重复运动的理由。')
            if selected['source_observation_id'].split(':')[0] != snapshot['observation_id'].split(':')[0]:
                raise ValueError('scene target expired after reset/prepare; resolve from a fresh observation')
            directory = snapshot['diagnostics'].get('scene_directory') or {}
            current = next((row for row in directory.get('objects', []) if row['object_id'] == selected['reference']['object_id']), None)
            if current is None: raise ValueError('referenced object no longer exists')
            old = selected['reference']
            position_drift = float(np.linalg.norm(np.subtract(current['origin_m'], old['origin_m'])))
            trace = np.trace(np.asarray(current['rotation_matrix']) @ np.asarray(old['rotation_matrix']).T)
            rotation_drift = float(np.arccos(np.clip((trace - 1) / 2, -1., 1.)))
            if position_drift > .003 or rotation_drift > .05:
                raise ValueError('scene reference moved after resolution; old target rejected before dispatch')
            clean = {key: value for key, value in args.items() if key != 'scene_target_id'}
            initiator = {**(context or {'type': 'direct-tool'}), 'scene_target': copy.deepcopy(selected)}
            result = super()._invoke(name, clean, role, initiator)
            return {**result, 'scene_target_id': selected['target_id'],
                'scene_reference_guard': {'position_drift_m': position_drift, 'rotation_drift_rad': rotation_drift,
                    'scope': 'pre-dispatch only; low-level planning has its own native state drift guard'}}
        result = super()._invoke(name, args, role, context)
        if role == 'execute' and name in {'observe', 'get_run_status'}:
            progress = self._task_progress(self.service.snapshot(), context)
            if progress:
                result = {**result, 'task_progress': progress}
        return result


def catalogue(snapshot, policy_type='pi05'):
    from .skills import catalogue as original
    from .agent import SKILL_TOOLS
    value = original(snapshot, policy_type)
    value['scene_directory_contract'] = copy.deepcopy(CONTRACT)
    directory = snapshot.get('diagnostics', {}).get('scene_directory', {})
    if directory:
        value['declared_assets'] = [{key: row[key] for key in ('object_id', 'label', 'kind', 'dynamic', 'affordances', 'physical_calibration')}
            for row in directory['objects']]
    specs = {tool['function']['name']: tool['function']['parameters'] for tool in SKILL_TOOLS}
    for skill in value['skills']:
        if skill['id'] in specs: skill['parameters'] = copy.deepcopy(specs[skill['id']])
    value['skills'] += [{'id': tool['function']['name'], 'description': tool['function']['description'],
        'parameters': copy.deepcopy(tool['function']['parameters']),
        'backends': ['tamp', 'curobo'], 'kind': 'scene_context', 'starts_motion': False} for tool in TOOLS]
    return value


def install_isolated():
    import harness.agent as agent
    agent.Harness = SceneHarness
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    motion = next(tool['function'] for tool in tools if tool['function']['name'] == 'move_to_pose')
    motion['parameters']['properties']['scene_target_id'] = {'type': 'string', 'minLength': 20, 'maxLength': 20,
        'description': 'Pass target_id from resolve_scene_target; its pose, arm, planner, epoch and reference drift are checked before dispatch.'}
    agent.SKILL_TOOLS = tools + copy.deepcopy(TOOLS)
    agent.catalogue = catalogue
