"""Immutable task-level object goals over individually replanned motor tools.

Switching after verified release changes only task identity. All legs share the
original simulator clock/budget, original object targets and one human goal.
The original single-object path remains available without this optional schema.
"""
import copy
import json
import time

import numpy as np

from .object_transport import ObjectTransportHarness
from .object_memory import ObjectMemoryWorkspace
from .objects import IDS, resolve_goal, SELECT
from .skills import validate_arguments

MOTIONS = {'pick', 'handover', 'place', 'relocate', 'move_relative', 'move_to_pose', 'set_gripper', 'run_policy'}


def task_scene_dependencies(kwargs, environment_factory):
    """Keep the task-scene fallback local while honoring an outer scene selection."""
    result=dict(kwargs)
    result.setdefault('environment_factory',environment_factory)
    return result


def object_snapshot(snapshot, ident):
    """A labelled oracle view for anchoring an unselected named object."""
    value = copy.deepcopy(snapshot); scene = value['diagnostics']['manipulation_scene']
    row = next(item for item in value['diagnostics']['scene_directory']['objects'] if item['object_id'] == ident)
    scene.update(object_id=ident, position_m=row['origin_m'], world_bounds=row['world_bounds'],
        upright=row['rotation_matrix'][2][2] >= .98, held_by=[])
    return value


def assess_tasks(plan, snapshot):
    states = snapshot['diagnostics'].get('task_objects', {}).get('objects', {}); rows = []
    native = snapshot['diagnostics'].get('task_objects', {}).get('goal_ledger') or {}
    keys = ('object_id', 'surface', 'target_xy_m', 'tolerance_m', 'anchor')
    expected = [{key: goal[key] for key in keys} for goal in plan['object_goals']]
    actual = [{key: goal.get(key) for key in keys} for goal in native.get('goals', [])]
    ledger_matches = native.get('root') == plan['turn_id'] and expected == actual
    for index, goal in enumerate(plan['object_goals']):
        state = states.get(goal['object_id'], {}); error = float('inf')
        if state: error = float(np.linalg.norm(np.subtract(state['origin_m'][:2], goal['target_xy_m'])))
        released = state and not any(state['touching'].values())
        stable = state.get('stable_seconds', {}).get(goal['surface'], 0.)
        complete = bool(ledger_matches and goal.get('placement_execution_id') and error <= goal['tolerance_m'] and released and stable >= 1.
            and not snapshot['diagnostics']['evaluation'].get('invalid'))
        rows.append({'object_id': goal['object_id'], 'surface': goal['surface'], 'target_xy_m': goal['target_xy_m'],
            'position_error_m': error if np.isfinite(error) else None, 'tolerance_m': goal['tolerance_m'],
            'confirmed': complete, 'stable_seconds': stable, 'placement_execution_id': goal.get('placement_execution_id')})
    directory = snapshot['diagnostics']['scene_directory']['objects']; task_ids = {goal['object_id'] for goal in plan['object_goals']}
    drift = {row['object_id']: float(np.linalg.norm(np.subtract(row['origin_m'], plan['task_object_anchors'][row['object_id']])))
        for row in directory if row['object_id'] in IDS and row['object_id'] not in task_ids}
    valid = all(value <= .003 for value in drift.values())
    return {'kind': 'multi_object_placement', 'source': 'native per-substep contact/stability and original task-goal ledger',
        'confirmed': all(row['confirmed'] for row in rows) and valid,
        'position_error_m': max((row['position_error_m'] or 0.) for row in rows),
        'tolerance_m': min(goal['tolerance_m'] for goal in plan['object_goals']), 'objects': rows,
        'native_original_ledger_matches': ledger_matches,
        'unlisted_object_displacement_m': drift, 'unlisted_objects_undisturbed': valid, 'transfer_ready': False}


class TaskWorkspace(ObjectMemoryWorkspace):
    @staticmethod
    def _step_done(step, feedback, plan):
        if plan.get('object_goals') and step.get('object_id'):
            if feedback.get('object_id') != step['object_id']: return False
            selected = next(goal for goal in plan['object_goals'] if goal['object_id'] == step['object_id'])
            return ObjectMemoryWorkspace._step_done(step, feedback, {**plan, 'placement_goal': selected})
        return ObjectMemoryWorkspace._step_done(step, feedback, plan)

    def save_plan(self):
        with self.lock:
            self._write('UPDATE plans SET body=? WHERE turn_id=?',
                (json.dumps(self.active_plan, ensure_ascii=False), self.active_plan['turn_id']))

    def task_receipt(self, snapshot):
        """Trusted current leg certificate; never a model-written completion."""
        plan = self.active_plan; ident = plan['object_goals'][plan['object_index']]['object_id']
        steps = [step for step in plan['steps'] if step.get('object_id') == ident]
        matches = [step for step in steps if step['checkpoint'] == 'placement_confirmed']
        for step in matches:
            execution = plan.get('stage_bindings', {}).get(step['id'])
            if not execution: continue
            try: receipt = self.execution_receipt('execute', execution, context=snapshot['memory_context'], include_oracle=True)
            except ValueError: continue
            after = receipt['evidence'].get('after', {}).get('observation_id', '')
            if (after.split(':')[0] == snapshot['observation_id'].split(':')[0]
                and self._step_done(step, receipt['feedback'], plan)):
                return execution
        return None

    def plan_status(self, snapshot):
        value = super().plan_status(snapshot)
        if not value or not value.get('object_goals'): return value
        for index, goal in enumerate(value['object_goals']):
            if index == value['object_index']:
                receipt = self.task_receipt(snapshot)
                if receipt:
                    goal['placement_execution_id'] = receipt
                    with self.lock:
                        self.active_plan['object_goals'][index]['placement_execution_id'] = receipt
            goal['status'] = 'done' if goal.get('placement_execution_id') else 'active' if index == value['object_index'] else 'pending'
        # Model deliberation is not a dispatched motor job. The base workspace
        # can mark the next pending stage running merely because self.busy is
        # true; that causes the model to poll a job which never started.
        for step in value['steps']:
            if step.get('skill') and step['status'] == 'running':
                job = snapshot.get('job') or {}
                if not job or value.get('stage_bindings', {}).get(step['id']) != job.get('execution_id'):
                    step['status'] = 'pending'
        value['goal_assessment'] = assess_tasks(value, snapshot)
        value['task_scope'] = 'one human goal, ordered immutable object targets, shared original clock and budget'
        return value

    def goal_keyframes(self, snapshot, plan=None):
        plan = plan or self.plan_status(snapshot)
        if not plan or not plan.get('object_goals'): return super().goal_keyframes(snapshot, plan)
        frames = []
        for goal in plan['object_goals']:
            epoch = goal.get('epoch')
            if not epoch: continue
            part = {**plan, 'epoch': epoch}
            for frame in super().goal_keyframes(snapshot, part):
                label = {'bottle': '瓶子', 'block': '长方体', 'can': '圆柱'}[goal['object_id']]
                frames.append({**frame, 'label': label + ' · ' + frame['label']})
        return sorted(frames, key=lambda frame: frame.get('sim_time', 0.))[:17]


class TaskHarness(ObjectTransportHarness):
    def _task_progress(self, snapshot, context=None):
        value = super()._task_progress(snapshot, context)
        plan = self.workspace.plan_status(snapshot)
        if value and plan and plan.get('object_goals'):
            value.update(object_index=plan['object_index'], current_object_id=plan['placement_goal']['object_id'],
                object_goals=[{key: goal[key] for key in ('object_id', 'surface', 'target_xy_m', 'status')}
                    for goal in plan['object_goals']],
                task_anchor_observation_id=plan['task_anchor_observation_id'],
                object_switch_rule='After current original place and all object stages are confirmed, observe then select ONLY the next declared object; do not prepare/reset or replace the plan.')
            current = plan['object_goals'][plan['object_index']]
            if current.get('status') == 'done' and plan['object_index'] + 1 < len(plan['object_goals']) and not snapshot.get('job'):
                following = plan['object_goals'][plan['object_index'] + 1]['object_id']
                value['next_action'] = ('No native job is running. Current object placement is confirmed. '
                    'Observe, then select_manipulation_object(' + following + ') to continue the SAME declared task; '
                    'do not keep polling the completed job or prepare/reset.')
        return value

    def _invoke(self, name, args, role, context=None):
        root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
        plan = self.workspace.active_plan or {}; queued = bool(plan.get('object_goals') and plan.get('turn_id') == root)
        if name == 'set_plan':
            if queued: raise ValueError('multi-object goal ledger is locked; cannot replace stages, budget or targets mid-task')
            if args.get('object_goals'):
                return self._queue_plan(args, role, context)
            if any('object_id' in step for step in args.get('steps', [])):
                selected = self.service.snapshot()['diagnostics']['manipulation_scene']['object_id']
                if any(step.get('object_id', selected) != selected for step in args['steps']):
                    raise ValueError('multiple stage objects require object_goals on the first set_plan')
                args = copy.deepcopy(args)
                for step in args['steps']: step.pop('object_id', None)
        if name == 'select_manipulation_object' and queued:
            return self._advance_object(args, role, context)
        if queued and name in MOTIONS:
            token = self.service.snapshot()['observation_id']
            if token.split(':')[0] != plan['epoch']: raise ValueError('task epoch changed outside the confirmed object transition')
            ledger = self.service.snapshot()['diagnostics'].get('task_objects', {}).get('goal_ledger') or {}
            if ledger.get('root') != root: raise ValueError('native original task-goal ledger is missing; no motion dispatched')
        result = super()._invoke(name, args, role, context)
        if queued and name == 'get_run_status':
            feedback = result.get('tool_feedback') or {}
            keys = ('status', 'object_id', 'primitive_success', 'verification_source', 'arm', 'held_by',
                'from_arm', 'to_arm', 'failure_code', 'message', 'position_error_m', 'rotation_error_rad', 'placement')
            result['tool_feedback'] = {key: feedback[key] for key in keys if key in feedback}
            result['native_job_active'] = bool(self.service.snapshot().get('job'))
            result['detailed_evidence'] = 'Full native planning/contact records remain archived; get_execution_record on demand.'
        if name == 'observe':
            result['active_scene_contract']['multi_object_task'] = (
                'For a requested multi-object placement, on FIRST set_plan declare '
                'object_goals in order and add object_id to each robot stage. All targets anchor at task start. '
                'The declaration performs fresh native inspection and first-object bookkeeping if needed; no robot motion. '
                'After native place confirms release/stability, observe and select_manipulation_object for the next declared object. '
                'Do not prepare/reset, replace the plan, add time or finish before all goals. Same clock/budget; one object at a time.')
        return result

    def _queue_plan(self, args, role, context):
        from .agent import tools_for
        spec = next(tool['function']['parameters'] for tool in tools_for(role) if tool['function']['name'] == 'set_plan')
        validate_arguments(spec, args)
        if role != 'execute' or not context: raise ValueError('multi-object plan belongs to the execution model')
        if 'placement_goal' in args: raise ValueError('use placement_goal for one object OR object_goals for multiple')
        snapshot = self.service.snapshot(); root = context.get('plan_turn_id', context['turn_id'])
        session = self.sessions.get(root, {})
        if session.get('actions') or self.workspace.active_plan: raise ValueError('declare all object goals before motion')
        # Declaring named geometric targets is itself an explicit oracle tool.
        # Reuse the existing fresh-observation gate inside the operation rather
        # than requiring the model to guess an inspection/selection ritual.
        super()._invoke('inspect_manipulation', {'observation_id': args['observation_id']}, role, context)
        requested = args['object_goals']; ids = [goal['object_id'] for goal in requested]
        if len(set(ids)) != len(ids): raise ValueError('each object has one original final goal')
        goals = [resolve_goal(goal, object_snapshot(snapshot, goal['object_id'])) for goal in requested]
        previous = -1; placements = set()
        for step in args['steps']:
            if not step.get('skill'):
                if step['checkpoint'] != 'visual_review' or step is not args['steps'][-1]:
                    raise ValueError('queued tasks use object-bound robot stages and an optional final visual review')
                continue
            ident = step.get('object_id')
            if ident not in ids or ids.index(ident) < previous: raise ValueError('stages must follow the declared object order')
            previous = ids.index(ident)
            if step['checkpoint'] == 'placement_confirmed' and step['skill'] == 'place': placements.add(ident)
        if placements != set(ids): raise ValueError('every requested object requires a native place/placement_confirmed stage')
        selection = None
        if ids[0] != snapshot['diagnostics']['manipulation_scene']['object_id']:
            selection = super()._invoke('select_manipulation_object', {'observation_id': args['observation_id'], 'object_id': ids[0]}, role, context)
            observation = super()._invoke('observe', {}, role, context)
            super()._invoke('inspect_manipulation', {'observation_id': observation['observation_id']}, role, context)
            snapshot = self.service.snapshot()
        clean = {k: copy.deepcopy(v) for k, v in args.items() if k != 'object_goals'}
        clean['observation_id'] = snapshot['observation_id']
        clean['placement_goal'] = requested[0]
        for step in clean['steps']: step.pop('object_id', None)
        result = super()._invoke('set_plan', clean, role, context)
        with self.workspace.lock:
            plan = self.workspace.active_plan
            for normalized, original in zip(plan['steps'], args['steps']):
                if original.get('object_id'): normalized['object_id'] = original['object_id']
            goals[0]['epoch'] = snapshot['observation_id'].split(':')[0]
            plan.update(object_goals=goals, object_index=0,
                task_anchor_observation_id=snapshot['observation_id'],
                task_object_anchors={row['object_id']: row['origin_m'] for row in snapshot['diagnostics']['scene_directory']['objects'] if row['object_id'] in IDS})
            plan['placement_goal'] = copy.deepcopy(goals[0]); self.workspace.save_plan()
        self.service.call('declare_object_goals', observation_id=snapshot['observation_id'], root=root, goals=goals)
        return {**result, 'object_goals': copy.deepcopy(goals), 'object_index': 0,
            'observation_id': snapshot['observation_id'], 'first_object_selection': selection,
            'contains_privileged_object_state': True, 'fresh_observation_required': True,
            'scope': 'immutable ordered object goals; native release gates, shared clock/budget, no automatic execution'}

    def _advance_object(self, args, role, context):
        validate_arguments(SELECT['function']['parameters'], args)
        if role != 'execute': raise ValueError('object transition belongs to the execution agent')
        snapshot = self.service.snapshot(); token = snapshot['observation_id']
        if args['observation_id'] != token or time.monotonic() - self.observed.get(token, -1e10) > 60:
            raise ValueError('observe the current release state before selecting the next object')
        plan = self.workspace.active_plan; index = plan['object_index']; goals = plan['object_goals']
        if index + 1 >= len(goals) or args['object_id'] != goals[index + 1]['object_id']:
            raise ValueError('only the next declared object may be selected; no undeclared or repeated selection')
        if (snapshot.get('job') or snapshot['diagnostics']['evaluation'].get('invalid')
            or any(snapshot['diagnostics']['evaluation']['touching'].values())):
            raise ValueError('native scene must be valid and released before advancing objects')
        certificate = self.workspace.task_receipt(snapshot)
        if not certificate: raise ValueError('current original placement has no matching native success receipt; cannot hide a failed object goal')
        current = self.workspace.plan_status(snapshot)
        if any(step['status'] != 'done' for step in current['steps'] if step.get('object_id') == goals[index]['object_id']):
            raise ValueError('finish all current-object stages before advancing')
        assessment = current['goal_assessment']['objects'][index]
        if not assessment['confirmed']: raise ValueError('current object no longer satisfies its original target/release/stability')
        before = copy.deepcopy(snapshot['observation']); result = self.service.call('select_object', **args)
        after = self.service.snapshot()
        for key in ('state', 'tcp', 'sim_time', 'task_window'):
            if before[key] != after['observation'][key]: raise RuntimeError('object switch changed native state or task clock/budget')
        epoch = after['observation_id'].split(':')[0]
        with self.workspace.lock:
            plan['object_goals'][index]['placement_execution_id'] = certificate
            plan['object_goals'][index + 1]['epoch'] = epoch
            plan.update(epoch=epoch, object_index=index + 1, execution_id=None, verdict=None,
                placement_goal=copy.deepcopy(plan['object_goals'][index + 1]))
            self.workspace.save_plan()
        root = context.get('plan_turn_id', context['turn_id']); self.sessions[root]['selected_object'] = args['object_id']
        self.workspace.index_object_id = args['object_id']; self.observed.clear(); self.privileged_epoch = epoch
        return {**result, 'object_index': index + 1, 'completed_placement_execution_id': certificate,
            'original_task_anchor': plan['task_anchor_observation_id'], 'shared_task_window_preserved': True,
            'scope': 'task identity transition only; no reset, pose edit, new budget or automatic robot motion'}


def install_isolated():
    import harness.agent as agent
    agent.Harness = TaskHarness; tools = copy.deepcopy(agent.EXECUTION_TOOLS)
    plan = next(tool['function'] for tool in tools if tool['function']['name'] == 'set_plan')
    goal = copy.deepcopy(plan['parameters']['properties']['placement_goal'])
    plan['parameters']['properties']['object_goals'] = {'type': 'array', 'minItems': 2, 'maxItems': 3, 'items': goal}
    plan['parameters']['properties']['steps']['items']['properties']['object_id'] = {'type': 'string', 'enum': IDS}
    plan['description'] += (' For a requested MULTI-object placement, use ordered object_goals instead of placement_goal; '
        'fresh native inspection and first-object bookkeeping happen inside this declaration, with no robot motion. '
        'Each robot stage needs object_id; every object needs a place/placement_confirmed stage. '
        'Targets and stage ledger stay locked; after actual release/stability select the next declared object, no prepare/reset or extra budget.')
    agent.EXECUTION_TOOLS = tools
    skills = copy.deepcopy(agent.SKILL_TOOLS)
    select = next(tool['function'] for tool in skills if tool['function']['name'] == 'select_manipulation_object')
    select['description'] = ('Select a registered free object before a single-object plan/motion. '
        'A multi-object set_plan can select its first declared object as bookkeeping. With object_goals declared, '
        'select_manipulation_object advances to ONLY the next object AFTER the current original placement and all stages '
        'are natively confirmed. Unknown IDs, any gripper contact, stale observations, missing release or undeclared objects reject. '
        'No physical state, original target, clock/budget or previous evidence is reset. Observe again after selection.')
    agent.SKILL_TOOLS = skills
    previous = agent.catalogue
    def catalogue(snapshot, policy_type='pi05'):
        value = previous(snapshot, policy_type)
        value['multi_object_tasks'] = 'optional ordered object_goals; shared clock/budget, native release/stability before object switches; not a symbolic or arbitrary-mesh planner'
        return value
    agent.catalogue = catalogue


def service_factory():
    from .object_handover import service_factory as relay
    from sim.task_scene import TaskScene
    base = relay()
    class GoalCommands:
        def __init__(self, original, owner): self.original, self.owner = original, owner
        def put(self, item): return self.original.put(item)
        def get(self, *args, **kwargs):
            while True:
                command, values, future = self.original.get(*args, **kwargs)
                if command != 'declare_object_goals': return command, values, future
                if not future.set_running_or_notify_cancel(): continue
                owner = self.owner
                try:
                    if owner.job or owner.decision_wait: raise ValueError('native owner is busy')
                    if values['observation_id'] != f'{owner.epoch}:{owner.env.tick}': raise ValueError('original task-goal observation expired')
                    owner.env.require_current_sources()
                    before = (owner.env.data.qpos.copy(), owner.env.data.qvel.copy(), owner.env.data.ctrl.copy(), owner.env.data.time)
                    owner.env.declare_goals(values['root'], values['goals'])
                    for old, new in zip(before[:3], (owner.env.data.qpos, owner.env.data.qvel, owner.env.data.ctrl)): np.testing.assert_array_equal(old, new)
                    assert before[3] == owner.env.data.time
                    owner._publish(); future.set_result({'accepted': True, 'physical_state_preserved': True})
                except Exception as exc: future.set_exception(exc)
    class TaskService(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**task_scene_dependencies(kwargs,TaskScene))

        def _worker(self):
            self.commands = GoalCommands(self.commands, self)
            return super()._worker()
    return TaskService
