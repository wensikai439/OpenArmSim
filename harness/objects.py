"""Isolated registered-object harness; language tools select real native IDs."""
import copy
import math
import time

from .scene import SceneHarness, catalogue as scene_catalogue
from .skills import OBS, function, validate_arguments
from .goals import same_xy, object_state_from_feedback as bottle_state
from .candidate import CandidateWorkspace
from .knowledge import consolidate as base_consolidate
from sim.placement import placement_tolerance

IDS = ['bottle', 'block', 'can']
SELECT = function('select_manipulation_object',
    'Select one registered free object BEFORE set_plan or motion. Bookkeeping only: no qpos/qvel/actuator/time change. '
    'Unknown IDs or any gripper contact reject selection. Selection gives a new observation identity; call observe and inspect_scene again. '
    'New upright box/cylinder objects support TAMP pick/place; handover/relocate and cuRobo carrying still only bottle. '
    'Do not select/reset to conceal a failed task.',
    {'observation_id': OBS, 'object_id': {'type': 'string', 'enum': IDS}}, ['observation_id', 'object_id'])


def object_environment_dependencies(kwargs, environment_factory):
    """Use registered-object physics locally unless an outer task scene was selected."""
    result=dict(kwargs)
    result.setdefault('environment_factory',environment_factory)
    return result


def consolidate_objects(receipts, forgotten=()):
    """Different object identities never share a claimed manipulation method."""
    rows = []; original_contexts = {}
    for item in receipts:
        value = copy.deepcopy(item)
        ident = value['arguments'].get('object_id', value['feedback'].get('object_id'))
        if value['tool'] in {'pick', 'handover', 'place', 'relocate'}:
            if ident not in IDS: continue
            scoped = str(value.get('context')) + '|object=' + ident
            original_contexts[scoped] = (value.get('context'), ident); value['context'] = scoped
        rows.append(value)
    lessons = base_consolidate(rows, forgotten)
    for lesson in lessons:
        if lesson['context'] in original_contexts:
            context, ident = original_contexts[lesson['context']]
            lesson['context'] = context; lesson['object_id'] = ident
            label = {'bottle': '瓶子', 'block': '长方体', 'can': '圆柱'}[ident]
            lesson['title'] += ' · ' + label
            lesson['summary'] = lesson['summary'].replace('持瓶', '持物').replace('瓶子', '物体')
            lesson['evidence_scope'] += ' 仅对应所选对象 ' + ident + ' 与当前模型版本。'
    return lessons


def resolve_goal(spec, snapshot, previous=None, initial_snapshot=None):
    if spec is None: return copy.deepcopy(previous)
    scene = snapshot['diagnostics']['manipulation_scene']
    if spec['object_id'] != scene['object_id']: raise ValueError('select the requested object, then inspect before planning')
    if 'target_xy_m' in spec and 'offset_xy_m' in spec: raise ValueError('choose absolute target OR offset')
    origin = initial_snapshot or snapshot
    if origin['observation_id'].split(':')[0] != snapshot['observation_id'].split(':')[0]:
        raise ValueError('goal-start identity expired; never replace a failed task anchor')
    initial_scene = origin['diagnostics']['manipulation_scene']
    if initial_scene['object_id'] != spec['object_id']: raise ValueError('goal anchor belongs to another object')
    anchor = previous['anchor']['object_xy_m'] if previous else initial_scene['position_m'][:2]
    surface = spec['surface']; area = scene['surfaces'][surface]
    if 'target_xy_m' in spec: target = spec['target_xy_m']
    elif 'offset_xy_m' in spec:
        if surface != 'table': raise ValueError('basket uses absolute cavity center')
        target = [a + b for a, b in zip(anchor, spec['offset_xy_m'])]
    elif previous: target = previous['target_xy_m']
    elif surface == 'basket': target = [(a + b) / 2 for a, b in zip(area['xy_min'], area['xy_max'])]
    else: raise ValueError('table goal requires a position or offset')
    if len(target) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in target):
        raise ValueError('invalid object target')
    bounds = initial_scene['world_bounds']; center = initial_scene['position_m']
    margin = [max(center[i] - bounds['min_m'][i], bounds['max_m'][i] - center[i]) + .010 for i in range(2)]
    if any(v < lo + m or v > hi - m for v, lo, hi, m in zip(target, area['xy_min'], area['xy_max'], margin)):
        raise ValueError('registered object footprint does not fit on the support surface')
    value = {'kind': 'object_placement', 'object_id': spec['object_id'], 'surface': surface, 'target_xy_m': list(target),
        'offset_xy_m': spec.get('offset_xy_m'), 'tolerance_m': placement_tolerance(surface, target, anchor),
        'anchor': {'observation_id': origin['observation_id'], 'object_xy_m': list(anchor), 'source': 'simulator_ground_truth'}}
    if previous:
        if any(value[k] != previous[k] for k in ('object_id', 'surface')) or not same_xy(target, previous['target_xy_m']):
            raise ValueError('object and original placement goal are locked; route recovery cannot change either')
        return copy.deepcopy(previous)
    return value


def bind_goal(args, goal):
    if args['object_id'] != goal['object_id'] or args['receptacle_id'] != goal['surface']:
        raise ValueError('requested object/support differs from the locked goal')
    proposed = args.get('target_xy_m')
    if 'offset_xy_m' in args:
        if proposed is not None: raise ValueError('choose only one placement target')
        proposed = [a + b for a, b in zip(goal['anchor']['object_xy_m'], args['offset_xy_m'])]
    if proposed is not None and not same_xy(proposed, goal['target_xy_m']): raise ValueError('placement target changed')
    return {**{k: v for k, v in args.items() if k not in ('target_xy_m', 'offset_xy_m')},
        'target_xy_m': goal['target_xy_m'], 'position_tolerance_m': goal['tolerance_m'],
        'goal_anchor_observation_id': goal['anchor']['observation_id']}


def object_state(skill, feedback):
    value = bottle_state(skill, feedback)
    if value:
        if feedback.get('object_id') not in IDS: return None
        value['object_id'] = feedback['object_id']
    return value


class ObjectWorkspace(CandidateWorkspace):
    @staticmethod
    def _step_done(step, feedback, plan):
        goal = plan.get('placement_goal')
        if goal and step['checkpoint'] in {'grasp_confirmed', 'placement_confirmed'}:
            if feedback.get('object_id') != goal['object_id']: return False
        return CandidateWorkspace._step_done(step, feedback, plan)


class ObjectHarness(SceneHarness):
    def _invoke(self, name, args, role, context=None):
        if name == 'select_manipulation_object':
            validate_arguments(SELECT['function']['parameters'], args)
            if role != 'execute': raise ValueError('object selection belongs to the execution agent')
            snapshot = self.service.snapshot(); token = args['observation_id']
            if token != snapshot['observation_id'] or time.monotonic() - self.observed.get(token, -1e10) > 60:
                raise ValueError('observe first; selection requires the current identity')
            root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
            session = self.sessions.get(root, {})
            if session.get('actions') or session.get('selected_object') or self.workspace.active_plan or snapshot.get('job'):
                raise ValueError('select once before planning/motion; never change object during recovery')
            result = self.service.call('select_object', **args)
            current = self.service.snapshot(); session['initial_snapshot'] = current
            session['selected_object'] = args['object_id']
            self.observed.clear(); self.privileged_epoch = current['observation_id'].split(':')[0]
            return result
        if name in {'pick', 'place', 'handover', 'relocate'}:
            selected = self.service.snapshot()['diagnostics']['manipulation_scene']['object_id']
            if args.get('object_id') != selected: raise ValueError('object ID differs from the current task selection')
            if name in {'handover', 'relocate'} and selected != 'bottle':
                raise ValueError('new-shape handover/composite transport is not yet validated')
        if name in {'move_to_pose', 'move_relative'} and args.get('planner') == 'curobo':
            snapshot = self.service.snapshot()
            if snapshot['diagnostics']['manipulation_scene']['object_id'] != 'bottle':
                raise ValueError('GPU obstacle/attachment adapter is still bottle-only; no silent backend substitute')
        result = super()._invoke(name, args, role, context)
        if name == 'observe':
            result['benchmark_task'] = 'Registered upright primitives: select object, plan, use native grasp/placement receipts; no legacy full-task score.'
        return result


def install_isolated():
    import harness.agent as agent
    import harness.knowledge as knowledge
    knowledge.consolidate = consolidate_objects
    agent.Harness = ObjectHarness
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    for tool in tools:
        value = tool['function']
        if value['name'] in {'pick', 'place'}:
            value['parameters']['properties']['object_id']['enum'] = IDS
            value['description'] = ('Pick/lift the SELECTED registered upright box/cylinder/bottle; actual native contacts/lift required. '
                'No arbitrary mesh/visual grasp claim. ' if value['name'] == 'pick' else
                'Place the SELECTED held object at the locked original table/basket goal; native support/release/upright/1s stability required. '
                'No new object selection after motion. ') + 'Observe/inspect first; the ID must match this task selection.'
    agent.SKILL_TOOLS = tools + [copy.deepcopy(SELECT)]
    execution = copy.deepcopy(agent.EXECUTION_TOOLS)
    next(t['function'] for t in execution if t['function']['name'] == 'set_plan')['parameters']['properties']['placement_goal']['properties']['object_id']['enum'] = IDS
    agent.EXECUTION_TOOLS = execution
    agent.resolve_placement = resolve_goal; agent.bind_place = bind_goal; agent.object_state_from_feedback = object_state
    def catalogue(snapshot, policy_type='pi05'):
        value = scene_catalogue(snapshot, policy_type)
        value['skills'].append({'id': 'select_manipulation_object', 'description': SELECT['function']['description'],
            'parameters': SELECT['function']['parameters'], 'backends': ['tamp'], 'kind': 'scene_context', 'starts_motion': False})
        value['object_manipulation_scope'] = 'registered upright primitives; handover, relocation and GPU carrying remain bottle-only'
        if snapshot.get('diagnostics', {}).get('manipulation_scene', {}).get('object_id') != 'bottle':
            backend = next(item for item in value['adapters'] if item['id'] == 'curobo')
            backend['connection_ready'] = backend['ready']; backend['ready'] = False
            backend['reason'] = 'GPU连接另行保留；当前所选新对象的携物适配尚未验证'
        return value
    agent.catalogue = catalogue


def service_factory():
    """Extend owner-thread commands without copying the physics worker loop."""
    import queue
    import uuid
    import sim.service as service
    from sim.objects import ObjectEnv
    from sim.object_planner import ObjectPlanner
    service.TampPlanner = ObjectPlanner
    original_recorder = service.Recorder
    class ObjectRecorder(original_recorder):
        def __init__(self, env, *args, **kwargs):
            super().__init__(env, *args, **kwargs)
            self.manifest['task'] = 'Manipulate selected registered object: ' + env.object_id
            self.manifest['details']['object_selection'] = env.selection()
            from sim.recording import dump
            dump(self.path / 'manifest.json', self.manifest)
    service.Recorder = ObjectRecorder
    class OwnerCommands:
        def __init__(self, original, owner): self.original, self.owner = original, owner
        def put(self, value): return self.original.put(value)
        def get(self, *args, **kwargs):
            while True:
                command, values, future = self.original.get(*args, **kwargs)
                if command != 'select_object': return command, values, future
                if not future.set_running_or_notify_cancel(): continue
                owner = self.owner
                try:
                    if owner.job or owner.decision_wait: raise ValueError('physics owner is busy')
                    if values['observation_id'] != f'{owner.epoch}:{owner.env.tick}': raise ValueError('object selection observation expired')
                    owner.env.require_current_sources(); selected = owner.env.select_object(values['object_id'])
                    owner.epoch = uuid.uuid4().hex[:12]; owner.keyframes = []; owner.last_run = None
                    owner.decision_receipt = None; owner._publish()
                    future.set_result({**selected, 'observation_id': owner.cache['observation_id'],
                        'physical_state_preserved': True, 'fresh_observation_required': True})
                except Exception as exc: future.set_exception(exc)
    class ObjectService(service.SimulationService):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**object_environment_dependencies(kwargs,ObjectEnv))

        def call(self, command, **arguments):
            if command in {'policy', 'replay'}: raise ValueError('multi-object cell has no matching VLA/replay evaluation contract')
            if command == 'tamp':
                selected = self.snapshot()['diagnostics']['manipulation_scene']['object_id']
                params = arguments.get('parameters') or {}
                if params.get('object_id', selected) != selected: raise ValueError('select the object explicitly before motion')
                if selected != 'bottle' and params.get('planner') == 'curobo':
                    raise ValueError('GPU new-object attachment adapter is not yet validated; no backend substitution')
            return super().call(command, **arguments)
        def _worker(self):
            self.commands = OwnerCommands(self.commands, self)
            return super()._worker()
        def _publish(self):
            super()._publish()
            with self.lock:
                directory = self.env.directory.snapshot(self.cache['observation_id'])
                for row in directory['objects']:
                    if row['object_id'] in IDS:
                        row['affordances']['pick_provider'] = 'registered_primitive_tamp'
                self.cache['diagnostics']['scene_directory'] = directory
                self.cache['task'] = 'Registered upright primitive manipulation · explicit simulator geometry'
                if self.env.object_id != 'bottle': self.cache['diagnostics']['manipulation_scene'].pop('radius_m', None)
                self.cache['policy_configured'] = False
                self.cache['policy_status'] = {'ready': False, 'reason': 'multi-object development cell has no validated learned-policy contract'}
        def _run(self, kind, args):
            if kind != 'tamp': raise ValueError('multi-object owner only accepts explicit TAMP development skills')
            return super()._run(kind, args)
    return ObjectService
