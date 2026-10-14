"""Expose verified primitive GPU motion while retaining object goal guards."""
import copy

from .objects import ObjectHarness, ObjectWorkspace
from .scene import SceneHarness
from .objects import consolidate_objects


def consolidate_gpu(receipts, forgotten=()):
    rows = []; scopes = {}
    for receipt in receipts:
        item = copy.deepcopy(receipt)
        if item['tool'] in {'move_relative', 'move_to_pose'} and item['arguments'].get('planner', 'tamp') == 'curobo':
            ident = item['feedback'].get('object_id'); payload = item['feedback'].get('payload_state', 'unknown')
            if ident not in ('bottle', 'block', 'can'): continue
            scoped = str(item.get('context')) + '|gpu_object=' + ident + '|payload=' + payload
            scopes[scoped] = (item.get('context'), ident, payload); item['context'] = scoped
        rows.append(item)
    lessons = consolidate_objects(rows, forgotten)
    for lesson in lessons:
        if lesson['context'] not in scopes: continue
        context, ident, payload = scopes[lesson['context']]
        lesson.update(context=context, object_id=ident, payload_state=payload)
        label = {'bottle': '瓶子', 'block': '长方体', 'can': '圆柱'}[ident]
        lesson['title'] += ' · ' + ('持' + label if payload == 'single_hold' else '空手' if payload == 'empty' else '持物条件未知')
        lesson['evidence_scope'] += ' GPU携物表示只对应 ' + ident + ' 与 ' + payload + ' 条件。'
    return lessons


class ObjectGPUWorkspace(ObjectWorkspace):
    @staticmethod
    def _step_done(step, feedback, plan):
        goal = plan.get('placement_goal')
        if goal and step['checkpoint'] == 'motion_reached' and feedback.get('object_id') != goal['object_id']:
            return False
        return ObjectWorkspace._step_done(step, feedback, plan)


class ObjectGPUHarness(ObjectHarness):
    def _invoke(self, name, args, role, context=None):
        if name in {'move_relative', 'move_to_pose'} and args.get('planner') == 'curobo':
            # Reuse scene-target/duplicate-action/freshness/role/budget checks;
            # only the older bottle-only capability gate is replaced.
            return SceneHarness._invoke(self, name, args, role, context)
        return super()._invoke(name, args, role, context)


def object_gpu_dependencies(kwargs, planner, metadata_provider):
    """Supply the legacy object backend locally, unless the service selected another one."""
    result=dict(kwargs)
    result.setdefault('curobo_planner_factory',planner)
    result.setdefault('curobo_metadata_provider',metadata_provider)
    return result


def install_isolated():
    import harness.agent as agent
    from .objects import install_isolated as objects
    objects(); previous = agent.catalogue
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    for tool in tools:
        value = tool['function']; name = value['name']
        if name in {'move_relative', 'move_to_pose'}:
            value['parameters']['properties']['planner']['description'] = (
                'tamp or actual curobo. GPU supports selected arm empty or alone holding registered bottle/block/can; '
                'other-arm and dual holds remain unsupported. Actual object volume cover, native collision/path/contact check; no fallback.')
        if name == 'select_manipulation_object':
            value['description'] = value['description'].replace('handover/relocate and cuRobo carrying still only bottle',
                'handover/relocate still only bottle; cuRobo bounded TCP motion supports selected registered primitives')
    agent.SKILL_TOOLS = tools
    def catalogue(snapshot, policy_type='pi05'):
        value = previous(snapshot, policy_type)
        value['object_manipulation_scope'] = 'registered upright primitives: TAMP pick/place, cuRobo TCP motion; new-shape handover/relocate unavailable'
        value['object_gpu_scope'] = 'registered native bottle/block/can geometry; bounded empty or single-held TCP motion'
        backend = next(item for item in value['adapters'] if item['id'] == 'curobo')
        backend.update(snapshot.get('curobo', {'ready': False, 'reason': 'object GPU worker unavailable'}))
        if backend.get('ready'): backend.pop('reason', None)
        for skill in value['skills']:
            replacement = next((tool['function'] for tool in tools if tool['function']['name'] == skill['id']), None)
            if replacement: skill.update(description=replacement['description'], parameters=replacement['parameters'])
        return value
    agent.catalogue = catalogue


def service_factory():
    from .objects import service_factory as objects
    from sim.object_curobo import ObjectCuRoboPlanner, metadata
    import sim.service as service
    base = objects()
    class ObjectGPUService(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**object_gpu_dependencies(kwargs,ObjectCuRoboPlanner,metadata))

        def call(self, command, **arguments):
            if command == 'tamp' and (arguments.get('parameters') or {}).get('planner') == 'curobo':
                selected = self.snapshot()['diagnostics']['manipulation_scene']['object_id']
                if (arguments.get('parameters') or {}).get('object_id', selected) != selected:
                    raise ValueError('GPU request differs from selected object identity')
                return service.SimulationService.call(self, command, **arguments)
            return super().call(command, **arguments)
    return ObjectGPUService
