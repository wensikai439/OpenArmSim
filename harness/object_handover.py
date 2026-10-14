"""Expose connected native relays without replacing the original bottle demo."""
import copy

import numpy as np

from .object_gpu import ObjectGPUHarness
from .scene import SceneHarness
from .objects import IDS
from sim.objects import ObjectEnv
from sim.object_planner import ObjectPlanner
from sim.object_handover import ObjectHandoverPlanner, ObjectRelayPickPlanner


def relay_environment_dependencies(kwargs, environment_factory):
    """Keep the protected-object fallback local and honor outer scene adapters."""
    result=dict(kwargs)
    result.setdefault('environment_factory',environment_factory)
    return result


class ProtectedObjectEnv(ObjectEnv):
    """Additional execution guard; model/controller/contact parameters intact."""
    def __init__(self, *args, **kwargs):
        self.protected_objects = {}; super().__init__(*args, **kwargs); self._protect()

    def _protect(self):
        if not hasattr(self, 'registry'): return
        self.protected_objects = {ident: self.data.xpos[row['body']].copy()
            for ident, row in self.registry.items() if ident != self.object_id}

    def reset(self, **kwargs):
        self.protected_objects = {}; value = super().reset(**kwargs); self._protect(); return value

    def select_object(self, ident):
        changed = ident != self.object_id; value = super().select_object(ident)
        if changed: self._protect()
        return value

    def _evaluate(self, dt):
        value = super()._evaluate(dt)
        protected = getattr(self, 'protected_objects', {})
        drift = {ident: float(np.linalg.norm(self.data.xpos[self.registry[ident]['body']] - origin))
            for ident, origin in protected.items()}
        displaced = [ident for ident, distance in drift.items() if distance > .003]
        value['non_target_object_guard'] = {'source': 'native registered-body displacement from task selection; hidden diagnostic',
            'displacement_m': drift, 'limit_m': .003, 'passed': not displaced,
            'scope': 'additional development execution gate; not a measured physical tolerance'}
        if displaced:
            value['invalid'] = True; value['success'] = False
            value['failure_reasons'] = value.get('failure_reasons', []) + ['non-target object displaced: ' + ', '.join(displaced)]
            if value.get('placement'): value['placement']['confirmed'] = False
        return value


class RelayPlanner:
    CONTRACT = {**ObjectPlanner.CONTRACT, 'registered_object_relay': ObjectHandoverPlanner.CONTRACT,
        'legacy_demo_preserved': True, 'non_target_displacement_limit_m': .003}

    def __new__(cls, env, task, **kwargs):
        parameters = kwargs.get('parameters') or {}
        if task == 'handover_bottle' and env.object_id != 'bottle':
            return ObjectHandoverPlanner(env, task, **kwargs)
        if task == 'pick_bottle' and env.object_id == 'can' and parameters.get('next_handover'):
            return ObjectRelayPickPlanner(env, task, **kwargs)
        return ObjectPlanner(env, task, **kwargs)


class ObjectRelayHarness(ObjectGPUHarness):
    def _task_progress(self, snapshot, context=None):
        value = super()._task_progress(snapshot, context)
        if value and snapshot['diagnostics']['evaluation'].get('invalid'):
            value['execution_stopped'] = True
            value['next_action'] = 'Finish failure or uncertain. Native execution is invalid; no further motion or reset within this goal.'
        return value

    def _invoke(self, name, args, role, context=None):
        if name in {'pick', 'handover', 'place', 'relocate', 'move_relative', 'move_to_pose', 'set_gripper', 'run_policy'}:
            if self.service.snapshot()['diagnostics']['evaluation'].get('invalid'):
                raise ValueError('原生执行已判无效；本目标不能继续运动或重置。请读取失败证据并finish failure/uncertain。')
        if name == 'handover':
            selected = self.service.snapshot()['diagnostics']['manipulation_scene']['object_id']
            if args.get('object_id') != selected: raise ValueError('handover object differs from this task selection')
            initiator = copy.deepcopy(context or {})
            goal = (self.workspace.active_plan or {}).get('placement_goal')
            if selected != 'bottle' and goal and goal['surface'] == 'table':
                initiator['relay_lookahead'] = {'surface': 'table', 'target_xy_m': copy.deepcopy(goal['target_xy_m'])}
            return SceneHarness._invoke(self, name, args, role, initiator or None)
        result = super()._invoke(name, args, role, context)
        if name == 'observe':
            result['active_scene_contract'] = {'registered_objects': IDS,
                'native_skills': ['pick', 'handover', 'move_to_pose', 'move_relative', 'place'],
                'handover': 'both directions are geometry hypotheses; tested block left→right and can right→left only',
                'can_pick_intent': 'for planned relay, next_handover.to_arm selects task-conditioned donor proposal ranking',
                'relocate': 'bottle-only composite; do not call it for block/can', 'learned_policy_available': False,
                'scope': 'explicit simulator geometry; tool schemas and this current scene contract qualify legacy bottle examples',
                'non_target_object_guard': '3mm position displacement limit; private diagnostic, no physical calibration claim'}
        return result


def install_isolated():
    import harness.agent as agent
    from .object_memory import install_isolated as memory
    memory(); agent.Harness = ObjectRelayHarness; previous = agent.catalogue
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    for tool in tools:
        value = tool['function']; name = value['name']
        if name == 'pick':
            value['parameters']['properties']['next_handover'] = {'type': 'object', 'additionalProperties': False,
                'properties': {'to_arm': {'type': 'string', 'enum': ['left', 'right']}}, 'required': ['to_arm']}
            value['description'] += (' If planning a can relay, declare next_handover.to_arm before pick to rank lower donor bands; '
                'the receiver must differ from arm. This is a grasp search hint, not a relay success guarantee.')
        if name == 'handover':
            value['parameters']['properties']['object_id']['enum'] = IDS
            value['description'] = ('Transfer the SELECTED upright registered object between distinct arms. '
                'Bottle retains the original directed route; box/cylinder search donor world yaw, receiver geometry and connected paths. '
                'Requires actual lifted donor hold and empty receiver. Native sustained dual contact, donor release/withdrawal and1s receiver-only hold required. '
                'A locked table goal adds scratch placement lookahead; next_placement_transit is only an unexecuted GPU transport proposal. '
                'Inspect fresh native feedback; no arbitrary-mesh or visual grasp claim. Composite relocate remains bottle-only.')
        if name == 'select_manipulation_object':
            value['description'] = value['description'].replace('handover/relocate still only bottle;',
                'registered box/cylinder directed handover available; composite relocate remains bottle-only;')
    agent.SKILL_TOOLS = tools
    def catalogue(snapshot, policy_type='pi05'):
        value = previous(snapshot, policy_type)
        value['object_manipulation_scope'] = 'registered upright bottle/box/cylinder: pick, directed native relay, cuRobo TCP transport, place; not arbitrary meshes'
        value['object_relay_evidence'] = {'native_cases': 'block left→right; can right→left; current candidate geometry',
            'scope': 'declared native development cases only; not a generalization rate or physical calibration'}
        for skill in value['skills']:
            replacement = next((tool['function'] for tool in tools if tool['function']['name'] == skill['id']), None)
            if replacement: skill.update(description=replacement['description'], parameters=replacement['parameters'])
        return value
    agent.catalogue = catalogue


def service_factory():
    from .object_gpu import service_factory as gpu
    import sim.service as service
    base = gpu(); service.TampPlanner = RelayPlanner
    class RelayService(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**relay_environment_dependencies(kwargs,ProtectedObjectEnv))

        def call(self, command, **arguments):
            if command == 'tamp' and arguments.get('task') == 'handover_bottle':
                lookahead = (arguments.get('initiator') or {}).get('relay_lookahead')
                if lookahead:
                    arguments = copy.deepcopy(arguments)
                    arguments['parameters'] = {**arguments.get('parameters', {}), 'next_place': copy.deepcopy(lookahead)}
            return super().call(command, **arguments)
    return RelayService
