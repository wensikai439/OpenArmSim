"""Expose the existing verified ACT endpoint, with its original skill bounds.

No arbitrary task language, oracle-taint bypass, backend substitution, extended
policy budget, action-range relaxation or automatic reset is introduced.
"""
import copy


def install_isolated(base):
    from . import agent
    previous = agent.catalogue
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    policy = next(item['function'] for item in tools if item['function']['name'] == 'run_policy')
    policy['description'] += (' The workbench exposes the local hash-verified ACT checkpoint as an experimental '
        'candidate-scene rollout. Select backend=act and skill_id=atomic_right_to_left_handover. '
        'ACT receives its own three camera images and encoder16 state. It is a trained whole-task policy, '
        'not a left-to-right transfer, general coordinate planner or arbitrary-object policy. '
        'Requires a fresh reset bottle scene, tick0, no oracle reads, and40s paused harness plan. '
        'Retain native action, collision and non-target-object gates. Interface readiness does not establish task success.')
    agent.SKILL_TOOLS = tools
    def catalogue(snapshot, policy_type='pi05'):
        value = previous(snapshot, policy_type)
        for skill in value['skills']:
            if skill['id'] == 'run_policy': skill['description'] = policy['description']
        backend = next(item for item in value['adapters'] if item['id'] == 'act')
        backend['connection_ready'] = snapshot.get('policy_status', {}).get('connection_ready', False)
        backend['performance_verified'] = False
        backend['scope'] = '本机 ACT 权重与接口已核验；候选场景内任务效果另测'
        value['policy_capability'] = snapshot.get('policy_status', {})
        return value
    agent.catalogue = catalogue
    class ActHarness(base):
        def _invoke(self, name, args, role, context=None):
            value = super()._invoke(name, args, role, context)
            if name == 'observe' and 'active_scene_contract' in value:
                snapshot = self.service.snapshot()
                value['active_scene_contract'].update(learned_policy_available=bool(snapshot.get('policy_configured')),
                    learned_policy=snapshot.get('policy_status'),
                    policy_prerequisites='reset bottle scene; no oracle inspection; original40s whole-policy plan')
            return value
    return ActHarness
