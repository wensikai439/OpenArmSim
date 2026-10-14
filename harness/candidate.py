"""Candidate-workbench execution and concise memory presentation.

Robot receipts remain the authoritative archive. User memory contains methods
and evidence links; the model's existing compact index is unchanged. Once a
relocation accepts a transport backend, recovery in the same goal retains it.
"""
import copy

from .composite import CompositeHarness, CompositeWorkspace, validate_composite_plan

CHECKPOINTS = {'observation', 'visual_review', 'tool_completed', 'motion_reached',
    'grasp_confirmed', 'handover_confirmed', 'placement_confirmed'}
CANDIDATE_RECOVERY = {
    'grasp_search_exhausted': ('已声明的几何候选未找到完整抓取路线；这不证明物理不可达。'
        '保留物体目标，用新观测检查候选与连接路径，尝试另一只手或明确扩展候选；不能重复同一失败搜索、改动物体或关闭检查。')
}


def validate_candidate_plan(values):
    steps = validate_composite_plan(values)
    for step in steps:
        if step['checkpoint'] not in CHECKPOINTS:
            raise ValueError('API 阶段只用当前技能回执：grasp_confirmed、handover_confirmed、motion_reached、placement_confirmed 或 tool_completed；旧演示的左右手事件不能用于通用计划。')
        if step['checkpoint'] == 'tool_completed' and not step.get('skill'):
            raise ValueError('tool_completed 阶段必须绑定实际 skill，不能用无关工具退出当作完成。')
    return steps

EXECUTION_CONTRACT = {
    'schema': 'candidate-execution/1',
    'backend_binding': 'first accepted relocate binds transport backend for this user goal, including continuation and plan revisions',
    'recovery': 'same backend may replan from fresh state; a new user goal may select another backend',
    'does_not_change': 'physical parameters, motion limits, pose/lift/stability checks or failure receipts',
    'stage_binding': 'generic observation/skill checkpoints only; fixed-demo direction events excluded from API plans',
    'scope': 'one transport backend per relocation goal; not arbitrary mixed-backend programs',
}


class CandidateWorkspace(CompositeWorkspace):
    def memory_recovery_rules(self):
        return {**super().memory_recovery_rules(), **CANDIDATE_RECOVERY}

    def public(self, *args, **kwargs):
        value = super().public(*args, **kwargs)
        # Default user memory needs provenance pointers, not repeated raw joint
        # arrays, camera evidence and detailed before/after poses.
        for lesson in value['memory']['knowledge']['lessons']:
            lesson['sources'] = [{'receipt_id': source['receipt_id'], 'run_id': source.get('run_id')}
                for source in lesson.get('sources', [])]
            if lesson.get('route'):
                lesson['route_label'] = lesson['route']
        value['memory']['presentation'] = {
            'raw_records_in_default_memory': False,
            'provenance': 'method evidence links; full receipts stay in experiment records and require explicit lookup',
        }
        return value


class CandidateHarness(CompositeHarness):
    def _invoke(self, name, args, role, context=None):
        root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
        session = self.sessions.get(root) if context and context.get('type') == 'llm' else None
        requested = (args.get('transport_planner', 'tamp') if name == 'relocate' else
            args.get('planner', 'tamp') if name in {'move_relative', 'move_to_pose'} else None)
        bound = session.get('transport_backend') if session else None
        if bound and requested and requested != bound:
            raise ValueError('本任务已绑定 ' + bound + ' 携物规划；失败后不能改用 ' + requested +
                ' 宣布路线完成。保留原目标，使用新观测在同一后端重规划，或如实结束；新的用户任务才能重新选择后端。')
        result = super()._invoke(name, args, role, context)
        if (session is not None and name == 'relocate' and result.get('execution_id')
                and result.get('state') == 'started'):
            session['transport_backend'] = requested
            result = {**result, 'execution_contract': copy.deepcopy(EXECUTION_CONTRACT),
                'bound_transport_planner': requested}
        return result


def install_isolated():
    import harness.agent as agent
    import harness.workspace as workspace
    workspace.Workspace = CandidateWorkspace
    workspace.validate_plan = agent.validate_plan = validate_candidate_plan
    tools = copy.deepcopy(agent.EXECUTION_TOOLS)
    plan = next(tool['function'] for tool in tools if tool['function']['name'] == 'set_plan')
    plan['parameters']['properties']['steps']['items']['properties']['checkpoint']['enum'] = sorted(CHECKPOINTS)
    plan['description'] += ' Use generic skill receipt checkpoints only. grasp_confirmed includes actual lift; handover_confirmed includes receiver-only hold. Fixed-demo right_lift/left_grasp events do not describe arbitrary directions.'
    agent.EXECUTION_TOOLS = tools
