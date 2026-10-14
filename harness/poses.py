"""Optional orientation-aware proposals and memory scope; legacy router intact."""
import copy
import time
import numpy as np

from .task_queue import TaskHarness, TaskWorkspace
from .object_memory import consolidate as base_consolidate


def consolidate(receipts, forgotten=()):
    """Split native pose conditions before grouping, not after retrieval."""
    scoped = []; contexts = {}
    for receipt in receipts:
        row = copy.deepcopy(receipt)
        start = row.get('feedback', {}).get('grasp_start', {})
        if (row['tool'] == 'pick' and row['evidence_kind'] == 'simulator_oracle'
            and start.get('orientation_class') == 'non_upright'):
            key = str(row.get('context')) + '|start_orientation=non_upright'
            contexts[key] = row.get('context'); row['context'] = key
        scoped.append(row)
    lessons = base_consolidate(scoped, forgotten)
    for lesson in lessons:
        if lesson['context'] in contexts:
            lesson['context'] = contexts[lesson['context']]
            lesson.setdefault('conditions', {})['start_orientation'] = 'non_upright'
            lesson['title'] = lesson['title'].replace('抓取', '倒卧/倾斜抓取', 1)
            lesson['evidence_scope'] += ' 原生起始外形姿态为非直立；不能作为直立或未知外形的泛化证据。'
    return lessons


class PoseHarness(TaskHarness):
    def _task_progress(self, snapshot, context=None):
        value = super()._task_progress(snapshot, context)
        start = (self.workspace.active_plan or {}).get('start_grasp_admission', {})
        if value and start.get('checked') and not start['valid']:
            value.update(execution_stopped=True,
                next_action='Original start has native object overlap. Preserve the original task and finish failure; do not reset, move the fixture or reuse it as a valid planning sample.')
        return value

    def _lookahead(self, snapshot, goal, arm):
        angle, execution_id = super()._lookahead(snapshot, goal, arm)
        if execution_id is not None: return angle, execution_id
        from .goals import same_xy
        for execution in reversed((self.workspace.active_plan or {}).get('execution_tools', [])):
            if execution['skill'] == 'handover': break  # The grasp relation changed.
            if execution['skill'] != 'pick': continue
            try:
                receipt = self.workspace.execution_receipt('execute', execution['id'],
                    context=snapshot['memory_context'], include_oracle=True)
            except ValueError: break
            feedback = receipt['feedback']; future = feedback.get('future_placement', {})
            after = receipt.get('evidence', {}).get('after', {}).get('observation_id', '')
            if (goal['surface'] != 'table' or receipt['evidence_kind'] != 'simulator_oracle'
                or after.split(':')[0] != snapshot['observation_id'].split(':')[0]
                or not feedback.get('primitive_success') or feedback.get('held_by') != arm
                or feedback.get('verification_source') != 'native_two_finger_contact_and_lift'
                or feedback.get('object_id') != goal['object_id']
                or feedback.get('grasp_start', {}).get('orientation_class') != 'non_upright'
                or not same_xy(future.get('target_xy_m', []), goal['target_xy_m'])): break
            yaw = future.get('yaw_rad')
            if not isinstance(yaw, (int, float)) or not np.isfinite(yaw): break
            obj = next(row for row in snapshot['diagnostics']['scene_directory']['objects']
                if row['object_id'] == goal['object_id'])
            matrix = np.asarray(obj['rotation_matrix']); current_yaw = np.arctan2(matrix[1,0], matrix[0,0])
            offset = float(np.arctan2(np.sin(yaw-current_yaw), np.cos(yaw-current_yaw)))
            return offset, execution['id']
        return angle, execution_id

    def _invoke(self,name,args,role,context=None):
        from .task_queue import MOTIONS
        plan = self.workspace.active_plan or {}; start = plan.get('start_grasp_admission', {})
        if name == 'finish' and start.get('checked') and not start['valid']:
            from .agent import tools_for
            from .skills import validate_arguments
            spec = next(t['function']['parameters'] for t in tools_for(role) if t['function']['name'] == 'finish')
            validate_arguments(spec, args); snapshot = self.service.snapshot(); token = snapshot['observation_id']
            root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
            if (role != 'execute' or root != plan['turn_id'] or token.split(':')[0] != plan['epoch']
                or args['observation_id'] != token or time.monotonic()-self.observed.get(token,-1e10) > 60
                or snapshot.get('job') or plan.get('execution_tools')):
                raise ValueError('无效起点的提前报告仍需本任务的新观测，且不能冒充已执行的动作')
            if args['verdict'] not in ('failure', 'uncertain'):
                raise ValueError('起点已拒绝且没有执行动作，不能提交success')
            evidence = {'early_stop_cause':'initial_object_overlap', 'native_start_admission':copy.deepcopy(start),
                'no_robot_motion_started':True, 'no_execution_receipt_created':True}
            self._log('independent_assessment', args, {'reported':args['verdict'], 'report_evidence':evidence,
                'run_id':None, 'privileged':True})
            self.workspace.assess(None, args['verdict'])
            with self.workspace.lock: self.workspace.save_plan()
            result = {'assessment_recorded':True, 'reported_verdict':args['verdict'], 'report_evidence':evidence,
                'transfer_ready':False, 'note':'Original invalid start reported; no motion or success receipt fabricated.'}
            self._log(name,args,result);return result
        if name in MOTIONS and start.get('checked') and not start['valid']:
            raise ValueError('本任务起点存在物体穿模；不能继续运动、重置或改原目标，需保留证据并提交失败')
        inspected = False
        if name == 'set_plan' and args.get('placement_goal') and not self.workspace.active_plan:
            from .agent import tools_for
            from .skills import validate_arguments
            spec = next(t['function']['parameters'] for t in tools_for(role) if t['function']['name'] == 'set_plan')
            validate_arguments(spec, args)
            snapshot = self.service.snapshot(); token = snapshot['observation_id']
            if (role != 'execute' or not context or args.get('observation_id') != token
                or time.monotonic() - self.observed.get(token, -1e10) > 60):
                raise ValueError('先读取当前观测，再声明本次对象目标')
            if any(step.get('skill') == 'run_policy' for step in args['steps']):
                raise ValueError('原生几何放置计划不能混入未匹配的学习策略')
            # Reuse the SAME inspection/freshness/VLA-taint gate for an already
            # geometric object goal. No selection, motion, reset or new time.
            super()._invoke('inspect_manipulation', {'observation_id': token}, role, context)
            inspected = True
        if name=='pick':
            goal=(self.workspace.active_plan or {}).get('placement_goal')
            if goal and goal['surface']=='table' and goal['object_id']==args.get('object_id'):
                context=copy.deepcopy(context or {})
                context['pose_pick_lookahead']={'surface':'table','target_xy_m':copy.deepcopy(goal['target_xy_m']),
                    'position_tolerance_m':goal['tolerance_m']}
        result=super()._invoke(name,args,role,context)
        if inspected:
            admission = copy.deepcopy(self.service.snapshot()['diagnostics'].get('grasp_admission', {}))
            with self.workspace.lock:
                self.workspace.active_plan['start_grasp_admission'] = admission
                self.workspace.save_plan()
            result.update(native_preconditions_checked=True, contains_privileged_object_state=True,
                start_grasp_admission=admission,
                inspection_scope='same fresh native object-goal gate; no automatic motion or visual localisation')
        if name=='observe':
            result['active_scene_contract']['oriented_grasp_scope']=(
                'For a currently non-upright registered box/cylinder/bottle, pick can generate object-axis 6D primitive candidates. '
                'A declared table goal supplies scratch-only future-place constraints. All native contact/lift/placement gates remain mandatory. '
                'This is an experimental geometry provider, not a trained GraspGenX model or visual grasp detector. '
                'A requested pick.hold_seconds performs native hold validation before the skill returns. '
                'For non-upright table tasks, resolve_placement_transit can use the confirmed pick original-goal yaw hint with fresh grasp geometry; '
                'execute its exact motion_request with cuRobo, then place. '
                'Native block/bottle development tasks completed this route; tilted cylinder tasks remain failed. '
                'Do not infer arbitrary actions, API reliability or real transfer from these fixed geometric tests.')
        return result


class PoseWorkspace(TaskWorkspace):
    """Native start-pose conditions stay distinct from successful upright picks."""
    pass


def install_isolated():
    from . import agent, knowledge, object_memory
    # ObjectMemoryWorkspace has its own bounded/filter-before-top-k path.
    # Both paths must use the same grouping and stable forgetting IDs.
    knowledge.consolidate = consolidate
    object_memory.consolidate = consolidate
    knowledge.RECOVERY['grasp_search_exhausted'] = (
        '当前姿态候选未找到完整抓取及原目标放置方案；搜索耗尽不证明物理不可达。保留原目标和失败证据，不能重置后称为本次恢复。')
    knowledge.RECOVERY['unsupported_grasp_geometry'] = (
        '当前抓取候选只覆盖已注册盒子与圆柱外形；未知网格需独立抓取候选与外形检查，不能套用已有成功经验。')
    knowledge.RECOVERY['initial_object_overlap'] = (
        '物体起点与其它碰撞体相交，不是有效规划样本。保留原场景和任务，停止运动；不能自动移动物体、重置或改目标制造成功。')
    agent.Harness = PoseHarness
    previous_catalogue=agent.catalogue
    def catalogue(snapshot,policy_type='pi05'):
        result=previous_catalogue(snapshot,policy_type)
        functions = {tool['function']['name']:tool['function'] for tool in agent.SKILL_TOOLS}
        for skill in result['skills']:
            if skill['id'] in functions:
                skill['description'] = functions[skill['id']]['description']
                skill['parameters'] = copy.deepcopy(functions[skill['id']]['parameters'])
        result['oriented_grasp_scope']='experimental object-axis primitive candidates with original-goal lookahead; non-upright full tasks not generally validated'
        return result
    agent.catalogue=catalogue
    skills=copy.deepcopy(agent.SKILL_TOOLS)
    next(t['function'] for t in skills if t['function']['name']=='pick')['description']+=(
        ' Non-upright known primitive poses have experimental 6D candidates with original-place lookahead. '
        'Proposals are not success or arbitrary-mesh perception; do not loosen gates or reset a failing task.')
    pick = next(t['function'] for t in skills if t['function']['name'] == 'pick')
    pick['parameters']['properties']['hold_seconds'] = {'type': 'number', 'minimum': 2, 'maximum': 10,
        'description': 'Optional requested native hold validation after lift. Keeps the same actuator command and checks actual contact/relative pose; simulation time counts against the original task budget.'}
    pick['description'] += ' When asked to hold, set hold_seconds (up to10); completion waits for native validation.'
    agent.SKILL_TOOLS=skills
    execution = copy.deepcopy(agent.EXECUTION_TOOLS)
    next(t['function'] for t in execution if t['function']['name']=='finish')['description'] += (
        ' A trusted native start_grasp_admission rejection also permits failure/uncertain before motion; '
        'success is forbidden and no execution receipt is invented. Observe again, preserve the original task.')
    agent.EXECUTION_TOOLS = execution


def service_factory():
    from .task_queue import service_factory as tasks
    from .object_handover import RelayPlanner
    from sim.oriented_grasp import OrientedPlanner, HoldAudit, initial_conflicts
    base=tasks()
    previous=getattr(base,'_tamp_planner_factory_default',None)
    if previous is None:
        raise RuntimeError('pose service requires an explicit base TAMP planner factory')
    class Router:
        CONTRACT={**previous.CONTRACT,'oriented_grasp':'optional known native box/cylinder geometry; original executor/physics'}
        def __new__(cls,env,task,**kwargs):
            parameters=kwargs.get('parameters') or {}
            if task in ('pick_bottle','reach_bottle') and parameters and env.data.xmat[env.bottle_id].reshape(3,3)[2,2]<.98:
                planner = OrientedPlanner(env,task,**kwargs)
            else: planner = previous(env,task,**kwargs)
            if 'hold_seconds' in parameters:
                if task != 'pick_bottle': raise ValueError('hold_seconds belongs only to pick')
                return HoldAudit(planner, parameters['hold_seconds'])
            return planner
    class PoseService(base):
        _tamp_planner_factory_default=Router
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('tamp_planner_factory',Router)
            super().__init__(*args,**kwargs)

        def _publish(self):
            super()._publish()
            checked = not any(self.env.evaluation['held'].values()) and not any(self.env.evaluation['touching'].values())
            conflicts = initial_conflicts(self.env) if checked else []
            with self.lock:
                self.cache['diagnostics']['grasp_admission'] = {'checked':checked, 'valid':checked and not conflicts,
                    'object_id':self.env.object_id, 'observation_id':self.cache['observation_id'], 'conflicts':conflicts,
                    'planning_penetration_margin_m':.0005, 'source':'current native contacts; no pose edit',
                    'scope':'fresh unheld-object start check; not physical calibration or a replacement for execution contact gates'}

        def call(self,command,**arguments):
            if command=='tamp' and arguments.get('task')=='pick_bottle':
                lookahead=(arguments.get('initiator') or {}).get('pose_pick_lookahead')
                if lookahead:
                    arguments=copy.deepcopy(arguments)
                    arguments['parameters']={**arguments.get('parameters',{}),'next_placement':copy.deepcopy(lookahead)}
            return super().call(command,**arguments)
    return PoseService
