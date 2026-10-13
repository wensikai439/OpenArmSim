"""Resolve object goals once; trajectories may change, user targets must not."""
import copy
import math
from sim.placement import placement_tolerance


PLACEMENT_SCHEMA={'type':'object','properties':{
    'object_id':{'type':'string','enum':['bottle']},
    'surface':{'type':'string','enum':['table','basket']},
    'target_xy_m':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},
    'offset_xy_m':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}},
    'required':['object_id','surface'],'additionalProperties':False}


def same_xy(left,right):
    return len(left)==len(right)==2 and all(abs(a-b)<=1e-6 for a,b in zip(left,right))


def resolve_placement(spec,snapshot,previous=None,initial_snapshot=None):
    if previous and spec is None:return copy.deepcopy(previous)
    if spec is None:return None
    if 'target_xy_m' in spec and 'offset_xy_m' in spec:raise ValueError('放置目标只选绝对位置或相对偏移之一')
    surface=spec['surface']
    scene=snapshot.get('diagnostics',{}).get('manipulation_scene',{})
    area=scene.get('surfaces',{}).get(surface)
    if not area or scene.get('object_id')!=spec['object_id']:raise ValueError('先 inspect_manipulation 获取当前对象与放置区域')
    origin=initial_snapshot or snapshot
    if origin['observation_id'].split(':')[0]!=snapshot['observation_id'].split(':')[0]:
        raise ValueError('用户目标的初始观测已因重置失效，请开启新任务')
    initial=previous['anchor']['bottle_xy_m'] if previous else origin['diagnostics']['manipulation_scene']['position_m'][:2]
    if 'target_xy_m' in spec:target=spec['target_xy_m']
    elif 'offset_xy_m' in spec:
        if surface!='table':raise ValueError('篮子目标使用容器中心或绝对位置，不接受相对瓶位偏移')
        target=[a+b for a,b in zip(initial,spec['offset_xy_m'])]
    elif previous:target=previous['target_xy_m']
    elif surface=='basket':target=[(a+b)/2 for a,b in zip(area['xy_min'],area['xy_max'])]
    else:raise ValueError('桌面目标需要 target_xy_m 或 offset_xy_m')
    if len(target)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in target):raise ValueError('invalid object target')
    margin=scene.get('radius_m',.028)+.01
    if any(v<lo+margin or v>hi-margin for v,lo,hi in zip(target,area['xy_min'],area['xy_max'])):
        raise ValueError('物体完整外形无法落在所选支撑面内')
    goal={'kind':'object_placement','object_id':spec['object_id'],'surface':surface,'target_xy_m':list(target),
          'offset_xy_m':spec.get('offset_xy_m'),'tolerance_m':placement_tolerance(surface,target,initial),
          'anchor':{'observation_id':origin['observation_id'],'bottle_xy_m':list(initial),'source':'simulator_ground_truth'}}
    if previous:
        if any(goal[k]!=previous[k] for k in ('object_id','surface')) or not same_xy(target,previous['target_xy_m']):
            raise ValueError('同一用户任务的物体目标已经锁定；恢复可改路线或机械臂，不能悄悄换目标')
        return copy.deepcopy(previous)
    return goal


def bind_place(arguments,goal):
    if arguments['object_id']!=goal['object_id'] or arguments['receptacle_id']!=goal['surface']:
        raise ValueError('放置对象或支撑面与本次任务目标不一致')
    proposed=arguments.get('target_xy_m')
    if 'offset_xy_m' in arguments:
        if proposed is not None:raise ValueError('放置只选一个目标参数')
        proposed=[a+b for a,b in zip(goal['anchor']['bottle_xy_m'],arguments['offset_xy_m'])]
    if proposed is not None and not same_xy(proposed,goal['target_xy_m']):
        raise ValueError('放置位置与已锁定物体目标不一致；可省略位置参数以使用原目标')
    resolved={k:v for k,v in arguments.items() if k not in ('offset_xy_m','target_xy_m')}
    resolved['target_xy_m']=list(goal['target_xy_m'])
    resolved['goal_anchor_observation_id']=goal['anchor']['observation_id']
    resolved['position_tolerance_m']=goal['tolerance_m']
    return resolved


def preserve_completed(previous,requested):
    completed=[] if previous is None else [{k:v for k,v in step.items() if k!='status'}
        for step in previous['steps'] if step.get('status')=='done' and step['checkpoint']!='visual_review']
    def binding(step):
        return (step['checkpoint'],None if step['checkpoint']=='observation' else step.get('skill'),step.get('occurrence',1))
    keys={binding(step) for step in completed};ids={step['id'] for step in completed};steps=completed[:]
    for value in requested:
        step=copy.deepcopy(value)
        # A model often renumbers a recovery plan from s1. Preserve actual
        # completed operations, but never silently discard a NEW skill.
        if binding(step) in keys and (step.get('skill') or step['checkpoint']=='observation'):continue
        original=step['id'];suffix=1
        while step['id'] in ids:
            step['id']=original[:23]+f'_replan{suffix}';suffix+=1
        ids.add(step['id']);steps.append(step)
    if len(steps)>8:raise ValueError('保留已完成阶段后超过八阶段；请合并尚未执行的阶段')
    return steps


def assess_placement(goal,snapshot):
    """Final physical goal check, independent of model verdict or old benchmark."""
    diagnostic=snapshot.get('diagnostics',{})
    evaluation=diagnostic.get('evaluation',{})
    placement=evaluation.get('placement',{})
    scene=diagnostic.get('manipulation_scene',{})
    position=scene.get('position_m',[])
    matching=placement.get('surface')==goal['surface'] and same_xy(placement.get('xy_m',[]),goal['target_xy_m'])
    matching=matching and placement.get('tolerance_m',float('inf'))<=goal['tolerance_m']
    error=math.dist(position[:2],goal['target_xy_m']) if len(position)>=2 else None
    confirmed=bool(matching and placement.get('confirmed') is True and not evaluation.get('invalid')
                   and not (snapshot.get('decision_timing') or {}).get('task_failed')
                   and error is not None and error<=goal['tolerance_m']
                   and not any(evaluation.get('touching',{}).values()))
    return {'confirmed':confirmed,'source':'native_support_release_and_stability',
            'target_matches':matching,'position_error_m':error,'surface':goal['surface'],
            'tolerance_m':goal['tolerance_m'],'stable_seconds':placement.get('stable_seconds',0),'transfer_ready':False}


def object_state_from_feedback(skill,feedback):
    """Current scene facts from a verified receipt; no fixed-demo assumptions."""
    if not feedback.get('primitive_success'):return None
    source=feedback.get('verification_source')
    if skill in {'pick','handover'}:
        expected={'pick':'native_two_finger_contact_and_lift','handover':'native_receiver_contact_then_donor_release'}[skill]
        arm=feedback.get('held_by')
        if source!=expected or arm not in ('left','right'):return None
        return {'object_id':'bottle','state':'held','arm':arm,'confidence':1.0}
    if skill in {'place','relocate'} and source=='native_support_release_and_stability' and feedback.get('placement',{}).get('confirmed'):
        surface=feedback['placement'].get('surface')
        if surface not in ('table','basket'):return None
        return {'object_id':'bottle','state':'on_surface' if surface=='table' else 'in_container',
                'arm':'none','container_id':surface,'confidence':1.0}
    return None
