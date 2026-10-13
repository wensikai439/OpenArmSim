"""Promote reusable, evidence-backed lessons; receipts are experiment records.

No model verdict, arbitrary prompt, image bytes or transient pose is promoted.
Grouping is by skill/arm/direction/failure cause, not individual target points.
"""
import hashlib
import json

NAMES={'move_relative':'末端相对移动','move_to_pose':'目标位姿运动','pick':'抓取','set_gripper':'夹爪开合',
       'handover':'交接','place':'放置','relocate':'连续搬运','run_policy':'学习策略'}
ARMS={'left':'左手','right':'右手'}
PROTOCOLS={'relocate': '每次锁定物体目标，以新原生状态重新规划并逐步验证；确认松手、撤离和稳定支撑后再用新相机复查。历史坐标不可复用；核对模型与底层规划等待的计时条件，旧记录不证明包含规划等待的性能。成功搬运不证明长期夹持或真机迁移。'}

def relocation_route(feedback):
    """Describe completed evidence, never the requested or failed route."""
    labels=[]
    for step in feedback.get('substeps',[]):
        if step.get('status')!='completed':continue
        args=step.get('parameters',{});skill=step.get('skill')
        arm=ARMS.get(args.get('arm'),'所选手')
        if skill=='pick':labels.append(arm+'抓取')
        elif skill=='handover':
            donor=ARMS.get(args.get('from_arm'),'供物手')
            receiver=ARMS.get(args.get('to_arm'),'接收手')
            labels.append(donor+'交给'+receiver)
        elif skill in ('move_relative','move_to_pose'):
            labels.append(('cuRobo' if step.get('backend')=='curobo' else 'TAMP')+'携物运动')
        elif skill=='place':
            surface={'table':'桌面','basket':'篮内'}.get(args.get('receptacle_id'),'目标处')
            labels.append(arm+surface+'放置')
    return ' → '.join(labels)

RECOVERY={
 'planning_state_changed':'规划等待期间状态变化；旧路径未执行。用新观测重新规划并保留原目标，不切换计时方式、加时或关闭检查。',
 'episode_finished':'新任务先 prepare_simulation 保留当前有效场景，再观察并制定计划；无效场景需明确重置。',
 'ik_unreachable':'目标位置和朝向需同时可达；换接近朝向或目标，重新规划，不重复同一个失败目标。',
 'grasp_pose_unreachable':'当前抓取位姿不可达；尝试另一种接近朝向，或由另一只手先递送。',
 'handover_unreachable':'交接需同时满足两只手的可达与碰撞条件；调整供物手姿态或交接区域。',
 'goal_collision':'目标处有碰撞；调整目标或接近方向，不能仅降低速度或关闭碰撞检查。',
 'start_collision':'起始状态存在碰撞；先检查并明确恢复场景，不能强行继续路径。',
 'grasp_lost':'失去持续双指接触后停止携物运动；重新观察，不把抬手当作抓起物体。',
 'dual_hold_not_confirmed':'接收手未确认夹持时，供物手不得释放。',
 'receiver_grasp_lost':'供物手释放后仍需验证接收手独立夹持；工具退出不能证明交接成功。',
 'donor_hold_required':'先抓取并抬升供物手中的瓶子，再调用交接。',
 'curobo_no_path':'当前几何与姿态未找到可执行路径；换目标或朝向，不自动改用另一规划器。',
 'distinct_arms_required':'供物手和接收手必须是两只不同机械臂。',
 'policy_gripper_range':'夹爪输出超出已核验动作范围时停止；先检查模型、归一化器和夹爪契约，不能靠放宽范围掩盖问题。',
 'policy_identity_mismatch':'推理服务的权重身份必须匹配配置；核对检查点与后端，不替换成另一模型继续执行。',
 'orientation_tracking_error':'位置接近目标仍可能朝向未到位；需要完整位姿反馈修正或更换可达姿态，不能只看位置或反复降低速度。',
 'tracking_error':'根据实测位置和朝向误差重新求解；守住碰撞与误差判据，不能用计划末端代替实际末端。',
 'placement_unreachable':'当前抓取姿态无法到达指定放置点时，可先交给另一只手再规划放置；保持原目标，不复用失败姿态。',
 'single_hold_required':'先用新场景状态确认哪只手持瓶；空手先抓取，双手共同夹持时先完成交接。',
 'placement_not_confirmed':'松手不等于完成；核对物体目标误差、实际支撑、夹爪脱离和连续稳定时间。',
 'release_pose_not_reached':'松手前检查瓶子本体误差；先修正携物位姿，末端到位不能代替物体到位。',
 'retreat_not_clear':'释放后夹爪仍可能包围物体；重新规划撤离并复核闭合外形，不得宣布已安全撤离。',
 'curobo_empty_gripper_required':'当前 cuRobo 接口仅支持空手运动；持物路径请使用具备携物碰撞检查的 TAMP，不自动切换后端。',
}

def consolidate(receipts,forgotten=()):
    groups={}
    for item in receipts:
        if item.get('compatible') is False or item['tool'] not in NAMES:continue
        outcome=item['outcome']
        if outcome in ('cancelled','unverified'):continue
        args=item['arguments'];feedback=item['feedback'];code=feedback.get('failure_code')
        if item['tool']=='relocate' and outcome=='success':
            placed=any(s.get('skill')=='place' and s.get('status')=='completed'
                and s.get('feedback',{}).get('primitive_success')
                and s.get('feedback',{}).get('verification_source')=='native_support_release_and_stability'
                for s in feedback.get('substeps',[]))
            if (feedback.get('verification_source')!='native_support_release_and_stability'
                    or not feedback.get('placement',{}).get('confirmed')
                    or not placed or not relocation_route(feedback)):
                continue
        message=str(feedback.get('message',feedback.get('error',''))).lower()
        if code in (None,'start_rejected','tool_contract_rejected'):
            if 'gripper' in message and ('range' in message or 'outside' in message):code='policy_gripper_range'
            elif 'checkpoint' in message and ('identity' in message or 'mismatch' in message):code='policy_identity_mismatch'
        if outcome in ('failure','rejected'):
            if code not in RECOVERY:continue  # low-signal/generic errors stay in diagnostics
        backend=args.get('transport_planner',args.get('planner',args.get('backend','tamp')))
        arm=args.get('arm');donor=args.get('from_arm');receiver=args.get('to_arm',args.get('via_arm'))
        surface=args.get('receptacle_id',feedback.get('surface')) if item['tool'] in {'place','relocate'} else None
        blocked=feedback.get('planning_diagnostics',{}).get('blocked_pairs',[])
        obstacles=sorted({tuple(sorted(p.get('geoms',[]))) if isinstance(p,dict) else (str(p),) for p in blocked})
        family=[item['tool'],backend,arm,donor,receiver,surface,outcome,code,obstacles,item.get('context'),item['evidence_kind']]
        if item['tool']=='relocate':
            route=[(s.get('skill'),s.get('backend'),s.get('parameters',{}).get('arm'),
                    s.get('parameters',{}).get('from_arm'),s.get('parameters',{}).get('to_arm'))
                   for s in feedback.get('substeps',[]) if s.get('status')=='completed']
            family.extend([bool(args.get('allow_handover_recovery')),route])
        key='knowledge:'+hashlib.sha256(json.dumps(family,sort_keys=True).encode()).hexdigest()[:24]
        if key in forgotten:continue
        if key not in groups:
            title=NAMES[item['tool']]+' · '+backend
            if arm:title+=' · '+ARMS[arm]
            if donor and receiver:title+=' · '+ARMS[donor]+' → '+ARMS[receiver]
            elif item['tool']=='relocate' and receiver:title+=' → '+ARMS[receiver]
            if surface:title+=' · '+{'table':'桌面','basket':'篮子'}.get(surface,surface)
            groups[key]={'id':key,'title':title,'tool':item['tool'],'backend':backend,'outcome':outcome,
                         'evidence_kind':item['evidence_kind'],'repeat_count':0,'updated':0.,'sources':[],
                         'timing_conditions':{},'planner_timing_conditions':{},
                         'failure_code':code,'context':item.get('context'),'surface':surface,'errors_mm':[]}
            if item['tool']=='relocate' and outcome=='success':
                groups[key]['protocol_id']='relocate'
                groups[key]['route']=relocation_route(feedback)
        group=groups[key];group['repeat_count']+=item.get('repeat_count',1)
        timing=feedback.get('decision_timing_mode','paused')
        group['timing_conditions'][timing]=group['timing_conditions'].get(timing,0)+item.get('repeat_count',1)
        planning=feedback.get('planning_timing',{})
        condition=(planning.get('schema','unknown')+':'+planning.get('mode','unknown')) if planning else 'legacy_unmodeled'
        group['planner_timing_conditions'][condition]=group['planner_timing_conditions'].get(condition,0)+item.get('repeat_count',1)
        group['updated']=max(group['updated'],item.get('created',0))
        if len(group['sources'])<3:
            evidence=item.get('evidence',{})
            group['sources'].append({'receipt_id':item['id'],'run_id':evidence.get('run_id'),
                                     'evidence':evidence})
        if feedback.get('position_error_m') is not None:
            group['errors_mm'].append(float(feedback['position_error_m'])*1000)
    lessons=[]
    for group in groups.values():
        count=group['repeat_count'];tool=group['tool'];code=group['failure_code']
        # One tiny successful move is a log entry, not durable knowledge.
        if group['outcome']=='success' and tool in ('move_to_pose','move_relative') and count<2:continue
        if code:summary=RECOVERY[code]
        elif tool in ('move_relative','move_to_pose'):
            summary='每次从新观测读取末端位姿，再求解并复核整条路径；历史目标坐标不可直接复用。'
        elif tool=='pick':summary='闭合后须确认持续双指接触和物体实际抬升；继续持瓶前还需检查瓶子相对夹爪的滑动。接触与抬升不代表持续稳定夹持。'
        elif tool=='handover':summary='先确认接收手夹持，再释放供物手，撤离后再次确认接收手独立持物。'
        elif tool=='set_gripper':summary='开合后核对编码器稳定与物体接触；夹爪闭合不等于已经抓起，放置后需确认夹爪已撤离。'
        elif tool=='relocate':summary='已验证路线：'+group['route']+'。'
        elif tool=='place':summary=('从当前持瓶姿态规划到桌面目标，松手并撤离后，核验直立、完整外形、支撑和连续稳定。' if group['surface']=='table' else
                                   '放置后核验容器内完整外形、支撑与稳定；松手不等于放置成功。')
        else:summary='仅记录匹配权重和动作契约下的执行；策略退出不能代替独立任务评分。'
        group['summary']=summary
        group['evidence_scope']=('仅对应这一失败原因及其仿真条件。' if code else {
            'pick':'已测候选场景的接触与抬升；未证明长期夹持、视觉定位或真机迁移。',
            'handover':'接收手接触、供物手释放与独立持物；未验证长期夹持。',
            'place':'本次物体目标、支撑、松手与连续稳定；不等于开放物体放置能力。',
            'relocate':'声明瓶子、目标与路线的原生组合验收；不证明长时间悬空夹持、开放物体或真实迁移。',
            'move_to_pose':'本次末端位姿反馈与路径检查；不能代替物体搬运。',
            'move_relative':'本次末端位姿反馈与路径检查；不能代替物体搬运。',
            'set_gripper':'编码器稳定，仅验证开合；不能证明持物。',
            'run_policy':'匹配权重与动作契约；任务退出不能代替独立评分。',
        }[tool])
        group['support']=f'{count} 次兼容记录支持；只覆盖已测场景，不代表泛化成功率。'
        group['kind']='failure_lesson' if code else 'skill_lesson'
        errors=group.pop('errors_mm')
        if errors:group['measured_error_mm']=[round(min(errors),2),round(max(errors),2)]
        lessons.append(group)
    return sorted(lessons,key=lambda g:(g['outcome']!='success',g['updated']),reverse=True)

def index_entry(lesson):
    item={k:lesson[k] for k in ('id','title','summary','support','outcome','tool','evidence_kind','updated','failure_code','repeat_count','evidence_scope','timing_conditions','planner_timing_conditions')}
    if lesson.get('protocol_id'):item['protocol_id']=lesson['protocol_id']
    return item
