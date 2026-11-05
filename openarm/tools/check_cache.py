"""Frozen input generation and isolated real-GPU cache consistency probes."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-cache'


def inputs():
    import mujoco
    import numpy as np
    from sim.environment import OpenArmEnv
    from sim.tamp import TampPlanner
    from sim.curobo import joint_state,scene,attachment
    from sim.recording import Recorder,dump
    path=OUT/'requests.json'
    if path.exists():raise ValueError('preserve frozen inputs')
    env=OpenArmEnv(images=False);env.reset(seed=0);rec=None
    def payload(e):
        body=e.ee_ids['right'];rot=e.data.xmat[body].reshape(3,3)
        point=e.data.xpos[body]+rot@np.array([0,.006,.075]);quat=np.empty(4)
        mujoco.mju_mat2Quat(quat,rot.ravel())
        return {'arm':'right','joints':joint_state(e),'scene':scene(e,carrying=True),
                'attachment':attachment(e,'right'),'position_m':(point+[.015,0,0]).tolist(),
                'quaternion_wxyz':quat.tolist()}
    try:
        preparation={'arm':'right','joints':joint_state(env),'scene':scene(env,carrying=True),'carrying':True}
        rec=Recorder(env,'tamp',root=OUT/'grasp',record_images=False,seed=0,
            details={'purpose':'frozen backend probe input, not a latency-aware manipulation evaluation'})
        planner=TampPlanner(env,'pick_bottle',parameters={'object_id':'bottle','arm':'right'})
        before=rec.observation(env)
        for action,diag in planner.actions():
            _,_,term,trunc,info=env.step_tamp(action);after=rec.observation(env)
            rec.append(before,action,after,info,terminated=term,truncated=trunc,diagnostics=diag);before=after
            if term or trunc:raise ValueError('native grasp failed before probes')
        assert env.evaluation['held']['right']
        requests={'picked':payload(env)}
        for name,delta in [('small_lock_change',.001),('large_lock_change',.15)]:
            shadow=copy.copy(env);shadow.data=mujoco.MjData(env.model)
            mujoco.mj_copyData(shadow.data,env.model,env.data)
            shadow.data.qpos[env.qids['left'][0]]+=delta
            mujoco.mj_forward(env.model,shadow.data)
            requests[name]=payload(shadow)
        dump(path,{'preparation':preparation,'requests':requests,'manifest':env.manifest(),
            'grasp_run':str(rec.path.relative_to(ROOT)),
            'scope':'current native grasp plus two scratch-MjData FK perturbations; GPU probes dispatch no robot action',
            'hardware_access':False,'transfer_ready':False})
        rec.finish('primitive_completed',env.info(),metrics={'tamp':planner.result})
        print(json.dumps({'frozen_requests':len(requests),'grasp_steps':env.tick}),flush=True)
    finally:
        if rec and not rec._closed:rec.finish('error',env.info(),error='input generation incomplete')
        env.close()


def gpu(output):
    from sim.curobo_server import Planner
    target=OUT/output
    if target.exists():raise ValueError('preserve previous GPU probe')
    raw=(OUT/'requests.json').read_bytes();data=json.loads(raw)
    worker=Planner();rows=[]
    started=time.monotonic()
    try:
        preparation=worker.prepare(data['preparation'])
    except Exception as exc:
        preparation={'error':str(exc),'wall_seconds':time.monotonic()-started}
    for key in ['picked','small_lock_change','large_lock_change','picked']:
        started=time.monotonic()
        try:response=worker.plan(data['requests'][key]);error=None
        except Exception as exc:response=None;error=str(exc)
        rows.append({'case':key,'wall_seconds':time.monotonic()-started,'response':response,'error':error})
        result={'input_sha256':hashlib.sha256(raw).hexdigest(),'preparation':preparation,'cases':rows,
            'worker_metadata':worker.metadata,'isolated_GPU_process':True,'robot_commands_dispatched':0,
            'source':'official cuRobo CUDA planner, not fake backend; no live manipulation during this probe',
            'hardware_access':False,'transfer_ready':False}
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'case':key,'error':error,'wall_seconds':rows[-1]['wall_seconds'],
                         'cache':(response or {}).get('cache')},ensure_ascii=False),flush=True)
    if worker.cached:worker.cached[1].destroy()


def collision(output):
    """Hypothetical payload self-collision; zero physics edits or dispatch."""
    import numpy as np
    from sim.curobo_server import Planner
    target=OUT/output
    if target.exists():raise ValueError('preserve previous collision diagnostic')
    raw=(OUT/'requests.json').read_bytes();data=json.loads(raw);worker=Planner()
    try:
        preparation=worker.prepare(data['preparation'])
        original=data['requests']['picked'];baseline=worker.plan(original)
        if not baseline['success']:raise ValueError('same frozen baseline must be feasible')
        full=worker.kin.compute_kinematics(worker.state(original))
        spheres=full.get_link_spheres().detach().cpu().numpy().reshape(-1,4)
        params=worker.kin.config.kinematics_config
        ids=params.get_sphere_index_from_link_name('openarm_right_link6').detach().cpu().numpy()
        tcp=full.tool_poses.get_link_pose('right_candidate_tcp')
        position=tcp.position.detach().cpu().numpy().reshape(3)
        quat=tcp.quaternion.detach().cpu().numpy().reshape(4)
        w,x,y,z=quat
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
            [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
            [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
        center=spheres[ids[np.argmin(np.linalg.norm(spheres[ids,:3]-position,axis=1))],:3]
        local=rotation.T@(center-position)
        probe=copy.deepcopy(original)
        probe['attachment']['spheres_tcp']=[[*local.tolist(),.04]]*len(original['attachment']['spheres_tcp'])
        started=time.monotonic();response=worker.plan(probe)
        overlap=response['attachment']['diagnostics']['checked_self_overlap_m']
        good=not response['success'] and overlap>0 and response['cache']['reused']
        result={'passed':good,'wall_seconds':time.monotonic()-started,'preparation':preparation,
            'baseline':baseline,'response':response,'diagnostic_request':probe,
            'input_sha256':hashlib.sha256(raw).hexdigest(),'worker_metadata':worker.metadata,
            'scope':'hypothetical payload sphere overlapping a checked forearm sphere; official CUDA collision rejection, no physical object edit',
            'robot_commands_dispatched':0,'hardware_access':False,'transfer_ready':False}
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'passed':good,'planner_success':response['success'],'checked_overlap_m':overlap}),flush=True)
        if not good:raise ValueError('cached CUDA planner did not reject declared overlap')
    finally:
        if worker.cached:worker.cached[1].destroy()


def native(output):
    """Real MuJoCo owner, including measured preparation/planner waits."""
    from sim.service import SimulationService
    from sim.recording import dump
    from tools.check_planning import CASES
    target=OUT/output
    if target.exists():raise ValueError('preserve previous native cache study')
    selected={name:copy.deepcopy(CASES[name]) for name in ('curobo','transfer')}
    selected['transfer'].update(transport_planner='curobo',transport_delta_m=[.015,0,0])
    protocol={'cases':selected,'timing_mode':'measured_delay','budget_s':120,'seed':0,'episode':0,
        'reset_before_each_condition':True,'physics_parameters_changed':False,'scoring_thresholds_changed':False,
        'cold_preparation_counted_as_physics_wait':True,'actual_LLM_API':False,
        'scope':'two declared development cases, single seed; no statistical generalization',
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hardware_access':False,'transfer_ready':False}
    dump(target.with_suffix('.protocol.json'),protocol)
    service=SimulationService();rows=[]
    try:
        for name,parameters in selected.items():
            service.call('reset',episode=0,seed=0);service.call('task_budget',seconds=120)
            initial=service.snapshot();manifest=service.call('manifest');started=time.monotonic()
            response=service.call('tamp',task='relocate_object',skill='relocate',parameters=parameters,
                observation_id=initial['observation_id'],timing_mode='measured_delay',realtime=False)
            deadline=time.monotonic()+300
            while True:
                state=service.snapshot()
                if not state.get('job'):break
                if time.monotonic()>deadline:raise TimeoutError('same native owner remains live; do not restart')
                time.sleep(.1)
            last=state['last_run'];summary=json.loads((ROOT/'runs'/last['run_id']/'summary.json').read_text())
            row={'case':name,'parameters':parameters,'response':response,'status':last['status'],'error':last['error'],
                'run_id':last['run_id'],'wall_seconds':time.monotonic()-started,
                'duration_sim_seconds':summary['duration_sim_seconds'],'metrics':last['metrics'],
                'terminal_evaluation':state['diagnostics']['evaluation'],
                'model_evidence_unchanged':manifest['model_evidence']==service.call('manifest')['model_evidence']}
            rows.append(row);dump(target,{'protocol':protocol,'cases':rows,'complete':len(rows)==len(selected)})
            print(json.dumps({k:row[k] for k in ('case','status','error','duration_sim_seconds')},ensure_ascii=False),flush=True)
    finally:service.close()


def isolated_api(output,*,arm='left',via_arm='right',distance_m=.06,experiment_details=None):
    """Same Harness and saved API, separate native owner/in-memory journal."""
    import sqlite3
    import os
    import numpy as np
    # Match the production launcher so historical evidence is admitted only
    # under the same policy contract, never by bypassing compatibility checks.
    os.environ.setdefault('OPENARM_POLICY_CONFIG',str(ROOT/'harness/policy.json'))
    from sim.service import SimulationService
    from sim.recording import dump
    from harness.agent import Harness
    from harness.workspace import Workspace
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    target=OUT/output
    if target.exists():raise ValueError('preserve previous isolated API task')
    if arm not in ('left','right') or via_arm not in (None,'left','right') or via_arm==arm:
        raise ValueError('valid selected and optional receiver arms required')
    if distance_m not in (.06,.08):raise ValueError('declared development distance required')
    labels={'left':'左手','right':'右手'}
    parameters={'object_id':'bottle','arm':arm,'receptacle_id':'table',
                'transport_planner':'curobo','transport_delta_m':[.015,0,0]}
    if via_arm:parameters['via_arm']=via_arm
    prompt=('只在本次独立仿真实验做一次连续搬运。第一个set_plan锁定timing_mode=measured_delay和time_budget_s=120。'
        f'用{labels[arm]}拿起瓶子，'+(f'交给{labels[via_arm]}，' if via_arm else '')+
        f'再用cuRobo携物向world正X移动1.5厘米，最后{labels[via_arm or arm]}放到任务起始瓶位向前{distance_m*100:g}厘米的桌面上。'
        '只调用一次relocate，由底层逐步确认交接、携物、放置、松手、撤离和稳定支撑，再用新观测复查。'
        '阶段由你拟定，搬运条件绑定placement_confirmed/relocate。目标不能改变，禁止任务中重置、加时或切回paused。'
        '明确路线参数：'+json.dumps(parameters,ensure_ascii=False))
    dump(target.with_suffix('.protocol.json'),{'prompt':prompt,'parameters':parameters,'seed':0,'episode':0,
        'budget_s':120,'timing_mode':'measured_delay','same_production_Harness':True,
        'independent_physics_owner':True,'main_page_commands':0,'production_memory_writes':False,
        'policy_config_sha256':hashlib.sha256((ROOT/'harness/policy.json').read_bytes()).hexdigest(),
        'experiment_details':experiment_details or {},
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hardware_access':False,'transfer_ready':False})
    service=SimulationService();workspace=Workspace();harness=None
    try:
        # Only historical memory is copied into RAM. Current user tasks and
        # their working state must not become this independent experiment.
        with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as source:
            for table in ('notes','episodes','experiences','recipes','forgotten_knowledge'):
                columns=[r[1] for r in source.execute('PRAGMA table_info('+table+')')]
                rows=source.execute('SELECT * FROM '+table).fetchall()
                if rows:workspace.db.executemany('INSERT INTO '+table+' ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',rows)
        workspace.db.commit()
        settings=ModelSettings(store=SecretStore())
        if not settings.public()['configured']:raise ValueError('saved API unavailable')
        harness=Harness(service,settings=settings,workspace=workspace)
        service.call('reset',episode=0,seed=0);initial=service.snapshot()
        memory=workspace.memory('execute',initial['observation_id'].split(':')[0],
            context=initial['memory_context'],compact=True)
        started=time.monotonic();response=harness.chat(prompt);root=response['turn_id']
        print(json.dumps({'started':True,'turn_id':root,'main_page_commands':0}),flush=True)
        deadline=time.monotonic()+600
        revision=None;stage_history=[];stage_key=None;published_stage_key=None
        while True:
            state=service.snapshot();agent=workspace.summary(state)
            turns=[t for t in workspace.public('execute',state['observation_id'].split(':')[0],state['memory_context'])['turns']
                   if t['id']==root or t.get('parent')==root]
            plan_now=agent.get('plan') or {};job_now=state.get('job') or {}
            milestones=[(s['id'],s['status']) for s in plan_now.get('steps',[])]
            key=(job_now.get('execution_id'),tuple(milestones))
            if key!=stage_key:
                stage_history.append({'observation_id':state['observation_id'],'sim_time':state.get('observation',{}).get('sim_time'),
                    'outer_execution_id':job_now.get('execution_id'),'stage_bindings':plan_now.get('stage_bindings',{}),
                    'steps':milestones,'completed_substeps':job_now.get('completed_substeps',[]),
                    'outer_running':bool(state.get('job'))})
                stage_key=key
            if revision!=agent['revision'] or key!=published_stage_key:
                dump(target.with_suffix('.progress.json'),{'status':'in_progress','turn_id':root,'turns':turns,
                    'agent':agent,'native_last_run':state.get('last_run'),
                    'memory_context':state['memory_context'],'main_page_commands':0,
                    'stage_history':stage_history,'production_memory_writes':False,'hardware_access':False})
                revision=agent['revision'];published_stage_key=key
            if (not state.get('job') and not agent['busy'] and turns
                    and all(t['status'] not in ('running','waiting') for t in turns)
                    and not harness.pending_review):break
            if time.monotonic()>deadline:raise TimeoutError('same isolated task requires inspection, no reset')
            time.sleep(.5)
        plan=agent.get('plan') or {};receipts=[]
        for execution in plan.get('execution_tools',[]):
            row=workspace.db.execute('SELECT feedback,evidence FROM experiences WHERE id=?',(execution['id'],)).fetchone()
            if row:receipts.append({'execution':execution,'feedback':json.loads(row[0]),'evidence':json.loads(row[1])})
        expected=np.asarray(initial['diagnostics']['manipulation_scene']['position_m'][:2])+[distance_m,0]
        declared=np.asarray((plan.get('placement_goal') or {}).get('target_xy_m',[]))
        target_unchanged=bool(declared.shape==(2,) and np.linalg.norm(declared-expected)<1e-6)
        moves=[r for r in receipts if r['execution']['skill']=='relocate']
        steps=moves[0]['feedback'].get('substeps',[]) if len(moves)==1 else []
        verified_curobo=any(s['backend']=='curobo' and s['status']=='completed'
            and s['feedback'].get('native_path_revalidation') and s['feedback'].get('fallback') is False for s in steps)
        verified_transfer=(via_arm is None or any(s['skill']=='handover' and s['status']=='completed'
            and s['parameters'].get('from_arm')==arm and s['parameters'].get('to_arm')==via_arm for s in steps))
        native_good=bool(len(moves)==1 and moves[0]['feedback'].get('primitive_success') and target_unchanged
            and verified_curobo and verified_transfer and (plan.get('goal_assessment') or {}).get('confirmed')
            and plan.get('timing_mode')=='measured_delay' and plan.get('time_budget_s')==120)
        review_complete=plan.get('verdict') in ('success','failure','uncertain')
        good=native_good and review_complete and all(t['status']=='completed' for t in turns)
        result={'status':'passed' if good else 'failed','turn_id':root,'turns':turns,'feedbacks':receipts,
            'native_goal_confirmed':native_good,'review_complete':review_complete,
            'actual_api_calls':sum(e['kind']=='api' for t in turns for e in t.get('trace',[])),
            'wall_seconds':time.monotonic()-started,'model':settings.public(),
            'initial_observation_id':initial['observation_id'],'expected_target_xy_m':expected.tolist(),
            'target_unchanged':target_unchanged,'verified_curobo':verified_curobo,'verified_transfer':verified_transfer,
            'terminal':{**state,'agent':agent},'model_verdict':plan.get('verdict'),'stage_history':stage_history,
            'default_memory':{'methods':len(memory['knowledge']['index']),'preferences':len(memory['notes']),
                'raw_experiences':len(memory['experiences']),'raw_episodes':len(memory['episodes']),
                'shared_protocols':len(memory['knowledge']['protocols'])},
            'independent_physics_owner':True,'main_page_commands':0,'production_memory_writes':False,
            'experiment_details':experiment_details or {},
            'scope':'single declared real-API development case in independent native owner; not statistical generalization or real transfer',
            'hardware_access':False,'transfer_ready':False}
        dump(target,result)
        print(json.dumps({k:result[k] for k in ('status','actual_api_calls','model_verdict','verified_curobo','verified_transfer')},ensure_ascii=False),flush=True)
    finally:
        if harness:harness.close()
        service.close();workspace.db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['inputs','gpu','collision','native','isolated_api'],required=True)
    parser.add_argument('--output',default='gpu.json');args=parser.parse_args()
    if '/' in args.output or not args.output.endswith('.json'):raise ValueError('simple JSON result filename required')
    if args.mode=='inputs':inputs()
    else:globals()[args.mode](args.output)
