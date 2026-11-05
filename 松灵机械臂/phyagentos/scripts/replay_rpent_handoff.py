"""Replay the proven handoff through official bound Core/Forge APIs, then enforce-verify."""
from __future__ import annotations
import os
import argparse,asyncio,hashlib,inspect,json,socket,sys,time,urllib.request
from pathlib import Path
from PhyAgentOS.config.loader import load_config
from PhyAgentOS.cli.commands import _make_provider,_make_forge_components
from PhyAgentOS.agent.experience.store import ExperienceStore
from PhyAgentOS.agent.experience.activation import SkillActivationManager
from PhyAgentOS.verification.contracts import TaskVerificationContract
ROOT=Path(str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--reset',action='store_true');p.add_argument('--route',default='runs/ik-handoff-20261004/handoff-06/validated-route.json');args=p.parse_args()
if not args.label or not all(c.isalnum() or c in '-_' for c in args.label):raise ValueError('unique label required')
route_path=(ROOT/args.route).resolve();route=json.loads(route_path.read_text())
if route.get('physical_complete') is not True:raise ValueError('route must be from an actually completed native episode')
out=ROOT/'runs/ik-handoff-20261004'/args.label;out.mkdir(parents=True,exist_ok=False)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));actions=[]

def save(name,value):
    (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')

def console(path,body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request('http://127.0.0.1:19081'+path,data=data,headers={'Content-Type':'application/json'})
    with opener.open(req,timeout=90) as r:return json.load(r)

def data(response):
    if response.get('ok') is False:raise RuntimeError(str(response.get('error')))
    return response.get('data',response)

def observation(response):
    return data(response)['response']['result']['outputs']['observation']

async def main():
    state=console('/api/state');save('before-console-state.json',state)
    rs=state.get('runtime',{}).get('state',{}) or {}
    if any(state.get(k) for k in ['job','active_task','control_busy','background_busy','reflection_busy']) or any(rs.get(k) for k in ['active_invocations','active_sessions','active_task_bindings']):raise RuntimeError('owned runtime is not idle; no reset or motion dispatched')
    if args.reset:save('reset-receipt.json',console('/api/runtime',{'action':'reset'}))
    save('route-source.json',{'path':str(route_path),'sha256':hashlib.sha256(route_path.read_bytes()).hexdigest(),'actions':route['actions'],'execution':'deterministic replay through official Task coordinator; no autonomous LLM planning'})
    cfg=load_config()
    # This standalone coordinator needs its own verifier session beside the live console.
    with socket.socket() as verifier_socket:
        verifier_socket.bind(('127.0.0.1',0))
        cfg.agents.verification.service_port=verifier_socket.getsockname()[1]
    provider=_make_provider(cfg)
    save('verifier-service.json',{'host':cfg.agents.verification.service_host,'port':cfg.agents.verification.service_port,'model':cfg.agents.verification.model or cfg.agents.defaults.model,'selection':'process-local ephemeral port; existing console verifier preserved'})
    client,ids,coord,availability=_make_forge_components(cfg,provider)
    activation_manager=SkillActivationManager(workspace=cfg.workspace_path,store=ExperienceStore(cfg.workspace_path),binding_resolver=coord.binding_resolver,runtime_availability_provider=availability)
    coord.set_activation_manager(activation_manager)
    session='console:ik-handoff-'+args.label
    description='Execute the independently proven 35-action assisted air-handoff recipe on the RPent dual-arm simulation through task-bound Forge tools: horizontal tabletop pickup, right-to-left bilateral-contact transfer, bucket-clearance carry, and physical release/settling. The old Learned Workflow for relative arm offsets does not apply to this object-transfer task; use general registered RPent primitive semantics.'
    activation_manager.begin_turn(session,description)
    activated=activation_manager.activate(session_key=session,name='rpent-piper-sim',role='primary')
    if inspect.isawaitable(activated):activated=await activated
    activation,content,lessons=activated
    save('activation.json',activation.model_dump(mode='json'))
    contract=TaskVerificationContract.model_validate({
        'mode':'enforce','goal':'Complete actual contact-assisted right-to-left airborne bottle transfer, then release and physically settle the bottle in the left-side bucket using the proven composable recipe.',
        'success_criteria':[
            'All 35 task-bound native Actions complete with primitive_completed=true and verified=true, with no failed/unknown/unresolved Action.',
            'The new same-episode composable_transfers entry has donor=right, receiver=left, at least two receiver contact geoms, qpos_change=0, table_contact=false, and bottle base z>0.78m at transfer.',
            'Final same-task observe reports bucket_bottom_contact=true, bucket_xy_error_m<0.055, bottle_speed_norm<0.05, both assist_active=false, failed=false, collision_stop=null, and every numerical_warnings counter=0.',
            'Final physics_step and sim_time_s increase relative to the initial same-task observe, with actual before/after robot-state and both camera evidence.'
        ],
        'constraints':[
            'Simulation only, project-owned RPent air-handoff instance; preserve native contact-triggered assistance and report assisted grasp, not pure friction.',
            'All Query/Action calls use this frozen task binding and registered Forge tools; no direct simulator SDK/RPC actions, hardware, qpos writes, database edits, relaxed thresholds, or replay of unknown effects.',
            'The user authorized this handoff recipe; the prior Learned Workflow scope for opening grippers/small offsets is not applicable and must not be claimed reused for handoff.',
            'Preserve native task_success=false and air_handoff_verified=false if returned. Those staged-macro-only flags are never set by composable actions; assess this contract from real composable_transfers, contacts and bucket settling, and never rewrite flags.'
        ],
        'evidence_policy':{'required_kinds':['rgb_image','robot_state'],'required_sources':['image/cam_right','image/overview'],'minimum_association':'best_effort'}
    })
    save('contract.json',contract.model_dump(mode='json'))
    task=await coord.create_task(task_description=description,verification=contract,activation_id=activation.activation_id,origin_session_key=session)
    save('task-created.json',task.model_dump(mode='json'));task_id=task.task_id;print('TASK '+task_id,flush=True)
    error=None
    try:
        initial_receipt=await coord.invoke_query(task_id,'rpent.observe',{},timeout_ms=30000);save('initial-query.json',initial_receipt);initial=observation(initial_receipt)
        if initial['composable_transfers'] or any(initial['assist_active'].values()) or initial['failed'] or initial['physics_step']!=0:raise RuntimeError('route replay requires a fresh factory episode; no motion dispatched')
        if sum((a-b)**2 for a,b in zip(initial['bottle_world_m'],route['initial_bottle_world_m']))>1e-10:raise RuntimeError('factory bottle layout does not match proven route')
        for step in route['actions']:
            row={'sequence':len(actions)+1,'label':step['label'],'tool_id':'rpent.'+step['id'],'arguments':step['payload'],'requested_at':time.time()};actions.append(row);save('action-records.json',actions)
            accepted=await coord.start_action(task_id,row['tool_id'],row['arguments'],timeout_ms=30000);row['accepted']=accepted;save('action-records.json',actions);iid=data(accepted)['invocation_id'];deadline=time.monotonic()+45
            while True:
                receipt=await client.invocation_status(iid);coord.observe_action(task_id,iid,receipt);row['status_receipt']=receipt;d=data(receipt);save('action-records.json',actions)
                if d.get('phase') in ['completed','failed','rejected','cancelled','expired']:break
                if time.monotonic()>deadline:raise TimeoutError('Action wait expired; preserved invocation; no replay: '+iid)
                await asyncio.sleep(.5)
            native=d.get('result',{}).get('outputs',{}).get('native_result',{})
            row['completed_at']=time.time();save('action-records.json',actions)
            print(json.dumps({'sequence':row['sequence'],'label':row['label'],'phase':d.get('phase'),'native_completed':native.get('primitive_completed'),'control_steps':native.get('control_steps')},ensure_ascii=False),flush=True)
            if d.get('result',{}).get('status')!='succeeded' or native.get('primitive_completed') is not True or native.get('verified') is not True:raise RuntimeError('Native/Forge Action failed; no replay: '+iid)
        final_receipt=await coord.invoke_query(task_id,'rpent.observe',{},timeout_ms=30000);save('final-query.json',final_receipt)
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)};save('execution-error.json',error)
    finally:
        current=coord.get_task(task_id)
        if all(r.terminal for r in current.execution_records):
            finalized=await coord.finalize_task(task_id);save('task-final.json',finalized.model_dump(mode='json'));print('FINAL '+finalized.status.value,flush=True)
        save('task-latest.json',coord.get_task(task_id).model_dump(mode='json'))
        save('task-events.json',coord.store.events(task_id,limit=200));coord.verifier.stop();await client.close();save('after-console-state.json',console('/api/state'))
    task=coord.get_task(task_id);save('summary.json',{'task_id':task_id,'status':task.status.value,'verification':task.verdict.model_dump(mode='json') if task.verdict else None,'actions':len(actions),'execution_error':error,'planning':'deterministic native-validated recipe','semantic_verifier_model':cfg.agents.verification.model or cfg.agents.defaults.model,'reflection':'not automatically scheduled by this thin coordinator client'})
    if error or task.status.value!='succeeded':raise RuntimeError('Task did not succeed; inspect preserved receipts')

try:asyncio.run(main())
except BaseException as exc:
    save('runner-error.json',{'type':type(exc).__name__,'message':str(exc)});print(type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
