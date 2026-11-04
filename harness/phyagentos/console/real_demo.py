"""One-run real demo view. Uses the existing relative-stage CLI; never retries motion."""
from pathlib import Path
from urllib.parse import urlparse
import asyncio, hashlib, json, time, re
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route
ROOT=Path(__file__).resolve().parents[1]
CAMERAS=('cam_middle','cam_left','cam_right')
JOB=None

def run_dir():
    run=Path(json.loads((ROOT/'config/real_demo.json').read_text())['run']).resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Demo run outside project runs')
    return run

def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def emit(run, kind, title, **data):
    with (run/'demo-events.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=time.time(),kind=kind,title=title,**data),ensure_ascii=False)+'\n')

def snapshot():
    run=run_dir();now=time.time();plan_path=run/'demo-plan.json';plan=read(plan_path)
    cameras={}
    for name in CAMERAS:
        m=read(run/(name+'-latest.json'),{})
        age=now-m.get('saved_wall_s',0)
        cameras[name]=dict(**m,age_s=age,fresh=0<=age<2.)
    events=[]
    if (run/'demo-events.jsonl').exists():
        events=[json.loads(l) for l in (run/'demo-events.jsonl').read_text().splitlines() if l][-120:]
    active=read(run/'demo-worker.json')
    stages=[]
    for p in sorted(run.glob('[0-9][0-9]-*')):
        if not p.is_dir():continue
        result=read(p/'forge-result.json')
        recorded=next((e for e in reversed(events) if e.get('stage')==p.name and e.get('kind')=='result'),{})
        native=(result or {}).get('result',{}).get('data',{}).get('result',{})
        stages.append(dict(name=p.name,phase=(result or {}).get('status',{}).get('data',{}).get('phase','failed' if recorded.get('exit_code',0) else 'pending'),invocation_id=(result or {}).get('invocation_id'),native_status=native.get('status'),physical_success=native.get('physical_success'),result_file=str((p/'forge-result.json').relative_to(run)) if result else None))
    return dict(run=run.name,goal='黑框取瓶 → 确认稳定夹持并提起 → 放入红桶',planner='当前 Codex · 基于实时画面逐段规划',chain='Forge piper_real.run_task → piper-real-runtime → native-four-can-replay/v1',server_wall_s=now,cameras=cameras,events=events,worker=active,stages=stages,plan=plan,plan_digest=hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan else None,can_execute=bool(plan and not (run/plan['stage']).exists() and all(c['fresh'] for c in cameras.values()) and not (JOB and not JOB.done())),feedback=read(run/'latest-feedback.json'),outcome=read(run/'demo-outcome.json'))

async def state(request):
    return JSONResponse(snapshot(),headers={'Cache-Control':'no-store'})

async def frame(request):
    name=request.path_params['camera']
    if name not in CAMERAS:return JSONResponse({'error':'Unknown camera'},404)
    p=run_dir()/(name+'-latest.jpg')
    if not p.exists():return JSONResponse({'error':'No camera frame'},404)
    return FileResponse(p,media_type='image/jpeg',headers={'Cache-Control':'no-store'})

async def artifact(request):
    run=run_dir();p=(run/request.query_params.get('path','')).resolve()
    if not p.is_relative_to(run) or not p.is_file() or p.suffix not in {'.jpg','.png','.json','.mp4','.md','.txt'}:
        return JSONResponse({'error':'Unavailable evidence'},404)
    return FileResponse(p,headers={'Cache-Control':'no-store'})

async def launch(run,plan,owner):
    log=run/(plan['stage']+'-dispatch.log')
    try:
        args=['/usr/bin/python3','-B',str(run/'dispatch-expert-stage.py'),plan['stage'],plan['prepared_id'],'--seconds',str(plan.get('seconds',8))]
        if plan.get('ik'):args.append('--ik')
        for side in ('left','right'):
            if plan.get('keep_'+side+'_closed'):args.append('--keep-'+side+'-closed')
            if side+'_gripper' in plan:args+=['--'+side+'-gripper',str(plan[side+'_gripper'])]
            if side+'_delta' in plan:args+=['--'+side+'-delta='+','.join(str(v) for v in plan[side+'_delta'])]
        (run/'demo-worker.json').write_text(json.dumps(dict(stage=plan['stage'],phase='running',started_wall_s=time.time(),command=args)))
        emit(run,'call','调用 Forge 真机 Skill',name='piper_real.run_task',stage=plan['stage'],arguments={'prepared_id':plan['prepared_id']},plan=plan)
        with log.open('x') as f:
            proc=await asyncio.create_subprocess_exec(*args,cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT)
            code=await proc.wait()
        result=read(run/plan['stage']/'forge-result.json')
        phase=(result or {}).get('status',{}).get('data',{}).get('phase','unknown' if code==0 else 'failed')
        emit(run,'result','原生调用返回 · '+phase,stage=plan['stage'],exit_code=code,result=result,log=log.name)
        (run/'demo-worker.json').write_text(json.dumps(dict(stage=plan['stage'],phase=phase,exit_code=code,ended_wall_s=time.time())))
    except Exception as exc:
        emit(run,'error','调用结果需要核查',stage=plan['stage'],error=str(exc))
        (run/'demo-worker.json').write_text(json.dumps(dict(stage=plan['stage'],phase='unknown',error=str(exc))))
    finally:
        owner.control_busy=False

async def execute(request):
    global JOB
    from console.server import S,check_origin
    try:
        check_origin(request)
        body=await request.json();run=run_dir();snap=snapshot();plan=snap['plan']
        if S.job or S.background_busy() or S.coordinator.store.active() or S.control_busy or (JOB and not JOB.done()):
            raise ValueError('已有活动任务，不能并发执行')
        if not plan or body.get('plan_digest')!=snap['plan_digest']:raise ValueError('计划已变化，请重新读取当前计划')
        if not snap['can_execute']:raise ValueError('画面过期或动作已提交；不重复下发')
        if not 0<=time.time()-plan['created_wall_s']<120:raise ValueError('计划已过期，需要按最新反馈重新规划')
        if not re.fullmatch(r'[0-9]{2}-[a-z0-9-]+',plan['stage']) or not re.fullmatch(r'[a-zA-Z0-9_-]+',plan['prepared_id']):raise ValueError('非法动作标识')
        if not 4<=plan.get('seconds',8)<=20:raise ValueError('动作时长不在已验证范围')
        if plan.get('ik'):
            ik_path=run/(plan['stage']+'-ik.json')
            if hashlib.sha256(ik_path.read_bytes()).hexdigest()!=plan.get('ik_sha256'):raise ValueError('IK 计划摘要已变化')
        marker=run/(plan['stage']+'.submitted')
        with marker.open('x') as f:f.write(json.dumps(body))
        S.control_busy=True
        JOB=asyncio.create_task(launch(run,plan,S))
        return JSONResponse(dict(accepted=True,stage=plan['stage']),status_code=202)
    except (ValueError,FileExistsError,KeyError) as exc:
        return JSONResponse({'error':str(exc)},status_code=409)

routes=[Route('/api/real-demo/state',state),Route('/api/real-demo/frame/{camera}',frame),Route('/api/real-demo/artifact',artifact),Route('/api/real-demo/execute',execute,methods=['POST'])]
