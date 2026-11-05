"""Command UI adapter over the existing Agent and one-use native Forge path."""
from pathlib import Path
import asyncio, base64, dataclasses, hashlib, inspect, json, math, os, re, shlex, shutil, time, uuid
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from PhyAgentOS.agent.tools.base import Tool
from PhyAgentOS.forge.tool_client import ForgeToolClient
from PhyAgentOS.verification.contracts import TaskVerificationContract


REAL_EXECUTION_POLICY = """
## Command console: managed Forge lifecycle
In this real-robot mode, real_observe and real_relative_action are the registered
console adapters. The backend of real_relative_action starts the selected Runtime,
activates piper-real-runtime, queries live binding context, creates an actual
AgentTask with a frozen Skill binding, submits its Action through the official
AgentTaskCoordinator, records the native result, and finalizes the motion-segment
task. The model does not need separate activate_skill or forge_task_create calls;
their absence from this mode's tool list is expected, not a missing capability.
The adapter returns actual task_id, binding_id, invocation_id and task_status;
an exception prevents dispatch. Never invent these IDs or claim this lifecycle
ran before the tool returns them. Segment verification is execution-only;
physical_success remains unknown until the next real_observe supplies evidence.
Use real_observe before each relative action. Respect this request's allow_motion,
stop/cancel and unknown-result guards. Do not bypass them with any other tools.
"""

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs/real-commands'
ARCHIVE_ROOT=Path(os.environ.get('PAOS_ARCHIVE_ROOT', str(ROOT/'runs')))
CAMERAS=('cam_middle','cam_left','cam_right')
ARCHIVE=[
 {'id':'console-demo-20261024','title':'黑框取瓶到红桶','result':'成功 · 含中断历史版','count':10,'directory':'console-bottle-demo-20261024T093533','preview':'video-preview-complete'},
 {'id':'left-reset-20261024','title':'先放回红桶，再尝试黑框','result':'后续未完成','count':8,'directory':'left-bottle-reset-then-black-box-20261024T021937','preview':'video-preview'},
 {'id':'bottle-to-black-box-20261024','title':'从红桶移往黑框','result':'交接失败','count':5,'directory':'bottle-to-black-box-20261024T014103','preview':'video-preview'},
]
TERMINAL={'completed','succeeded','failed','cancelled','unknown','rejected','expired'}

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def validate_step(left_delta=None,right_delta=None,seconds=8.,left_gripper=None,right_gripper=None,translation_m=None,side='right',frame='base',keep_closed=None):
    values=locals().copy()
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 4<=seconds<=20:
        raise ValueError('每段时长须为4–20秒')
    for name in ('left_delta','right_delta'):
        q=values[name]
        if q is not None and (not isinstance(q,list) or len(q)!=6 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in q)):
            raise ValueError(name+'必须是6个有限数值，单位度')
    for name in ('left_gripper','right_gripper'):
        v=values[name]
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=.094):
            raise ValueError('夹爪目标须为0–0.094米')
    if translation_m is not None:
        if side not in ('left','right') or frame not in ('base','tool'):
            raise ValueError('无效的手臂或相对坐标系')
        if not isinstance(translation_m,list) or len(translation_m)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in translation_m):
            raise ValueError('相对平移须为3个有限数值，单位米')
        if values[side+'_delta'] is not None:raise ValueError('同一手臂不能同时指定关节增量与IK')
        if math.sqrt(sum(v*v for v in translation_m))>.06:raise ValueError('每段局部平移不超过60毫米')
    if keep_closed is not None and (not isinstance(keep_closed,list) or any(v not in ('left','right') for v in keep_closed)):
        raise ValueError('无效的保持夹持手臂')
    if all(values[k] is None for k in ('left_delta','right_delta','left_gripper','right_gripper','translation_m')):
        raise ValueError('请指定本次相对动作')
    return values

class RealCommands:
    def __init__(self,owner):
        self.owner=owner;self.current=None;self.camera_run=None;self.camera_unit=None
        self.invocation=None;self.client=None;self.task_id=None;self.cancel_requested=False;self.halted=False
        self.action_lock=asyncio.Lock();self.camera_lock=asyncio.Lock();self.last_vision=0
        self.camera_health={};self.camera_checked=0
        RUNS.mkdir(exist_ok=True)
        saved=read(RUNS/'camera.json',{})
        if saved.get('run'):self.camera_run=Path(saved['run']);self.camera_unit=saved.get('unit')

    def cameras(self):
        result={}
        for name in CAMERAS:
            meta=read(self.camera_run/(name+'-latest.json'),{}) if self.camera_run else {}
            age=time.time()-meta.get('saved_wall_s',0)
            result[name]={**meta,'age_s':age if meta else None,'fresh':bool(meta and 0<=age<2)}
        return result

    def configuration_busy(self):
        return bool(self.invocation or read(RUNS/'active.json'))

    def state(self):
        from console.providers import active_model
        cfg=read(ROOT/'config/real_runtime.json',{})
        active=read(RUNS/'active.json')
        current=read(self.current/'command.json') if self.current else None
        return {'model':active_model(self.owner.config),'skill':'piper-real-runtime','profile':'staged',
          'runtime_mode':cfg.get('mode'),'busy':bool(self.owner.job or self.owner.control_busy or self.invocation or self.owner.background_busy() or self.owner.reflection_busy() or self.owner.coordinator.store.active()),
          'current':current,'unresolved':active if active and not self.owner.job else None,
          'cameras':self.cameras(),'camera_unit':self.camera_unit,'camera_health':self.camera_health,'archive':archive_index(),
          'invocation_id':self.invocation,'motion_control':'启动反馈 + 相对增量',
          'physical_success':None}

    async def emit(self,kind,**data):
        job=self.owner.job or {}
        await self.owner.emit(kind,session=job.get('session'),request_id=job.get('request_id'),**data)

    async def shell(self,args,*,env=None):
        p=await asyncio.create_subprocess_exec(*args,cwd=ROOT,env=env,
          stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        output,_=await p.communicate()
        if p.returncode:raise RuntimeError(output.decode(errors='replace')[-3500:])
        return output.decode(errors='replace')

    def require_job(self,motion=False):
        job=self.owner.job or {}
        if not self.current or job.get('real_command_run')!=str(self.current):
            raise ValueError('只能由前端当前命令调用此技能')
        if motion and (not job.get('allow_motion') or self.cancel_requested or self.halted):
            raise ValueError('本次未授权动作，或动作已停止；不能继续下发')
        return job

    def capture(self):
        self.require_job()
        cams=self.cameras()
        if not all(c['fresh'] for c in cams.values()):raise ValueError('三路现场画面未开启或已过期，请先开启现场观察')
        dest=self.current/'observations'/str(time.time_ns());dest.mkdir(parents=True)
        for name in CAMERAS:
            raw=(self.camera_run/(name+'-latest.jpg')).read_bytes()
            (dest/(name+'.jpg')).write_bytes(raw)
            (self.current/(name+'-latest.jpg')).write_bytes(raw)
            save(self.current/(name+'-latest.json'),cams[name])
        return dest,cams

    async def observe(self,question='查看瓶子、夹爪、容器的位置和遮挡，区分已观察到的事实与不确定内容。'):
        self.last_vision=0
        dest,cams=self.capture()
        env=dict(os.environ,PAOS_COMMAND_RUN=str(self.current))
        feedback=await self.shell([os.environ.get('PIPER_CONTROL_PYTHON','/usr/bin/python3'),'-B','-c',"import scripts.real_command_stage as s,json; print(json.dumps(s.observe()))"],env=env)
        feedback=json.loads(feedback)
        save(dest/'feedback.json',feedback)
        messages=[{'type':'text','text':question+'\n三张图片依次为全景、左腕、右腕。以下为同次观察读取的真实被动遥测，单位见字段：\n'+json.dumps(feedback,ensure_ascii=False)+'\n仅报告图像与遥测事实；图像无法判断的内容明确说明。负开度是原始反馈异常，不可当作物理负宽度。不从程序完成推断抓取成功，动作决策由外层Agent负责。'}]
        for name in CAMERAS:
            messages.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode((dest/(name+'.jpg')).read_bytes()).decode()}})
        result=await self.owner.agent.provider.chat_with_retry(messages=[{'role':'user','content':messages}],max_tokens=1200)
        if getattr(result,'finish_reason',None)=='error' or not result.content:raise RuntimeError('当前模型未能读取现场画面：'+str(result.content))
        save(dest/'observation.json',{'cameras':cams,'analysis':result.content})
        self.last_vision=time.time()
        images=['/api/real-command/artifact?run='+self.current.name+'&path=observations/'+dest.name+'/'+name+'.jpg' for name in CAMERAS]
        await self.emit('progress',content=result.content,media=images)
        return {'observation':result.content,'feedback':feedback,'images':images,'physical_success':None}


    async def bind_stage(self,target,job):
        # Each bounded segment has its own real activation and immutable runtime binding.
        # Runtime restarts invalidate activation candidates; use the segment's own scope.
        scope=job['session']+':'+target.name
        summary='Relative motion segment only; physical object outcome is not verified. '+read(self.current/'command.json')['content']
        activation_manager=self.owner.agent.skill_activation
        activation_manager.begin_turn(scope,summary)
        activation=activation_manager.activate(session_key=scope,name='piper-real-runtime',role='primary')
        if inspect.isawaitable(activation):activation=await activation
        activation,_,_=activation
        task=self.owner.coordinator.create_task(task_description=summary,
          verification=TaskVerificationContract(mode='off'),
          activation_id=activation.activation_id,origin_session_key=scope)
        if inspect.isawaitable(task):task=await task
        self.task_id=task.task_id
        binding=task.primary_skill_binding
        if binding is None:raise RuntimeError('动作任务缺少冻结的真机Skill绑定，未下发')
        data={'task_id':task.task_id,'activation_id':activation.activation_id,
          'binding_id':binding.binding_id,'runtime_instance_id':binding.runtime_instance_id,
          'parent_session':job['session'],'task':task.model_dump(mode='json')}
        save(target/'agent-task-binding.json',data)
        await self.emit('tool_result',name='forge_task_create',arguments={'skill':'piper-real-runtime','stage':target.name},result=json.dumps(data,ensure_ascii=False))
        return task

    async def action(self,**kwargs):
        params=validate_step(**kwargs)
        job=self.require_job(motion=True)
        if not 0<=time.time()-self.last_vision<30:raise ValueError('每个动作前须重新调用real_observe读取当前画面与反馈')
        async with self.action_lock:
            self.require_job(motion=True)
            stage='stage-'+uuid.uuid4().hex[:12];prepared_id='web-'+uuid.uuid4().hex
            env=dict(os.environ,PAOS_COMMAND_RUN=str(self.current))
            args=[os.environ.get('PIPER_CONTROL_PYTHON','/usr/bin/python3'),'-B',str(ROOT/'scripts/real_command_stage.py'),stage,prepared_id,'--seconds',str(params['seconds'])]
            if params['translation_m'] is not None:
                argsik=[os.environ.get('PIPER_CONTROL_PYTHON','/usr/bin/python3'),'-B',str(ROOT/'scripts/real_command_ik.py'),stage,'--side',params['side'],'--frame',params['frame'],'--translation='+','.join(map(str,params['translation_m']))]
                await self.shell(argsik,env=env)
                params[params['side']+'_delta']=read(self.current/(stage+'-ik.json'))['delta_deg']
            for side in ('left','right'):
                if params[side+'_delta'] is not None:args+=['--'+side+'-delta='+','.join(map(str,params[side+'_delta']))]
                if params[side+'_gripper'] is not None:args+=['--'+side+'-gripper',str(params[side+'_gripper'])]
                if side in (params['keep_closed'] or []):args+=['--keep-'+side+'-closed']
            self.capture()
            await self.shell(args,env=env)
            target=self.current/stage
            # This confirmation is the actual browser request; the model cannot supply it.
            cfgpath=ROOT/'config/real_runtime.json';cfg=read(cfgpath)
            if cfg.get('mode')!='offline' or cfg.get('release') is not None:raise ValueError('控制链已被其他任务占用')
            report=await asyncio.to_thread(self.owner.manager.status,'piper-real-runtime')
            if report.state and report.state.status!='stopped':raise ValueError('真机Skill已由其他会话持有')
            raw=(ROOT/'prepared/real_tasks'/(prepared_id+'.json')).read_bytes()
            release={'prepared_id':prepared_id,'sha256':hashlib.sha256(raw).hexdigest(),'release_id':'web-'+uuid.uuid4().hex,
              'valid_until_unix':time.time()+240,'operator_confirmation_ref':str(self.current/'command.json')}
            live={**cfg,'mode':'live','release':release}
            save(target/'release.json',release);save(cfgpath,live)
            started=False;known=False;submission_attempted=False;self.task_id=None
            self.last_vision=0
            try:
                self.require_job(motion=True)
                await asyncio.to_thread(self.owner.manager.start,'piper-real-runtime','staged');started=True
                async with ForgeToolClient('http://127.0.0.1:19014') as client:
                    self.client=client
                    task=await self.bind_stage(target,job)
                    self.require_job(motion=True)
                    save(target/'dispatch-intent.json',{'prepared_id':prepared_id,'at':time.time(),'request_id':job['request_id']})
                    await self.emit('tool',content='调用 piper_real.run_task',name='piper_real.run_task',arguments={'prepared_id':prepared_id,'relative_action':params})
                    submission_attempted=True
                    submission=await self.owner.coordinator.start_action(task.task_id,'piper_real.run_task',{'prepared_id':prepared_id})
                    self.invocation=submission['data']['invocation_id']
                    save(target/'submission.json',submission)
                    save(RUNS/'active.json',{'run':self.current.name,'stage':stage,'invocation_id':self.invocation})
                    deadline=time.monotonic()+135
                    while True:
                        status=await client.invocation_status(self.invocation)
                        if status['data']['phase'] in TERMINAL:
                            result=await client.invocation_result(self.invocation)
                            known=status['data']['phase']!='unknown'
                            break
                        if time.monotonic()>deadline:raise RuntimeError('动作结果尚未核实，已锁定后续下发；不能重放')
                        await asyncio.sleep(.3)
                    self.owner.coordinator.observe_action(task.task_id,self.invocation,status)
                    self.owner.coordinator.observe_action(task.task_id,self.invocation,result)
                    finalized=await self.owner.coordinator.finalize_task(task.task_id)
                    save(target/'agent-task-final.json',finalized.model_dump(mode='json'))
                    receipt={'task_id':task.task_id,'binding_id':task.primary_skill_binding.binding_id,'task_status':finalized.status.value,'invocation_id':self.invocation,'status':status,'result':result}
                    save(target/'forge-result.json',receipt)
                    await self.emit('tool_result',name='piper_real.run_task',arguments={'prepared_id':prepared_id},result=json.dumps(receipt,ensure_ascii=False))
                    phase=status['data']['phase']
                    if phase not in ('completed','succeeded'):
                        self.halted=True;raise RuntimeError('原生动作'+phase+'，停止本次后续动作；请检查回执')
                    # Completion is motion completion only, never an object verdict.
                    return {'stage':stage,'task_id':task.task_id,'binding_id':task.primary_skill_binding.binding_id,'task_status':finalized.status.value,'invocation_id':self.invocation,'phase':phase,'physical_success':None,'next':'重新观察物体实际状态后才能规划下一段'}
            except BaseException:
                self.halted=True
                raise
            finally:
                if self.task_id:
                    recorded=self.owner.coordinator.get_task(self.task_id)
                    if not recorded.terminal:
                        if not recorded.execution_records:
                            await self.owner.coordinator.cancel_task(self.task_id,reason='segment stopped before Action admission')
                            known=True
                        elif all(e.terminal and e.status!='unknown' for e in recorded.execution_records):
                            finalized=await self.owner.coordinator.finalize_task(self.task_id)
                            save(target/'agent-task-final.json',finalized.model_dump(mode='json'))
                            known=True
                        else:
                            known=False
                if known or not submission_attempted:
                    if started:await asyncio.to_thread(self.owner.manager.stop,'piper-real-runtime')
                    if read(cfgpath)==live:save(cfgpath,cfg)
                    self.invocation=None;self.client=None;self.task_id=None
                else:
                    self.halted=True
                    save(RUNS/'active.json',{'run':self.current.name,'stage':stage,'invocation_id':self.invocation,'phase':'unknown','task_id':self.task_id,'do_not_resubmit':True})

    async def process(self,content,session,rid):
        run=self.current
        try:
            result=await self.owner.process(content,session,rid,propagate_errors=True)
            meta=read(run/'command.json');meta.update(phase='stopped' if self.cancel_requested else 'failed' if self.halted else 'completed',answer=result,ended_wall_s=time.time())
            save(run/'command.json',meta)
        except asyncio.CancelledError:
            meta=read(run/'command.json');meta.update(phase='stopped',ended_wall_s=time.time());save(run/'command.json',meta)
            raise
        except Exception as exc:
            meta=read(run/'command.json');meta.update(phase='failed',error=str(exc),ended_wall_s=time.time());save(run/'command.json',meta)
        finally:
            if not self.invocation:
                active=read(RUNS/'active.json',{})
                if active.get('run')==run.name and active.get('phase')!='unknown':
                    (RUNS/'active.json').unlink(missing_ok=True)

    async def submit(self,body):
        content=str(body.get('content','')).strip();rid=body.get('request_id','');session=body.get('session','')
        if not content or len(content)>12000 or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',rid) or not re.fullmatch(r'console:real-[A-Za-z0-9_-]{1,60}',session):
            raise ValueError('无效的任务、会话或请求编号')
        run=RUNS/('command-'+rid)
        if run.exists():
            old=read(run/'command.json',{})
            if old.get('content')!=content or old.get('session')!=session or old.get('allow_motion')!=body.get('allow_motion'):raise ValueError('请求编号已绑定其他内容')
            return {'accepted':True,'duplicate':True,'run':run.name}
        if self.owner.job or self.owner.control_busy or self.owner.background_busy() or self.owner.coordinator.store.active() or read(RUNS/'active.json'):
            raise ValueError('已有活动任务或待核对结果，请先查看原任务')
        if body.get('allow_motion') not in (True,False):raise ValueError('须明确本次为只读观察或真机执行')
        if body['allow_motion'] and not all(c['fresh'] for c in self.cameras().values()):raise ValueError('请先开启现场观察，三路画面就绪后再执行')
        await self.owner.set_agent_mode('real')
        self.current=run;run.mkdir();self.cancel_requested=False;self.halted=False;self.last_vision=0
        meta={'request_id':rid,'content':content,'session':session,'allow_motion':body['allow_motion'],'source':'actual_frontend_user_request','created_wall_s':time.time(),'phase':'running'}
        save(run/'command.json',meta);save(RUNS/'active.json',{'run':run.name,'phase':'planning'})
        self.owner.job={'session':session,'request_id':rid,'real_command_run':str(run),'allow_motion':body['allow_motion'],'started_at':time.time()}
        await self.owner.emit('accepted',session=session,request_id=rid,content=content)
        if content.lower() in ('/help','/status','/状态','帮助'):
            answer=('输入自然语言目标后点击发送。只读模式可检查状态与现场；真机执行模式按当前反馈规划相对动作。/status 查看状态，/stop 请求停止。' if content.lower() in ('/help','帮助') else '模型：'+self.owner.config.agents.defaults.model+'\n控制方式：当前反馈 + 相对增量\n动作链路：'+str(read(ROOT/'config/real_runtime.json',{}).get('mode'))+'\n现场画面：'+str(sum(c['fresh'] for c in self.cameras().values()))+'/3 路就绪\n历史录像：3组，共23份\n本命令未调用模型或下发动作。')
            await self.owner.emit('answer',session=session,request_id=rid,content=answer)
            meta.update(phase='completed',answer=answer,ended_wall_s=time.time());save(run/'command.json',meta)
            self.owner.job=None;(RUNS/'active.json').unlink();await self.owner.refresh()
        else:
            self.owner.running=asyncio.create_task(self.process(content,session,rid))
        return {'accepted':True,'run':run.name,'request_id':rid}

    async def stop(self):
        self.cancel_requested=True;self.halted=True
        if self.invocation and self.client:
            result=await self.owner.coordinator.cancel_action(self.task_id,self.invocation) if self.task_id else await self.client.cancel_invocation(self.invocation)
            await self.emit('progress',content='已请求原生动作停止，等待原生终态。')
            return {'requested':True,'invocation_id':self.invocation,'native':result}
        if self.owner.job and self.owner.job.get('real_command_run'):
            await self.owner.cancel_agent_tasks()
            return {'requested':True,'phase':'stopped'}
        return {'requested':False,'phase':'idle'}

    async def refresh_camera_health(self,force=False):
        if not self.camera_unit:
            self.camera_health={};return
        if not force and time.monotonic()-self.camera_checked<2:return
        self.camera_checked=time.monotonic()
        try:
            raw=await self.shell(['systemctl','--user','show',self.camera_unit,'--property=ActiveState,SubState,Result,ExecMainStatus'])
            self.camera_health=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        except RuntimeError as exc:
            self.camera_health={'ActiveState':'unavailable','error':str(exc)}

    async def camera(self,action):
        async with self.camera_lock:
            await self.refresh_camera_health(force=True)
            return await self._camera(action)

    async def _camera(self,action):
        if action=='start':
            if self.owner.job:raise ValueError('请等待当前命令结束再调整相机')
            if self.camera_health.get('ActiveState') in ('active','activating'):
                return {'reused':True,'pending':not all(c['fresh'] for c in self.cameras().values())}
            if any(c['fresh'] for c in self.cameras().values()):raise ValueError('上一观察进程的画面尚未过期，请稍后再开启')
            mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            if mem<6*1024**3:raise ValueError('可用RAM不足6GiB，保留系统资源，未启动观察')
            run=RUNS/('camera-'+uuid.uuid4().hex[:12]);run.mkdir()
            unit='paos-real-command-camera-'+uuid.uuid4().hex[:10]
            camera_env=['--setenv=PAOS_COMMAND_CAMERA_RUN='+str(run),
                        '--setenv=PAOS_ROOT='+str(ROOT),
                        '--setenv=PIPER_CONTROL_ROOT='+os.environ.get('PIPER_CONTROL_ROOT',str(ROOT/'external/robot-control'))]
            camera_cmd='source '+shlex.quote(os.environ.get('PIPER_ROS_SETUP','/opt/ros/humble/setup.bash'))+' && exec '+shlex.join([os.environ.get('PIPER_CONTROL_PYTHON','/usr/bin/python3'),'-u','-B',str(ROOT/'scripts/real_command_observe.py')])
            await self.shell(['systemd-run','--user','--unit',unit,'--property=MemoryHigh=1500M','--property=MemoryMax=2G','--property=MemorySwapMax=0',*camera_env,'/bin/bash','-c',camera_cmd])
            self.camera_run=run;self.camera_unit=unit;save(RUNS/'camera.json',{'run':str(run),'unit':unit})
            await self.refresh_camera_health(force=True)
            return {'started':True,'run':run.name}
        if action=='stop':
            if self.owner.job or self.invocation:raise ValueError('当前任务使用现场观察，请先停止任务并等待终态')
            if self.camera_unit:await self.shell(['systemctl','--user','stop',self.camera_unit])
            await self.refresh_camera_health(force=True)
            return {'stopped':True}
        raise ValueError('无效的相机操作')

class ObserveTool(Tool):
    name='real_observe'
    description='读取当前三路真实相机和执行臂反馈，并用当前视觉模型回答问题。每段相对动作前必须重新观察。不启动相机，不下发动作。'
    parameters={'type':'object','properties':{'question':{'type':'string'}},'additionalProperties':False}
    def __init__(self,controller):self.controller=controller
    async def execute(self,**kwargs):
        try:return json.dumps(await self.controller.observe(**kwargs),ensure_ascii=False)
        except Exception as exc:return json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False)

class RelativeTool(Tool):
    name='real_relative_action'
    description='为本次前端明确允许的真机任务执行单个相对动作。后端先实际激活Skill并创建冻结绑定的AgentTask，再经官方Coordinator下发Forge/原生控制链；返回真实task_id和回执，模型无需单独创建。当前反馈为起点。每段前必须real_observe。任一动作失败后本次不能继续。不能设置绝对关节、世界坐标或从图片臆造标定。关节单位度，平移/夹爪单位米。'
    parameters={'type':'object','properties':{
      'left_delta':{'type':'array','items':{'type':'number'},'minItems':6,'maxItems':6},
      'right_delta':{'type':'array','items':{'type':'number'},'minItems':6,'maxItems':6},
      'seconds':{'type':'number','minimum':4,'maximum':20},
      'left_gripper':{'type':'number','minimum':0,'maximum':.094},
      'right_gripper':{'type':'number','minimum':0,'maximum':.094},
      'side':{'type':'string','enum':['left','right']},
      'translation_m':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3},
      'frame':{'type':'string','enum':['base','tool']},
      'keep_closed':{'type':'array','items':{'type':'string','enum':['left','right']}}},
      'additionalProperties':False}
    def __init__(self,controller):self.controller=controller
    async def execute(self,**kwargs):
        try:return json.dumps(await self.controller.action(**kwargs),ensure_ascii=False)
        except Exception as exc:
            self.controller.halted=True
            return json.dumps({'ok':False,'error':str(exc),'physical_success':None},ensure_ascii=False)

CONTROLLER=None
def attach(owner):
    global CONTROLLER
    CONTROLLER=getattr(owner,'real_commands',None) or RealCommands(owner)
    for tool in (ObserveTool(CONTROLLER),RelativeTool(CONTROLLER)):owner._all_tools[tool.name]=tool
    owner.apply_tool_mode()
    return CONTROLLER

async def api_state(request):
    await CONTROLLER.refresh_camera_health()
    return JSONResponse(CONTROLLER.state(),headers={'Cache-Control':'no-store'})
async def mutate(request):
    from console.server import check_origin
    try:
        check_origin(request);body=await request.json()
        action=request.path_params['action']
        if action=='message':
            if body.get('content','').strip().lower()=='/stop':result=await CONTROLLER.stop()
            else:result=await CONTROLLER.submit(body)
        elif action=='stop':result=await CONTROLLER.stop()
        elif action=='camera':result=await CONTROLLER.camera(body.get('action'))
        else:return JSONResponse({'error':'Unknown operation'},404)
        return JSONResponse(result,status_code=202)
    except (ValueError,RuntimeError,FileExistsError) as exc:return JSONResponse({'error':str(exc)},409)

async def camera_frame(request):
    name=request.path_params['camera']
    if name not in CAMERAS or not CONTROLLER.camera_run:return JSONResponse({'error':'当前未开启现场观察'},404)
    path=CONTROLLER.camera_run/(name+'-latest.jpg')
    if not path.exists():return JSONResponse({'error':'等待画面'},404)
    return FileResponse(path,media_type='image/jpeg',headers={'Cache-Control':'no-store'})

async def artifact(request):
    run=(RUNS/request.query_params.get('run','')).resolve()
    path=(run/request.query_params.get('path','')).resolve()
    if not run.is_relative_to(RUNS.resolve()) or not path.is_relative_to(run) or not path.is_file() or path.suffix not in ('.jpg','.png','.json','.md'):
        return JSONResponse({'error':'Unavailable artifact'},404)
    return FileResponse(path,headers={'Cache-Control':'no-store'})

def archive_index():
    result=[]
    for item in ARCHIVE:
        base=ARCHIVE_ROOT/item['directory']
        files=[]
        for path in sorted(base.rglob('*.mp4')):
            if not path.is_file():continue
            rel=str(path.relative_to(base));folder=path.parent.name;stem=path.stem
            if folder.startswith('scene-recording'):
                part='连续原片2' if folder.endswith('-2') else '连续原片1'
                label=part+' · '+{'cam_middle':'全景','cam_left':'左腕','cam_right':'右腕'}.get(stem,stem)
            else:
                label=('全部动作 · 四倍速' if stem=='task-all-stages-4x' else '关键过程 · 原速')
                if item['id']=='console-demo-20261024' and folder=='video-preview':label+=' · 中断历史版'
            files.append({'path':rel,'title':label,'bytes':path.stat().st_size,
              'url':'/api/real-command/video?group='+item['id']+'&path='+rel})
        result.append({**item,'count':len(files),'files':files})
    return result

async def video(request):
    item=next((g for g in ARCHIVE if g['id']==request.query_params.get('group')),None)
    if item is None:return JSONResponse({'error':'Unknown recording group'},404)
    base=(ARCHIVE_ROOT/item['directory']).resolve()
    path=(base/request.query_params.get('path','')).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.suffix!='.mp4':
        return JSONResponse({'error':'Unknown recording'},404)
    return FileResponse(path,media_type='video/mp4')

def routes():
    playback=[]
    for item in ARCHIVE:
        directory=ARCHIVE_ROOT/item['directory']/item['preview']
        if directory.exists():playback.append(Mount('/recordings/'+item['id'],StaticFiles(directory=directory,html=True,follow_symlink=True)))
    # Preserve the three existing links, including the successful historical demo page.
    for item in ARCHIVE:
        directory=ARCHIVE_ROOT/item['directory']/('demo-preview' if item['id']=='console-demo-20261024' else item['preview'])
        if directory.exists():playback.append(Mount('/'+item['id'],StaticFiles(directory=directory,html=True,follow_symlink=True)))
    return [Route('/api/real-command/state',api_state),Route('/api/real-command/video',video),
      Route('/api/real-command/{action}',mutate,methods=['POST']),
      Route('/api/real-command/frame/{camera}',camera_frame),
      Route('/api/real-command/artifact',artifact),*playback]
