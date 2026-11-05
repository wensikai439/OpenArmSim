"""Official Forge Tool endpoint for the independent shared PiPER simulation asset."""
from __future__ import annotations
import asyncio, base64, dataclasses, hashlib, json, os, queue, signal, subprocess, sys, threading, time, urllib.request, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
import pyarrow as pa
from dora import Node
from forge_msgs import CompressedImage
from forge_tool import (TOOL_ENDPOINT_PROTOCOL, ToolEndpointDescriptor, ToolOperationDescriptor,
    ToolEndpointHandler, ToolAccepted, ToolResult, ToolError, ToolEndpointError,
    ToolExecutionStatus, ToolResultResponse, ToolControlResponse, ToolEvent)
from forge_tool._tool_message import ToolMessage
from forge_tool.dora import DoraToolEndpointBinding, tool_message_to_envelope, tool_envelope_to_message
from endpoint_lease import EndpointLease
from session_handler import SessionEndpointHandler

ROOT = Path(str(Path(os.environ["PAOS_ROOT"])))
SIM = Path(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(os.environ["PAOS_ROOT"]) / "external/piper_robocasa")))))
ACTIONS = {'set_gripper', 'move_arm', 'grasp_bottle', 'release_bottle', 'return_arm_to_initial', 'place_bottle_at_origin'}
ENDPOINT = 'piper.dual_sim'

class RpcRejected(RuntimeError): pass

class Simulation:
    def __init__(self):
        self.instance = 'paos-piper-' + uuid.uuid4().hex[:12]
        self.run = ROOT / 'runs/piper-runtime' / self.instance
        self.run.mkdir(parents=True)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.log = (self.run/'server.log').open('w')
        command = [str(SIM/'.venv/bin/python'), str(Path(__file__).with_name('native_shim.py')),
            '--transport','http','--host','127.0.0.1','--port','19023','--instance-id',self.instance,
            '--output',str(self.run/'sim'),'--a-source-root',str(SIM/'src'),
            '--task-variant','air-handoff','--render','--enable-actions','--parent-watch']
        env = os.environ.copy()
        env.update(MUJOCO_GL='egl', PYTHONPATH=str(Path(__file__).parent)+os.pathsep+str(SIM/'src'), PYTHONDONTWRITEBYTECODE='1',
            PAOS_CANCEL_REQUEST=str(self.run/'cancel-request.json'))
        self.process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.PIPE,
            stdout=self.log,stderr=subprocess.STDOUT)
        (self.run/'launch.json').write_text(json.dumps({'command':command,'pid':self.process.pid,
            'instance_id':self.instance,'runtime_instance_id':os.getenv('PAOS_RUNTIME_INSTANCE_ID')},indent=2))
        try:
            deadline = time.monotonic()+60
            while time.monotonic()<deadline:
                if self.process.poll() is not None: raise RuntimeError('PiPER SIM exited; see '+str(self.run/'server.log'))
                try:
                    self.meta=self.rpc('env.get_env_meta'); break
                except (OSError,urllib.error.URLError): time.sleep(.5)
            else: raise RuntimeError('PiPER SIM startup deadline')
        except BaseException:
            self.close(); raise
        (self.run/'metadata.json').write_text(json.dumps(self.meta,indent=2))
    def rpc(self,method,**kwargs):
        request=urllib.request.Request('http://127.0.0.1:19023/call',
            data=json.dumps({'method':method,'args':[],'kwargs':kwargs}).encode(),
            headers={'Content-Type':'application/json'})
        with self.opener.open(request,timeout=120) as response: data=json.load(response)
        if data.get('ok') is not True: raise RpcRejected(json.dumps(data,ensure_ascii=False))
        return data['result']
    def observe(self,node):
        captured=self.rpc('env.capture_frames',cameras=['cam_right','overview'])
        for frame in captured['frames']:
            rgb=frame.pop('rgb')
            raw=base64.b64decode(rgb['__ndarray__'])
            frame['rgb_sha256']=hashlib.sha256(raw).hexdigest()
            # Use the simulator's own PNG. No synthetic render or substituted image.
            path=Path(frame['path']) if 'path' in frame else Path(frame['image_path'])
            path = self.run/'sim'/path if not path.is_absolute() else path
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != frame['sha256']: raise RuntimeError('Native frame SHA mismatch')
            node.send_output('image/'+frame['camera_id'],CompressedImage(format='png',data=data).to_arrow())
        state=self.rpc('env.observe')
        status={'asset_root':str(SIM),'adapter_source_root':str(Path(__file__).parent),'instance_id':self.instance,
            'task_variant':'air-handoff','physics_paused_between_requests':True,
            'interruptible_actions':True,'cancel_boundary':'native control tick; reset required after cancellation','evidence_root':str(self.run),'observation':state}
        node.send_output('runtime_status',pa.array(list(json.dumps(status).encode()),type=pa.uint8()))
        (self.run/'latest-state.json').write_text(json.dumps(state,indent=2))
        return {'metadata':self.meta,'observation':state,'frames':captured['frames'],
            'physics_paused_between_requests':True,'source_root':str(SIM)}
    def journal(self,event):
        with (self.run/'actions.jsonl').open('a') as f:f.write(json.dumps(event,default=str)+'\n')
    def close(self):
        if self.process.poll() is None:
            try:self.rpc('shutdown')
            except (OSError,RuntimeError) as error:self.journal({'shutdown_error':str(error)})
            if self.process.stdin:self.process.stdin.close()
        try:code=self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.journal({'shutdown':'unresolved','pid':self.process.pid});return
        self.log.close()
        (self.run/'exit.json').write_text(json.dumps({'exit_code':code,'pid':self.process.pid}))

class Endpoint:
    def __init__(self,sim,node):
        self.sim,self.node=sim,node
        self.executions={}; self.active=None; self.completed=queue.Queue(); self.stream=None

    async def query(self,request,context):
        if self.active is not None: raise ToolEndpointError(ToolError('BUSY','Native action is running; Query waits for its control loop to return'))
        if context.operation == 'check_pose':
            output = await asyncio.to_thread(self.sim.rpc, 'env.check_pose', **dict(request.arguments))
            self.sim.journal({'type':'pose_query','arguments':dict(request.arguments),'result':output})
            return ToolResult(status='succeeded',outputs=output)
        if request.arguments: raise ToolEndpointError(ToolError('INVALID_ARGUMENT','observe takes no arguments'))
        return ToolResult(status='succeeded',outputs=await self.observe())

    async def start(self,request,context,events):
        if context.operation == 'observe_stream':
            return self.start_stream(request,context,events)
        if self.active is not None:raise ToolEndpointError(ToolError('BUSY','PiPER simulator has an active action'))
        if context.operation not in ACTIONS:raise ToolEndpointError(ToolError('UNSUPPORTED','Unsupported PiPER primitive'))
        key=context.execution_key
        execution_id=key.invocation_id+':'+key.attempt_id
        self.executions[key]={'phase':'running','result':None,'events':events,'kind':'action','cancel_requested':False}
        self.active=key
        action={'id':context.operation,'payload':dict(request.arguments)}
        self.sim.journal({'type':'request','invocation_id':key.invocation_id,'attempt_id':key.attempt_id,'action':action})
        def work():
            try:
                raw=self.sim.rpc('env.step_action',action=action,execution_id=execution_id)
                if raw.get('adapter_cancelled') is True and raw.get('adapter_execution_id') == execution_id:
                    result=ToolResult(status='cancelled',outputs={'native_result':raw,'instance_id':self.sim.instance,'reset_required':True})
                elif raw.get('primitive_completed') is True:
                    result=ToolResult(status='succeeded',outputs={'native_result':raw,'instance_id':self.sim.instance})
                else:result=ToolResult(status='failed',outputs={'native_result':raw},error=ToolError('PIPER_INCOMPLETE','Native primitive did not complete'))
            except RpcRejected as error:
                result=ToolResult(status='failed',error=ToolError('PIPER_RPC_ERROR',str(error)))
            except Exception as error:
                result=ToolResult(status='unknown',error=ToolError('RPC_OUTCOME_UNKNOWN',str(error)))
            self.completed.put((key,result))
        threading.Thread(target=work,daemon=True).start()
        return ToolAccepted(details={'instance_id':self.sim.instance,'interruptible':True,'cancel_boundary':'native control tick','reset_required_after_cancel':True})

    def start_stream(self,request,context,events):
        if self.stream is not None:raise ToolEndpointError(ToolError('BUSY','One observation stream is already active'))
        args=dict(request.arguments)
        interval=args.pop('interval_ms',500);duration=args.pop('max_duration_s',30)
        if (args or isinstance(interval,bool) or not isinstance(interval,int) or not 500<=interval<=5000
                or isinstance(duration,bool) or not isinstance(duration,(int,float)) or not 1<=duration<=60):
            raise ToolEndpointError(ToolError('INVALID_ARGUMENT','interval_ms is 500..5000; max_duration_s is 1..60'))
        key=context.execution_key
        now=time.time();deadline=now+duration
        if context.deadline_ms is not None:deadline=min(deadline,context.deadline_ms/1000)
        path=self.sim.run/('observe-stream-'+hashlib.sha256((key.invocation_id+':'+key.attempt_id).encode()).hexdigest()[:20]+'.jsonl')
        path.touch(exist_ok=False)
        self.executions[key]={'phase':'running','result':None,'events':events,'kind':'session',
            'samples':0,'first_state':None,'last_state':None,'last_frames':[],
            'manifest':path,'interval_s':interval/1000,'next_sample':now,'deadline':deadline}
        self.stream=key
        self.sim.journal({'type':'session_start','invocation_id':key.invocation_id,'attempt_id':key.attempt_id,'manifest':str(path),'deadline':deadline})
        return ToolAccepted(details={'instance_id':self.sim.instance,'manifest_path':str(path),'interval_ms':interval,
            'physics_paused_between_requests':True,'sampling_paused_during_native_actions':True})

    async def observe(self):
        data=self.sim.observe(self.node)
        if self.stream is not None:
            key=self.stream;r=self.executions[key];now=time.time()
            if now>=r['next_sample'] and now<r['deadline']:
                sample={'sequence':r['samples']+1,'captured_at_unix_s':now,
                    'instance_id':self.sim.instance,'observation':data['observation'],'frames':data['frames']}
                with r['manifest'].open('a') as f:f.write(json.dumps(sample)+'\n')
                r['samples']+=1;r['next_sample']=now+r['interval_s']
                if r['first_state'] is None:r['first_state']=data['observation']
                r['last_state']=data['observation'];r['last_frames']=data['frames']
                await r['events'].emit(ToolEvent(type='progress',data={'samples':r['samples'],'physics_step':data['observation'].get('physics_step'),'manifest_path':str(r['manifest'])}))
        return data

    async def finish_stream(self,key,reason):
        r=self.executions[key]
        outputs={'instance_id':self.sim.instance,'samples':r['samples'],'stop_reason':reason,
            'manifest_path':str(r['manifest']),'manifest_sha256':hashlib.sha256(r['manifest'].read_bytes()).hexdigest(),
            'first_state':r['first_state'],'last_state':r['last_state'],'last_frames':r['last_frames'],
            'physics_paused_between_requests':True,'sampling_paused_during_native_actions':True}
        r.update(phase='stopped',result=ToolResult(status='stopped',outputs=outputs));self.stream=None
        self.sim.journal({'type':'session_stopped','invocation_id':key.invocation_id,'attempt_id':key.attempt_id,'result':dataclasses.asdict(r['result'])})
        await r['events'].emit(ToolEvent(type='stopped',data={'samples':r['samples'],'manifest_path':str(r['manifest'])}))

    async def drain(self):
        while not self.completed.empty():
            key,result=self.completed.get_nowait();record=self.executions[key]
            record.update(result=result,phase={'succeeded':'completed','failed':'failed','unknown':'unknown','cancelled':'cancelled'}[result.status])
            self.active=None
            self.sim.journal({'type':'result','invocation_id':key.invocation_id,'attempt_id':key.attempt_id,'result':dataclasses.asdict(result)})
            await self.observe()
            if result.status in {'succeeded','failed','cancelled'}:
                event={'succeeded':'executor_completed','failed':'executor_failed','cancelled':'cancelled'}[result.status]
                await record['events'].emit(ToolEvent(type=event,data={'native_result_available':True,'reset_required':result.status=='cancelled'}))
        if self.stream is not None and time.time()>=self.executions[self.stream]['deadline']:
            await self.finish_stream(self.stream,'duration_or_invocation_deadline')

    async def status(self,key):
        r=self.executions.get(key)
        if r is None:raise ToolEndpointError(ToolError('NOT_FOUND','Unknown execution key'))
        result=r['result']
        details={'cancel_requested':r.get('cancel_requested',False)}
        if r['kind']=='session':details.update(samples=r['samples'],manifest_path=str(r['manifest']),sampling_paused=self.active is not None)
        return ToolExecutionStatus(phase=r['phase'],error=result.error if result else None,details=details)

    async def result(self,key):
        r=self.executions.get(key)
        return ToolResultResponse(status='not_found' if r is None else 'available' if r['result'] else 'pending',result=r['result'] if r else None)

    async def cancel(self,key,reason=None):
        r=self.executions.get(key)
        if r is None or r['kind']!='action':return ToolControlResponse(command='cancel',status='rejected',error=ToolError('NOT_FOUND','No such action'))
        if r['result'] is not None:return ToolControlResponse(command='cancel',status='terminal')
        if self.active!=key or self.sim.process.poll() is not None:
            return ToolControlResponse(command='cancel',status='rejected',error=ToolError('CANCEL_UNAVAILABLE','Owned active simulator process is unavailable'))
        if not r['cancel_requested']:
            payload={'execution_id':key.invocation_id+':'+key.attempt_id,'reason':reason,'requested_at':time.time()}
            pending=self.sim.run/'cancel-request.tmp';pending.write_text(json.dumps(payload));pending.replace(self.sim.run/'cancel-request.json')
            self.sim.process.send_signal(signal.SIGUSR1)
            r['cancel_requested']=True;r['phase']='stopping'
            self.sim.journal({'type':'cancel_requested','invocation_id':key.invocation_id,'attempt_id':key.attempt_id,'pid':self.sim.process.pid})
        return ToolControlResponse(command='cancel',status='accepted',details={'requested':True,'terminal':False,
            'reason':'Await native control-tick cancellation receipt; completion may win the race','reset_required_if_cancelled':True})

    async def stop(self,key,reason=None):
        r=self.executions.get(key)
        if r is None or r['kind']!='session':return ToolControlResponse(command='stop',status='rejected',error=ToolError('NOT_FOUND','No such observation session'))
        if r['result'] is not None:return ToolControlResponse(command='stop',status='terminal')
        await self.finish_stream(key,reason or 'requested')
        return ToolControlResponse(command='stop',status='accepted',details={'samples':r['samples']})

def main():
    node=Node()
    sim=Simulation()
    try:
        endpoint=Endpoint(sim,node)
        descriptor=ToolEndpointDescriptor(protocol_version=TOOL_ENDPOINT_PROTOCOL,endpoint_id=ENDPOINT,
            operations=(ToolOperationDescriptor(name='observe',semantics='query'),
                ToolOperationDescriptor(name='check_pose',semantics='query'),
                ToolOperationDescriptor(name='observe_stream',semantics='session',status_supported=True,stoppable=True),
                *(ToolOperationDescriptor(name=op,semantics='action',status_supported=True,cancellable=True) for op in sorted(ACTIONS))))
        handler=SessionEndpointHandler(descriptor,endpoint_instance_id=sim.instance,operations={op:endpoint for op in ['observe','check_pose','observe_stream',*ACTIONS]})
        async def sink(batch):
            e=tool_message_to_envelope(ToolMessage.from_arrow(batch))
            sim.journal({'type':'wire_out','at':time.time(),'message_type':e.message_type,'request_id':e.request_id,'invocation_id':e.invocation_id})
            node.send_output('tool_out',batch)
        binding=DoraToolEndpointBinding(handler,event_sink=sink)
        lease=EndpointLease(descriptor,sim.instance);runner=asyncio.Runner();last=0.
        def send(envelope):node.send_output('tool_out',tool_envelope_to_message(envelope).to_arrow())
        sim.observe(node)
        for event in node:
            if event['type'] in {'STOP','ERROR'}:break
            request=lease.poll(time.monotonic())
            if request:send(request)
            runner.run(endpoint.drain())
            if event['type']=='INPUT' and event['id']=='tool_in':
                envelope=tool_message_to_envelope(ToolMessage.from_arrow(event['value']))
                if not envelope.message_type.startswith('endpoint.'):
                    sim.journal({'type':'wire_in','at':time.time(),'message_type':envelope.message_type,'request_id':envelope.request_id,'invocation_id':envelope.invocation_id})
                if envelope.message_type=='endpoint.registry.response':lease.acknowledge(envelope,time.monotonic())
                else:
                    for batch in runner.run(binding.dispatch_input(event['value'])):runner.run(sink(batch))
            if time.monotonic()-last>=.5 and endpoint.active is None:
                runner.run(endpoint.observe());last=time.monotonic()
        if endpoint.active is not None:
            # No false cancellation receipt: let the owned synchronous RPC reach a result.
            deadline=time.monotonic()+120
            while endpoint.active is not None and time.monotonic()<deadline:
                runner.run(endpoint.drain());time.sleep(.05)
        if endpoint.stream is not None:
            runner.run(endpoint.stop(endpoint.stream,'runtime_shutdown'))
        unregister=lease.stop()
        if unregister:send(unregister)
        runner.close()
    finally:sim.close()

if __name__=='__main__':main()
