"""Real Forge Gateway smoke, offline connection only. Never starts robot-control."""
import asyncio,dataclasses,json,os,time
from pathlib import Path
from PhyAgentOS.skill_runtime.manager import RuntimeManager
from PhyAgentOS.forge.tool_client import ForgeToolClient,ForgeToolAPIError

ROOT=Path(str(Path(__file__).resolve().parents[1]))
EVIDENCE=ROOT/'runs/real-runtime-adaptation-20261023'
os.environ['PATH']=':'.join(str(ROOT/p) for p in ['.venv/bin','toolchains/bin','toolchains/dora-0.4.1/dora-cli-x86_64-unknown-linux-gnu'])+':'+os.environ['PATH']

def find(value,key):
    if isinstance(value,dict):
        if key in value:return value[key]
        for child in value.values():
            result=find(child,key)
            if result is not None:return result
    elif isinstance(value,list):
        for child in value:
            result=find(child,key)
            if result is not None:return result
    return None

async def main():
    cfg_path=ROOT/'config/real_runtime.json';cfg=cfg_path.read_bytes()
    assert json.loads(cfg)['mode']=='offline' and json.loads(cfg).get('release') is None
    manager=RuntimeManager();before=manager.state_store.load('piper-real-runtime')
    assert before is None or before.status=='stopped'
    sim_before=manager.status('piper-dual-sim')
    native_manifest=Path(str(Path(os.environ.get("PIPER_CONTROL_ROOT", str(Path(__file__).resolve().parents[1] / "external/robot-control"))) / 'run/unified_runtime.json'))
    native_before=json.loads(native_manifest.read_text())
    results={'scope':'actual Forge Gateway and provider, OFFLINE connection; no hardware execution',
             'sim_runtime_before':dataclasses.asdict(sim_before.state),'native_runtime_before':{k:native_before.get(k) for k in ['runtime_instance_id','pid','stage','demo_enabled']}}
    started=False
    try:
        state=manager.start('piper-real-runtime','staged');started=True
        results['skill_started']=dataclasses.asdict(state)
        async with ForgeToolClient('http://127.0.0.1:19014') as client:
            results['tools']=await client.list_tools()
            results['run_task_context']=await client.get_tool_context('piper_real.run_task')
            results['capabilities']=await client.invoke_query_tool('piper_real.capabilities',{})
            assert find(results['capabilities'],'mode')=='offline'
            assert find(results['capabilities'],'motion_ready') is False
            prepared_id='offline-protocol-'+str(time.time_ns())
            program={'schema_version':'independent-pick-place/v1','request':'Offline Forge software check; no motion or field authorization',
                     'arm':'right','steps':[{'skill':'observe','args':{}}]}
            results['prepared']=await client.invoke_query_tool('piper_real.prepare_task',{'prepared_id':prepared_id,'task_type':'single_arm_program','program':program})
            assert find(results['prepared'],'field_approved') is False
            assert find(results['prepared'],'native_io_performed') is False
            results['prepared_step']=await client.invoke_query_tool('piper_real.prepare_task',{'prepared_id':prepared_id+'-step','task_type':'visual_servo_step','step':{'skill':'move_right_tcp_relative','dx_m':0.005,'dy_m':0.,'dz_m':0.}})
            assert find(results['prepared_step'],'field_approved') is False
            for name,op,args in [('state','state',{}),('offline_run','run_task',{'prepared_id':prepared_id}),
                                  ('offline_cancel','cancel_task',{'receipt_reference':'0'*64})]:
                try:
                    reply=await (client.invoke_query_tool('piper_real.'+op,args) if op=='state' else client.invoke_action('piper_real.'+op,args))
                    results[name]={'response':reply}
                    invocation=find(reply,'invocation_id')
                    if invocation:
                        deadline=time.monotonic()+5
                        while True:
                            status=await client.invocation_status(invocation);results[name]['status']=status
                            phase=find(status,'phase') or find(status,'status')
                            if phase in {'succeeded','failed','cancelled','unknown','rejected','expired'}:break
                            if time.monotonic()>deadline:break
                            await asyncio.sleep(.1)
                        results[name]['result']=await client.invocation_result(invocation)
                except ForgeToolAPIError as exc:
                    results[name]={'api_error':str(exc),'payload':exc.payload,'status_code':exc.status_code}
            assert find(results['offline_run'],'phase')=='dispatching'
            assert results['offline_run']['status']['data']['phase']=='failed'
            assert results['offline_run']['result']['data']['result']['error']['code']=='FIELD_RELEASE_REQUIRED'
            assert 'OFFLINE_ONLY' in json.dumps(results['state'])
            assert results['offline_cancel']['status']['data']['phase']=='failed'
            ready=manager.status('piper-real-runtime');assert ready.ready
            results['readiness']=dataclasses.asdict(ready)
    finally:
        if started:results['stop_result']=dataclasses.asdict(manager.stop('piper-real-runtime'))
        assert cfg_path.read_bytes()==cfg
        native_after=json.loads(native_manifest.read_text())
        results['native_runtime_after']={k:native_after.get(k) for k in ['runtime_instance_id','pid','stage','demo_enabled']}
        assert results['native_runtime_before']==results['native_runtime_after']
        sim_after=manager.status('piper-dual-sim');results['sim_runtime_after']=dataclasses.asdict(sim_after.state)
        assert sim_before.state.runtime_instance_id==sim_after.state.runtime_instance_id
        assert sim_before.state.gateway_identity==sim_after.state.gateway_identity
        results['native_runtime_identity_and_stage_unchanged']=True
        results['simulation_runtime_identity_unchanged']=True
        results['connection_configuration_unchanged']=True
        (EVIDENCE/'forge-offline-smoke.json').write_text(json.dumps(results,indent=2,default=str))
    print(json.dumps({'evidence':str(EVIDENCE/'forge-offline-smoke.json'),'offline_caps':results['capabilities'],
        'offline_run':results['offline_run'],'offline_cancel':results['offline_cancel'],'stopped':results['stop_result']},indent=2,default=str))

if __name__=='__main__':asyncio.run(main())
