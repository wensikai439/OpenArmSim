"""Thin CLI for the existing Forge Skill. Never creates operator releases or retries actions."""
import argparse, asyncio, dataclasses, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(str(Path(__file__).resolve().parents[1]))
os.environ['PATH']=':'.join(str(ROOT/p) for p in ['.venv/bin','toolchains/bin','toolchains/dora-0.4.1/dora-cli-x86_64-unknown-linux-gnu'])+':'+os.environ['PATH']
from PhyAgentOS.skill_runtime.manager import RuntimeManager
from PhyAgentOS.forge.tool_client import ForgeToolClient
sys.path.insert(0,str(ROOT/'bundles/piper-real-runtime/profiles/runtime'))
from runtime_bridge import Bridge, identifier
TERMINAL={'completed','succeeded','failed','cancelled','unknown','rejected','expired'}

async def read_terminal(client, invocation, timeout_s):
    deadline=time.monotonic()+timeout_s
    while True:
        status=await client.invocation_status(invocation)
        if status['data']['phase'] in TERMINAL:
            return {'status':status,'result':await client.invocation_result(invocation)}
        if time.monotonic()>=deadline:
            raise TimeoutError('Owned action outcome unresolved; inspect its receipt, never resubmit')
        await asyncio.sleep(.25)

async def main(args):
    bridge=Bridge();cfg=bridge.config();original=bridge.config_path.read_bytes()
    if args.command=='run':
        prepared_id=identifier(args.prepared_id)
        bridge.preflight(prepared_id,'cli-preflight-'+str(time.time_ns()))
        task=json.loads(bridge.task_path(prepared_id).read_text())
        trajectory=json.loads((ROOT/'prepared/native_replays'/(task['registered_replay_id']+'.json')).read_text())
        if trajectory.get('execution_mode') not in ('follower-relative-j1','follower-relative-j1-cycle','follower-relative-j1-travel','follower-relative-bottle','follower-relative-expert'):
            raise RuntimeError('This CLI only accepts a registered bounded follower-relative task')
        release_id=identifier(cfg['release']['release_id'])
        if (bridge.run/'releases'/(release_id+'.used')).exists():
            raise RuntimeError('This human release was already consumed; never replay it')
        with (bridge.run/('cli-'+release_id+'.once')).open('x') as f:f.write(str(time.time()))
    manager=RuntimeManager();before=manager.state_store.load('piper-real-runtime')
    if before is not None and before.status!='stopped':raise RuntimeError('Existing real Skill is owned by another session')
    started=False;report={'mode':args.command,'hardware_action_submissions':0}
    try:
        started_state=manager.start('piper-real-runtime','staged');started=True
        async with ForgeToolClient('http://127.0.0.1:19014') as client:
            if args.command=='run':
                report['hardware_action_submissions']=1
                report['submission']=await client.invoke_action('piper_real.run_task',{'prepared_id':prepared_id})
                invocation=report['submission']['data']['invocation_id']
            else:invocation=args.invocation_id
            report['invocation_id']=invocation
            report.update(await read_terminal(client,invocation,130. if args.command=='run' else 5.))
    finally:
        if args.command=='run' and bridge.config_path.read_bytes()==original:
            closed={**cfg,'mode':'offline','release':None}
            tmp=bridge.config_path.with_suffix('.once.tmp');tmp.write_text(json.dumps(closed,indent=2)+'\n');tmp.replace(bridge.config_path)
        if started:report['skill_stopped']=dataclasses.asdict(manager.stop('piper-real-runtime'))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,default=str)
        print(json.dumps({'evidence':str(args.output),'mode':args.command,'hardware_action_submissions':report['hardware_action_submissions'],
            'phase':report.get('status',{}).get('data',{}).get('phase')},ensure_ascii=False))
    return 0 if report.get('result',{}).get('data',{}).get('result',{}).get('status')=='succeeded' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('prepared_id')
    p=sub.add_parser('inspect');p.add_argument('invocation_id')
    raise SystemExit(asyncio.run(main(parser.parse_args())))
