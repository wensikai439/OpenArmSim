"""Run a declared task through the saved, real LLM API and archive its receipts.

Each invocation explicitly resets a new test scene once, before the user goal.
No reset, pose edit, planner substitution or secret entry occurs within a goal.
"""
import argparse
import hashlib
import json
import sqlite3
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def http(route,body=None):
    request=urllib.request.Request('http://127.0.0.1:8765'+route,
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=90) as response:return json.load(response)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('small','transfer','recovery','timing','carry','carry-timing'),required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('use a new report path under runs')
    status=http('/api/status')
    if status['job'] or status['agent']['busy'] or status['learning_job'].get('running'):raise RuntimeError('platform is busy; leave its task untouched')
    if not status.get('llm'):raise RuntimeError('saved API is not configured')
    prompt=('只在仿真里：用右手抓起瓶子，把瓶子放到本次任务开始时瓶位向前 1 厘米的桌面位置，然后松手并完全撤离。保持目标不变，任务中不能重置，不要用空手移动代替瓶子移动。'
        if args.case=='small' else '只在仿真里：先用左手拿起瓶子，交给右手，再放到本次任务开始时瓶位向前 6 厘米的桌面位置，松手并完全撤离。必须完成左手到右手交接，保持目标不变，任务中不能重置。')
    if args.case=='recovery':
        prompt='只在仿真里：开始前在第一个计划锁定60秒预算。先用左手拿起瓶子并尝试左手直接放到本次任务起始瓶位前方6厘米的桌面位置；若放置失败，根据当前状态恢复，可以换手，但必须保持原目标和原场景，禁止重置或失败后加时。最后松手、撤离并复核。'
    if args.case=='timing':
        prompt='只在仿真里做一次包含模型等待的实验：第一个set_plan声明timing_mode=measured_delay和time_budget_s=120。右手抓起瓶子，放到本次用户任务起始瓶位向前8厘米的桌面位置，松手并撤离，最后复核。保持原目标，不得重置、加时或切回paused。计时声明前的初始规划仍暂停，不把它记为真机连续时间。'
    if args.case in ('carry','carry-timing'):
        prompt='只在仿真里做一次后端组合实验：第一个set_plan锁定time_budget_s=120和timing_mode=paused。用右手抓起瓶子，保持持瓶，用cuRobo的move_relative把右手末端沿world正X移动3厘米。然后用TAMP的place，把瓶子放到本次用户任务起始瓶位向前8厘米的桌面位置，松手并撤离，最后复核。cuRobo这一步必须真实调用，不得用TAMP代替；它是持瓶末端移动，不是精确瓶位放置。保持起始目标不变，禁止任务中重置或加时。'
        if args.case=='carry-timing':prompt=prompt.replace('timing_mode=paused','timing_mode=measured_delay')+' 模型请求期间必须保持旧执行器目标并实际推进物理，不能中途切回paused；首个计划前的请求仍暂停，低层规划耗时不在本次模型等待范围。'
    out.parent.mkdir(parents=True,exist_ok=True)
    expected=[('pick',{'arm':'right'}),('place',{'arm':'right','receptacle_id':'table'})] if args.case in ('small','timing','carry','carry-timing') else [
        ('pick',{'arm':'left'}),('handover',{'from_arm':'left','to_arm':'right'}),('place',{'arm':'right','receptacle_id':'table'})]
    if args.case=='recovery':expected.insert(1,('place',{'arm':'left','receptacle_id':'table'}))
    if args.case in ('carry','carry-timing'):expected.insert(1,('move_relative',{'arm':'right','frame':'world','delta_m':[.03,0,0],'planner':'curobo'}))
    protocol={'case':args.case,'prompt':prompt,'required_motion_sequence':expected,
              'sequence_evidence':'completed native executions for carried-motion cases; failed attempts remain in trace',
              'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scene_reset':'explicit once before goal only',
              'API':'existing saved model; no fake provider','hardware_access':False,'transfer_ready':False}
    out.with_suffix('.protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    http('/api/command',{'command':'reset','arguments':{'episode':0,'seed':0}})
    initial=http('/api/status');started=time.monotonic()
    response=http('/api/chat',{'message':prompt});root=response.get('turn_id')
    print(json.dumps({'case':args.case,'turn_id':root,'started':True},ensure_ascii=False),flush=True)
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        status=http('/api/status');plan=status['agent'].get('plan') or {}
        if not status['job'] and not status['agent']['busy'] and plan.get('turn_id')==root and plan.get('verdict'):break
        if not status['job'] and not status['agent']['busy']:
            with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
                latest=db.execute('SELECT status FROM turns WHERE id=? OR parent=? ORDER BY created DESC LIMIT 1',(root,root)).fetchone()
            if latest and latest[0] in {'error','cancelled','interrupted','budget_exhausted'}:break
        time.sleep(.5)
    else:raise TimeoutError('task still needs attention; do not reset or restart an active task')
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        turns=[dict(r) for r in db.execute('SELECT * FROM turns WHERE id=? OR parent=? ORDER BY created',(root,root))]
        for turn in turns:
            turn['trace']=[{**dict(e),'detail':json.loads(e['detail'])} for e in db.execute('SELECT * FROM trace WHERE turn_id=? ORDER BY id',(turn['id'],))]
    calls=sum(e['kind']=='api' for turn in turns for e in turn['trace'])
    events=[e for turn in turns for e in turn['trace'] if e['kind']=='tool' and e['detail'].get('result',{}).get('execution_id')]
    native={}
    if args.case in ('carry','carry-timing'):
        with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
            for event in events:
                execution=event['detail']['result']['execution_id']
                archived=db.execute('SELECT run_id FROM episodes WHERE id=?',(execution,)).fetchone()
                if not archived or not archived[0]:continue
                folder=(ROOT/'runs'/archived[0]).resolve()
                if not folder.is_relative_to(ROOT/'runs'):raise ValueError('native receipt outside runs')
                manifest=json.loads((folder/'manifest.json').read_text());summary=json.loads((folder/'summary.json').read_text())
                feedback=summary.get('metrics',{}).get('tamp',{})
                native[execution]={'execution_id':execution,'run_id':archived[0],'manifest':manifest,'summary':summary,
                    'native_carried_motion_confirmed':bool(summary.get('transitions',0)>0
                        and feedback.get('primitive_success') and feedback.get('native_path_revalidation')
                        and feedback.get('fallback') is False and feedback.get('plans')
                        and all(p.get('success') and p.get('attachment',{}).get('object_id')=='bottle' for p in feedback['plans'])
                        and summary.get('evaluation',{}).get('held',{}).get('right'))}
        events=[e for e in events if (receipt:=native.get(e['detail']['result']['execution_id']))
                and receipt['summary'].get('metrics',{}).get('tamp',{}).get('primitive_success')
                and (e['name']!='move_relative' or receipt['native_carried_motion_confirmed'])]
    matched=0;matched_events=[]
    for event in events:
        tool,required=expected[matched]
        if event['name']==tool and all(event['detail']['arguments'].get(k)==v for k,v in required.items()):
            matched+=1;matched_events.append(event)
        if matched==len(expected):break
    target=plan.get('placement_goal') or {};initial_xy=initial['diagnostics']['manipulation_scene']['position_m'][:2]
    expected_xy=[initial_xy[0]+(.01 if args.case=='small' else .08 if args.case in ('timing','carry','carry-timing') else .06),initial_xy[1]]
    unchanged_goal=len(target.get('target_xy_m',[]))==2 and max(abs(a-b) for a,b in zip(target['target_xy_m'],expected_xy))<1e-8
    passed=bool(matched==len(expected) and unchanged_goal and plan.get('epoch')==initial['observation_id'].split(':')[0]
        and plan.get('goal_assessment',{}).get('confirmed') and all(s['status']=='done' for s in plan['steps'] if s['checkpoint']!='visual_review'))
    if args.case=='recovery':passed=passed and plan.get('time_budget_s')==60
    if args.case in ('timing','carry-timing'):
        waits=[e for turn in turns for e in turn['trace'] if e['kind']=='timing']
        passed=passed and plan.get('time_budget_s')==120 and plan.get('timing_mode')=='measured_delay' and bool(waits) and all(not e['detail']['task_failed'] for e in waits)
    backend_evidence=None
    if args.case in ('carry','carry-timing'):
        carried=next((e for e in matched_events if e['name']=='move_relative'),None)
        if carried:
            execution=carried['detail']['result']['execution_id']
            backend_evidence=native.get(execution)
        passed=passed and plan.get('time_budget_s')==120 and plan.get('timing_mode')==('paused' if args.case=='carry' else 'measured_delay') and bool(backend_evidence and backend_evidence['native_carried_motion_confirmed'])
    report={**protocol,'status':'passed' if passed else 'failed','turn_id':root,'actual_api_calls':calls,
        'wall_seconds':time.monotonic()-started,'initial_observation_id':initial['observation_id'],
        'initial_bottle_position_m':initial['diagnostics']['manipulation_scene']['position_m'],
        'required_sequence_matched':matched==len(expected),'goal_unchanged':unchanged_goal,'backend_execution_evidence':backend_evidence,
        'response':response,'plan':plan,'turns':turns,'terminal':status}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'case':args.case,'status':report['status'],'actual_api_calls':calls,'goal':plan.get('goal_assessment')},ensure_ascii=False),flush=True)
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
