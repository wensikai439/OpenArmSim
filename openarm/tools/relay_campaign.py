"""Frozen parallel simulation trials with disjoint development/heldout groups."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.visual_benchmark import verify, save
from tools.relay_motion import measure


def report(folder,protocol,rows):
    esc=html.escape;counts={}
    for group in protocol['groups']:
        selected=[r for r in rows if r['group']==group]
        counts[group]={'passed':sum(r['full_task_success'] for r in selected),'completed':len(selected),
            'declared':sum(r['group']==group for r in protocol['cases'])}
    save(folder/'summary.json',{'groups':counts,'passed':sum(r['full_task_success'] for r in rows),
        'total':len(rows),'declared':len(protocol['cases']),'complete':len(rows)==len(protocol['cases']),'cases':rows})
    table=[]
    for r in rows:
        video=f'<a href="{r["name"]}/rollout.mp4">原速录像</a>' if (folder/r['name']/'rollout.mp4').exists() else '已留图像与轨迹'
        table.append(f'<tr><td>{esc(r["name"])}</td><td>{"通过" if r["full_task_success"] else "失败"}</td><td>{r.get("sim_seconds",0):.1f}</td><td>{video}</td><td>{esc(r.get("error") or "")}</td></tr>')
    title=' · '.join(f'{g} {c["passed"]}/{c["completed"]}（声明 {c["declared"]}）' for g,c in counts.items())
    (folder/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OpenArm · 场景验收</title><style>
    body{max-width:1180px;margin:45px auto;padding:0 24px;font:14px/1.7 system-ui;color:#37352f;background:#f7f5ef}h1{font-size:28px}p{color:#8a7c6c}a{color:#997354}table{border-collapse:collapse;width:100%;background:#fffefa}td,th{text-align:left;padding:12px;border-bottom:1px solid #e7e1d6}td:last-child{font-size:12px;max-width:400px}header{margin-bottom:25px}
    </style><header><h1>场景与完整动作验收</h1><p>'''+esc(title)+'''</p><p>独立物理评分；所有初始化、运动和执行失败均计入分母。部分场景只保留图像与实测轨迹，避免重复录像开销。</p></header>
    <table><thead><tr><th>场景</th><th>结果</th><th>仿真秒</th><th>记录</th><th>失败原因</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table><p><a href="protocol.json">冻结协议</a> · <a href="summary.json">完整结果</a> · <a href="events.jsonl">关键过程</a></p></html>''')


def run_one(folder,protocol,spec):
    verify(protocol);case=folder/'cases'/(spec['name']+'.json');save(case,spec['case']);out=folder/spec['name'];start=time.monotonic()
    try:
        proc=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','harness.aspire.visual_adaptive',
            '--case',str(case),'--output',str(out),'--program',str(ROOT/protocol['program'])],cwd=ROOT,
            env={**os.environ,'MUJOCO_GL':'egl','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
        out.mkdir(exist_ok=True);(out/'process.txt').write_text(proc.stdout+proc.stderr)
        result=json.loads((out/'result.json').read_text())
        if proc.returncode:
            result={**result,'full_task_success':False,'error':'process failed after or during execution; inspect process.txt'}
        if (out/'controller.json').exists():
            motion=measure(out);save(out/'motion.json',motion)
            result['motion']=motion['actual'];result['steady_holds']=motion['steady_holds']
    except Exception as exc:
        result={'full_task_success':False,'error':type(exc).__name__+': '+str(exc)}
        out.mkdir(exist_ok=True);save(out/'result.json',result)
    return {**spec,**result,'wall_seconds':time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--groups',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program',default='skills/bottle_visual/adaptive/program.py');p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();folder=a.output.resolve()
    if folder.exists() or not folder.is_relative_to(ROOT/'runs') or not 1<=a.jobs<=3:p.error('new runs/ folder; 1..3 jobs')
    splits=json.loads(a.splits.read_text());cases=[]
    for group in a.groups:
        for spec in splits['groups'][group]:cases.append({**spec,'group':group})
    base=json.loads((ROOT/'skills/bottle_visual/relay/evaluation.json').read_text());sources=list(base['sources'])+[
        'harness/aspire/visual_adaptive.py','harness/aspire/adaptive_checks.py','tools/relay_campaign.py',a.program]
    protocol={**base,'schema':'adaptive-rgb-relay/1','program':a.program,'groups':a.groups,'cases':cases,
        'splits_sha256':hashlib.sha256(a.splits.read_bytes()).hexdigest(),
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in dict.fromkeys(sources)},
        'training_method':'model-reviewed program and engineered closed-loop controller; no neural/RL training',
        'model_role':'reviewed engineered draft and bounded waits; see campaign model requests',
        'motion_comparison':{'baseline':'runs/20261026-smooth/validation',
            'paired_regression_gates':{'all_native_tasks_pass':True,'median_time_change_percent_max':-15,
                'median_jerk_p95_change_percent_max':10,'median_jerk_rms_change_percent_max':20},
            'note':'six matched previously used cases; new heldout and stress have no matched previous controller baseline'},
        'recording':'all RGB/keyframe/controller traces; overview video only for predeclared cases'}
    verify(protocol);folder.mkdir(parents=True);(folder/'cases').mkdir();save(folder/'protocol.json',protocol)
    for name in protocol['sources']:
        target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    rows=[];order={s['name']:i for i,s in enumerate(cases)};report(folder,protocol,rows)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        pending={pool.submit(run_one,folder,protocol,s):s for s in cases}
        for future in as_completed(pending):
            spec=pending[future]
            try:row=future.result()
            except Exception as exc:row={**spec,'full_task_success':False,'error':str(exc)}
            rows.append(row);rows.sort(key=lambda r:order[r['name']]);report(folder,protocol,rows)
            event={'time_utc':datetime.now(timezone.utc).isoformat(),'case':row['name'],'group':row['group'],
                'pass':row['full_task_success'],'seconds':row.get('sim_seconds'),'error':row.get('error')}
            with (folder/'events.jsonl').open('a') as stream:stream.write(json.dumps(event,ensure_ascii=False)+'\n')
            print(json.dumps(event,ensure_ascii=False),flush=True)
    verify(protocol)


if __name__=='__main__':main()
