"""Thin sequential wrapper for the already proven normal-reset and official expert commands. No replay."""
import os
from pathlib import Path
import argparse,hashlib,json,subprocess,datetime,sys,urllib.request,shlex
ROOT=Path(str(Path(__file__).resolve().parents[1]))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ledger(run,current=None,blocker=None):
 entries=[]
 for td in sorted((run/'formal').iterdir()):
  p=td/'summary.json'
  if not p.exists():continue
  d=json.loads(p.read_text())
  entries.append({'output':str(td),'task_id':d.get('task_id'),'actual_physical_attempt':bool(d.get('task_id') and d.get('actions_dispatched',0)),
   'status':d.get('status'),'physical_goal_met':d.get('physical_goal_met'),'official_verdict':(d.get('formal_verdict') or {}).get('verdict'),
   'program_sha256':d.get('program_sha256'),'reviewer':d.get('reviewer','deepseek-flash'),'bc_pool_admitted':bool(d.get('physical_goal_met') and d.get('status')=='succeeded' and (d.get('formal_verdict') or {}).get('verdict')=='success')})
 value={'updated_at':utc(),'planned_formal_physical_attempts':30,'finished_physical_attempts':sum(x['actual_physical_attempt'] for x in entries),
  'bc_pool_episodes':sum(x['bc_pool_admitted'] for x in entries),'current':current,'blocker':blocker,'entries':entries}
 save(run/'formal-ledger.json',value)
 p=ROOT/'PROJECT.md';before=p.read_text();a=before.index('## 独立仿真回合扩充（执行中）');b=before.index('\n## ',a+4)+1
 section=f"""## 独立仿真回合扩充（执行中）：2026-10-12

实际完成：正式固定30次尝试已结束{value['finished_physical_attempts']}次，物理+官方enforce/success准入{value['bc_pool_episodes']}次；当前配置{current or '无正在派发的配置'}。train-01失败保留、不重跑凑成功数。expert-v8 SHA256 648ded7c256b304157e48aa0cb160eba2b624f0792de2a799f9eab07e062ff7d采用训练失败反馈的先X后Y退让，原1mm环境碰撞/IK/物理判据保持。train-02/task_8e140d4495d94d85实际155动作、3662控制步/366200积分、空中双指交接后落桶，桶心误差.0486250467m，官方succeeded/Flash success。v7四pilot均物理成功但无官方Task、0数据准入；15次诊断失败与版本全部保留。制品0.3.10+evo1及共享主资产不变。

证据路径：{run}/formal-ledger.json、formal/、formal-logs/、lifecycle/、expert-freeze-v8.json、recorded-receiver-clear.json及inference-interface-smoke.json；后者只验证旧state12检查点实际离线推理，0动作/0新训练，不能视为新策略性能。scenario-manifest SHA256 09831b1b34415eb7608c4bef008b700b7debdf2a969adfb7c9c441206c292fef保持。

当前阻塞：{blocker or '其余固定尝试尚在执行，独立sim数据导出及新对照训练/策略测试未完成。'}。验收来源逐Task记录：已有Flash回合保持，后续按用户要求由Codex实际读取证据回复，官方enforce/schema/准入保持；使用Codex后不再依赖DS余额。真实UNKNOWN/开合转换监督缺项、辅助未知、相机和继承插值限制保持，cross_domain_mixing_ready=false。

下一步：继续剩余固定配置；仅成功Task导出，完整回合划分20train/5validation/5test，训练失败可修订、失败不重放，进入validation/test前冻结专家。导出准备/LeRobot/20Hz/二值视图后做固定2000更新纯真实/混训ACT对照，再让两策略在冻结test初态实际运行；验证/测试排除训练与归一化。不新建环境、不改共享资产/RPent/ROS/ACT/π0.5/PiPER底座、不连接真机。

"""
 assert p.read_text()==before,'Concurrent PROJECT modification'
 p.write_text(before[:a]+section+before[b:])
 return value
def command(run,name,phase,argv):
 stem=run/'formal-logs'/(name+'-'+phase)
 save(stem.with_suffix('.command.json'),{'argv':argv,'cwd':str(ROOT),'automatic_replays':0})
 with stem.with_suffix('.log').open('x') as f:
  p=subprocess.Popen(['bash','-c','source scripts/env.sh\nexec '+shlex.join(argv)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  save(run/'batch-active.json',{'scenario':name,'phase':phase,'pid':p.pid,'started_at':utc()})
  rc=p.wait()
 save(stem.with_suffix('.exit.json'),{'exit_code':rc,'automatic_replays':0,'at':utc()})
 return rc
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--program',type=Path,required=True)
 ap.add_argument('--from-scenario',required=True);ap.add_argument('--reviewer',choices=('deepseek-flash','codex'),default='deepseek-flash');ap.add_argument('--codex-review-url');a=ap.parse_args();run=a.run_dir.resolve();program=a.program.resolve()
 sys.path.insert(0,str(ROOT/'scripts'));from run_sim_expert import load_program
 _,admission=load_program(program)
 manifest=json.loads((run/'scenario-manifest.json').read_text());rows=[x for x in manifest['scenarios'] if x['split']!='pilot']
 start=next(i for i,r in enumerate(rows) if r['scenario_id']==a.from_scenario)
 from PhyAgentOS.config.loader import load_config
 cfg=load_config();
 if a.reviewer=='deepseek-flash':assert cfg.agents.defaults.model in ('deepseek-flash','custom/deepseek-flash')
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for row in rows[start:]:
  name=row['scenario_id'];out=run/'formal'/('paos-20261012-'+name);assert not out.exists(),'Existing output; no replay'
  assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
  assert hashlib.sha256(program.read_bytes()).hexdigest()==admission['sha256']
  if row['split']!='train' and not (run/'expert-freeze-heldout.json').exists():
   save(run/'expert-freeze-heldout.json',{'program':str(program),'sha256':admission['sha256'],'before_first_heldout':name,'at':utc(),'heldout_used_for_revision':False})
  if a.reviewer=='deepseek-flash':
   req=urllib.request.Request(cfg.providers.custom.api_base.rstrip('/')+'/user/balance',headers={'Authorization':'Bearer '+cfg.providers.custom.api_key})
   with opener.open(req,timeout=20) as f:balance=json.load(f)
   save(run/'formal-logs'/(name+'-balance.json'),{'model':cfg.agents.defaults.model,'paid_chat_requests':0,'read_only':True,'at':utc(),'balance':balance})
   if not balance.get('is_available'):
    ledger(run,blocker='DeepSeek Flash余额不可用；未reset/创建Task/重放该配置。');return 4
  ledger(run,current=name)
  rc=command(run,name,'reset',[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/reset_sim_scene.py'),'--scene-config',row['path'],'--audit-dir',str(run/'lifecycle'/name)])
  if rc:ledger(run,blocker=name+'正常reset未确认；停止，须核副作用。');return rc
  rc=command(run,name,'enforce',[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_sim_expert.py'),'--program',str(program),'--out',str(out),'--max-actions','240','--require-joint-targets','--require-binary-gripper','--scene-config',row['path'],'--verification-mode','enforce','--reviewer',a.reviewer]+(['--codex-review-url',a.codex_review_url] if a.reviewer=='codex' else [])+['--run'])
  summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
  text=(run/'formal-logs'/(name+'-enforce.log')).read_text()
  verdict=summary.get('formal_verdict') or {}
  # A known terminal invalid model response is a failed formal attempt, not
  # permission to retry the paid request or to admit its physically successful data.
  invalid_verifier=False
  if summary.get('official_task_status')=='failed' and not verdict:
   taskfile=out/'task-final.json'
   if taskfile.exists():
    task_record=json.loads(taskfile.read_text())
    attempts=task_record.get('verification_attempts') or []
    invalid_verifier=bool(task_record.get('status')=='failed' and attempts
      and (attempts[-1].get('error') or '').startswith('invalid verifier response:')
      and all(e.get('status') in {'succeeded','failed','rejected','cancelled','expired'}
        for e in task_record.get('execution_records',[]) if e.get('semantics')=='action'))
    if invalid_verifier:save(run/'formal-logs'/(name+'-invalid-verifier.json'),{
      'task_id':summary['task_id'],'official_task_status':'failed','bc_pool_admitted':False,
      'known_terminal_verification_error':attempts[-1]['error'],'automatic_paid_replays':0,
      'continuation':'Preserve failed formal attempt; proceed to next frozen scenario once.'})
  if not summary.get('task_id') or (verdict.get('verifier_status')!='completed' and not invalid_verifier) or 'Insufficient Balance' in text:
   ledger(run,blocker=name+'官方入口/模型验收未完成；已记录尝试，不重放。');return rc or 4
  final_file=out/'final-query.json'
  final=json.loads(final_file.read_text())['data']['response']['result']['outputs']['observation'] if final_file.exists() else {}
  first_action_error=""
  if summary.get("actions_dispatched")==1 and final.get("control_step")==0:
   action_rows=json.loads((out/"actions.json").read_text())
   first_action_error=(action_rows[0].get("terminal_status",{}).get("data",{}).get("result",{})
     .get("outputs",{}).get("native_result",{}).get("error",""))
  initial_guard_failure=(summary.get("actions_dispatched")==1 and final.get("control_step")==0 and final.get("failed") is True
    and "Moving geometry penetration exceeds 10 mm" in first_action_error)
  if rc and row['split']=='train' and not initial_guard_failure and not invalid_verifier:
   ledger(run,blocker=name+'训练配置真实失败；保留失败，需要按真实证据修订专家后推进下一配置。');return rc
  value=ledger(run)
  print(json.dumps({'scenario_completed':name,'exit_code':rc,'finished':value['finished_physical_attempts'],'success':value['bc_pool_episodes']},ensure_ascii=False),flush=True)
 ledger(run);save(run/'batch-completed.json',{'at':utc(),'all_remaining_fixed_attempts_finished':True});return 0
if __name__=='__main__':raise SystemExit(main())
