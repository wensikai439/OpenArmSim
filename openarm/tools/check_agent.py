#!/usr/bin/env python3
"""Actual physics and deterministic local API checks; never an LLM benchmark."""
import html
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault('MUJOCO_GL','egl')
from harness.agent import Harness
from harness.workspace import Workspace, redact
from sim.service import SimulationService
from sim.recording import dump


class FastSimulation(SimulationService):
    def call(self,command,**args):
        if command=='tamp':args['realtime']=False
        return super().call(command,**args)


def main():
    calls=[]
    scenario={'name':'chain', 'recovery_calls':0}
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            names=[t['function']['name'] for t in payload['tools']]
            assert 'run_tamp' not in names and 'run_pi05_vla' not in names
            assert '"evaluation"' not in json.dumps(payload)
            observation=None
            for message in payload['messages']:
                parts=message.get('content',[])
                if isinstance(parts,str):parts=[{'type':'text','text':parts}]
                for part in parts:
                    text=part.get('text','')
                    if '\nCurrent observation:\n' in text:
                        observation=json.JSONDecoder().raw_decode(text.split('\nCurrent observation:\n',1)[1])[0]
            assert observation
            obs=observation['observation_id'];index=len(calls)%10 if scenario['name']=='chain' else scenario['recovery_calls']
            stages=[{'id':'observe','title':'观察场景','checkpoint':'observation'},
                    {'id':'pick','title':'右手抓起','checkpoint':'grasp_confirmed','skill':'pick'},
                    {'id':'transfer','title':'交给左手','checkpoint':'handover_confirmed','skill':'handover'},
                    {'id':'place','title':'放入篮子','checkpoint':'placement_confirmed','skill':'place'},
                    {'id':'review','title':'视觉复查','checkpoint':'visual_review'}]
            actions=[('inspect_manipulation',{'observation_id':obs}),('set_plan',{'observation_id':obs,'steps':stages,'placement_goal':{'object_id':'bottle','surface':'basket'}}),
                     ('pick',{'observation_id':obs,'object_id':'bottle','arm':'right'}),('get_run_status',{}),
                     ('handover',{'observation_id':obs,'object_id':'bottle','from_arm':'right','to_arm':'left'}),('get_run_status',{}),
                     ('place',{'observation_id':obs,'object_id':'bottle','receptacle_id':'basket','arm':'left'}),('get_run_status',{}),
                     ('finish',{'observation_id':obs,'verdict':'uncertain','reason':'Contract provider is not a visual verifier.'})]
            if scenario['name']=='recovery':
                for message in payload['messages']:
                    if message.get('role')=='tool':
                        feedback=json.loads(message['content'])
                        if feedback.get('skill')=='move_to_pose':
                            assert feedback['feedback']['failure_code']=='workspace_bounds'
                        elif feedback.get('skill')=='move_relative':
                            assert feedback['feedback']['primitive_success'] and feedback['evidence'].get('camera_keyframes')
                actions=[('set_plan',{'observation_id':obs,'steps':[{'id':'move','title':'末端到位','checkpoint':'motion_reached','skill':'move_to_pose'}]}),
                    ('move_to_pose',{'observation_id':obs,'arm':'right','position_m':[8,0,1.2],'quaternion_xyzw':[0,0,0,1],'frame':'world'}),
                    ('get_run_status',{}),('get_execution_record',{'execution_id':(work.active_plan or {}).get('execution_id','not-started'),'include_oracle':True}),
                    ('set_plan',{'observation_id':obs,'steps':[{'id':'recovery','title':'向上移动两厘米','checkpoint':'motion_reached','skill':'move_relative'}]}),
                    ('move_relative',{'observation_id':obs,'arm':'right','delta_m':[0,0,.02],'frame':'world'}),
                    ('get_run_status',{}),('get_execution_record',{'execution_id':(work.active_plan or {}).get('execution_id','not-started'),'include_oracle':True}),
                    ('finish',{'observation_id':obs,'verdict':'uncertain','reason':'FK arrival is independently verified; provider is not a camera verifier.'})]
                scenario['recovery_calls']+=1
            if index<len(actions):
                name,args=actions[index]
                message={'role':'assistant','tool_calls':[{'type':'function','id':f'call{len(calls)}','function':{'name':name,'arguments':json.dumps(args)}}]}
            else:name,args='final',{};message={'role':'assistant','content':'技能链结束；测试端不做视觉成功判定。'}
            calls.append({'tool':name,'arguments':args,'observation_id':obs,'scenario':scenario['name']})
            data=json.dumps({'model':'local-contract-provider','choices':[{'message':message}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    provider=ThreadingHTTPServer(('127.0.0.1',0),Provider)
    thread=threading.Thread(target=provider.serve_forever,daemon=True);thread.start()
    report={'status':'running','scope':'local protocol provider + actual native MuJoCo, not external LLM capability',
            'cases':[],'motions':[],'recovery':None,'hardware_access':False,'transfer_ready':False,'external_llm_tested':False}
    folder=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-harness-check');folder.mkdir(parents=True)
    service=None;agent=None
    try:
        with tempfile.TemporaryDirectory(prefix='openarm-harness-') as tmp:
            service=FastSimulation();work=Workspace(Path(tmp)/'agent.sqlite3');agent=Harness(service,workspace=work)
            agent.configure_model({'base_url':f'http://127.0.0.1:{provider.server_port}/v1','model':'local-contract-provider',
                                   'api_key':'local-test-placeholder','vision':True})
            for episode in range(3):
                service.call('reset',episode=episode,seed=0)
                response=agent.chat('用右手抓起瓶子，交给左手，再放入篮子。')
                assert [c['name'] for c in response['tools']]==['inspect_manipulation','set_plan','pick']
                deadline=time.monotonic()+120
                while time.monotonic()<deadline:
                    plan=work.active_plan or {}
                    if plan.get('verdict')=='uncertain' and not work.busy and not agent.pending_review and not service.snapshot().get('job'):break
                    time.sleep(.05)
                else:raise TimeoutError('skill chain exceeded wall-time budget')
                snapshot=service.snapshot();run=snapshot['last_run'];assert run['status']=='primitive_completed',run
                ids=work.active_plan['executions'];assert len(ids)==3
                experiences=work.memory('execute',snapshot['observation_id'].split(':')[0],include_oracle=True)['experiences']
                actual=[e for e in experiences if e['id'] in ids]
                assert {e['tool'] for e in actual}=={'pick','handover','place'}
                assert all(e['evidence_kind']=='simulator_oracle' for e in actual)
                assert all(e['outcome']=='success' for e in actual)
                assert all(len(e['evidence']['camera_keyframes'])==2 for e in actual)
                assert not work.memory('execute',snapshot['observation_id'].split(':')[0])['experiences']
                case={'episode':episode,'status':run['status'],'execution_ids':ids,'last_run':run['run_id'],
                      'runs':[dict(work.db.execute('SELECT run_id,tool FROM episodes WHERE id=?',(ident,)).fetchone()) for ident in ids],
                      'model_verdict':'uncertain','independent_simulator_success':snapshot['diagnostics']['evaluation']['success']}
                report['cases'].append(case);dump(folder/'summary.json',report)
                print(json.dumps({'episode':episode,'status':'passed','skills':['pick','handover','place']},ensure_ascii=False),flush=True)
            service.call('reset',episode=0,seed=0);scenario['name']='recovery'
            response=agent.chat('将右手末端向上移动两厘米，目标不可达时根据反馈修正方案。')
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if (work.active_plan or {}).get('verdict')=='uncertain' and not work.busy and not agent.pending_review and not service.snapshot().get('job'):break
                time.sleep(.05)
            else:raise TimeoutError('recovery chain exceeded wall-time budget')
            snapshot=service.snapshot();epoch=snapshot['observation_id'].split(':')[0]
            ids=work.active_plan['executions'];assert len(ids)==2
            memory=work.memory('execute',epoch,context=snapshot['memory_context'],include_oracle=True)
            receipts={e['id']:e for e in memory['experiences']}
            assert receipts[ids[0]]['feedback']['failure_code']=='workspace_bounds'
            assert receipts[ids[0]]['outcome']=='failure' and receipts[ids[1]]['outcome']=='success'
            rejected = receipts[ids[0]]
            assert rejected['feedback']['motion_started'] is False
            assert len(rejected['evidence']['camera_keyframes'])==2
            rejected_run = ROOT/'runs'/rejected['evidence']['run_id']
            rejected_summary = json.loads((rejected_run/'summary.json').read_text())
            assert rejected_summary['transitions']==0
            assert work.plan_status(snapshot)['steps'][0]['status']=='done'
            assert service.snapshot()['last_run']['metrics']['tamp']['position_error_m']<=.012
            report['recovery']={'status':'passed','execution_ids':ids,'failure_code':'workspace_bounds',
                'corrected_skill':'move_relative','raw_receipt_inspected_explicitly':True,'after_observation_id':snapshot['observation_id'],
                'feedback':receipts[ids[1]]['feedback'],'evidence':receipts[ids[1]]['evidence'],
                'failed_attempt':{'feedback':rejected['feedback'],'evidence':rejected['evidence'],'transitions':0}}
            print(json.dumps({'recovery':'passed','archive':'failure + success receipts explicitly inspected; default memory stays compact'}),flush=True)
            for arm in ('right','left'):
                service.call('reset',episode=0,seed=0);obs=agent.invoke('observe',{})
                if arm=='right':
                    result=agent.invoke('move_relative',{'observation_id':obs['observation_id'],'arm':arm,'delta_m':[0,0,.02],'frame':'world'})
                else:
                    pose=obs['tcp'][arm];position=pose['position_m'].copy();position[2]+=.02
                    result=agent.invoke('move_to_pose',{'observation_id':obs['observation_id'],'arm':arm,'position_m':position,
                                                      'quaternion_xyzw':pose['quaternion_xyzw'],'frame':'world'})
                deadline=time.monotonic()+60
                while service.snapshot().get('job') and time.monotonic()<deadline:time.sleep(.05)
                run=service.snapshot()['last_run'];feedback=run['metrics']['tamp'];assert run['status']=='primitive_completed',run
                assert feedback['position_error_m']<=.012 and feedback['rotation_error_rad']<=.08
                report['motions'].append({'arm':arm,'execution_id':result['execution_id'],'run_id':run['run_id'],'feedback':feedback})
            report.update(status='passed',provider_requests=len(calls),calls=calls,
                          memory=redact(work.memory('execute',service.snapshot()['observation_id'].split(':')[0],public=True,
                              context=service.snapshot()['memory_context'])))
    except Exception as exc:report.update(status='failed',error=str(exc));raise
    finally:
        if agent:agent.close()
        if service:service.close()
        provider.shutdown();provider.server_close();thread.join(2);dump(folder/'summary.json',report)
        cards=[]
        for case in report['cases']:
            links=' '.join(f'<a href="../{r["run_id"]}/summary.json">{html.escape(r["tool"])} ↗</a>' for r in case['runs'])
            frames=sorted((ROOT/'runs'/case['last_run']/'keyframes').glob('*.jpg'))
            gallery=''.join(f'<a href="../{case["last_run"]}/keyframes/{p.name}"><img src="../{case["last_run"]}/keyframes/{p.name}" alt="原生仿真结束帧"></a>' for p in frames[-3:])
            cards.append(f'<article><h2>初态 {case["episode"]+1}</h2><p>pick → 新观测 → handover → 新观测 → place → 独立仿真评估</p><p>{links}</p><div class="gallery">{gallery}</div></article>')
        (folder/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Harness 工程反馈</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;background:#f7f5f0;color:#403d38;font:15px/1.7 system-ui}article{background:#fffdf9;border:1px solid #ded8cf;padding:20px;border-radius:12px;margin:20px 0}.gallery{display:flex;gap:12px}.gallery a{width:33%}img{width:100%;border-radius:8px}a{color:#93674e}</style>'
            f'<h1>Harness · {html.escape(report["status"])}</h1><p>三个记录初态，真实 MuJoCo 技能链。测试 API 验证协议，不代表真实 LLM 成功率或真机迁移。</p><p><a href="summary.json">完整反馈、动作误差与记忆来源</a></p>'+''.join(cards)+'</html>')
        if report.get('recovery'):
            recovery=report['recovery'];failed=recovery['failed_attempt']['evidence']['run_id']
            success=recovery['evidence']['run_id'];error=recovery['feedback']['position_error_m']*1000
            extra=f'<article><h2>反馈与记忆恢复</h2><p>目标越界 → 读取失败经验 → 修改计划 → 相对移动 → 检索成功经验</p><p>越界尝试：0 次物理动作，前后三路相机证据保留。修正后末端误差 {error:.2f} mm。</p><p><a href="../{failed}/summary.json">失败证据 ↗</a> · <a href="../{success}/summary.json">恢复证据 ↗</a></p></article>'
            page=folder/'index.html';page.write_text(page.read_text().replace('</html>',extra+'</html>'))
    print(json.dumps({'status':report['status'],'report':str(folder/'index.html'),'provider_requests':len(calls)},ensure_ascii=False))


if __name__=='__main__':main()
