"""Reproducible native TAMP development check across three recorded starts.

No external API, hardware, model training or calibration mutation. All cases,
including failures, are retained. These are development cases, not a benchmark.
"""
import html
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
from sim.service import SimulationService
from sim.recording import dump


def main():
    folder=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-tamp-check-'+uuid.uuid4().hex[:6])
    folder.mkdir()
    report={'scope':'three recorded arm starting poses, same candidate object scene; development check, not generalization benchmark',
            'status':'running','cases':[], 'external_llm_tested':False,'hardware_access':False,'transfer_ready':False}
    dump(folder/'summary.json',report)
    service=SimulationService()
    try:
        for episode in range(3):
            service.call('reset',episode=episode,seed=0)
            service.call('tamp',task='handover_to_basket',observation_id=service.snapshot()['observation_id'],realtime=False,
                         initiator={'type':'development-check','case':episode})
            deadline=time.monotonic()+120
            while service.snapshot().get('job') and time.monotonic()<deadline:
                time.sleep(.05)
            if service.snapshot().get('job'):
                service.call('stop')
                raise TimeoutError('case exceeded wall-time budget')
            result=service.snapshot()['last_run']
            summary=json.loads((ROOT/'runs'/result['run_id']/'summary.json').read_text())
            row={'episode':episode,'run_id':result['run_id'],'status':result['status'],
                 'steps':summary['transitions'],'sim_seconds':summary['sim_seconds'],
                 'wall_seconds':summary['wall_seconds'],'evaluation':summary['evaluation'],
                 'events':summary['events'],'planning':summary['metrics']['tamp']}
            report['cases'].append(row)
            dump(folder/'summary.json',report)
            print(json.dumps({k:row[k] for k in ('episode','run_id','status','steps','sim_seconds')},ensure_ascii=False),flush=True)
        report['status']='passed' if all(c['status']=='success' for c in report['cases']) else 'failed'
    except Exception as exc:
        report.update(status='error',error=str(exc))
        raise
    finally:
        service.close()
        dump(folder/'summary.json',report)
        cards=[]
        for case in report['cases']:
            run=case['run_id'];evaluation=case['evaluation']
            frames=sorted((ROOT/'runs'/run/'keyframes').glob('*.jpg'))
            gallery=''.join(f'<a href="../{run}/keyframes/{p.name}"><img loading="lazy" src="../{run}/keyframes/{p.name}" alt="原生仿真关键帧"></a>' for p in frames)
            stages=' → '.join(e['name'] for e in case['events'])
            cards.append(f'<section><h2>初态 {case["episode"]+1} · {html.escape(case["status"])}</h2>'
                f'<p>{case["steps"]} 控制步 · {case["sim_seconds"]:.2f} s 仿真 · {case["wall_seconds"]:.2f} s 计算 · '
                f'峰值穿入 {evaluation["max_penetration_m"]*1000:.2f} mm</p><p>{stages}</p>'
                f'<a href="../{run}/summary.json">完整记录与失败反馈</a><div class="gallery">{gallery}</div></section>')
        (folder/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm TAMP 验证</title>'
            '<style>body{font:15px/1.7 system-ui,sans-serif;background:#f3f2ee;color:#333b38;max-width:1200px;margin:40px auto;padding:0 24px}'
            'h1{font-size:28px}h2{font-size:20px}section{padding:22px;margin:22px 0;border:1px solid #ddd;border-radius:14px;background:#fffefa}'
            'a{color:#83604c}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:18px}'
            'img{width:100%;border-radius:7px}p{margin:8px 0}</style>'
            '<h1>OpenArm · TAMP 原生物理验证</h1><p>右手抓起 → 双手交接 → 左手放入篮子</p>'
            '<p>三个记录初态、同一候选物体场景。已知仿真场景信息的开发检查，不能当作视觉策略成绩、泛化成功率或真机迁移证明。'
            '物体始终由自由刚体接触积分，阶段由独立评估器确认。点击关键帧可查看原图。</p>'
            '<a href="summary.json">结构化报告</a>'+''.join(cards),encoding='utf-8')
    print(json.dumps({'status':report['status'],'report':str(folder/'index.html')},ensure_ascii=False))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
