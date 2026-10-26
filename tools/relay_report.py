"""Summarize expanded skill experiments without hiding failures or speed cost."""
import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.relay_motion import compare


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    development=json.loads((root/'development/round-2/summary.json').read_text())
    validation=json.loads((root/'validation/summary.json').read_text())
    rows=[*development['cases'],*validation['cases']];groups={}
    for group in ('development','regression','heldout','stress'):
        selected=[r for r in rows if r['group']==group];successful=[r for r in selected if r['full_task_success']]
        groups[group]={'passed':len(successful),'total':len(selected),
            'median_seconds':float(np.median([r['sim_seconds'] for r in successful])) if successful else None,
            'min_seconds':min((r['sim_seconds'] for r in successful),default=None),
            'max_seconds':max((r['sim_seconds'] for r in successful),default=None),
            'failures':[{'name':r['name'],'error':r.get('error')} for r in selected if not r['full_task_success']]}
    paired={r['name']:compare(ROOT/'runs/20261026-smooth/validation'/r['name'],root/'development/round-2'/r['name'])
        for r in development['cases'] if r['group']=='regression'}
    motion={'median_time_change_percent':float(np.median([100*(r['candidate']['seconds']/r['baseline']['seconds']-1) for r in paired.values()])),
        'median_jerk_p95_change_percent':float(np.median([r['change_percent']['jerk_rad_s3']['p95'] for r in paired.values()])),
        'median_jerk_rms_change_percent':float(np.median([r['change_percent']['jerk_rad_s3']['rms'] for r in paired.values()]))}
    motion['gates_passed']=bool(groups['regression']['passed']==6 and motion['median_time_change_percent']<=-15 and
        motion['median_jerk_p95_change_percent']<=10 and motion['median_jerk_rms_change_percent']<=20)
    usage=json.loads((root/'model/usage.json').read_text())
    summary={'groups':groups,'motion':motion,'paired':paired,'unique_declared_cases':sum(g['total'] for g in groups.values()),
        'model_requests':usage['requests'],'model_reported_tokens':usage['total_tokens'],
        'model_role':'review of engineered program and bounded waits; not per-control-cycle inference or neural training',
        'scope':'known upright black bottle, fixed white basket, two synthetic RGB views and virtual tool marks; no hardware',
        'first_development':json.loads((root/'development/round-1/summary.json').read_text())['groups'],
        'final_sources_frozen':True}
    save(root/'analysis.json',summary)
    # Joint-position coverage: every declared case is shown, failures in red.
    plots=[]
    palette={'development':'#b9a58d','regression':'#8f9290','heldout':'#63816b','stress':'#ab815e'}
    for group in groups:
        points=[]
        for r in rows:
            if r['group']!=group:continue
            x,y=r['case']['bottle_xy_m'];cx=55+(x-.45)/.10*360;cy=245-(y+.27)/.12*200
            color=palette[group] if r['full_task_success'] else '#bb5c4a'
            title=html.escape(f'{r["name"]}: {"pass" if r["full_task_success"] else "FAIL"}; x={x:.3f}, y={y:.3f}')
            points.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.8" fill="{color}" opacity=".85"><title>{title}</title></circle>')
        plots.append(f'<g transform="translate({len(plots)%2*470},{len(plots)//2*295})"><text x="55" y="25" fill="#625b50" font-size="15">{group} {groups[group]["passed"]}/{groups[group]["total"]}</text><path d="M55 45V245H415" stroke="#cfc5b6" fill="none"/><text x="55" y="267">x 0.45 m</text><text x="343" y="267">0.55 m</text><text x="3" y="50">−0.15</text><text x="3" y="248">−0.27</text>'+''.join(points)+'</g>')
    (root/'coverage.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 590" style="font:12px system-ui;background:#fffefa;color:#8d8171">'+''.join(plots)+'</svg>')
    table=[]
    for group,g in groups.items():
        seconds=f'{g["median_seconds"]:.1f} 秒' if g['median_seconds'] is not None else '—'
        table.append(f'<tr><td>{group}</td><td>{g["passed"]} / {g["total"]}</td><td>{seconds}</td></tr>')
    failures=[r for r in validation['cases'] if not r['full_task_success']]
    failure_text='；'.join(html.escape(r['name']+': '+str(r.get('error'))) for r in failures) or '本次留出与压力测试没有执行失败。'
    held=groups['heldout'];reg=groups['regression'];stress=groups['stress']
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 扩大泛化与提速</title><style>
    :root{font:14px/1.7 Inter,"Noto Sans CJK SC",system-ui;color:#39362f;background:#f7f5ef}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:45px auto;padding:0 25px}.eyebrow{color:#a28a70;font-size:11px;letter-spacing:2px}h1{font-size:33px;line-height:1.3;letter-spacing:-1px;margin:14px 0}p{color:#847b6d}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.metric,.card{background:#fffefa;border:1px solid #e7e0d5;border-radius:15px;padding:19px}.metric strong{font-size:28px;display:block;font-weight:550}.metric span{font-size:12px;color:#918674}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.label{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px}.badge{background:#edf1e7;color:#6d7f59;border-radius:15px;padding:1px 9px;font-size:11px}video{display:block;width:100%;border-radius:9px;background:#6b746c}.card p{font-size:12px;margin-bottom:0}.note{border-left:3px solid #bba286;padding-left:17px;margin:25px 0}a{color:#997252;text-underline-offset:4px}details{border-top:1px solid #e5ded2;padding-top:17px;margin-top:21px;font-size:13px;color:#857a6a}summary{cursor:pointer;color:#625a4e}table{width:100%;border-collapse:collapse;margin:17px 0}td,th{border-bottom:1px solid #e8e1d5;text-align:left;padding:10px}th{font-weight:500;color:#9a8b78}img{width:100%;border-radius:10px}.links{display:flex;gap:23px;flex-wrap:wrap;margin:25px 0}footer{margin:30px 0;color:#9b8f7e;font-size:12px}@media(max-width:780px){main{padding:0 16px}.grid{grid-template-columns:1fr}.metrics{gap:8px}.metric{padding:13px}.metric strong{font-size:24px}h1{font-size:28px}}
    </style><main><div class="eyebrow">OPENARM LAB / EXPANDED VISUAL SKILL</div><h1>扩大覆盖，让整套动作更快</h1><p>远处根据 RGB 与机器人反馈估计相对目标，近处保留 6 mm 视觉纠偏；接瓶先检查整段接近路径。</p>'''
    page+=f'<section class="metrics"><div class="metric"><strong>{held["passed"]} / {held["total"]}</strong><span>新留出场景 · 完整任务</span></div><div class="metric"><strong>{abs(motion["median_time_change_percent"]):.0f}%</strong><span>相同六个场景 · 运动耗时缩短</span></div><div class="metric"><strong>{summary["unique_declared_cases"]}</strong><span>声明场景 · 开发、留出、压力与回归分开</span></div></section>'
    page+='''<section class="grid"><article class="card"><div class="label">完整抓取、交接、入篮<span class="badge">原速录像</span></div><video controls playsinline preload="metadata" src="development/round-2/nominal/rollout.mp4"></video><p>与上一版同场景比较；没有加速视频、放宽物理评分或操作真机。</p></article><article class="card"><div class="label">新场景与闭爪指令丢失<span class="badge">留出测试</span></div><video controls playsinline preload="metadata" src="validation/heldout-01/rollout.mp4"></video><p>联合改变瓶位、两相机视角、观察距离、曝光与测量噪声；首次闭爪指令丢失。</p></article></section>'''
    page+=f'<div class="note"><p>留出区域约 7 × 9 cm；边界压力另算 {stress["passed"]}/{stress["total"]}。离散测试通过不代表区域内所有位置、所有物体或真机都已验证。</p></div><table><thead><tr><th>阶段</th><th>完整任务通过</th><th>成功案例耗时中位数</th></tr></thead><tbody>'+''.join(table)+'</tbody></table>'
    page+='''<div class="links"><a href="validation/index.html">查看全部新测试与失败 →</a><a href="development/round-2/index.html">开发与配对回归 →</a><a href="../20261026-smooth/index.html">上一版录像 →</a></div>
    <details><summary>泛化覆盖与失败位置</summary><img src="coverage.svg" alt="全部声明瓶位与测试结果；红色为失败"/><p>开发与留出位置采用分层采样，额外联合改变相机公共及独立角度、距离、曝光、像素噪声与首次闭爪丢失。相机与物体在每次任务内仍保持固定，瓶型、尺寸、篮子和机器人动力学保持不变。</p><p>'''+failure_text+'</p></details>'
    page+=f'''<details><summary>关键调优过程与运动数据</summary><p>第一轮开发 22/24；两例暴露“目标点可达，但中间接近路径不可达”。修正为整段走廊检查后，重新跑全部开发条件。没有放宽 IK 精度、8 mm 实际到位检查或原生接触评分。</p><p>六例配对：耗时中位变化 {motion['median_time_change_percent']:+.1f}%；实际关节 jerk 95 分位变化 {motion['median_jerk_p95_change_percent']:+.1f}%，RMS 变化 {motion['median_jerk_rms_change_percent']:+.1f}%。运动与接触指标均由原始实测关节记录计算，不以视频观感代替。</p><p>Haiku API {summary['model_requests']} 次，服务报告 {summary['model_reported_tokens']:,} token（不包含 Codex 自身用量）；模型审阅工程程序并调整有界等待，底层视觉与控制器由工程实现。不是每个控制周期调用 API，也没有训练神经网络。</p><p><a href="记录.md">实验记录</a> · <a href="analysis.json">完整统计与配对指标</a> · <a href="splits.json">预声明场景</a> · <a href="model/review.json">模型修改审查</a></p></details><footer>仅本地仿真 · 官方 ASPIRE 执行器与工具注册 · 旧 TAMP 演示保留 · 真机未操作</footer></main></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps({'groups':groups,'motion':motion},ensure_ascii=False))


if __name__=='__main__':main()
