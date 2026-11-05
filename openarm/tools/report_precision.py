"""Build a compact evidence page from frozen precision/API trial receipts."""
import hashlib
import html
import json
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261008-precision'
def read(path):return json.loads(path.read_text())
def esc(value):return html.escape(str(value))
def sources(path):
    manifest=read(path/'manifest.json')
    changed=[name for name,value in manifest['hashes'].items()
        if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value]
    return {'run_id':manifest['run_id'],'manifest':str((path/'manifest.json').relative_to(ROOT)),
            'matches_current_sources':not changed,'changed':changed}

def main():
    rows=read(OUT/'final/summary.json');variation=read(OUT/'variation/summary.json')
    apis=[read(OUT/'api-small.json'),read(OUT/'api-transfer.json')]
    signature=[sources(ROOT/row['run_path']) for row in rows+variation]
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
        for api in apis:
            for execution in api['plan']['execution_tools']:
                record=db.execute('SELECT run_id FROM episodes WHERE id=?',(execution['id'],)).fetchone()
                if not record or not record[0]:raise ValueError('API execution has no frozen run identity')
                signature.append(sources(ROOT/'runs'/record[0]))
    expected={'small':[('pick',{'arm':'right'}),('place',{'arm':'right','receptacle_id':'table'})],
              'transfer':[('pick',{'arm':'left'}),('handover',{'from_arm':'left','to_arm':'right'}),('place',{'arm':'right','receptacle_id':'table'})]}
    sequence_checks=[]
    for api in apis:
        required=expected[api['case']];matched=0
        for turn in api['turns']:
            for event in turn['trace']:
                if matched==len(required):break
                tool,args=required[matched];detail=event['detail']
                if event['kind']=='tool' and event['name']==tool and detail.get('result',{}).get('execution_id') and all(detail.get('arguments',{}).get(k)==v for k,v in args.items()):matched+=1
        sequence_checks.append({'case':api['case'],'required_sequence_matched':matched==len(required),
            'source':'archived tool requests plus native stage receipts; no new motion'})
    assert all(row['matches_current_sources'] for row in signature)
    assert all(check['required_sequence_matched'] for check in sequence_checks)
    assert 'Ran 142 tests' in (OUT/'checks.log').read_text() and (OUT/'checks.log').read_text().rstrip().endswith('OK')
    audit={'status':'verified_progress','goal_status':'active','checks_passed':142,
        'actual_api_calls':sum(a['actual_api_calls'] for a in apis),'required_sequence_checks':sequence_checks,
        'precision_cases':{'passed':sum(r['status']=='passed' for r in rows),'total':len(rows)},
        'variation_cases':{'seed':2718,'passed':sum(r['status']=='passed' for r in variation),'total':len(variation)},
        'source_signatures':signature,'scope':'known upright bottle, simulator oracle; development checks, not an independent benchmark',
        'physical_calibration_verified':False,'transfer_ready':False,'hardware_access':False}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    names={'right-1cm':'右手前移 1 cm','right-8cm':'右手前移 8 cm','left-transfer-6cm':'左手 → 右手 → 前移 6 cm',
           'right-left-4cm':'右手横移 4 cm','right-back-4cm':'右手后移 4 cm'}
    table=''.join(f'<tr><td>{names[r["case"]["id"]]}</td><td>{r["steps"][-1]["feedback"]["position_error_m"]*1000:.2f} mm</td><td>{r["evaluation"]["placement"]["tolerance_m"]*1000:.1f} mm</td><td>{r["sim_seconds"]:.2f} s</td><td><a href="final/{r["case"]["id"]}.json">原生回执 ↗</a></td></tr>' for r in rows)
    api_table=''.join(f'<tr><td>{"前移 1 cm" if a["case"]=="small" else "左取 → 右接 → 前移 6 cm"}</td><td>{a["actual_api_calls"]}</td><td>{a["plan"]["goal_assessment"]["position_error_m"]*1000:.2f} mm</td><td>{a["plan"]["verdict"]}</td><td><a href="api-{a["case"]}.json">调用与阶段 ↗</a></td></tr>' for a in apis)
    cards=''.join(f'<figure><a href="final/{r["case"]["id"]}.jpg"><img loading="lazy" src="final/{r["case"]["id"]}.jpg" alt="{names[r["case"]["id"]]}完成后的原生仿真画面"></a><figcaption>{names[r["case"]["id"]]}</figcaption></figure>' for r in rows)
    errors=''.join(f'<tr><td>{esc(r["case"]["id"])} · 初态 {r["case"]["episode"]}</td><td>{"完成" if r["status"]=="passed" else "未完成"}</td><td>{"—" if r["status"]=="passed" else esc(r["failure"]["code"])}</td><td><a href="variation/{r["case"]["id"]}.json">详情 ↗</a></td></tr>' for r in variation)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 物体精度验收</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f8f6f1;color:#302f2a;font:15px/1.75 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:38px 28px 60px}a{color:#926749;text-underline-offset:4px}header{display:flex;justify-content:space-between;gap:18px;align-items:center;border-bottom:1px solid #ded8cd;padding-bottom:18px}.eyebrow{letter-spacing:2px;color:#8b8171;font-size:12px}h1{font-size:30px;letter-spacing:-.5px;margin:13px 0 9px}h2{font-size:19px;margin:0 0 12px}p{margin:9px 0;color:#696255}section{border:1px solid #e5dfd4;border-radius:16px;padding:24px;background:#fffefa;margin:22px 0}.pill{background:#edf1e8;border:1px solid #d6dfcd;border-radius:24px;padding:4px 13px;color:#536547;font-size:13px;white-space:nowrap}.stats{display:flex;gap:34px;flex-wrap:wrap;margin:24px 0}.stats b{display:block;font-size:28px;line-height:1.4}.stats span{color:#827866;font-size:13px}table{width:100%;border-collapse:collapse;text-align:left;font-size:14px}th{font-weight:500;color:#8a7c69}td,th{padding:12px 9px;border-bottom:1px solid #eae4da}.scroll{overflow:auto}img{max-width:100%;display:block;border-radius:8px}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}figcaption{color:#857a68;font-size:13px;margin:6px 0}details{margin:18px 0}summary{cursor:pointer;color:#77634b}.small{font-size:13px}.sources{display:flex;gap:16px;flex-wrap:wrap}@media(max-width:700px){main{padding:20px 14px}section{padding:18px}.gallery{grid-template-columns:1fr}td,th{white-space:nowrap}header{align-items:flex-start}h1{font-size:24px}}
    </style><main><header><div><div class="eyebrow">OPENARM LAB / 2026.10.08</div><h1>把物体放准，再判断完成</h1><p>验证瓶子本体，而不是仅看机械臂末端是否到位。</p></div><span class="pill">有实测进展 · 目标继续</span></header>
    <div class="stats"><div><b>142</b><span>回归检查通过</span></div><div><b>26</b><span>真实模型 API 请求</span></div><div><b>5 / 5</b><span>固定开发动作验收</span></div><div><b>5 / 6</b><span>新位置 / 初态开发检查</span></div></div>
    <section><h2>真实 API 自主调用</h2><p>使用已配置的 deepseek-flash。每项试验只在用户目标开始前明确重置一次，任务中不重置、不改目标。阶段由模型提交；原生物理验收与模型判断分开记录。</p><div class="scroll"><table><tr><th>用户任务</th><th>API 次数</th><th>最终物体 XY 误差</th><th>模型判断</th><th>证据</th></tr>'''+api_table+'''</table></div><p class="small">1 cm 位移允许 2.5 mm 误差；原地不动不能通过。左 → 右 → 放置耗时 35.97 s，保持原有 40 s 预算。均确认支撑、脱离双手、连续稳定 ≥1 s。</p></section>
    <section><h2>控制与验收的改动</h2><p>在瓶子离开桌面时按 30 Hz 控制步修正实际物体误差，每一步重新求解与检查携物碰撞。松手前确认物体位置，释放后检查稳定支撑和撤离。所有运动通过执行器与原生接触发生。</p><p>目标锚点固定在用户任务开始，交接和重规划不能改变它。桌面误差上限收紧到 10 mm，小位移按位移的四分之一进一步收紧；篮内验收与原固定演示保留。</p><div class="scroll"><table><tr><th>原生开发动作</th><th>最终 XY 误差</th><th>允许误差</th><th>仿真耗时</th><th>证据</th></tr>'''+table+'''</table></div></section>
    <section><h2>页面实际验收</h2><p>保留左侧对话、右上场景、右下模型阶段。完成后显示瓶子的实际误差与允许误差。</p><a href="platform.jpg"><img src="platform.jpg" alt="真实 API 左手到右手放置，四阶段完成，误差2.9毫米"></a></section>
    <section><h2>精炼记忆与实验记录分开</h2><p>默认只加载不含旧目标坐标的方法索引，按需展开证据。两次右手桌面放置合并为一条方法，左右手抓取和交接方向分别保留。新增夹爪开合方法与放置、撤离、后端边界的失败教训。</p><p>规划器、视角、记录和诊断代码更新不使物理契约一致的方法失效；几何、控制、精度、相机或权重契约变化仍会隔离旧记录。历史支持次数不代表当前规划版本的成功率。</p><details><summary>查看实际记忆卡片</summary><a href="memory.jpg"><img loading="lazy" src="memory.jpg" alt="五条合并后的偏好与方法记忆，无原始记忆切换"></a></details></section>
    <section><h2>仍有一个失败，迁移尚未验证</h2><p>种子 2718 的六组初态/瓶位/目标检查完成五组。最后一组左手抓起后，放置中的逆解失败；场景仍有效，失败回执保留。这些是已知直立瓶子的工程开发检查，不能推导开放物体、视觉定位或真机成功率。</p><details><summary>六组结果与失败证据</summary><div class="scroll"><table><tr><th>场景</th><th>结果</th><th>失败原因</th><th>证据</th></tr>'''+errors+'''</table></div></details><p>独立基座/TCP、三路相机内外参与时间同步、夹爪传动和力、物体动力学、采集时控制器版本与内部状态仍待核验。软件 FK 或此次放置误差不是物理标定精度。cuRobo 仍只接空手运动；兼容的 OpenArm π0.5 未配置，ACT 尚未通过当前完整任务。</p></section>
    <details><summary>原生完成画面</summary><div class="gallery">'''+cards+'''</div></details>
    <details><summary>开发过程与完整审计</summary><p>收紧误差后，早期整段末端修正方法失败或超时，已全部保留：<a href="first/summary.json">首轮</a>、<a href="above-table/summary.json">离桌修正</a>、<a href="free-release/summary.json">离桌释放（3组后源版本变化中止）</a>、<a href="steady-release/summary.json">固定姿态释放</a>、<a href="object-servo/summary.json">逐控制步修正</a>、<a href="final/summary.json">当前版本复核</a>。</p><p><a href="checks.log">142项检查日志</a> · <a href="audit.json">源码与语义审计</a> · <a href="variation/protocol.json">位置检查协议</a>。只整理本轮离线记录目录，原始轨迹未改；线上 API 回执地址保留。</p></details>
    <footer><p class="small">设计参考，未声称完整复现或达到论文指标：</p><div class="sources"><a href="https://lianegalanti.github.io/Pigey/">Pigey · 观察 / 调用 / 验证 / 恢复</a><a href="https://code.claude.com/docs/en/memory">Claude Code · 小索引与按需细节</a><a href="https://arxiv.org/abs/2607.18060">RoboHarness · 能力与记忆编排</a><a href="/">返回平台 ↗</a></div></footer></main></html>'''
    (OUT/'index.html').write_text(page)
    print(json.dumps({'status':audit['status'],'sources_checked':len(signature),'actual_API_requests':audit['actual_api_calls'],'report':str((OUT/'index.html').relative_to(ROOT))}))

if __name__=='__main__':main()
