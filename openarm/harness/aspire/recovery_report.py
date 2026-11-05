"""A quiet standalone report; detailed feedback stays inside experiment files."""
import argparse
import json
from pathlib import Path

from .runtime import ROOT, dump


LABEL={'no_recovery':'不恢复','fixed':'固定规则恢复','condensed':'凝练 skill'}
GROUP={'none':'正常执行','right_close_loss':'右夹爪命令丢失',
    'stale_grasp':'感知高度偏差','receiver_close_loss':'接收手未夹住'}


def publish(folder):
    assessment=json.loads((folder/'assessment.json').read_text())
    results=json.loads((folder/'results.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    verifier=json.loads((folder/'verification.json').read_text())
    cards=''
    for method,label in LABEL.items():
        rows=assessment[method];passed=sum(v['passed'] for v in rows.values());total=sum(v['total'] for v in rows.values())
        cards+=f'<article class="card"><span>{label}</span><strong>{passed}<small> / {total}</small></strong><div class="bar"><i style="width:{100*passed/total:.1f}%"></i></div></article>'
    matrix=''
    for group,label in GROUP.items():
        cells=''.join(f'<td>{assessment[m][group]["passed"]} / {assessment[m][group]["total"]}</td>' for m in LABEL)
        matrix+=f'<tr><th>{label}</th>{cells}</tr>'
    trials=''
    for case in protocol['cases']:
        value=json.loads((ROOT/case['path']).read_text());seed=value['seed']
        pair={r['method']:r for r in results if r['case']['seed']==seed}
        cells=''
        for method in LABEL:
            r=pair[method];url=Path(r['trial']).relative_to(folder.relative_to(ROOT))/'result.json'
            cells+=f'<td><a class="pill {"yes" if r["task_success"] else "no"}" href="{url}">{"通过" if r["task_success"] else "未完成"}</a></td>'
        trials+=f'<tr><th>{seed} · {GROUP[case["group"]]}<small>({value["bottle_xy_m"][0]:.3f}, {value["bottle_xy_m"][1]:.3f})</small></th>{cells}</tr>'
    gallery=''
    for group,label in GROUP.items():
        row=next((r for r in results if r['method']=='condensed' and r['group']==group and r['task_success']),None)
        if row is None:continue
        base=ROOT/row['trial'];images=sorted((base/'media').glob('failure-*-overview.jpg'))+list((base/'media').glob('final-overview.jpg'))
        for path in images[:2]:
            rel=path.relative_to(folder);name='恢复前' if path.name.startswith('failure') else '结束状态'
            gallery+=f'<figure><button class="image" aria-label="放大{label}{name}"><img src="{rel}" alt="{label} · {name}" loading="lazy"></button><figcaption>{label} · {name}</figcaption></figure>'
    usage=[]
    for path in folder.rglob('usage.json'):
        r=json.loads(path.read_text());usage.append({'path':str(path.relative_to(folder)),
            'requests':r['requests'],'tokens':r['total_tokens'],'seconds':sum(v['seconds'] for v in r['usage']),
            'response_models':sorted({v.get('response_model','unknown') for v in r['usage']})})
    dump(folder/'api-summary.json',{'groups':usage,'requests':sum(v['requests'] for v in usage),
        'tokens':sum(v['tokens'] for v in usage),'api_wait_seconds':sum(v['seconds'] for v in usage)})
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>抓瓶 · 反馈恢复</title><style>
:root{{--ink:#302e2a;--muted:#7b786e;--line:#e6e1d8;--paper:#fffefb;--accent:#a76d50}}
*{{box-sizing:border-box}}body{{margin:0;background:#f6f4ef;color:var(--ink);font:15px/1.7 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 32px 70px}}nav{{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;border-bottom:1px solid var(--line);padding-bottom:18px}}a{{color:inherit;text-decoration-color:#c9b6a3;text-underline-offset:4px}}header{{padding:45px 0 30px}}h1{{font-size:36px;letter-spacing:-1px;margin:0 0 15px}}h2{{font-size:20px;margin:0 0 18px}}p{{margin:10px 0;max-width:900px}}.lead{{font-size:17px;color:#625b51}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.card,section{{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:24px}}.card>span{{color:var(--muted)}}strong{{display:block;font-size:38px;font-weight:550;margin:8px 0}}strong small{{font-size:20px;color:var(--muted);font-weight:400}}.bar{{height:4px;background:#eee9df;border-radius:3px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--accent)}}section{{margin-top:22px}}table{{width:100%;border-collapse:collapse;text-align:left}}th,td{{padding:13px 12px;border-top:1px solid var(--line)}}thead th{{border:0;color:var(--muted);font-weight:450}}tbody th{{font-weight:500}}th small{{display:block;color:var(--muted);font-weight:400}}.pill{{display:inline-block;border-radius:20px;padding:2px 12px;font-size:13px;text-decoration:none}}.yes{{color:#49624e;background:#edf1e9}}.no{{color:#98644c;background:#f5ece5}}.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}}.steps b{{display:block;margin-bottom:8px}}.steps p{{color:var(--muted);font-size:14px}}.gallery{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}}figure{{margin:0}}.image{{border:0;padding:0;background:#e8e8e1;border-radius:9px;overflow:hidden;width:100%;cursor:zoom-in}}img{{display:block;width:100%}}figcaption{{padding:7px 0;font-size:13px;color:var(--muted)}}details{{margin-top:18px}}summary{{cursor:pointer}}code{{font-size:13px;background:#f1ede5;border-radius:5px;padding:3px 6px}}dialog{{border:0;border-radius:14px;padding:12px;max-width:90vw;background:var(--paper)}}dialog::backdrop{{background:#191813b3}}dialog button{{float:right;border:1px solid var(--line);border-radius:6px;background:transparent;padding:6px 14px;margin-bottom:8px;cursor:pointer}}dialog img{{max-height:80vh;width:auto;max-width:85vw}}footer{{color:var(--muted);font-size:13px;margin-top:30px}}@media(max-width:700px){{main{{padding:22px 16px}}.cards,.steps{{grid-template-columns:1fr}}.gallery{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:28px}}section{{overflow:auto}}}}
</style><main><nav><span>OpenArm Lab / Skill Research</span><a href="/">返回工作台 ↗</a></nav>
<header><h1>抓瓶，从执行走向恢复</h1><p class="lead">按实测反馈决定补闭爪、退回检查点重抓、或保留供物手补夹。所有恢复都发生在同一物理回合。</p></header>
<div class="cards">{cards}</div>
<section><h2>相同条件，三种方法</h2><table><thead><tr><th>条件</th><th>不恢复</th><th>固定规则恢复</th><th>凝练 skill</th></tr></thead><tbody>{matrix}</tbody></table>
<p>冻结后对 3 个新瓶位 × 4 种条件做 36 次隔离原生测试；不是 12 个独立位置，也不是开放世界成功率。感知高度偏差幅度为 4.5 / 5.5 / 6.5 cm。固定规则获得相同工具和观测，不是弱化的对照。</p>
<p>凝练程序与固定规则的差异以本表为准；即使打平，也不能宣称 LLM 优于规则。候选抓姿、运动规划与接触判定的能力来自工程工具。</p></section>
<section><h2>恢复如何发生</h2><div class="steps"><div><b>01 · 闭爪没有执行</b><p>编码器仍接近张开端点，允许一次原位补闭爪。只有真实夹持成立才抬升。</p></div><div><b>02 · 闭爪却抓空</b><p>打开空夹爪，实际运动回已验证的接近点；新观测、新候选、新路径，不恢复世界快照。</p></div><div><b>03 · 接收手未夹住</b><p>右手继续持瓶；左手补夹，确认双手夹持后才松右手。未知碰撞、失稳和超时停止。</p></div></div></section>
<section><h2>原生关键帧</h2><div class="gallery">{gallery}</div></section>
<section><h2>复用与验收</h2><p>新 API 上下文仅凭技能说明和工具契约重建程序：原生测试通过 {verifier['passed']} / {verifier['total']}。这是技能说明的可复用性检查，不增加留出测试的样本数。</p><p>独立运行：<code>./run aspire recover --fault stale_grasp</code>。冻结程序运行不再调用 API；新技能尚未替换工作台的旧演示或旧宏工具。</p>
<p><a href="skill.md">技能说明</a> · <a href="candidate.py">程序</a> · <a href="protocol.json">冻结协议</a> · <a href="assessment.json">统计结果</a> · <a href="verification.json">重建验收</a></p></section>
<section><h2>证据边界</h2><p>这是一份带专家纠正的程序凝练实验，没有训练神经网络权重。模型在线修复在开发的三类故障中成功两类；接收手故障时误选动作，之后通过专家轨迹和静态审查纠正。所有失败稿和原始记录保留。</p><p>输入仍是仿真几何真值。感知偏差在首次失败后撤除，属于合成诊断，不能当作真实视觉重定位。规划 / API 等待期间仿真暂停；没有验证连续实时闭环、倒瓶重抓、任意物体或真机迁移。远程电脑和硬件均未操作。</p>
<details><summary>查看逐案例结果与内部证据</summary><table><thead><tr><th>案例</th><th>不恢复</th><th>固定规则</th><th>凝练 skill</th></tr></thead><tbody>{trials}</tbody></table><p><a href="results.json">完整原生结果与连续性审计</a> · <a href="api-summary.json">模型调用与等待时间</a> · <a href="check.log">回归检查</a>。首次评估曾因 CLI 文件并行修改触发源码保护，已中止并保存在 interrupted；同一冻结程序与同一组案例完整重测，未按结果筛选位置。</p></details></section>
<footer>OpenArm · 2026-10-24 · 候选仿真模型 · hardware_access = false · transfer_ready = false</footer></main>
<dialog id="viewer"><form method="dialog"><button>关闭</button></form><img alt="原生关键帧放大"></dialog><script>const d=document.querySelector('#viewer');document.querySelectorAll('.image').forEach(b=>b.onclick=()=>{{d.querySelector('img').src=b.querySelector('img').src;d.showModal()}});d.onclick=e=>{{if(e.target===d)d.close()}};</script></html>'''
    (folder/'skill.md').write_bytes((ROOT/'skills/bottle_recovery/SKILL.md').read_bytes())
    (folder/'index.html').write_text(content)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    f=a.folder.resolve()
    if not f.is_relative_to(ROOT/'runs'):p.error('output must stay inside runs/')
    publish(f)
