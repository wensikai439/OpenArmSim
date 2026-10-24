"""Publish diagnosis, paired outcomes and the honest deployment boundary."""
import json
from pathlib import Path
from statistics import mean

from .runtime import ROOT,dump


def publish(folder):
    rows=json.loads((folder/'results.json').read_text())
    assessment=json.loads((folder/'assessment.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    totals={m:{'passed':sum(v['passed'] for v in assessment['methods'][m].values()),
        'total':sum(v['total'] for v in assessment['methods'][m].values())} for m in ('baseline','precision')}
    metrics={}
    for m in ('baseline','precision'):
        successful=[r for r in rows if r['method']==m and r['task_success']]
        metrics[m]={'successful_case_count':len(successful),
            'success_only_mean_xy_error_mm':1000*mean(r['pose_error_m'] for r in successful) if successful else None,
            'success_only_max_xy_error_mm':1000*max(r['pose_error_m'] for r in successful) if successful else None,
            'success_only_mean_sim_seconds':mean(r['sim_seconds'] for r in successful) if successful else None}
    shared=assessment['paired_success_xy_error_m']
    paired={m:1000*mean(r[m] for r in shared) if shared else None for m in ('baseline','precision')}
    dump(folder/'metrics.json',{'methods':metrics,'paired_success_count':len(shared),
        'paired_success_mean_xy_error_mm':paired,'success_only_metrics_exclude_failed_trials':True})
    table='';failures=''
    for c in protocol['cases']:
        value=json.loads((ROOT/c['path']).read_text());seed=value['seed']
        pair={r['method']:r for r in rows if r['case']['seed']==seed}
        cells=''.join(f'<td><a class="{"ok" if pair[m]["task_success"] else "no"}" href="heldout/{m}/{seed}/result.json">{"通过" if pair[m]["task_success"] else "未完成"}</a></td>' for m in ('baseline','precision'))
        table+=f'<tr><th>{seed} · {"位置" if c["group"]=="position" else "组合压力"}</th>{cells}</tr>'
        if not pair['precision']['task_success']:
            failure=pair['precision'].get('native_failures') or pair['precision'].get('error') or pair['precision'].get('executions',[])[-1].get('error')
            failures+=f'<li>{seed}：{failure}</li>'
    pictures=''
    for row in [r for r in rows if r['method']=='precision' and r['task_success']][:4]:
        p=Path(row['trial']).relative_to(folder.relative_to(ROOT))/'media/final-overview.jpg'
        pictures+=f'<figure><img src="{p}" alt="案例{row["case"]["seed"]}的最终原生状态"><figcaption>{row["case"]["seed"]} · {1000*row["pose_error_m"]:.2f} mm</figcaption></figure>'
    a,b=totals['baseline'],totals['precision']
    group_rows=''.join(f'<tr><th>{"新位置" if g=="position" else "组合压力"}</th><td>{assessment["methods"]["baseline"][g]["passed"]}/{assessment["methods"]["baseline"][g]["total"]}</td><td>{assessment["methods"]["precision"][g]["passed"]}/{assessment["methods"]["precision"][g]["total"]}</td></tr>' for g in ('position','combined'))
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>抓瓶精度 · 本地接入包</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f6f4ef;color:#302e2a;font:15px/1.75 system-ui,sans-serif}}main{{max-width:1040px;margin:auto;padding:38px 28px 65px}}nav{{color:#787367;font-size:13px;border-bottom:1px solid #e6e0d5;padding-bottom:18px}}h1{{font-size:34px;margin:38px 0 12px}}h2{{font-size:20px;margin:0 0 15px}}p{{margin:12px 0}}section,.card{{border:1px solid #e6e0d5;border-radius:16px;background:#fffefa;padding:24px;margin-top:20px}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card b{{display:block;font-size:36px;font-weight:550;margin-top:10px}}.card span{{color:#787367}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:12px 15px;border-top:1px solid #e6e0d5}}thead th{{color:#787367;font-weight:450;border:0}}tbody th{{font-weight:450}}a{{color:inherit;text-underline-offset:4px}}.ok,.no{{display:inline-block;padding:2px 12px;border-radius:18px;text-decoration:none}}.ok{{background:#edf2e9;color:#50684b}}.no{{background:#f4e9e0;color:#966347}}.gallery{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}figure{{margin:0}}img{{width:100%;border-radius:9px}}figcaption{{font-size:13px;color:#787367}}details{{margin-top:15px}}summary{{cursor:pointer}}code{{background:#f0ece4;padding:3px 6px;border-radius:4px}}li{{margin:8px 0}}@media(max-width:700px){{main{{padding:20px 16px}}.cards{{grid-template-columns:1fr}}.gallery{{grid-template-columns:1fr 1fr}}section{{overflow:auto}}}}</style>
<main><nav>OpenArm Lab / 精度与迁移准备</nav><h1>先修执行误差，再准备真机接入</h1><p>瓶体位姿分段运动、受检的供物手归位。当前评估仍是候选仿真模型，本轮所有工作均在本地完成。</p>
<div class="cards"><div class="card"><span>原恢复工具</span><b>{a['passed']} / {a['total']}</b></div><div class="card"><span>改进执行工具</span><b>{b['passed']} / {b['total']}</b></div></div>
<section><h2>差在哪里，改了什么</h2><ol><li><b>带载放置：</b>原关节插值没有沿途约束瓶体位姿，出现持续夹持中断。新版每段按瓶体目标和最新手—瓶关系重新规划。</li><li><b>恢复后的撤离：</b>原恢复分支使用旧撤离路径，与正常现代交接不一致。新版复用现代流程的受检空手归位，接收手夹持保持验证。</li><li><b>仅降速不够：</b>降速和刚性轴向撤离开发候选仍失败；最终改进通过五个开发案例。这些案例已用于调试，不再当新泛化证据。</li></ol><p>技能分支程序保持相同，本轮提升来自底层工具工程，没有追加模型 API 调用或训练模型权重。</p></section>
<section><h2>新条件成对测试</h2><table><thead><tr><th>条件</th><th>原工具</th><th>改进工具</th></tr></thead><tbody>{group_rows}</tbody></table><p>{len(protocol['positions_m'])}个新瓶位，分别测试标称与组合压力，共{len(protocol['cases'])}条件、{len(rows)}次原生测试。组合压力改变篮位、瓶子朝向、质量、摩擦，并包含一次命令/感知故障；这些范围不是实测真机分布。</p><p>修复 {len(assessment['improvements'])} 例；退步 {len(assessment['regressions'])} 例。所有失败均保留。原120秒仿真预算、64次适配器调用预算、碰撞/持瓶/释放/稳定验收保持。</p><p>先前候选的位置测试8/8、组合压力2/8，出现退步，未采用。它暴露了“目标已更新、碰撞预测仍用旧抓持关系”的错误；修正后另起本页的新随机条件，旧结果未删除。</p></section>
<section><h2>准确性与耗时</h2><p>两者都成功的 {len(shared)} 个案例，平均最终 XY 误差：原工具 {paired['baseline'] or 0:.2f} mm，改进工具 {paired['precision'] or 0:.2f} mm。该比较排除了失败案例，不能代替成功率。</p><p>新工具全部成功案例的平均XY误差 {metrics["precision"]["success_only_mean_xy_error_mm"] or 0:.2f} mm，最大 {metrics["precision"]["success_only_max_xy_error_mm"] or 0:.2f} mm；不能与旧工具排除失败后的均值直接比较。</p><p>改进工具成功案例的平均仿真用时 {metrics['precision']['success_only_mean_sim_seconds'] or 0:.1f} 秒。分段运动更慢，存在120秒预算压力；规划期间仿真暂停，不能当作真实实时运行速度。</p><div class="gallery">{pictures}</div></section>
<section><h2>本地接入包</h2><p><code>./run real preflight</code> 检查缺项；<code>./run real replay</code> 读取已有真实关节记录；<code>./run real demo</code> 用合成快照做影子规划，实际运动命令与仿真物理步数均为0。</p><p>没有三维感知服务时，默认模板明确拒绝规划；包内没有真机执行入口。本轮未连接、上传或修改真机电脑。接入说明随包提供。</p></section>
<section><h2>保留的失败与证据</h2><ul>{failures or '<li>本次改进工具全部通过；样本量有限，不保证任意位置或真机成功。</li>'}</ul><details><summary>逐案例结果</summary><table><thead><tr><th>案例</th><th>原工具</th><th>改进工具</th></tr></thead><tbody>{table}</tbody></table></details><p><a href="protocol.json">冻结协议</a> · <a href="assessment.json">完整统计</a> · <a href="metrics.json">误差/耗时</a> · <a href="check.log">回归检查</a></p><p>输入为仿真几何真值。没有验证任意物体、倒瓶、视觉定位、实时闭环或 sim-to-real；新版没有替换旧固定 TAMP 演示。</p></section></main></html>'''
    (folder/'index.html').write_text(html)


if __name__=='__main__':
    import sys
    publish(Path(sys.argv[1]).resolve())
