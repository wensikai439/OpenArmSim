"""Render retained relay evidence and memory QA without issuing commands."""
import html
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261009-handover'


def read(path): return json.loads((OUT / path).read_text())
def esc(value): return html.escape(str(value))
def link(path, label): return f'<a href="{esc(path)}">{esc(label)}</a>'


def main():
    current = read('try-8/results.json')['cases']; api = read('api-relay/result.json'); ui = read('workbench-2.json')
    full = read('api-full-1/result.json')
    ui_full = read('workbench-full/result.json') if (OUT / 'workbench-full/result.json').exists() else None
    assert all(case['passed'] for case in current) and api['passed'] and ui['passed'] and full['passed']
    labels = ['长方体 · 左→右', '圆柱 · 右→左']
    drift = [case['hold']['arms'][case['case']['receiver']]['max_window_relative_drift_m'] * 1000 for case in current]
    errors = [case['steps'][-1]['feedback']['position_error_m'] * 1000 for case in current]
    with tempfile.TemporaryDirectory(prefix='openarm-relay-plot-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False,
            'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout='constrained', facecolor='#fffefa')
        bars = axes[0].bar(labels, drift, color=['#819488', '#b49378'], width=.55)
        axes[0].bar_label(bars, fmt='%.3f', padding=4)
        axes[0].axhline(1, color='#ae7862', linestyle='--', linewidth=1, label='原有 1 mm 门槛')
        axes[0].set(ylim=(0, 1.15), ylabel='最差 2 秒窗口相对夹爪漂移 / mm', title='当前新版：实际交接后持物 10 秒')
        axes[0].legend(frameon=False, fontsize=8)
        bars = axes[1].bar(labels, errors, color=['#819488', '#b49378'], width=.55)
        axes[1].bar_label(bars, fmt='%.3f', padding=4)
        axes[1].set(ylim=(0, 1.5), ylabel='原目标放置误差 / mm', title='当前新版：交接 → cuRobo → 原目标放置')
        for axis in axes:
            axis.set_facecolor('#fffefa'); axis.grid(axis='y', color='#e6e0d5', linewidth=.6); axis.set_axisbelow(True)
        fig.savefig(OUT / 'results.png', dpi=180); plt.close(fig)
    trials = []
    for folder in sorted(OUT.glob('try-*')):
        for case in read(folder.name + '/results.json')['cases']:
            relay = case['checks'].get('verified_directed_handover'); hold = case['checks'].get('receiver_10s_hold')
            place = case['checks'].get('downstream_original_object_placement')
            error = (case.get('error') or {}).get('code', '—')
            trials.append(f'<tr><td>{folder.name}</td><td>{esc(case["case"]["name"])}</td>'
                f'<td>{"通过" if relay else "未通过"}</td><td>{"通过" if hold else "未通过"}</td>'
                f'<td>{"未测试" if place is None else "通过" if place else "未通过"}</td><td>{esc(error)}</td>'
                f'<td>{link(folder.name+"/"+case["case"]["name"]+"/result.json", "回执")}</td></tr>')
    gallery = ''.join(f'<a href="try-8/{esc(case["case"]["name"])}/{frame}.jpg" target="_blank" rel="noopener">'
        f'<img src="try-8/{esc(case["case"]["name"])}/{frame}.jpg" alt="{esc(labels[i])} · {label}">'
        f'<span>{esc(labels[i])} · {label}</span></a>' for i, case in enumerate(current)
        for frame, label in [('start', '起始'), ('transferred', '交接确认'), ('final', '原目标放置')])
    style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261009-scene/index.html').read_text(), re.S)[1]
    style += '.gallery{grid-template-columns:repeat(3,1fr)}table{display:block;overflow:auto}.stats b{font-size:26px}details{margin:18px 0} @media(max-width:650px){.gallery{grid-template-columns:1fr 1fr}}'
    memory = (ui_full or ui)['memory']['knowledge']['lessons']
    memory_rows = ''.join(f'<tr><td>{esc(item["title"])}</td><td>{"失败教训" if item["outcome"]!="success" else "验证方法"}</td>'
        f'<td>{item["repeat_count"]}</td><td>{esc(item["summary"])}</td></tr>' for item in memory)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 交接与记忆</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.09</div><h1>交接从固定高度，走向连接约束</h1><p>对象几何、供物手朝向、接收抓取、原生接触与非目标物体净空。</p></div><a class="pill" href="http://127.0.0.1:8767/">多物体工作台 ↗</a></header>
<div class="stats"><div><b>2 条完整路线</b><span>交接 · 持稳 · GPU · 放置</span></div><div><b>{full['actual_api_calls']} 次 API</b><span>模型实际组合四个运动工具</span></div><div><b>31 项检查</b><span>相关契约与回归通过</span></div><div><b>未标定</b><span>不构成任意动作或真实迁移</span></div></div>
<section><h2>这轮落地的变化</h2><p>接收点由实际长方体／圆柱外形与夹爪开口生成，不复用瓶子的固定交接高度。供物手世界朝向也参与搜索，预抓取、抓取和双臂路径必须连接。双手持续接触后才释放供物手；撤离后要求接收手独立持物至少 1 秒。圆柱的交接前抓取考虑接收手需要的高度空间。</p>
<p>若原计划包含桌面放置，先用 scratch MjData 检查供物手撤离、接收姿态与原放置目标的兼容性；它只是预测，后续实际放置仍要单独通过。新增 12 mm 的非目标物体路径净空约束与 3 mm 的非目标物体位移停止门槛；这些是开发规划／执行约束，不是实物标定精度。机器人、物体、控制器、接触与原放置／夹持评分参数没有放宽。原瓶子演示与主页面保留。</p>
<div class="flow"><div>几何抓取候选<span>对象 ID、开口、高度和朝向</span></div><div>连接双臂阶段<span>供物朝向、预抓取、接近路径</span></div><div>原生接触交接<span>接收确认、供物释放与撤离</span></div><div>独立证据<span>持物漂移、非目标物体与原目标</span></div></div></section>
<section><h2>当前完整组合验收</h2><img class="chart" src="results.png" alt="当前交接持稳与原目标放置误差"><p>当前 12 mm 整段净空约束下，两条完整原生路线分别用时 {current[0]["sim_time"]:.3f} s、{current[1]["sim_time"]:.3f} s，均计入本机底层规划等待，实际交接后持物 10 秒，再由真实 cuRobo 携物运输、TAMP 放在锁定的原桌面目标。对应落点误差 {errors[0]:.3f}／{errors[1]:.3f} mm，非目标物体检查均通过。圆柱用时接近原 120 秒预算，不能据此承诺加入不同 API 等待后也一定完成。该原生脚本不是 LLM 成绩。</p><div class="gallery">{gallery}</div></section>
<section><h2>模型自己组合工具</h2><p>新增只读“计算运输位姿”：用当前所选物体与实际 TCP 的相对关系，将锁定的物体目标换算成末端位姿。只接受单手实际持物，绑定本任务、版本、物体、原目标与当前抓取关系；旧物体、旧 TCP、换手、改目标或过期状态会在派发前被拒。它只生成假设，模型仍须显式调用运动、读取原生结果、再放置。有效交接回执只提供兼容朝向提示，旧世界位姿不会复用。</p><p>独立保存的模型 API 使用 {full['actual_api_calls']} 次请求完成长方体左抓→右接→新接口解析→真实 cuRobo→原目标放置，{full['terminal']['observation']['sim_time']:.3f} s，误差 {full['receipts'][-1]['feedback']['position_error_m']*1000:.3f} mm，5/5 自定阶段通过。原 120 秒 measured_delay、原目标、原生接触、真实 GPU 工作器与所有绑定检查通过，没有任务中重置或替换后端。该 API 版本在下一步记忆按物体筛选之前，源码独立冻结。</p><div class="links">{link('api-full-1/result.json','完整真实 API 回执')}{link('api-full-1/protocol.json','预声明协议与源码')}{link('transport-checks.log','31 项相关检查')}</div>
<p>实际页面完整任务：{'已通过' if ui_full and ui_full['passed'] else '未通过或尚未完成'}。{'模型使用 '+str(ui_full['actual_api_calls'])+' 次 API，'+format(ui_full['terminal']['observation']['sim_time'],'.3f')+' s；全部验收以原始回执为准。' if ui_full else '记录将在同一预声明任务结束后只读收集。'}</p><div class="links">{link('workbench-full/result.json','页面完整任务记录')}{link('workbench-full/protocol.json','页面协议与源码')}</div>{'<a href="transport.png"><img class="chart" src="transport.png" alt="真实模型交接运输放置界面"></a>' if (OUT/'transport.png').exists() else ''}</section>
<section><h2>所有原生尝试</h2><details><summary>展开不同版本的结果与失败回执</summary><p>每版预先声明的案例和源码留在对应目录。初始失败没有删除或改判；try-7 只验收单独交接与持稳，try-8 才是当前整段净空约束下的完整复验。</p><table><thead><tr><th>版本</th><th>声明案例</th><th>交接</th><th>10秒持稳</th><th>原目标放置</th><th>停止原因</th><th>证据</th></tr></thead><tbody>{''.join(trials)}</tbody></table></details></section>
<section><h2>真实模型与页面</h2><p>独立 API 场景实际完成长方体左手抓起→右手接收，{api["actual_api_calls"]} 次调用。页面用普通中文提出圆柱右手抓起→左手接收→保持夹持，模型自行制定 3 个阶段并选工具，{ui["actual_api_calls"]} 次调用，{ui["terminal"]["observation"]["sim_time"]:.3f} s。对象、方向、执行绑定、原生独立持物、原 120 秒 measured_delay 计时与非目标物体检查均通过。这是已知仿真几何工具验收，不是视觉抓取泛化率。</p><div class="links">{link('api-relay/result.json','独立 API 回执')}{link('workbench-2.json','实际页面回执')}{link('workbench.json','此前页面失败')}{link('final-regression.log','24 项相关检查')}</div><a href="workbench.png"><img class="chart" src="workbench.png" alt="实际 API 交接完成页面"></a>
<details><summary>保留采集故障与停止证据</summary><p>独立 API 已完成动作和复查，但原采集器末尾读取不存在的 snapshot.manifest，发生 KeyError。原日志与进度留存；使用同轮 API 轨迹、原生 summary 和 manifest 做只读重评，没有重跑 API 或机器人动作。此前页面测试在碰动旁边瓶子约 4.57 mm 时停止；该失败没有计入成功。之后以新代码版本另开预先声明的同目标测试，未在失败任务内重置、加时或改目标。</p><div class="links">{link('api-relay.log','原采集器日志')}{link('api-relay/progress.json','原始 API 进度')}{link('workbench-failed.png','位移停止页面')}</div></details></section>
<section><h2>记忆：短索引，明确条件，按需证据</h2><p>物体、空手／持物、机械臂与交接方向保留为结构化适用条件。普通抓取与交接前抓取分开命名；同条件的重复记录合并计数，重复操作规则只显示一次。当前模型默认索引先按明确选择的本任务对象过滤，再排序取最多 8 条，索引与公共规则总计最多 8192 UTF-8 字节。近期无关物体记录不会挤掉旧的相关方法；详细经验按 ID 读取。忘记方法后不会由旧回执自动恢复，但原始实验档案保留。</p><p>非目标物体被碰动被提炼为关键失败教训；场景已判无效时，Harness 在派发新动作前拒绝并引导提交失败。原始轨迹和调用日志继续作为实验档案，默认记忆与模型索引没有原始回执、相机链接或旧坐标。以下是用户精炼方法面板，允许查看多物体方法；它与模型只取当前对象的短索引分别呈现。</p><table><thead><tr><th>方法／教训</th><th>类别</th><th>支持记录</th><th>精炼内容</th></tr></thead><tbody>{memory_rows}</tbody></table><a href="memory-transport.png"><img class="chart" src="memory-transport.png" alt="精炼记忆面板"></a></section>
<section><h2>参考与剩余边界</h2><p>参考 <a href="https://lianegalanti.github.io/Pigey/">Pigey</a> 的观测、工具、复核闭环，<a href="https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/connecting_stages.html">MoveIt Task Constructor</a> 的连接阶段思想，以及 <a href="https://code.claude.com/docs/en/memory">Claude Code</a> 的精简索引和主题按需读取。这里是原生 MuJoCo 工程实现，没有声称移植其完整系统或复现其基准。</p><p>当前新增形状为合成开发资产，未独立实测质量、惯量、摩擦和柔顺。机械臂／相机物理标定、MIT 控制响应、真实延迟与传感器反馈尚不完整；“任意物体、任意动作”与 real→sim→real 均未验收。ACT／π0.5 的兼容闭环与迁移仍待完成；本轮不连接真机，SSH 公司电脑零操作。</p><div class="links">{link('audit-transport.json','本轮源代码与模型审计')}{link('memory-transport.json','当前对象短记忆索引审计')}{link('cleanup.json','此前仅清理自建重复进度')}</div></section>
</main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print('Report rendered; no API, robot or remote commands')


if __name__ == '__main__': main()
