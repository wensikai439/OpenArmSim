"""Render retained multi-object evidence; never issue a robot or model command."""
import html
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261009-objects'


def read(name): return json.loads((OUT / name).read_text())
def esc(value): return html.escape(str(value))
def link(path, label): return f'<a href="{esc(path)}">{esc(label)}</a>'


def main():
    native = read('results.json'); api = read('api.json'); ui = read('workbench.json')
    verified = read('workbench-checked.json'); legacy = read('legacy.json')
    assert native['complete'] and all(c['passed'] for c in native['cases'])
    assert api['passed'] and verified['passed'] and legacy['passed']
    cases = native['cases']
    errors = [c['steps'][-1]['feedback']['position_error_m'] * 1000 for c in cases]
    drifts = [c['hold']['arms'][c['case']['arm']]['max_window_relative_drift_m'] * 1000 for c in cases]
    api_error = api['terminal']['agent']['plan']['goal_assessment']['position_error_m'] * 1000
    ui_error = ui['terminal']['agent']['plan']['goal_assessment']['position_error_m'] * 1000
    labels = ['长方体 · 左手', '圆柱 · 右手', '长方体 · 改初态', '圆柱 · 改初态']
    with tempfile.TemporaryDirectory(prefix='openarm-objects-plot-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False,
            'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained', facecolor='#fffefa')
        bars = axes[0].bar(range(4), errors, color=['#7a8f81', '#b29378'] * 2, width=.6)
        axes[0].bar_label(bars, fmt='%.2f', padding=3)
        axes[0].set(xticks=range(4), xticklabels=labels, ylim=(0, 2.3), ylabel='放置误差 / mm',
            title='四个预先声明的原生开发案例')
        bars = axes[1].bar(range(4), drifts, color='#7a8f81', width=.6)
        axes[1].bar_label(bars, fmt='%.3f', padding=3)
        axes[1].axhline(1, color='#b29378', linestyle='--', linewidth=1, label='原有 1 mm 门槛')
        axes[1].set(xticks=range(4), xticklabels=labels, ylim=(0, 1.18), ylabel='最差 2 秒窗口相对漂移 / mm',
            title='每次实际持物 10 秒，检查相对夹爪漂移')
        axes[1].legend(frameon=False, fontsize=8)
        for ax in axes:
            ax.set_facecolor('#fffefa'); ax.grid(axis='y', color='#e6e0d5', linewidth=.6); ax.set_axisbelow(True)
        fig.savefig(OUT / 'results.png', dpi=180); plt.close(fig)

    rows = ''.join(f'<tr><td>{esc(labels[i])}</td><td>{esc(c["case"]["delta_m"])} m · {c["case"]["yaw_rad"]:+.2f} rad</td>'
        f'<td>{errors[i]:.3f} mm</td><td>{drifts[i]:.3f} mm</td><td>{link(c["case"]["name"]+"/result.json","原生回执")}</td></tr>'
        for i, c in enumerate(cases))
    gallery = ''.join(f'<a href="{esc(c["case"]["name"])}/{frame}.jpg" target="_blank" rel="noopener">'
        f'<img src="{esc(c["case"]["name"])}/{frame}.jpg" alt="{esc(labels[i])} · {label}">'
        f'<span>{esc(labels[i])} · {label}</span></a>' for i,c in enumerate(cases[:2])
        for frame,label in [('start','初态'),('held','真实夹持'),('final','松手放置')])
    ui_frames = ''.join(f'<a href="{esc(f["url"])}" target="_blank" rel="noopener"><img src="{esc(f["url"])}" alt="{esc(f["label"])}">'
        f'<span>{esc(f["label"])} · {f["sim_time"]:.1f}s</span></a>' for f in ui['terminal']['agent']['goal_keyframes'])
    style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261009-scene/index.html').read_text(), re.S)[1]
    style += '.gallery{grid-template-columns:repeat(3,1fr)}.tag{color:#82776a;font-size:12px}.stats b{font-size:27px}table{display:block;overflow:auto} @media(max-width:650px){.gallery{grid-template-columns:1fr 1fr}}'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 多物体抓取与记忆</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.09</div><h1>从瓶子专用动作，走向对象契约</h1><p>长方体与圆柱的实际抓取、持稳、放置，以及真实模型工具调用。</p></div><a class="pill" href="http://127.0.0.1:8767/">多物体工作台 ↗</a></header>
<div class="stats"><div><b>4 个案例</b><span>原生抓取 · 持物 10 秒 · 放置</span></div><div><b>2 次任务</b><span>真实 API，各自抓取与放置</span></div><div><b>30 项</b><span>相关边界与回归检查通过</span></div><div><b>未标定</b><span>已知形状开发测试</span></div></div>
<section><h2>对象身份进入整个动作闭环</h2><p>原瓶子保持独立自由刚体，场景新增两个真正的自由关节对象：长方体 block 和圆柱 can。选择对象只更换任务身份与观测 epoch，不改 qpos、qvel、控制目标或时间。抓取候选从当前形状、朝向及夹爪开度生成，再连接预抓取、抓取与路径；接触、抬升、速度、携物预测和放置判据均绑定所选对象。实际执行仍走原生关节执行器与接触动力学。</p>
<div class="flow"><div>选择注册对象<span>原生 body / free joint / collision geoms</span></div><div>生成形状抓取<span>夹爪宽度、朝向、逆解与连接路径</span></div><div>原生接触执行<span>双指持续接触、抬升与持物漂移</span></div><div>锁定对象目标<span>完整外形、松手、支撑与连续稳定</span></div></div>
<div class="geometry"><div class="snapshot"><b>长方体 block</b><p>48 × 42 × 180 mm，0.20 kg。沿实际几何的面与对角方向、三层高度和四种俯仰生成候选。</p></div><div class="snapshot"><b>圆柱 can</b><p>直径 48 mm，高 180 mm，0.20 kg。24 个周向角、三层高度和四种俯仰生成候选。</p></div></div>
<p class="scope">新增形状是合成开发资产，不是真机扫描重建。原瓶子质量 0.55 kg；本轮新物体更轻，结果不能外推至所有负载。TAMP 使用仿真几何真值，不构成纯视觉抓取。</p>
<div class="links">{link('source/sim/objects.py','原生对象契约')}{link('source/sim/object_planner.py','形状抓取生成')}{link('source/workbench/harness/objects.py','工具与任务绑定')}{link('protocol.json','预声明协议')}</div></section>
<section><h2>先验证物理原语，再让模型调用</h2><p>四个声明案例均完成 pick → 保持原控制目标 10 秒 → place。改初态案例在任务开始前改变位置与 yaw；执行中不重置、不改物体位姿、不焊住物体。原有 2 秒滚动窗口 ≤1 mm／0.02 rad 的持稳门槛、10 mm 放置容差和 ≥1 秒支撑稳定判据保留。</p><img src="results.png" alt="四个原生案例的放置误差及持物漂移"><table><thead><tr><th>案例</th><th>初态变化</th><th>放置误差</th><th>最差持物漂移</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table>
<p class="provenance">原生四例使用 paused 规划计时，属于几何与接触诊断；下面两项真实模型任务使用 measured_delay，将首份计划后的模型及规划等待积分为保持原控制目标的物理时间。四例不是随机泛化成功率。未选物体的自然沉降仍有记录，不声称它们在物理上完全零位移。</p><div class="gallery">{gallery}</div></section>
<section><h2>真实 API 与页面自然语言，分别执行</h2><div class="geometry"><div class="snapshot"><b>独立 API · 左手长方体前放 8 cm</b><p>{api['actual_api_calls']} 次模型请求，恰好抓取与放置两次运动，3/3 阶段完成。</p><p>落点误差 {api_error:.3f} mm，仿真耗时 {api['terminal']['observation']['sim_time']:.2f} s。</p><p>{link('api.json','完整模型调用与原生回执')}</p></div><div class="snapshot"><b>页面任务 · 右手圆柱前放 6 cm</b><p>{ui['actual_api_calls']} 次模型请求，模型自行选择对象和制定阶段；恰好抓取与放置两次运动，3/3 完成。</p><p>落点误差 {ui_error:.3f} mm，仿真耗时 {ui['terminal']['observation']['sim_time']:.2f} s。</p><p>{link('workbench-checked.json','按原始目标复核')}</p></div></div><p>两项第一份计划均声明 120 秒仿真预算；执行中没有重置、加时、换对象或改变目标。物理结论来自对应对象和执行编号的原生回执，模型视觉判断单独保存。观测仍是固定的头部与左右腕三路相机。</p><div class="gallery">{ui_frames}</div>
<details><summary>检查器修正与历史元数据，全部保留</summary><p>页面模型将起始 can 位置 [0.503, −0.345] m 加前向 0.06 m，声明等价绝对目标 [0.563, −0.345] m。首版收集器只认 offset_xy_m 字段，判为未通过；原记录未覆盖。只读复评分由原始协议重新计算目标，并核对对象、锚点、执行编号、原生来源、机械臂、松手、稳定与原 10 mm 容差。沿用既有 1e−6 m 目标数值一致性精度，没有重跑动作或放宽物理门槛。三个对抗检查证明错目标、错锚点、错对象、额外动作和未绑定回执不能借缓存布尔值通过。</p><p>独立 API 的原生回执已正确记录 block；但其较早版 manifest.task 文本仍是旧瓶子描述。随后修复记录器，页面 can 的两个 manifest 均明确写入实际对象和自由关节。这是元数据修复，旧记录保持原样。</p><div class="links">{link('workbench.json','原始收集器未通过记录')}{link('workbench-verified.json','首次只读复评分')}{link('workbench-checked.json','字段级复核')}{link('workbench.protocol.json','页面原协议')}{link('api.protocol.json','独立API原协议')}{link('score-contracts.log','3项评分边界检查')}{link('source/api/harness/objects.py','API历史源码')}</div></details></section>
<section><h2>精炼记忆按对象归并，回执归入实验</h2><p>同一对象、技能、后端和兼容条件下的重复经验合为稳定方法卡；不同对象的方法保留各自范围。每条卡片说明验证来源、兼容记录数和边界，不把瓶子的经验自动推广成所有物体都可抓。当前任务进度是短期状态，成功阶段须绑定正确对象的原生回执。</p><p>新工作台现在只有两条圆柱方法：抓取与放置。默认没有原始经历、episode 或轨迹入口；完整证据通过实验链接按需查看。旧工作台已有偏好与方法仍保留，没有把历史原始记录重新灌入新索引。</p><div class="geometry"><a href="skills.png"><img src="skills.png" alt="技能按后端切换，场景与对象选择折叠显示"></a><a href="memory.png"><img src="memory.png" alt="仅有两条精炼圆柱方法，没有原始记忆入口"></a></div><div class="links">{link('workbench.png','实际任务页面')}{link('regression.log','27项相关检查')}{link('contracts.log','9项对象契约检查')}{link('audit.json','源码与运行状态审计')}</div></section>
<section><h2>固定演示保留，新的能力有明确边界</h2><p>没有新增对象的对照中，原固定演示与对象适配器使用相同编译模型，{legacy['commands']} 条实际指令的每步 qpos／qvel 和终点完全一致。两者同样在 goal_collision 停止；这里只证明兼容，不把旧演示记成整项通过。{link('legacy.json','对照证据')}。</p><p>新对象目前支持 TAMP 抓取与桌面放置；新形状交接、连续 relocate、cuRobo 携物表示和 ACT／π0.5 对象输入尚未验证，因此拒绝调用。cuRobo GPU 服务仍在线；当前对象适配器不支持时页面不显示为可用，也不以 TAMP 替代它。任意网格、球、横放物体和视觉自动定位尚未覆盖。</p>
<p>结构参考 <a href="https://lianegalanti.github.io/Pigey/">Pigey</a> 的观察、工具反馈与继续决策闭环，以及 <a href="https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html">MoveIt Task Constructor</a> 的目标生成、连接与分阶段执行。这里是设计借鉴，不是它们的完整复现或等价评测。</p>
<div class="limits"><p><b>下一步：把携物碰撞表示统一到对象契约，再核验新对象的 cuRobo 和双臂交接。</b>随后加入未知位姿、负载和障碍变化的独立留出测试，避免固定案例成为新的演示模板。</p><p>Real → Sim → Real 仍需底座、关节零位、TCP、相机与时序的独立标定，以及驱动响应、质量惯量、摩擦和负载辨识。当前接触候选 dim4／NoSlip3 未经实物标定；仿真成功与回放一致不能替代这些证据。</p></div>
<p class="small">开发入口：<code>./run objects</code> → <a href="http://127.0.0.1:8767/">8767</a>，前台 Ctrl+C 停止。主平台 8765 和原动作工作台 8766 的已停止场景保持；未新增定时启动。公司 SSH／真机操作：本轮均为 0。总目标继续进行。</p></section></main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print('object report rendered from retained evidence', flush=True)


if __name__ == '__main__': main()
