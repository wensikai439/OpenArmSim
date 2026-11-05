"""Rebuild the scene-reference report from immutable native/API/UI evidence."""
import html
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261009-scene'


def read(name): return json.loads((OUT / name).read_text())
def esc(value): return html.escape(str(value))
def link(path, title): return f'<a href="{esc(path)}">{esc(title)}</a>'


def main():
    native = read('native-timed.json')['cases']
    api = read('api.json'); browser = read('workbench-final.json'); failed = read('workbench.json')
    assert read('api-verified.json')['passed'] and read('workbench-verified.json')['passed']
    api_errors = [row['feedback']['position_error_m'] * 1000 for row in api['receipts']]
    ui_errors = [row['feedback']['position_error_m'] * 1000 for row in browser['receipts']]
    with tempfile.TemporaryDirectory(prefix='openarm-scene-plot-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False, 'font.size': 9,
            'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained', facecolor='#fffefa')
        batches = [('原生工具', [row['native_feedback']['position_error_m'] * 1000 for row in native]),
            ('独立 API', api_errors), ('页面 API', ui_errors)]
        for offset, color, index, label in [(-.17, '#7a8f81', 0, '右手 · 瓶子参考 · TAMP'),
            (.17, '#b29378', 1, '左手 · 篮子参考 · cuRobo')]:
            values = [row[1][index] for row in batches]
            bars = axes[0].bar(np.arange(3) + offset, values, width=.30, color=color, label=label)
            axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        axes[0].set(xticks=range(3), xticklabels=[row[0] for row in batches], ylim=(0, 1.85),
            ylabel='原生 FK 到声明目标的误差 / mm', title='同一目标协议，三层执行证据')
        axes[0].legend(frameon=False, fontsize=8)
        x = np.arange(2); width = .3
        bars = axes[1].bar(x - .17, [failed['actual_api_calls'], browser['actual_api_calls']], width,
            color='#b9b7a6', label='实际 API 请求')
        axes[1].bar_label(bars, padding=3)
        bars = axes[1].bar(x + .17, [len(failed['receipts']), len(browser['receipts'])], width,
            color='#7a8f81', label='实际运动')
        axes[1].bar_label(bars, padding=3)
        axes[1].set(xticks=x, xticklabels=['首次页面任务 · 未通过', '补齐进度后 · 通过'],
            ylim=(0, 36), ylabel='次数', title='相同提示的开发对照，非统计收益评测')
        axes[1].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.set_facecolor('#fffefa'); axis.grid(axis='y', color='#e6e0d5', linewidth=.6); axis.set_axisbelow(True)
        fig.savefig(OUT / 'results.png', dpi=180); plt.close(fig)

    rows = ''
    for label, values, artifact in [('原生工具', native, 'native-timed.json'),
        ('独立 API · 12 次', api['receipts'], 'api-verified.json'),
        ('页面 API · 9 次', browser['receipts'], 'workbench-verified.json')]:
        errors = [row.get('native_feedback', row.get('feedback'))['position_error_m'] * 1000 for row in values]
        rows += f'<tr><td>{esc(label)}</td><td>两次到位</td><td>{errors[0]:.3f} mm</td><td>{errors[1]:.3f} mm</td><td>{link(artifact,"核验记录")}</td></tr>'
    gallery = ''.join(f'<a href="{esc(frame["url"])}" target="_blank" rel="noopener"><img src="{esc(frame["url"])}" alt="{esc(frame["label"])}"><span>{esc(frame["label"])} · {frame["sim_time"]:.1f}s</span></a>'
        for frame in browser['terminal']['agent']['goal_keyframes'])
    style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261008-ik/index.html').read_text(), re.S)[1]
    style += '.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}.flow div{border:1px solid #e6dfd4;border-radius:10px;padding:15px;background:#fffefa;font-size:13px}.flow span{display:block;font-size:11px;color:#82776a;margin-top:7px}.geometry{display:grid;grid-template-columns:1fr 1fr;gap:18px}.geometry p{margin:8px 0;font-size:13px}.snapshot{border:1px solid #e6dfd4;border-radius:12px;padding:15px;background:#fffefa}.snapshot b{display:block;margin-bottom:7px}code{font-size:12px;overflow-wrap:anywhere}img{max-width:100%}.gallery a{text-decoration:none;border:1px solid #e6dfd4;border-radius:10px;overflow:hidden}.gallery img{width:100%}.gallery span{padding:8px 12px}.limits{border-left:3px solid #b29378;padding-left:16px}.provenance{font-size:12px;color:#82776a}@media(max-width:700px){.flow{grid-template-columns:1fr 1fr}.geometry{grid-template-columns:1fr}}'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 场景目标与任务进度</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.09</div><h1>让模型按物体关系选目标</h1><p>命名场景、坐标解析、原生位姿回执和精简任务进度，接入同一个闭环。</p></div><a class="pill" href="http://127.0.0.1:8766/">动作工作台 ↗</a></header>
<div class="stats"><div><b>2 次</b><span>页面实际运动 · 无重复</span></div><div><b>9 次</b><span>真实模型 API 请求</span></div><div><b>3 / 3</b><span>模型阶段与回执核验</span></div><div><b>未标定</b><span>仍属仿真几何候选</span></div></div>
<section><h2>补齐从语言到运动之间的一层</h2><p>模型现在可读取瓶子、篮子和桌面的命名目录，再按物体原点、外形中心或顶部生成末端目标。目标绑定所选机械臂、朝向、规划后端和场景版本；物体在规划前明显移动、目标被改写或版本过期时拒绝派发。位置匹配使用原有 1e−9 m 数值精度，运动验收仍为原有 12 mm／0.08 rad。</p>
<div class="flow"><div>场景目录<span>当前仿真几何，有来源标签</span></div><div>解析相对目标<span>世界轴／物体轴，保持当前 TCP 朝向</span></div><div>TAMP / cuRobo<span>原生逆解、路径碰撞与执行器运动</span></div><div>原生到位回执<span>绑定阶段，再用新观测复查</span></div></div>
<div class="geometry"><div class="snapshot"><b>本次右手目标 · TAMP</b><p>瓶子原点 + 物体轴偏移 <code>[−0.13, −0.06, +0.32] m</code>。</p><p>物体轴旋转会旋转偏移：<code>p = p₀ + R · offset</code>。这不是抓取动作。</p></div><div class="snapshot"><b>本次左手目标 · cuRobo</b><p>篮子外形顶部 + 世界轴偏移 <code>[−0.12, 0, +0.20] m</code>。</p><p>篮子顶部是外形边缘，不是内腔底部；世界轴偏移不随物体旋转。</p></div></div>
<p class="scope">目录来自 MuJoCo 真值与候选碰撞体，不是相机识别或实物测量。解析目标本身既不证明可达，也不证明可抓。</p>
<div class="links">{link('native-timed.protocol.json','原生协议')}{link('api.protocol.json','独立API协议')}{link('workbench-final.protocol.json','页面协议')}{link('source/sim/scene.py','场景目录实现')}{link('source/harness/scene.py','目标绑定与任务进度')}</div></section>
<section><h2>相同声明目标，三层分别核验</h2><p>原生工具两次运动、独立真实 API 两次运动和页面真实 API 两次运动分别留档。cuRobo 回执必须有成功的 GPU 提案、原生路径复核且无 TAMP 替代；两只手的实际 FK 都达标。第一份计划声明 120 秒、计入模型及规划等待；任务中没有重置、加时、换目标或改物理参数。</p>
<img src="results.png" alt="三层位姿误差以及首次失败和进度修复后的模型请求数、实际运动数对照"><table><thead><tr><th>验收层</th><th>动作</th><th>右手 TAMP</th><th>左手 cuRobo</th><th>来源</th></tr></thead><tbody>{rows}</tbody></table>
<p class="provenance">这些是同一已声明两目标协议下的开发案例，不是跨物体泛化成功率，也不是记忆收益的统计评测。位置误差来自仿真 FK，不是独立实物精度。</p><div class="gallery">{gallery}</div>
<details><summary>保留首次失败与检查器修正</summary><p>首次页面任务的两项指定运动已到位，但续接模型未获得清晰的整项进度，重复运动至 8 次，31 次请求后报告不确定。整项任务判为未通过，不能拿第一次到位掩盖重复动作和错误总结。</p><p>现在新观测与状态工具附带精简 task_progress：各阶段、绑定执行编号、待办项和本任务摘要回执；已计划运动全部完成后，重复的场景目标运动会拒绝。阶段只由原生回执判定，模型仍可提交相机证据不足。</p><p>最初原生检查曾把实际 paused 误写成 measured_delay；该记录只用于动作诊断，另按原协议重新执行。独立 API 首个检查器错误要求恰好两次只读目标解析；后来改为检查两个实际使用目标。页面最终检查器又被 −0.233 和 −0.23299999999999998 的二进制表示差异误判；只读复评分使用现有派发精度 1e−9 m，没有更改物理门槛、重跑动作或覆盖原始结果。</p>
<div class="links">{link('workbench.json','首次页面整项失败')}{link('workbench-final.json','最终页面原始记录')}{link('workbench-verified.json','最终只读复评分')}{link('api.json','独立API原始记录')}{link('api-verified.json','独立API只读核验')}{link('native.json','首次计时不一致记录')}{link('protocol-audit.json','计时审计')}{link('contracts-verified.log','15项场景边界检查')}{link('regression.log','21项相关回归')}</div></details></section>
<section><h2>任务状态、方法记忆与实验档案分开</h2><p>当前任务状态负责告诉模型哪些步骤已经做完；长期记忆只放用户偏好、合并的方法和失败教训，不用过去坐标替代新观测。原始回执属于实验档案，按执行编号查询；不再提供“原始记忆”的默认入口。页面目前是 1 条偏好、5 条合并方法，多次同类运动没有产生重复卡片。</p><p>技能窗口依然按 TAMP、cuRobo、π0.5、ACT 切换。新增只读目录和计算工具合并为一行可折叠的“场景与目标”；主布局、自由观察视角与模型三路固定相机维持原结构。</p><div class="geometry"><a href="skills.png"><img src="skills.png" alt="cuRobo技能目录：运动技能与折叠场景目标说明"></a><a href="memory.png"><img src="memory.png" alt="记忆默认只有一条偏好与五条方法，无原始记忆切换"></a></div><div class="links">{link('workbench.png','实际任务页面')}{link('audit.json','源码、物理与主页面审计')}</div></section>
<section><h2>与成熟项目的关系，以及下一步</h2><p>参照 <a href="https://lianegalanti.github.io/Pigey/">Pigey</a> 的观察、工具反馈、继续决策与恢复结构；参照 <a href="https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html">MoveIt Task Constructor</a> 将目标生成、连接路径与执行阶段分开。<a href="https://github.com/sea-bass/pyrobosim">PyRoboSim</a> 的世界与对象建模也提供结构参考，但它是 2D 移动机器人原型工具，不能替代本平台的三维接触动力学。这里是设计借鉴，未引入这些仓库作为完整实现或宣称与其评测等价。</p>
<div class="limits"><p><b>“移动到物体附近”已经接通；“任意物体都能抓”尚未实现。</b>当前真实场景仍只有瓶子、篮子、桌面。单元检查中的方块和球仅验证自动发现、旋转坐标和外形边界；没有在 OpenArm 中执行它们的抓取。抓取、接触判据、物体目标验收和携物碰撞表示仍有瓶子专用分支。</p><p>下一步应把物体注册、形状候选、夹爪可夹范围、接触与放置验收统一为对象契约，先在多个已声明形状上核验原生动作，再接语言与视觉定位。原有固定 TAMP 演示保留；本轮没有改它的物理执行。</p><p>Real → Sim → Real 仍缺独立的底座／关节零位／TCP／相机内外参／时序标定、驱动响应及负载辨识、物体质量惯量与摩擦接触测量，以及独立任务留出评测。当前 NoSlip 接触候选未经物理标定；回放一致和 FK 达标不能补齐这些证据。</p></div><p class="small">公司 SSH 读取与硬件操作：本轮均为 0。主平台 8765 的持瓶状态保持不变；独立候选工作台使用 8766。当前总目标继续进行。</p></section>
</main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print('scene report rebuilt from retained evidence', flush=True)


if __name__ == '__main__': main()
