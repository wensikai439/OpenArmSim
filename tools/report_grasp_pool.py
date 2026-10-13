"""Rebuild the grasp, route and memory report from retained evidence."""
import html
import json
import os
import re
import statistics
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261008-ik'


def read(name): return json.loads((OUT / name).read_text())
def esc(value): return html.escape(str(value))
def link(path, text): return f'<a href="{esc(path)}">{esc(text)}</a>'


def main():
    probes = [('预抓取', read('results.json')['cases']), ('最终夹持', read('grasp/results.json')['cases'])]
    native = read('native/summary.json')['cases']
    routes = read('routes.json')['cases']
    api = read('api-shifted-bound.json')
    browser = read('workbench-final.json')
    audit = read('audit.json')
    plan = api['terminal']['agent']['plan']
    error = plan['goal_assessment']['position_error_m'] * 1000
    with tempfile.TemporaryDirectory(prefix='openarm-grasp-report-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False, 'font.size': 9,
            'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained', facecolor='#fffefa')
        for backend, offset, color, label in [('original', -.16, '#758d81', '原求解器'), ('bounded', .16, '#b4957b', '边界 TRF')]:
            values = [statistics.median(r['wall_s'] for r in rows if r['backend'] == backend) * 1000 for _, rows in probes]
            axes[0].bar(np.arange(2) + offset, values, width=.29, color=color, label=label)
            for x, value, (_, rows) in zip(np.arange(2) + offset, values, probes):
                found = sum(r['proposal_found'] for r in rows if r['backend'] == backend)
                axes[0].text(x, value + 7, f'{value:.0f}ms\n{found}/10提案', ha='center', fontsize=8)
        axes[0].set(xticks=range(2), xticklabels=[p[0] for p in probes], ylabel='每个固定问题的中位求解时间 / ms',
            title='换求解器没有增加可行提案', ylim=(0, 470))
        axes[0].legend(frameon=False, loc='upper left')
        for side, offset, color, label in [('left', -.16, '#758d81', '左手'), ('right', .16, '#b4957b', '右手')]:
            rows = [row for row in native if row['case']['arm'] == side]
            x = [i + offset for i, row in enumerate(rows) if row['passed']]
            y = [row['stability']['max_window_relative_drift_m'] * 1000 for row in rows if row['passed']]
            axes[1].bar(x, y, width=.29, color=color, label=label)
            for i, row in enumerate(rows):
                if not row['passed']:
                    axes[1].scatter(i + offset, 1.05, marker='x', color='#ad6c51', s=45)
                    axes[1].text(i + offset, 1.10, '未找到路线', ha='center', fontsize=7)
        axes[1].axhline(1, color='#ad6c51', linestyle='--', linewidth=1)
        axes[1].set(xticks=range(5), xticklabels=['原位', '先前失败位', '内侧', '外侧', '横移'],
            ylabel='完整10秒中最差2秒相对漂移 / mm', ylim=(0, 1.27), title='声明条件内的真实抓取与保持')
        axes[1].legend(frameon=False, loc='upper left')
        errors = [row['metrics']['tamp']['position_error_m'] * 1000 for row in routes] + [error]
        bars = axes[2].bar(range(4), errors, color=['#b1b6a9'] * 3 + ['#758d81'])
        axes[2].bar_label(bars, fmt='%.2f', padding=3)
        axes[2].axhline(10, color='#ad6c51', linestyle='--', linewidth=1)
        axes[2].set(xticks=range(4), xticklabels=['原位左→右', '新位左→右', '原位右手', 'API新位左→右'],
            ylabel='实际瓶子落点误差 / mm', ylim=(0, 12), title='原目标与10mm验收容差保留')
        axes[2].tick_params(axis='x', labelsize=7)
        for ax in axes:
            ax.set_facecolor('#fffefa'); ax.grid(axis='y', color='#e6e0d5', linewidth=.6); ax.set_axisbelow(True)
        fig.savefig(OUT / 'grasp.png', dpi=180); plt.close(fig)

    native_rows = ''
    for row in native:
        name = row['case']['name']; feedback = row['native_feedback'] or {}; search = feedback.get('grasp_search') or {}
        detail = f'{row["stability"]["max_window_relative_drift_m"] * 1000:.5f}mm / {row["actual_hold_s"]:.3f}s' if row['passed'] else row['failure']['code']
        native_rows += f'<tr><td>{esc(name)}</td><td>{esc(row["case"]["delta_m"])}</td><td>{"通过" if row["passed"] else "未通过"}</td><td>{esc(detail)}</td><td>{search.get("candidate_index", "—")}</td><td>{link("native/" + name + "/result.json", "完整证据")}</td></tr>'
    route_rows = ''
    for row in routes:
        route_rows += f'<tr><td>{esc(row["case"])}</td><td>{"通过" if row["passed"] else "未通过"}</td><td>{row["metrics"]["tamp"]["position_error_m"] * 1000:.3f}mm</td><td>{row["duration_sim_seconds"]:.3f}s</td><td>{link("/runs/" + row["run_id"] + "/summary.json", "原生回执")}</td></tr>'
    gallery = ''.join(f'<a href="{esc(frame["url"])}" target="_blank" rel="noopener"><img src="{esc(frame["url"])}" alt="{esc(frame["label"])}"><span>{esc(frame["label"])} · {frame["sim_time"]:.1f}s</span></a>'
        for frame in browser['terminal']['agent']['goal_keyframes'])
    style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261008-grip/index.html').read_text(), re.S)[1]
    style += '.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.gallery img{margin:0}.gallery span{font-size:12px;display:block;padding:6px}details>summary{cursor:pointer;margin:12px 0}.scope{color:#8b6e57}@media(max-width:640px){.gallery{grid-template-columns:1fr}}'
    style += 'header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px}.eyebrow{font-size:10px;letter-spacing:.12em;color:#8b7d6e;margin-bottom:12px}h1{margin:0 0 10px}header p{margin:0;color:#797367}.pill{white-space:nowrap;border:1px solid #ddd4c8;border-radius:24px;padding:8px 16px;text-decoration:none;font-size:12px;background:#fffefa}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:28px 0}.stats>div{border:1px solid #e6dfd4;border-radius:12px;padding:16px;background:#fffefa}.stats b{display:block;font-size:25px;font-weight:600;line-height:1.5}.stats span{display:block;font-size:11px;color:#82776a}.links{display:flex;flex-wrap:wrap;gap:7px 16px;padding-top:12px;border-top:1px solid #eee8df;margin-top:16px;font-size:12px}.small{font-size:12px;color:#82776a}.gallery a{background:#fffefa}details{border-top:1px solid #eee8df;margin-top:16px;padding-top:4px}details>summary{font-size:13px;color:#806146}@media(max-width:700px){header{display:block}.pill{display:inline-block;margin-top:16px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 几何抓取与交接</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.08–09</div><h1>新瓶位抓取，完整交接落地</h1><p>扩大物体几何候选，并用原生动作、模型调用和页面阶段共同核验。</p></div><a class="pill" href="http://127.0.0.1:8766/">动作工作台 ↗</a></header>
<div class="stats"><div><b>9 / 10</b><span>声明抓取条件通过 · 非成功率</span></div><div><b>{error:.2f} mm</b><span>新瓶位API前放6cm误差</span></div><div><b>2 / 2</b><span>页面左→右交接阶段</span></div><div><b>未标定</b><span>真机迁移未验证</span></div></div>
<section><h2>为何小幅换位置就失败</h2><p>原左手工具把少量抓取朝向与固定高度逐一求逆解；预抓取可达并不代表夹持点与连接路径同时可达。先用相同12个种子、原限位和原位置／旋转门槛比较两种求解器：预抓取都只找到4/10提案，最终夹持都只有1/10，边界TRF更慢，因此没有作为默认。软件求解失败不等于物理不可达。</p><p>新参数化抓取从当前碰撞圆柱的中心与轴取候选，覆盖24个接近方位和4个夹爪俯仰；先验证夹持逆解，再用它热启动预抓取逆解，并检查起点连接路径和接近段。只在scratch MjData搜索；真实闭合、抬升和保持均由执行器与接触产生。</p><p class="scope">对象来源仍是仿真几何，不是“模型看见任意物体即可抓”。只覆盖已知直立圆柱瓶子；未知形状、视觉定位、横卧物体及通用目标注册尚未实现。</p><img src="grasp.png" alt="原逆解和边界TRF的固定条件对照、全部10条原生抓取保持结果、四条完成路线的实际落点误差"><div class="links">{link('protocol.json','预抓取固定协议')}{link('results.json','全部20个预抓取提案')}{link('grasp/protocol.json','夹持固定协议')}{link('grasp/results.json','全部20个夹持提案')}{link('native/protocol.json','真实抓取协议')}</div></section>
<section><h2>新瓶位与持续夹持</h2><p>位置偏移在任务开始前声明，过程中没有改动物体或重置。原位、先前失败位、内侧、外侧与横移，左右手各一次，抬升后保持原控制目标10秒。原2秒窗口1mm／0.02rad验收保留。先前失败位为(+1.5,-2.0)cm：左手第25个候选以-45°方位、-15°俯仰完成抓取，实际抬升约11.89cm，完整保持中的最差2秒漂移0.590mm。</p><p>外侧左手未找到完整无碰撞路线，未启动动作，全部候选拒绝原因保留。9/10只是这些预先指定开发条件的结果，不能当作随机泛化成功率。这组几何诊断的搜索时间暂停；下表完整路线及API才计入底层规划等待。</p><div class="scroll"><table><thead><tr><th>条件</th><th>初态偏移 / m</th><th>结果</th><th>保持或失败</th><th>候选序号(从0)</th><th>依据</th></tr></thead><tbody>{native_rows}</tbody></table></div></section>
<section><h2>完整路线与真实模型调用</h2><p>三组连续路线均使用普通SimulationService、measured_delay及首次声明的120秒预算，cuRobo冷准备、逆解、路径规划和反馈修正耗时实际积分。原有目标、碰撞、夹持与放置门槛不变，原生记录器的参数守卫保留。</p><div class="scroll"><table><thead><tr><th>路线</th><th>验收</th><th>实际瓶子落点误差</th><th>原生时长</th><th>依据</th></tr></thead><tbody>{route_rows}</tbody></table></div><p>独立API实例中，已保存的deepseek-flash用{api['actual_api_calls']}次请求完成新瓶位左抓→右接→真实cuRobo携瓶+X 1.5cm→原始瓶位前方6cm桌面放置，误差{error:.3f}mm；目标未改、未用TAMP替代cuRobo。模型请求等待也计入物理。阶段由模型给出，并绑定同一组合动作的原生子回执。最终通用阶段约束随后又在页面交接任务中验收；这条API是约束前版本，源码分别冻结。</p><div class="links">{link('routes.protocol.json','完整路线协议')}{link('routes.json','三条完整原生路线')}{link('api-shifted-bound.protocol.json','API任务协议')}{link('api-shifted-bound.json','全部7次API与回执')}</div><details><summary>保留的API失败与Harness修正</summary><p>第一次同条件API成功抓取和交接，cuRobo首段路径实际执行，但反馈修正未找到路径。模型随后改用TAMP完成放置，并把目标完成报告为成功；指定cuRobo路线没有完成，整条实验记为失败。新约束绑定第一次已启动连续搬运的规划器，同一用户目标的续跑和计划修订不能换后端；新用户目标可另选后端。它适用于一次搬运选一个后端，并非任意混合后端程序。</p><p>两次API的延迟和调用选择不同，不能把第二次通过归因于后端绑定，更不能推断cuRobo已稳定。没有放宽GPU、网格碰撞或位姿误差检查。</p>{link('api-shifted-left.json','第一次失败与完整恢复记录')}{link('candidate-final-checks.log','拒绝换后端的对抗检查')}</details></section>
<section><h2>页面：只做要求的左→右交接</h2><p>从页面自然语言提交左手拿起→交给右手。最终{browser['actual_api_calls']}次API调用，工具仅pick和handover，右手独立持瓶后停止，没有多做放置。模型自行给出两阶段，两个方向无关的原生技能回执确认2/2。</p><a href="workbench.png"><img src="workbench.png" alt="更新后页面左侧真实模型调用、右上机器人、右下2/2左手到右手交接阶段"></a><div class="gallery">{gallery}</div><details><summary>保留的阶段绑定缺陷</summary><p>第一轮物理交接实际完成，但模型把右手独立持瓶绑定到了旧演示left_grasp，页面显示3/4；该次页面验收未通过。通用API计划现只接受观察、抓取、交接、末端运动、放置、工具完成和视觉复查回执。方向专属的旧演示事件从API schema与验证器中排除，原固定演示保留。</p>{link('workbench.json','首次页面验收未通过')}{link('stage-issue.png','3/4误绑定截图')}</details><div class="links">{link('workbench-final.protocol.json','页面提交前协议')}{link('workbench-final.json','最终页面验收')}{link('workbench-final.collector.json','只读收集范围')}</div></section>
<section><h2>记忆：一条方法，多次证据</h2><p>默认面板为3条方法和1条偏好。重复左抓与左→右交接各合并为一个方法，支持次数仍保留；默认模型读取小索引，详情按需检索。页面现在只传方法摘要、适用范围和证据链接，重复的关节数组、前后位姿与原始相机证据不再嵌在记忆数据中，当前知识部分约{audit['services']['8766']['knowledge_json_bytes']/1024:.1f}KiB。</p><p>完整回执仍属于实验档案：没有它们就无法检查某条“成功经验”来自什么动作，也无法纠正错误摘要。它们不作为第二份默认记忆注入；特权回执依旧需要显式检索。新增grasp_search_exhausted会归为同一有边界的失败教训，不记成“机械臂绝对不可达”。删除方法仍按原遗忘规则，不因原记录留存而自动复活。</p><a href="memory.png"><img src="memory.png" alt="三条方法与一条偏好，重复交接归并为一条方法，证据按需展开"></a><p class="small">设计参考<a href="https://code.claude.com/docs/en/memory">Claude Code官方memory文档</a>的简短入口、主题细节按需读取，以及<a href="https://github.com/letta-ai/letta">Letta</a>的持久状态与检索分工；没有安装其服务，也没有宣称等同复现。</p></section>
<section><h2>保留与边界</h2><p>38项相关回归和6项新增对抗检查通过。96组几何旋转／平移检查通过。原固定演示与新类固定分支逐一比较679条目标及每个原生qpos/qvel端点，完全一致；两者随后都在同一目标碰撞处停止。因此“保留演示”证明的是行为未改，不能说这个候选里的固定全任务已通过。</p><p>8765原主页面仍保持原参数与原左手持瓶状态；本轮新能力在独立8766工作台启用。两个服务的代码守卫与参数守卫均正常。没有访问或改动SSH公司电脑，没有硬件运动、定时器或自动启动。接触候选仍是此前dim4/NoSlip3，摩擦、质量、惯量、限位、控制器和评分没有再调。</p><p>物体资产／视觉定位、独立base/TCP/相机与时序标定、真实夹爪开口力与物体摩擦惯量、控制器等价性，以及匹配OpenArm的π0.5与ACT完整任务，仍有缺项。仿真pass不能推出真机pass。</p><div class="links">{link('contract/result.json','固定演示逐步等价性')}{link('regression.log','38项相关回归')}{link('candidate-final-checks.log','6项候选对抗检查')}{link('audit.json','参数、源码与服务审计')}{link('source/sim/grasp.py','抓取实现快照')}</div><p class="small">方法借鉴<a href="https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html">MoveIt Task Constructor</a>的候选生成、逆解和连接阶段组合；<a href="https://lianegalanti.github.io/Pigey/">Pigey</a>提供观测→参数化工具→反馈→恢复的目标结构。本轮是其中部分接口和开发案例的推进，不是Pigey性能复现。</p></section>
</main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print(json.dumps({'report': str(OUT / 'index.html'), 'native_conditions': len(native),
        'native_passed': sum(row['passed'] for row in native), 'API_status': api['status'], 'UI_passed': browser['passed']}, ensure_ascii=False))


if __name__ == '__main__': main()
