"""Render all fixed bottle tasks, including failures and collector corrections."""
import html
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261009-bottle'

def read(path):return json.loads(path.read_text())
def esc(value):return html.escape(str(value))


def main():
    native={}
    for folder in sorted(OUT.iterdir()):
        path=folder/'results.json'
        if path.is_file():native[folder.name]=read(path)
    api=read(OUT/'api-forward/result.json') if (OUT/'api-forward/result.json').exists() else None
    left=read(OUT/'api-left-right/recovery-2/assessment.json') if (OUT/'api-left-right/recovery-2/assessment.json').exists() else None
    cross_paths=sorted(OUT.glob('api-cross*/result.json'),
        key=lambda p:1 if p.parent.name=='api-cross' else int(p.parent.name.rsplit('-',1)[1]))
    cross_folder=cross_paths[-1].parent.name if cross_paths else None
    cross=read(cross_paths[-1]) if cross_paths else None
    front_paths=sorted(OUT.glob('api-front-*/result.json'),key=lambda p:int(p.parent.name.rsplit('-',1)[1]))
    front_html=''
    if front_paths:
        path=front_paths[-1];trial=read(path);folder=path.parent.name
        plan=trial['terminal']['agent'].get('plan') or {}
        placement=next((r['feedback'] for r in trial.get('receipts',[]) if r['execution']['skill']=='place'),{})
        measured=placement.get('position_error_m')
        if measured is None:measured=(plan.get('goal_assessment') or {}).get('position_error_m')
        error_label=f'{measured*1000:.2f} mm' if measured is not None else 'UNKNOWN'
        checks=''.join(f'<li>{esc(k)}：{"通过" if v else "未通过"}</li>' for k,v in trial['checks'].items())
        iterations=' · '.join(f'<a href="{p.parent.name}/result.json">{esc(p.parent.name)}：{"通过" if read(p)["passed"] else "未完成"}</a>' for p in front_paths)
        stages=' → '.join(s['title'] for s in plan.get('steps',[]))
        front_html=(f'<section><h2>左手前放8cm · 真实 API 闭环</h2><div class="stats">'
            f'<article><small>实际 API 请求</small><strong>{trial["actual_api_calls"]}</strong></article>'
            f'<article><small>累计仿真时间</small><strong>{trial["terminal"]["observation"]["sim_time"]:.2f} s</strong></article>'
            f'<article><small>原目标 XY 误差</small><strong>{error_label}</strong></article>'
            f'<article><small>原整项验收</small><strong>{"完整通过" if trial["passed"] else "未完成"}</strong></article></div>'
            f'<blockquote>{esc(read(path.parent/"protocol.json")["prompt"])}</blockquote><p>{esc(stages)}</p>'
            '<p>左侧起点 [0.44, 0.18] 米，目标 [0.52, 0.18] 米。模型自行规划与选择工具，TAMP抓取/放置、真实cuRobo携物，目标与120秒预算锁定。'
            '第一轮动作完成但漏计等待，协议失败；第二轮拒绝过期路径后重试，松手后漂出10mm目标区，完整任务失败。原始失败与冻结源码均保留。</p>'
            f'<div class="pair"><a href="{folder}/operator-start.jpg"><img src="{folder}/operator-start.jpg" alt="左手前移原始场景"><span>原始场景</span></a>'
            f'<a href="{folder}/operator-final.jpg"><img src="{folder}/operator-final.jpg" alt="左手前移实际终态"><span>实际终态</span></a></div>'
            '<details><summary>完整验收与所有 API 迭代</summary>'
            f'<ul>{checks}</ul><p>{iterations}</p><p><a href="{folder}/protocol.json">预先声明协议与源码</a></p></details></section>')
    page_html=''
    if (OUT/'front-ui/result.json').is_file():
        trial=read(OUT/'front-ui/result.json');plan=trial['terminal']['agent']['plan']
        place=next(r['feedback'] for r in trial['receipts'] if r['execution']['skill']=='place')
        page_html=(f'<section><h2>页面短指令 · 左手交右手</h2><div class="stats">'
            f'<article><small>实际 API 请求</small><strong>{trial["actual_api_calls"]}</strong></article>'
            f'<article><small>累计仿真时间</small><strong>{trial["terminal"]["observation"]["sim_time"]:.2f} s</strong></article>'
            f'<article><small>原目标 XY 误差</small><strong>{place["position_error_m"]*1000:.2f} mm</strong></article>'
            f'<article><small>原整项验收</small><strong>{"完整通过" if trial["passed"] else "未完成"}</strong></article></div>'
            f'<blockquote>{esc(read(OUT/"front-ui/protocol.json")["prompt"])}</blockquote>'
            '<p>从实际页面发送普通中文指令，模型自行选择120秒measured_delay与五个阶段，完成原位左抓→右接→GPU→原桌面目标放置→复查。未在提示中指定预算或工具协议。此任务没有请求10秒持物；完整持物检查在四项原生回归中另列。</p>'
            '<a href="front-ui/page.png"><img src="front-ui/page.png" alt="实际工作台五阶段完成截图"></a>'
            '<p><a href="front-ui/result.json">页面实际 API、原生回执和全部验收</a> · <a href="front-ui/memory.png">当前对象三条精炼方法</a></p></section>')
    rows=[]; chart=[]
    for revision,trial in native.items():
        assessment=read(OUT/revision/'assessment.json') if (OUT/revision/'assessment.json').exists() else None
        corrected={r['case']['name']:r for r in assessment['cases']} if assessment else {}
        for row in trial['cases']:
            name=row['case']['name'];passed=corrected.get(name,row)['passed']
            source=read(OUT/revision/'protocol.json')['source_sha256']
            sha=source.get('sim/bottle_routing.py',source['sim/object_handover.py'])[:12]
            status='完整通过' if passed else '未完成'
            folder=f'{revision}/{name}';error=(row.get('error') or {}).get('code','—')
            correction=f' · <a href="{folder}/assessment.json">独立评分</a>' if name in corrected else ''
            rows.append(f'<tr><td>{esc(revision)}</td><td>{esc(name)}</td><td class="{"pass" if passed else "fail"}">{status}</td>'
                f'<td>{row["sim_time"]:.3f}</td><td>{esc(row["phase"])}</td><td>{esc(error)}</td><td><code>{sha}</code></td>'
                f'<td><a href="{folder}/result.json">原始记录</a>{correction}</td></tr>')
            chart.append((revision+' / '+name,row['sim_time'],passed))
    with tempfile.TemporaryDirectory(prefix='openarm-bottle-chart-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
        if not font.is_file():raise RuntimeError('Report requires the installed CJK font; no missing-glyph fallback')
        font_manager.fontManager.addfont(str(font))
        plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
            'font.size':9,'axes.unicode_minus':False})
        fig,axis=plt.subplots(figsize=(10.5,max(4,len(chart)*.38)),layout='constrained',facecolor='#fffefa')
        bars=axis.barh(range(len(chart)),[r[1] for r in chart],height=.6,
            color=['#829986' if r[2] else '#bd9e85' for r in chart])
        axis.set(yticks=range(len(chart)),yticklabels=[r[0] for r in chart],xlabel='累计仿真时间 / 秒',
            title='原目标与原验收：开发案例，不是泛化成功率',xlim=(0,130))
        axis.bar_label(bars,fmt='%.1f',padding=3);axis.invert_yaxis()
        axis.axvline(120,color='#9a624b',linestyle='--',linewidth=1)
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',color='#e8e1d7',linewidth=.5);axis.set_axisbelow(True)
        fig.savefig(OUT/'results.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(10.5,5.8),layout='constrained',facecolor='#fffefa',sharex=True)
        for name,color in [('goal-1','#b77856'),('goal-2','#638571')]:
            paths=list((OUT/name/'left-right/episode').glob('*/transitions.jsonl'))
            if not paths:continue
            points=[json.loads(line) for line in paths[0].read_text().splitlines()]
            times=[r['next_observation']['sim_time'] for r in points]
            axes[0].plot(times,[r['diagnostics']['native_object']['position_m'][2] for r in points],label=name,color=color)
            axes[1].plot(times,[r['evaluation']['non_target_object_guard']['displacement_m']['block']*1000 for r in points],label=name,color=color)
        axes[0].set(ylabel='瓶子原点 Z / 米',title='实际运输反馈：接触驱动的自由物体')
        axes[1].set(ylabel='非目标方块位移 / mm',xlabel='累计仿真时间 / 秒');axes[1].axhline(3,color='#9a624b',linestyle='--',linewidth=1)
        for axis in axes:axis.legend();axis.spines[['top','right']].set_visible(False);axis.grid(color='#e8e1d7',linewidth=.5)
        fig.savefig(OUT/'feedback.png',dpi=170);plt.close(fig)
    api_html='<p>真实 API 验收尚未结束。</p>'
    if api:
        place=next((r['feedback'] for r in api['receipts'] if r['execution']['skill']=='place'),{})
        plan=api['terminal']['agent'].get('plan') or {}
        stages=' → '.join(s['title'] for s in plan.get('steps',[]))
        checks=''.join(f'<li>{esc(k)}：{"通过" if v else "未通过"}</li>' for k,v in api['checks'].items())
        error=place.get('position_error_m')
        error_label=f'{error*1000:.2f} mm' if error is not None else 'UNKNOWN'
        api_html=(f'<div class="stats"><article><small>实际 API 请求</small><strong>{api["actual_api_calls"]}</strong></article>'
            f'<article><small>累计仿真时间</small><strong>{api["terminal"]["observation"]["sim_time"]:.2f} s</strong></article>'
            f'<article><small>原目标 XY 误差</small><strong>{error_label}</strong></article>'
            f'<article><small>原生任务验收</small><strong>{"通过" if api["passed"] else "未完成"}</strong></article></div>'
            f'<blockquote>{esc(read(OUT/"api-forward/protocol.json")["prompt"])}</blockquote><p>{esc(stages)}</p>'
            '<p>提示没有提供世界目标坐标或工具调用顺序；模型自行声明原目标和运行阶段。原生评估、模型判断与相机证据分别保留。单次固定场景通过不能支持开放物体、稳定成功率或真机迁移结论。</p>'
            '<div class="pair"><a href="api-forward/operator-start.jpg"><img src="api-forward/operator-start.jpg" alt="API 原始场景"><span>原始场景</span></a>'
            '<a href="api-forward/operator-final.jpg"><img src="api-forward/operator-final.jpg" alt="API 实际终态"><span>实际终态</span></a></div>'
            f'<details><summary>完整验收条件</summary><ul>{checks}</ul></details>'
            '<p><a href="api-forward/result.json">原始 API、阶段与原生回执</a> · <a href="api-forward/protocol.json">预先固定的实验条件</a></p>')
    calibration=read(ROOT/'sim/calibration.json')
    left_html='<p>左→右真实 API 尚未完成。</p>'
    if left:
        place=next(s['feedback'] for s in left['steps'] if s['task']=='place')
        relay=next(s['feedback'] for s in left['steps'] if s['task']=='handover')
        status='原目标完整通过' if left['passed'] else '原目标未完成'
        left_html=(f'<div class="stats"><article><small>实际 API 请求</small><strong>{left["actual_api_calls"]}</strong></article>'
            f'<article><small>原目标 XY 误差</small><strong>{place["position_error_m"]*1000:.2f} mm</strong></article>'
            f'<article><small>实际独立持物</small><strong>{relay["hold_audit"]["actual_seconds"]:.2f} s</strong></article>'
            f'<article><small>原生任务验收</small><strong>{status}</strong></article></div>'
            '<p>固定左侧起点 [0.44, 0.18] 米；目标仍为 [0.56, −0.13] 米。模型自行制定阶段并调用抓取、交接、cuRobo 和放置。总预算120秒，未重置或替换原目标。</p>'
            '<div class="pair"><a href="api-left-right/operator-start.jpg"><img src="api-left-right/operator-start.jpg" alt="左→右原始操作者视图"><span>原始操作者视图</span></a>'
            '<a href="api-left-right/overview-final.jpg"><img src="api-left-right/overview-final.jpg" alt="左→右实际最终全景留帧"><span>实际最终全景留帧</span></a></div>'
            '<details><summary>结果汇总修复与原始证据</summary><p>动作与正式模型复查已完成，汇总程序随后因变量覆盖报错，目标元数据也写错了路径。独立复核使用原始请求、冻结的初始案例、模型派发记录和四份原生执行回执；没有新动作、重跑任务或改写原始轨迹。丢失的内存 SQL 回执未伪造。首次复核、修正评分和原始程序错误均保留。</p>'
            '<p><a href="api-left-right/recovery-2/assessment.json">14项独立复核与源码校验</a> · <a href="api-left-right/progress.json">原始模型阶段与调用</a> · <a href="api-left-right/protocol.json">原始协议及错误元数据</a> · <a href="api-left-right/assessment.json">首次复核记录</a></p></details>')
    gaps=''.join(f'<tr><td>{esc(p["label"])}</td><td>{esc(p["status"])}</td><td>{esc(p["needed"])}</td></tr>' for p in calibration['parameters'])
    cross_html=''
    if cross:
        model_plan=cross.get('terminal',{}).get('agent',{}).get('plan') or {}
        cross_checks=''.join(f'<li>{esc(k)}：{"通过" if v else "未通过"}</li>' for k,v in cross.get('checks',{}).items())
        placement=next((r['feedback'] for r in cross.get('receipts',[]) if r['execution']['skill']=='place'),{})
        hold=next((r['feedback'].get('hold_audit',{}) for r in cross.get('receipts',[]) if r['execution']['skill']=='handover'),{})
        error=placement.get('position_error_m')
        error_label=f'{error*1000:.2f} mm' if error is not None else '未完成'
        cross_html=(f'<section><h2>原位跨侧抓瓶 · 真实 API</h2><div class="stats">'
            f'<article><small>实际 API 请求</small><strong>{cross.get("actual_api_calls","UNKNOWN")}</strong></article>'
            f'<article><small>原目标 XY 误差</small><strong>{error_label}</strong></article>'
            f'<article><small>实际独立持物</small><strong>{hold.get("actual_seconds",0):.2f} s</strong></article>'
            f'<article><small>原生整项任务验收</small><strong>{"完整通过" if cross["passed"] else "未完成"}</strong></article></div>'
            '<p>瓶子保持原始右侧位置 [0.503, −0.173] 米，由左手跨侧抓起后交给右手；原目标仍为桌面 [0.56, −0.13] 米。'
            '120秒预算、连续持物、cuRobo、松手支撑及稳定性要求不变。此单次开发案例不是视觉泛化或真机迁移成绩。</p>'
            f'<p>模型复查：{esc(model_plan.get("verdict","UNKNOWN"))}。模型阶段：{esc(" → ".join(s["title"] for s in model_plan.get("steps",[])))}。</p>'
            f'<div class="pair"><a href="{cross_folder}/operator-start.jpg"><img src="{cross_folder}/operator-start.jpg" alt="原位跨侧抓瓶初始场景"><span>原始右侧瓶子</span></a>'
            f'<a href="{cross_folder}/operator-final.jpg"><img src="{cross_folder}/operator-final.jpg" alt="原位跨侧抓瓶实际终态"><span>实际终态</span></a></div>'
            '<details><summary>原始 API、模型阶段与任务验收</summary>'
            f'<p><a href="{cross_folder}/result.json">完整原始结果</a> · <a href="{cross_folder}/protocol.json">执行前固定协议</a></p>'
            f'<ul>{cross_checks}</ul>'
            '<p>API 开发迭代均保留：'
            + ' · '.join(f'<a href="{p.parent.name}/result.json">{esc(p.parent.name)}：{"通过" if read(p)["passed"] else "未完成"}</a>' for p in cross_paths)
            + '</p></details></section>')
    comparison_html=''
    if (OUT/'cross-assessment.json').exists():
        comparison=read(OUT/'cross-assessment.json')['cases']
        first=next(r for r in comparison if r['revision']=='cross-1')
        last=comparison[-1]
        comparison_html=(f'<section><h2>原位跨侧任务的搜索效率</h2><p>同一原始瓶子位置、初始关节状态、编译模型、120秒预算和最终目标。'
            f'交接规划等待由 {first["relay_planning_wait_s"]:.2f} 秒降至 {last["relay_planning_wait_s"]:.2f} 秒，'
            f'接收抓取被拒候选由 {first["receiver_candidates_rejected"]} 个降至 {last["receiver_candidates_rejected"]} 个。'
            '先检查离供物手更远的轴向位置；候选集、路径和碰撞门槛均保留。</p>'
            '<p>随后仍有两轮放置失败：负载下的末端补偿目标不可达。可选放置路线在已执行完整下降段、持续夹持、'
            '当前无碰撞且物体接近原目标时，才允许原有的小步物体对齐继续；位置逆解另查预测物体直立轴。'
            '松手前追加低速与连续6个控制周期到位检查，避免瞬时到点后仍在漂移。'
            '最终目标、物体精度、松手、撤离、直立及稳定性门槛不变。原始失败均保留。</p>'
            '<p><a href="cross-assessment.json">原位开发迭代与模型/初始状态一致性核验</a></p></section>')
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>瓶子前移与双臂交接 · OpenArm</title><style>
:root{{color-scheme:light;font:15px/1.7 system-ui,sans-serif;color:#38332d;background:#f7f5f0}}body{{max-width:1130px;margin:0 auto;padding:36px 22px 70px}}h1{{font-size:30px;font-weight:550;margin:8px 0}}h2{{font-size:21px;font-weight:550}}small,.muted{{color:#8c8377}}section{{background:#fffefa;border:1px solid #e5ded4;border-radius:15px;padding:24px;margin:22px 0}}a{{color:#926b50;text-underline-offset:4px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.stats article{{background:#f7f5ef;padding:16px;border-radius:10px}}strong{{display:block;font-size:23px;font-weight:550}}blockquote{{border-left:3px solid #b99171;margin:24px 0;padding:5px 18px;background:#faf7f1}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{max-width:100%;height:auto;border-radius:10px}}.pair a{{display:flex;flex-direction:column;gap:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:11px 8px;border-bottom:1px solid #ebe4da;text-align:left;vertical-align:top}}th{{color:#897e70;font-weight:500}}.scroll{{overflow-x:auto}}.pass{{color:#50735a}}.fail{{color:#986c4f}}details{{border-top:1px solid #e9e2d9;padding:16px 0}}summary{{cursor:pointer}}code{{font-size:12px}}@media(max-width:650px){{.stats,.pair{{grid-template-columns:1fr 1fr}}section{{padding:16px}}body{{padding:20px 12px}}}}
</style><header><small>OpenArm Lab · 固定开发案例</small><h1>瓶子前移与双臂交接</h1><p class="muted">保留原目标、失败轨迹和原验收。原演示仍在；实验提供者尚未覆盖所有方向。</p></header>
<section><h2>简短指令 · 真实 API</h2>{api_html}</section>
{front_html}
{page_html}
<section><h2>左→右交接与原目标放置 · 真实 API</h2>{left_html}</section>
{cross_html}
{comparison_html}
<section><h2>相同任务的完整实验记录</h2><p>goal系列使用固定左侧瓶子起点，cross-1至cross-6保留原始右侧瓶子起点并要求左手跨侧抓取。cross-control保留旧版左手前移失败；front-1与front-control是后续的抓取目的排序和低位稳定释放检查。front-control的四项完整通过，均含10秒持物、真实GPU、原目标放置及非目标物体检查。每项原起点、最终放置目标与120秒预算均固定，禁止任务中重置、修改物体姿态或调低验收要求。不同起点与程序版本不能混合计算成功率。</p><img src="results.png" alt="全部已结束原生案例与仿真时间"><div class="scroll"><table><thead><tr><th>版本</th><th>原任务</th><th>结果</th><th>仿真秒</th><th>停止阶段</th><th>原因</th><th>源码</th><th>证据</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">baseline 的 GPU 判别字符串写错；独立评分仅核对实际 planner/backend_metadata、原生路径复核、冻结的 worker 哈希和原位姿阈值。原始回执未改写，负例测试也保留在 assessment.json。</p></section>
<section><h2>运输过程中发生了什么</h2><img src="feedback.png" alt="瓶子实际高度与非目标物体位移反馈"><p>goal-1 已找到端点，但运输途中瓶子碰动方块。goal-2 增加场景高度与物体净空后停止于实际路径净空不足，方块未被碰动。之后的高位运输暴露了负载反馈修正与关节限位问题。goal-8 实现交接与10秒持物，但后续运输的反馈修正路径撞到供物手；goal-9 让空出的供物手沿原生检查路径退出后，完成原放置目标。早期失败均保留，不能算完成。</p></section>
<section><h2>与 Pigey 的差距</h2><p><a href="https://github.com/lianegalanti/Pigey">Pigey</a> 的真机 TAMP 依赖 <a href="https://github.com/tiptop-robot/tiptop">TiPToP</a> 的感知、抓取预测和 cuRobo；其仿真接口没有向模型暴露物体真值位置和接触状态。当前平台仍是明确标注的已注册仿真几何路线，尚不能声称复现这些感知与迁移能力。<a href="https://nvlabs.github.io/curobo/latest/api/curobo.motion_planner.html">cuRobo 官方接口</a> 将抓取目标集、接近、抓取和抬升分开；运动可达还需要抓取、接触及任务级闭环验证。</p></section>
<section><h2>物理标定仍未完成</h2><p>仿真通过不能替代独立真机标定。源码、FK 与坐标一致仅是软件证据；以下项目仍需实际测量或留出真机数据验证。</p><div class="scroll"><table><thead><tr><th>参数</th><th>证据状态</th><th>仍需什么</th></tr></thead><tbody>{gaps}</tbody></table></div></section></html>'''
    memory_html=('<section><h2>记忆：精炼方法与适用条件</h2><p>借鉴<a href="https://code.claude.com/docs/en/memory">Claude Code 的短索引与按需读取</a>。默认模型索引最多8条、8192字节，先按对象与兼容契约过滤，再按相关性检索；旧位姿、原始轨迹和相机文件不进入索引。</p>'
        '<p>此次新增“面向原桌面目标的抓取”条件，不把不同坐标拆成重复经验；交接方向、后端、持稳请求时长和连续/末段审核保持分开。未完成的连续持物审核不能误标成末段成功。忘记精炼方法不会因重新归并旧回执而复活。</p>'
        '<p>原始回执只在实验档案中供追溯，不能作为当前感知或永久重复注入模型。方法精炼不证明记忆提高成功率，尚需受控消融。</p>'
        '<p>抓取提案与原生验证分开，参考<a href="https://moveit.picknik.ai/main/doc/examples/moveit_grasps/moveit_grasps_tutorial.html">MoveIt Grasps</a>的提案、IK/碰撞筛选与运动规划分层。当前重心/接触宽度排序是本地几何启发式，未调用GraspGen或证明抓取动力学泛化。</p></section>')
    content=content.replace('<section><h2>物理标定仍未完成</h2>',memory_html+'<section><h2>物理标定仍未完成</h2>')
    (OUT/'index.html').write_text(content)
    print(OUT/'index.html')


if __name__=='__main__':main()
