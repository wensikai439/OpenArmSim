"""Summarize every declared carried-motion group; never change success criteria."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261008-carry'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
NAMES={'empty-right':'右手空手抬升','carry-right-up':'右手持瓶抬升','carry-right-forward':'右手持瓶前移',
       'carry-left-up':'左手持瓶抬升','carry-right-yaw':'右手持瓶转腕','carry-right-table-collision':'向桌下移动（拒绝）'}

def load(path):return json.loads((OUT/path).read_text())

def measures(row):
    case=row['case'];arm=case['arm'];before=row['before'];after=row['after'];feedback=row['feedback']
    displacement=np.asarray(after['bottle_position_m'])-before['bottle_position_m']
    error=float(np.linalg.norm(displacement-case['delta_m'])) if case['hold'] and 'yaw_rad' not in case and not case.get('expect_rejection') else None
    return {'id':case['id'],'status':row['status'],'commands':row['commands'],
        'tcp_position_error_mm':feedback.get('position_error_m',0)*1000 if row['status']=='passed' else None,
        'bottle_displacement_m':displacement.tolist(),'bottle_displacement_error_mm':None if error is None else error*1000,
        'feedback_corrections':len(feedback.get('plans',[])),
        'held_after':row['evaluation']['held'][arm],'failure':row['failure'],
        'meaning':'TCP goal and native contact only; no precision object-placement or long-hold claim'}

def main():
    rows=load('fitted/summary.json')+load('orientation/summary.json')
    values=[measures(r) for r in rows]
    fig,ax=plt.subplots(figsize=(8.6,3.9),layout='constrained');fig.patch.set_facecolor('#fcfaf6');ax.set_facecolor('#fcfaf6')
    plotted=[v for v in values if v['tcp_position_error_mm'] is not None];x=np.arange(len(plotted))
    a=ax.bar(x-.17,[v['tcp_position_error_mm'] for v in plotted],.32,color='#ae7457',label='末端位置误差')
    object_values=[np.nan if v['bottle_displacement_error_mm'] is None else v['bottle_displacement_error_mm'] for v in plotted]
    b=ax.bar(x+.17,object_values,.32,color='#6e8b78',label='瓶子位移偏差（诊断）')
    for bars in (a,b):
        for bar in bars:
            h=bar.get_height()
            if np.isfinite(h):ax.text(bar.get_x()+bar.get_width()/2,h+.13,f'{h:.2f}',ha='center',fontsize=10)
    ax.set_xticks(x,[NAMES[v['id']] for v in plotted],fontproperties=FONT)
    ax.set_ylabel('毫米',fontproperties=FONT);ax.set_ylim(0,6.4)
    ax.set_title('固定开发案例：末端误差与物体反馈分开记录',fontproperties=FONT,pad=13)
    ax.legend(prop=FONT,frameon=False,ncol=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for spine in ('top','right'):ax.spines[spine].set_visible(False)
    fig.savefig(OUT/'feedback.png',dpi=190);plt.close(fig)
    api=load('api.json');rotation=load('orientation/rotation-audit.json');migration=load('memory-migration.json')
    result={'engineering_cases':values,'groups':['fitted/protocol.json','orientation/protocol.json'],
        'not_a_generalization_success_rate':True,'original_groups_preserved':['native','debug','feasibility','constraints','pairs','pairs-fixed','instrumented','ready'],
        'startup_timeout_correction':'pairs/pairs-fixed/instrumented began before worker readiness; not planning failures; readiness barrier now required',
        'geometry':load('geometry-final/summary.json'),'rotation':rotation,
        'api':{'status':api['status'],'turn_id':api['turn_id'],'calls':api['actual_api_calls'],'wall_seconds':api['wall_seconds'],
            'sim_seconds':api['terminal']['observation']['sim_time'],'timing_mode':api['plan']['timing_mode'],
            'goal':api['plan']['goal_assessment'],'required_sequence_matched':api['required_sequence_matched'],
            'native_carried_motion_confirmed':api['backend_execution_evidence']['native_carried_motion_confirmed']},
        'memory_migration':migration,'hardware_access':False,'transfer_ready':False}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    trs=''.join(f"<tr><td>{NAMES[v['id']]}</td><td>{'通过' if v['status']=='passed' else '拒绝符合预期'}</td><td>{v['commands']}</td><td>{v['tcp_position_error_mm']:.2f} mm</td><td>{v['bottle_displacement_error_mm']:.2f} mm</td></tr>" if v['bottle_displacement_error_mm'] is not None else
        f"<tr><td>{NAMES[v['id']]}</td><td>{'通过' if v['status']=='passed' else '拒绝符合预期'}</td><td>{v['commands']}</td><td>{str(round(v['tcp_position_error_mm'],2))+' mm' if v['tcp_position_error_mm'] is not None else '—'}</td><td>—</td></tr>" for v in values)
    degrees=lambda value:f'{np.degrees(value):.2f}°'
    goal=api['plan']['goal_assessment'];memory_count=len(migration['migrated'])
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>cuRobo 持瓶运动与记忆兼容</title>
<style>body{{margin:0;background:#f7f5f0;color:#34322e;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1000px;margin:auto;padding:38px 24px}}h1{{font-size:30px;line-height:1.3}}h2{{font-size:21px;margin-top:0}}a{{color:#9f674d}}section{{background:#fffefa;border:1px solid #e4dfd5;border-radius:16px;padding:25px;margin:22px 0}}small,.muted{{color:#777369}}.stats{{display:flex;gap:24px;flex-wrap:wrap}}.stats strong{{font-size:28px;display:block;color:#9f674d}}img{{width:100%;border-radius:9px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 7px;border-bottom:1px solid #eee9df;text-align:left}}details{{margin:12px 0}}summary{{cursor:pointer}}.scroll{{overflow-x:auto}}footer{{font-size:13px;color:#777369}}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 固定开发实验 · 仅本机仿真</p><h1>cuRobo 持瓶运动与记忆兼容</h1><p>从实际失败定位碰撞近似，再验证真实模型的工具组合。原生物理、碰撞判据和放置精度门槛保持不变。</p>
<section><h2>真实 API 完成了后端组合</h2><p>右手抓取 → cuRobo 持瓶沿世界 +X 前移 3 cm → TAMP 放到任务起始瓶位前方 8 cm → 松手、撤离和复核。模型自行提交四个阶段，4/4 完成；目标与场景在任务中保持不变。</p>
<div class="stats"><div><strong>{api['actual_api_calls']}</strong>真实 API 请求</div><div><strong>{goal['position_error_m']*1000:.2f} mm</strong>最终瓶位误差 / 10 mm</div><div><strong>110</strong>cuRobo 原生控制指令</div><div><strong>{api['terminal']['observation']['sim_time']:.2f} s</strong>原生仿真时间</div></div>
<p>计时：规划时暂停，首个计划锁定 120 s；墙钟 {api['wall_seconds']:.2f} s。放置连续稳定支撑 {goal['stable_seconds']:.3f} s。携物 GPU 回执、完整原生路径复核与控制指令均有记录，没有用 TAMP 替代 cuRobo。</p><p>结果只证明本次已声明瓶子任务完成，不证明长时间持瓶稳定、开放物体泛化或真机迁移。</p><a href="api.json">逐次 API、工具、物理结果</a> · <a href="api.protocol.json">执行前协议</a><details><summary>查看实际工作台</summary><img src="platform.jpg" alt="真实 API 组合任务完成，四阶段与关键帧"></details></section>
<section><h2>运动案例与物体反馈</h2><img src="feedback.png" alt="末端位置误差与瓶子位移偏差分开记录"><div class="scroll"><table><thead><tr><th>案例</th><th>结果</th><th>控制指令</th><th>末端位置误差</th><th>瓶子位移偏差</th></tr></thead><tbody>{trs}</tbody></table></div>
<p>五项位置／拒绝案例与后来单独声明的转腕案例属于两份开发协议，不能合并为泛化成功率。末端成功门槛仍为 12 mm / 0.06 rad；瓶子位移偏差是独立诊断，并未作为精确物体放置成绩。空手移动不计为瓶子移动。</p>
<p>转腕要求 10°，实际末端 {degrees(rotation['actual_tcp_rotation_rad'])}，实际瓶子 {degrees(rotation['actual_bottle_rotation_rad'])}；瓶子相对理想刚性转动仍有 {degrees(rotation['bottle_rotation_error_from_rigid_rotation_rad'])} 差异。按已归档原生指令重放与终点位置一致，是软件可复现性检查，不是实物标定。</p>
<a href="fitted/summary.json">位置案例原始结果</a> · <a href="orientation/summary.json">转腕原始结果</a> · <a href="orientation/rotation-audit.json">瓶子朝向审计</a></section>
<section><h2>失败原因与边界</h2><p>首轮三个携瓶正向案例都在执行前被拒绝。旧夹指球每指 10 个，其中 8 个伸出 CAD 凸包；持瓶开口时产生约 0.835 mm 的虚假夹指相交。新方案只改 GPU 夹指近似：每指 84 / 90 个内接球，未修改原生机器人、接触参数、关节限位或碰撞忽略表。</p>
<p>内接球没有保守覆盖整个表面：两种夹指采样表面到球的最大距离仍为 10.12 / 7.10 mm。因此 GPU 结果只能提出候选，完整原生网格路径复核仍是必需步骤。附加瓶子采用 19 个球覆盖圆柱截面；它只用于规划，物理中的瓶子保持自由体，没有焊接或位姿覆盖。</p>
<p>部分诊断检查在工作进程尚未启动完成时就开始，得到连接超时。后续日志与堆栈确认它们未进入规划；不算成求解失败或 GPU 挂起。检查程序已增加就绪等待；原失败目录和更正均保留。</p>
<p>仅开放所选手空手或独立持声明瓶子的单臂运动。另一手持物、双手同时持物、任意物体视觉抓取尚未通过。cuRobo 负责位置与朝向路径；抓取、交接、放置仍由 TAMP 完成。</p>
<a href="native/summary.json">首轮失败</a> · <a href="geometry-final/summary.json">几何与未覆盖范围</a> · <a href="candidate/cancellation.json">超时判断更正</a></section>
<section><h2>记忆更少，也更可追溯</h2><p>当前页面是 5 条合并方法／失败教训与 1 条用户偏好，没有原始记忆切换入口；模型读取小索引，需要时才读取单项证据。实验原始回执继续归档，供重评分和审计。</p>
<p>物理兼容契约升为 openarm-memory/4，规划服务代码与物理条件分开。核对源码、数据、原生验证器及反馈后，迁移 {memory_count} 份兼容 TAMP 回执；缺少原生记录的 1 份拒绝回执不迁移。时间、原始反馈和运行 ID 不改；旧场景笔记、策略经验及不兼容物理版本不迁移。删除标记保留机制有专门回归检查。</p><a href="memory-migration.json">逐条迁移审计</a><details><summary>查看精炼记忆</summary><img src="memory.jpg" alt="关键方法与偏好，适用条件默认折叠"></details><details><summary>查看 cuRobo 技能分类</summary><img src="skills.jpg" alt="cuRobo 后端只显示运动技能"></details></section>
<section><h2>验收与未完成工作</h2><p>163 项回归检查通过，另有 25 项针对性检查。页面已实际核对技能分组、关键记忆、模型阶段与关键帧；布局保持左对话、右上场景、右下阶段。</p><p>持续夹持尚有滑动问题，含模型等待的 cuRobo 组合任务没有完成本轮验收。独立 TCP／基座、相机内外参与同步、夹爪宽度／力、接触材料及采集时控制器身份仍缺实测证据；ACT 完整任务未通过，π0.5 缺匹配 OpenArm 权重。real-to-sim-to-real 尚未通过。</p><a href="regression.log">163 项回归</a> · <a href="contracts-final.log">针对性检查</a> · <a href="audit.json">运行版本与部署审计</a></section>
<footer>设计参考官方 <a href="https://github.com/NVlabs/curobo/blob/main/curobo/_src/collision/attachment_manager.py">cuRobo AttachmentManager</a> 的携物表示，以及 <a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey</a> 的新观测、工具反馈和复查；不是完整复现。公司 SSH 与真机本轮均未访问。</footer></main></html>'''
    (OUT/'index.html').write_text(html)
    print(json.dumps({'results':'runs/20261008-carry/results.json','report':'runs/20261008-carry/index.html'},ensure_ascii=False))

if __name__=='__main__':main()
