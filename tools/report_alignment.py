"""Publish negative grasp evidence and actual API outcomes without rescoring."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-alignment'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
load=lambda path:json.loads((OUT/path).read_text())

def label(case):
    if case.get('baseline'):return '原抓取方案'
    text=f"h={round(case['height_m']*1000)} mm"
    for key,name in (('yaw_deg','偏航'),('roll_deg','滚转'),('pitch_deg','俯仰')):
        if case.get(key):text+=f" · {name}{case[key]:+g}°"
    if not any(case.get(k) for k in ('yaw_deg','roll_deg','pitch_deg')):text+=' · 朝向修正'
    if case['position_tolerance_m']==.003:text+=' · 3 mm到位'
    return text

def main():
    original=load('rescored.json')['cases'];roll=load('roll/summary.json')
    fig,axes=plt.subplots(1,2,figsize=(11.7,8.2),layout='constrained');fig.patch.set_facecolor('#faf8f3')
    for ax,rows,title in zip(axes,(original,roll),('位置、朝向反馈与偏航 · 14项','滚转与俯仰 · 17项')):
        ax.set_facecolor('#faf8f3');ax.set_xscale('log');ax.set_xlim(.65,60)
        for i,row in enumerate(rows):
            if row.get('grasp_stability'):
                right=row['grasp_stability']['arms']['right'];value=right['max_window_relative_drift_m']*1000
                ax.barh(i,value,color='#b07758',height=.62);ax.text(value*1.08,i,f'{value:.2f}',va='center',fontsize=8)
            else:ax.text(1.12,i,'未完成抓取',fontproperties=FONT,fontsize=8,color='#777166',va='center')
        ax.axvline(1,color='#6f8677',ls='--',lw=1.2)
        ax.set_yticks(range(len(rows)),[label(r['case']) for r in rows],fontproperties=FONT,fontsize=9)
        ax.invert_yaxis();ax.set_title(title,fontproperties=FONT,pad=15);ax.set_xlabel('2秒窗口内相对位置最大范围 / mm',fontproperties=FONT)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for spine in ('top','right'):ax.spines[spine].set_visible(False)
    fig.savefig(OUT/'grasp.png',dpi=180);plt.close(fig)
    assert len(original)==14 and len(roll)==17 and not any(r['status']=='stable' for r in original+roll)
    reports=[(name,load(name+'.json')) for name in ('api','api-final','api-terminal')]
    rows=''.join(f"<tr><td>{'初次等待验收' if i==0 else '报告通道修复后' if i==1 else '终止流程修复后'}</td><td>{r['status']}</td><td>{r['actual_api_calls']}</td><td>{r['plan'].get('verdict') or '未提交'}</td><td>{'已执行持瓶路径' if (r.get('backend_execution_evidence') or {}).get('native_carried_motion_confirmed') else '未确认执行'}</td><td><a href='{name}.json'>原始记录</a></td></tr>" for i,(name,r) in enumerate(reports))
    data={'grasp_batches':[{'protocol':'proposals/protocol.json','evaluated':14,'stable':0,'diagnostic_version':'original v1 preserved; v2 replay audit available'},
        {'protocol':'roll/protocol.json','evaluated':17,'stable':0,'diagnostic_version':'v2'}],
        'not_independent_generalization_trials':True,'geometry_candidates_deployed':False,
        'API':[{'file':name+'.json','status':r['status'],'calls':r['actual_api_calls'],'goal':r['plan'].get('goal_assessment'),
            'model_verdict':r['plan'].get('verdict'),'carried_motion':bool((r.get('backend_execution_evidence') or {}).get('native_carried_motion_confirmed'))} for name,r in reports],
        'native_physics_changed':False,'hardware_access':False,'transfer_ready':False}
    (OUT/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>持续夹持与失败报告</title>
<style>body{{margin:0;background:#f7f5f0;color:#34322e;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1060px;margin:auto;padding:36px 24px}}h1{{font-size:31px;line-height:1.3}}h2{{font-size:21px;margin-top:0}}section{{background:#fffefa;border:1px solid #e4dfd5;border-radius:16px;padding:25px;margin:22px 0}}a{{color:#9f674d}}img{{width:100%;border-radius:9px}}.muted,footer{{color:#777166}}details{{margin:12px 0}}summary{{cursor:pointer}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 7px;text-align:left;border-bottom:1px solid #eee9df}}footer{{font-size:13px}}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 负面实验也保留 · 当前候选物理模型</p><h1>持续夹持仍未通过，失败报告更可靠</h1><p>本轮没有用调整摩擦、夹持力、NoSlip 或成功门槛来换取通过。抓取候选、真实模型等待与报告通道分别验收。</p>
<section><h2>两个抓取几何批次都未通过</h2><p>首批 14 项比较抓取高度、朝向反馈和偏航；第二批 17 项比较夹爪滚转／俯仰。均为 episode 0、seed 0 的固定开发条件，两个批次重复原方案作对照，不是 31 份独立泛化测试。</p><img src="grasp.png" alt="两个批次的夹持滑动范围，绿色虚线是1毫米门槛"><p>抓取后保持原执行器目标 10 秒，要求每个 2 秒窗口的物体相对夹爪位置范围 ≤1 mm、旋转范围 ≤0.02 rad，且不丢失持续双指接触。两个批次各自均为 0 项通过。受阻候选可能已执行开爪或接近动作，不能把整段尝试称为零动作。</p>
<p>仅强化朝向可降低某些候选的滑动，仍达不到原标准；较大滚转被完整网格碰撞或逆解约束拒绝。没有候选替换生产抓取。</p><a href="proposals/protocol.json">14项原始协议</a> · <a href="proposals/summary.json">原始结果</a> · <a href="roll/protocol.json">17项协议</a> · <a href="roll/summary.json">滚转／俯仰结果</a></section>
<section><h2>稳定性诊断修正</h2><p>旧版本只比较窗口首尾，可能漏掉滑出后返回；浮点边界也可能让某些窗口未被计算。新版 grasp-stability/2 比较窗口内全部采样位姿的最大位置与旋转范围，并保留边界样本。窗口范围实际跨度也有记录，阈值没有放宽。</p><p>对首批已完成抓取的候选，用归档原生指令重新积分，并逐控制步核对机器人状态与瓶子位置，重现误差均小于 1e-9。原始结果保留，新诊断另存；这是软件重现证据，不能代替实物标定。</p><a href="rescored.json">新版诊断与原判据并列</a></section>
<section><h2>真实 API 加入等待后的验收</h2><div class="scroll"><table><thead><tr><th>运行</th><th>任务</th><th>API请求</th><th>模型报告</th><th>cuRobo携物</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table></div><p>各次均要求右手抓取 → cuRobo 沿 world +X 移动 3 cm → TAMP 放到起始瓶位前方 8 cm → 松手撤离。首个计划锁定 measured_delay 与 120秒预算；任务内不重置、不加时、不切回暂停，也不替换 cuRobo。不同模型输出、速度选择和请求延迟分别记录，不能据此做受控增益比较。</p><p>初次运行两次携物请求因场景漂移被拒绝，未执行 cuRobo 和放置。随后一次完成抓取与实际 cuRobo 携物，但模型等待期间脱手，瓶子跌到工作面以下，物理回合失效。单段工具成功不能代替完整任务成功。</p><p>本轮修复了报告通道：同一任务的旧证据只可提交失败／不确定并明确标记历史来源。未完成的计划只有真实执行失败、等待失效或至少两次经日志验证的物理漂移拒绝，才能提交非成功结论；缺证据不能提前结束，成功与运动仍要求新观测。跨场景和未来观测拒绝。</p><p>finish 接受后立即结束模型循环，避免额外 API 请求与额外等待；最后一个调用上已提交结果也不会再误标为调用预算耗尽。受控提供者回归验证此终止语义，真实 API 结果单独列出。</p><details><summary>查看最终工作台</summary><img src="platform.jpg" alt="保持原布局，模型阶段和真实失败报告"></details></section>
<section><h2>已部署与仍缺的证据</h2><p>最终回归与运行版本见 <a href="audit.json">部署审计</a>，全部原始检查日志保留。页面仍是左对话、右上场景、右下模型阶段；记忆继续只呈现关键方法与偏好，旧回执归档按需读取。</p><p>持续夹持尚未通过，连续模型等待下的组合任务尚无可靠泛化结论。独立 TCP／基座和相机标定、真实夹爪宽度／力、瓶子尺寸质量及接触材料仍缺实测证据。ACT 完整仿真任务和匹配 OpenArm π0.5 也未完成，real-to-sim-to-real 目标保持进行中。</p><a href="regression-complete.log">168项最终回归</a> · <a href="report-checks-final.log">报告通道针对性检查</a> · <a href="api-terminal.protocol.json">最终真实API协议</a></section>
<footer>方法参考 <a href="https://github.com/NVlabs/GraspDataGen/blob/main/docs/components/grasp-sim.md">GraspDataGen 的物理抓取验证</a>；本轮未复现其扰动力测试，也没有调用训练好的 GraspGen。<a href="https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip">MuJoCo 文档</a>说明软接触可能产生慢滑移；数值求解器敏感性不等于实物接触标定。本轮未访问 SSH 或操作真机。</footer></main></html>'''
    (OUT/'index.html').write_text(html)
    print('runs/20261008-alignment/index.html')

if __name__=='__main__':main()
