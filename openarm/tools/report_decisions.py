"""Visualize the failed real API trial and independent native delay controls."""
import html
import json
import os
import re
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261008-decision'


def main():
    api=json.loads((OUT/'api-first-failure.json').read_text())
    replay=json.loads((OUT/'replay.json').read_text())
    native=json.loads((OUT/'native/summary.json').read_text())
    waits=[r for r in replay['records'] if r['kind']=='wait'][1:]
    with tempfile.TemporaryDirectory(prefix='openarm-decision-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained',facecolor='#fffefa')
        x=np.arange(len(waits));width=.25
        for offset,field,label,color in [(-width,'tcp','Right TCP','#68796d'),(0,'body','Bottle in world','#b78361'),(width,'relative','Bottle relative to gripper','#897d73')]:
            values=[(r['tcp_drift_m']['right'] if field=='tcp' else r['object_drift_m'] if field=='body' else r['grasp_relative_drift_m']['right'])*1000 for r in waits]
            axes[0].bar(x+offset,values,width,label=label,color=color)
        axes[0].axhline(3,color='#b78361',lw=1,ls='--',label='World object gate: 3 mm')
        axes[0].set_xticks(x,[f'W{i+2}\n{r["sim_seconds"]:.2f} s' for i,r in enumerate(waits)])
        axes[0].set(title='Real API / native holds after grasp',ylabel='Position change (mm)')
        axes[0].legend(frameon=False,fontsize=8)
        values=[r['position_error_m']*1000 for r in native]
        bars=axes[1].bar(['Paused','2 s / wait','5 s / wait'],values,color=['#68796d','#b78361','#897d73'],width=.55)
        axes[1].bar_label(bars,fmt='%.2f mm',padding=4)
        axes[1].set(title='Paired native placement controls',ylabel='Final XY error (mm)',ylim=(0,max(values)*1.5+1))
        for ax in axes:
            ax.set_facecolor('#fffefa');ax.grid(axis='y',color='#e5dfd4',lw=.5);ax.set_axisbelow(True)
        fig.savefig(OUT/'feedback.png',dpi=180);plt.close(fig)
    style=re.search(r'<style>(.*?)</style>',(ROOT/'runs/20261008-recovery/index.html').read_text(),re.S)[1]
    rows=''.join(f'<tr><td>{html.escape(r["condition"]["id"])}</td><td>{r["status"]}</td><td>{r["sim_seconds"]:.2f} s</td><td>{r["position_error_m"]*1000:.2f} mm</td><td><a href="native/{r["condition"]["id"]}/summary.json">全部回执 ↗</a></td></tr>' for r in native)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 模型等待验收</title><style>{style}</style><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.08</div><h1>模型在等待，物理继续运行</h1><p>把以前冻结的夹持状态放回时间轴，保留真实失败。</p></div><a class="pill" href="/">返回工作台 ↗</a></header>
<div class="stats"><div><b>156</b><span>回归检查通过</span></div><div><b>{api['actual_api_calls']}</b><span>本轮真实 API 请求</span></div><div><b>{api['waiting_seconds']:.2f} s</b><span>实际积分的模型等待</span></div><div><b>未完成</b><span>真实 API 的前放任务</span></div></div>
<section><h2>新增等待计时</h2><p>首个 set_plan 可声明 timing_mode=measured_delay；之后每次 API 请求期间，唯一物理线程持续保持上一条执行器目标，按原生 1 ms 物理步积分。返回的新指令不会提前应用。原来的 paused、固定演示和 VLA 独立计时契约保留。</p><p>计时方式和预算在执行前锁定。动作／完成判断需要匹配本次模型请求的观测；小漂移通过后用当前观测调度并保留原始观测编号，明显漂移拒绝。等待期间的停止、超时和预算到期都不会应用迟到的动作。</p><p class="small">首个计划声明之前的规划仍暂停，不计入新窗口；低层规划计算、真实相机曝光、CAN、驱动看门狗时序还没有等价实现。该模式不是完整真机实时性证明，也尚未设为默认。</p></section>
<section><h2>真实 API 未完成：抓取后发生漂移</h2><p>120 秒预算、目标为任务起始瓶位前方 8 cm；整个目标未重置。抓取原语通过。随后一次工具参数格式错误、两次场景漂移拒绝触发失败预算，放置未开始。六段模型等待累计 {api['waiting_seconds']:.3f} s；没有把拒绝记录成动作成功。</p><a href="feedback.png"><img src="feedback.png" alt="实际 API 等待期间的瓶子、夹爪位移及三组原生对照结果"></a><p>实际指令序列在同一候选模型中逐步重放，终点残差为 {replay['reproduction_residual_m']:.1e} m。抓取后的较长等待中，右手 TCP 仅漂移约 0.2–0.5 mm，瓶子相对夹爪漂移约 3.6–6.3 mm，并向下移动。说明夹持相对位姿不稳定；原来的暂停模式会隐藏这段变化。软件重放一致不等于物理标定。</p><div class="links"><a href="api-first-failure.json">完整 API 与失败记录 ↗</a><a href="api-timing.protocol.json">事先声明的目标 ↗</a><a href="replay.json">指令重放与相对位姿 ↗</a></div></section>
<section><h2>原生对照：三个条件分别保留</h2><p>相同初态、目标、原生控制、120 秒预算；抓取前、抓取／放置之间和放置后分别插入 0、2 或 5 秒的目标保持。三组均完成前放 8 cm。它们是注入延迟的工程对照，不是真实 API 成绩，不是开放泛化成功率，也不能抵消上面的模型失败。</p><div class="scroll"><table><thead><tr><th>条件</th><th>结果</th><th>仿真用时</th><th>物体误差</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table></div><p><a href="native/protocol.json">开始前固定全部条件 ↗</a></p></section>
<section><h2>页面与反馈</h2><a href="platform-failure.jpg"><img src="platform-failure.jpg" alt="真实 API 未完成：已抓取，放置待执行，复查失败"></a><p>保留左侧对话、右上相机、右下阶段。只增加“含模型等待”标识和简短状态；详细原生控制、时延和漂移证据留在实验档案。停止按钮支持模型处理阶段，迟到动作的丢弃通过受控协议测试。</p><p>旧 API 验收轮询曾只等待 finish；模型已终止错误却没有正式 finish 时会空等。检查器已改为识别“无运行任务、模型空闲、最新相关回合已终止错误”的组合。此次只停止了已确认无后续工作的验收轮询进程，未借此重置机器人。</p></section>
<section><h2>还需要解决的具体问题</h2><p>先增强抓取后的持续稳定性，验证夹爪内相对位姿和载荷，而不是只看一瞬间的双指接触。保持几何、质量、摩擦、碰撞及精度条件，重新测同一等待流程；同时区分格式错误、观测重新定位和实际动作失败。尚未通过的模式不替换默认模式。</p><p>本轮没有完成开放物体、带物 cuRobo、ACT 完整任务、OpenArm π0.5、实物标定或迁移验收。FK 与重放的吻合仅为软件证据。远程公司电脑未修改，未运行硬件，未创建定时启动。</p><details><summary>检查与设计来源</summary><div class="links"><a href="checks-final.log">156 项检查 ↗</a><a href="contracts.log">更新后的接口检查 ↗</a><a href="audit.json">审计 ↗</a></div><p><a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey agent.ts</a> 的新观测、工具、失败分类与后续验证作为设计参考；<a href="https://mujoco.readthedocs.io/en/stable/programming/simulation.html">MuJoCo 原生积分与控制说明</a> 用于保持控制与物理线程的契约。不是对应项目完整复现。</p></details></section></main></html>'''
    (OUT/'index.html').write_text(page)
    audit={'goal_status':'active','checks_passed':156,'actual_api_calls':api['actual_api_calls'],'api_task_passed':False,
           'native_controls':[{'id':r['condition']['id'],'status':r['status'],'position_error_m':r['position_error_m']} for r in native],
           'reproduction_residual_m':replay['reproduction_residual_m'],'wait_seconds':api['waiting_seconds'],
           'physical_calibration_verified':False,'transfer_ready':False,'hardware_access':False,
           'mode_scope':'after first accepted plan only; earlier planning paused; generic TAMP only',
           'next_issue':'sustained grasp-relative stability, no relaxed drift gate'}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
