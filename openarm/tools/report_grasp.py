"""One concise report, preserving successful tasks and unstable-hold evidence."""
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261008-grasp'


def main():
    api=json.loads((OUT/'api-timing.json').read_text())
    mechanics=json.loads((OUT/'mechanics.json').read_text())
    candidates=json.loads((OUT/'candidates/summary.json').read_text())
    solver=json.loads((OUT/'solver-hold/summary.json').read_text())
    timing=[e['detail'] for turn in api['turns'] for e in turn['trace'] if e['kind']=='timing']
    errors=[e for turn in api['turns'] for e in turn['trace'] if e['kind']=='tool' and e['status']=='error']
    samples=[s for record in mechanics['records'] for s in record['samples'] if s['held']['right'] and record['kind']=='wait']
    with tempfile.TemporaryDirectory(prefix='openarm-grasp-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained',facecolor='#fffefa')
        start=samples[0]
        axes[0].plot([s['sim_seconds']-start['sim_seconds'] for s in samples],
                     [1000*np.linalg.norm(np.subtract(s['relative_m'],start['relative_m'])) for s in samples],color='#ad785c')
        axes[0].set(title='Archived API / relative grip drift',xlabel='Native hold time (s)',ylabel='Object-in-hand change (mm)')
        axes[1].bar([str(int(c['case']['height_m']*1000)) if c['case']['id']!='production' else 'Current' for c in candidates],
                    [1000*c['max_two_second_relative_drift_m'] for c in candidates],color=['#68796d']+['#bca792']*5)
        axes[1].axhline(1,color='#ad785c',ls='--',lw=1)
        axes[1].set(title='Grasp candidates / all unstable',xlabel='TCP height above bottle origin (mm)',ylabel='Max drift over 2 seconds (mm)')
        bars=axes[2].bar([str(s['noslip_iterations']) for s in solver],
                         [1000*s['relative_drift_m'] for s in solver],color=['#68796d']+['#bca792']*3)
        axes[2].bar_label(bars,fmt='%.2f',padding=3)
        axes[2].set(title='Solver study / NOT APPLIED',xlabel='NoSlip iterations (same initial hold)',ylabel='Drift over 10 seconds (mm)')
        for ax in axes:
            ax.set_facecolor('#fffefa');ax.grid(axis='y',color='#e5dfd4',lw=.5);ax.set_axisbelow(True)
        fig.savefig(OUT/'feedback.png',dpi=180);plt.close(fig)
    style=re.search(r'<style>(.*?)</style>',(ROOT/'runs/20261008-recovery/index.html').read_text(),re.S)[1]
    scope=api['plan']['goal_assessment'];seconds=api['terminal']['observation']['sim_time']
    waits=sum(t['integrated_seconds'] for t in timing)
    rows=''.join(f'<tr><td>{html.escape(c["case"]["id"])}</td><td>{c["max_two_second_relative_drift_m"]*1000:.2f} mm</td><td>未通过</td><td><a href="candidates/{c["case"]["id"]}.json">原始力与位姿 ↗</a></td></tr>' for c in candidates)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 夹持与反馈</title><style>{style}</style><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.08</div><h1>任务完成，夹持缺陷仍保留</h1><p>把动作结果、连续稳定性和记忆证据分别核验。</p></div><a class="pill" href="/">返回工作台 ↗</a></header>
<div class="stats"><div><b>159</b><span>回归检查通过</span></div><div><b>{api['actual_api_calls']}</b><span>真实 API 请求</span></div><div><b>{scope['position_error_m']*1000:.2f} mm</b><span>前放 8 cm 的物体误差</span></div><div><b>未验证</b><span>物理标定与真机迁移</span></div></div>
<section><h2>这次实际 API 完成了什么</h2><p>右手抓瓶 → 起始瓶位前方 8 cm 桌面 → 松手与撤离 → 新相机复核。模型等待期间物理持续运行，7 段等待累计 {waits:.3f} s，原生时钟最终 {seconds:.3f} s。首个计划锁定 120 秒；目标内没有重置、加时或切回暂停。{len(errors)} 次 place 提案因场景漂移被拒绝；模型收到具体漂移反馈后，用新观测重新提出动作。拒绝没有启动运动。</p><p>最终物体误差 {scope['position_error_m']*1000:.3f} mm / 原有 10 mm 容差，连续支撑 {scope['stable_seconds']:.3f} s；3/3 模型阶段完成。原来的碰撞、夹持、放置与 3 mm 决策漂移门槛保留。这是一条开发任务的完成，不是成功率，也不能推断“任何动作”。</p><a href="platform.jpg"><img src="platform.jpg" alt="真实 API 完成前放 8 cm，左侧调用记录与右侧三个阶段"></a><div class="links"><a href="api-timing.json">全部 API 与动作回执 ↗</a><a href="api-timing.protocol.json">执行前协议 ↗</a><a href="../20261008-decision/api-first-failure.json">上一轮失败记录 ↗</a></div><p class="small">两次 API 的响应时长、工具选择及记忆输入不同，因此本页不把改善归因于某一项修改。计时声明之前的规划仍暂停，低层规划耗时、曝光、CAN 与硬件看门狗未实现等价计时。页面曾短暂显示连接中断，随后自动恢复，未重启或重放动作；浏览器错误日志为空。</p></section>
<section><h2>为什么接触还不等于稳定夹持</h2><p>对上一轮实际动作做原生重放，读取已求解的法向／切向接触力：两侧法向力约 13 N，夹爪执行器约 26–27 N，远未到候选 333 N 上限，竖直合力约 5.40 N。瓶子相对夹爪仍缓慢滑动，因此不通过加力、加摩擦或降低精度来处理。333 N 仍不是经过实物测量的参数。</p><a href="feedback.png"><img src="feedback.png" alt="相对夹持漂移、全部抓取候选失败以及未应用的求解器敏感性对比"></a><p>新增 grasp-stability/1 只读诊断：按 2 秒窗口审计相对位姿（1 mm / 0.02 rad）。区分时间不足、稳定窗口、滑动和接触中断。本次实际 API 在持瓶等待中也记录了滑动／短暂接触中断；新诊断没有把旧接触抬升评分改写成“持续夹持通过”，也没有把整条已完成放置任务改写成未完成。</p><p>五个新抓取高度及原始位置，均未通过事先声明的 1 mm / 2 秒检查。候选全开夹爪使用已有 TAMP 原生范围，闭合仍为 +0.05；不扩大学习策略范围。全部保留，未部署替代抓取。</p><div class="scroll"><table><thead><tr><th>候选</th><th>最大窗口漂移</th><th>结果</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table></div><div class="links"><a href="mechanics.json">实际指令重放接触力 ↗</a><a href="candidates/protocol.json">抓取对照协议 ↗</a><a href="solver-hold/protocol.json">同一夹持初态的数值对照 ↗</a><a href="solver-hold/summary.json">数值对照结果 ↗</a><a href="solver/summary.json">抓取前改变求解器的补充对照 ↗</a></div><p>仅在隔离实验中改变 NoSlip 迭代数，其他参数和起始夹持状态保持一致，滑动量对求解器设置有响应。它支持“数值接触模型会影响滑动”的判断，不证明真机应该无滑动，也不提供经过物理标定的设置；没有自动应用。<a href="https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip">MuJoCo 官方说明</a>也指出软接触可在摩擦锥内产生持续慢滑，NoSlip 不保证零滑且有逆动力学与稳定性代价。</p></section>
<section><h2>反馈与记忆：少重复，保留边界</h2><p>工具格式错误、观测重新定位、接口约束分别使用有界重试计数，实际物理动作失败仍按原控制与恢复规则处理；32 次模型调用与每轮 8 次请求的总预算保留。ObservationChanged 返回 source/current 观测编号、瓶子与 TCP 漂移及原门槛，模型不再只收到一条失败字符串。相同被拒参数仍禁止重复。</p><p>默认记忆是按需索引，当前界面为 5 条方法／失败教训和 1 条偏好。方法按技能、机械臂、交接方向与失败原因合并，详情列出计时条件及究竟验证了什么。原始实验回执继续保存在档案，默认索引和记忆面板都不重复显示它们；需要审计一条动作才显式 get_execution_record。抓取方法明确不等同于长期稳定夹持。</p><a href="memory.jpg"><img src="memory.jpg" alt="精炼记忆详情：适用范围与实验依据，不重复显示原始轨迹"></a><p class="small">截图是在任务进行时打开记忆后的状态；当前新增的右手抓取与放置记录已归入相应方法。历史时间暂停与含等待条件分列支持次数；旧物理契约继续隔离。参考 <a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey agent.ts</a> 的观测、调用、反馈和失败边界，没有宣称完整复现其性能。</p></section>
<section><h2>下一项验收</h2><p>先建立可复现的持续夹持测试和接触数值收敛报告，再依据真实夹爪／物体测量判断物理参数；不能直接用某个数值对照的更小漂移替换生产参数。当前瓶子候选几何、摩擦、质量、TCP、相机与控制时序仍缺独立标定。cuRobo 仅空手运动，ACT 完整任务和兼容 OpenArm π0.5 仍未通过。本轮没有访问或修改远程公司电脑，没有硬件运动或定时启动。</p><div class="links"><a href="checks.log">159 项回归检查 ↗</a><a href="contracts.log">19 项最终反馈与记忆检查 ↗</a><a href="audit.json">版本与证据审计 ↗</a></div></section></main></html>'''
    (OUT/'index.html').write_text(page)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        ROOT/'sim/decision_timing.py',ROOT/'harness/failures.py',ROOT/'harness/agent.py',ROOT/'harness/workspace.py',
        ROOT/'harness/knowledge.py',ROOT/'tools/check_grasp.py',ROOT/'tools/check_contact_solver.py']}
    audit={'goal_status':'active','checks_passed':159,'API_calls':api['actual_api_calls'],'API_task_status':api['status'],
           'API_root':api['turn_id'],'object_error_m':scope['position_error_m'],'native_clock_seconds':seconds,
           'decision_wait_seconds':waits,'rejected_proposals':len(errors),'geometry_candidates_applied':False,
           'solver_candidates_applied':False,'physical_parameters_changed':False,'default_timing_mode':'paused',
           'grasp_stability':'diagnostic only, no sustained-grasp pass claimed',
           'hardware_access':False,'transfer_ready':False,'physical_calibration_verified':False,'hashes':hashes}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
