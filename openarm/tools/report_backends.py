"""Render measured backend checks without converting diagnostics into scores."""
from pathlib import Path
import json
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261007-platform-upgrade'
read=lambda name:json.loads((OUT/(name+'.json')).read_text())
moves=read('motion-tests');strict=read('act-rollouts');clipped=read('act-rollouts-clipped')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
def move_label(result):
    delta=np.asarray(result['delta_m'])*100
    axis=int(np.argmax(np.abs(delta)))
    direction=('+' if delta[axis]>=0 else '-')+'XYZ'[axis]
    arm={'right':'R','left':'L'}[result['arm']]
    return f"{result['backend']}\n{arm} {direction} {abs(delta[axis]):g} cm"
labels=[move_label(r) for r in moves]
errors=[r['feedback']['position_error_m']*1000 for r in moves]
axes[0].bar(range(5),errors,color=['#b97859']*3+['#849579']*2,width=.65)
axes[0].set_xticks(range(5),labels,fontsize=8);axes[0].set_ylabel('Position error (mm)')
axes[0].set_title('Five native Cartesian checks');axes[0].set_ylim(0,1.2)
for i,v in enumerate(errors):axes[0].text(i,v+.025,f'{v:.2f}',ha='center')
axes[1].bar(range(4),[r['sim_seconds'] for r in strict+clipped],color=['#8a918f']*2+['#b97859']*2,width=.65)
axes[1].set_xticks(range(4),['Strict 0\nrange stop','Strict 1\nrange stop','Diagnostic 0\n900-step limit','Diagnostic 1\n900-step limit'],fontsize=8)
axes[1].set_ylabel('Simulated seconds');axes[1].set_title('ACT: task did not pass in any run')
fig.suptitle('Candidate simulation only; different checks, no benchmark or hardware transfer claim',fontsize=11)
fig.savefig(OUT/'results.png',dpi=180);plt.close(fig)
rows=''.join(f"<tr><td>{r['backend']}</td><td>{r['arm']}</td><td>{html.escape(str(r['delta_m']))}</td><td>{r['feedback']['position_error_m']*1000:.2f} mm</td><td>{r['status']}</td></tr>" for r in moves)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm 后端验收</title><style>
body{font:15px/1.7 system-ui,sans-serif;background:#f8f6f1;color:#363630;margin:0}main{max-width:1000px;margin:40px auto;padding:0 22px}h1{font-size:28px}h2{font-size:19px;margin-top:28px}section{background:white;border:1px solid #e6dfd4;border-radius:14px;padding:22px;margin:20px 0}a{color:#a46345}img{width:100%;border-radius:10px}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #eee8df;padding:9px}small{color:#73756e}code{background:#f6f2ec;padding:2px 5px}li{margin:7px 0}</style><main>
<small>OpenArm Lab · 2026-10-07 · 本机仿真，SSH 只读</small><h1>技能、记忆与后端验收</h1>
<p>保留左对话、右上视图、右下运行阶段。后端卡片可点击切换，只显示各自技能；方向键和 Home/End 可切换焦点。固定 TAMP 演示保留在 TAMP 面板。</p>
<section><h2>实际运行</h2><img src="results.png" alt="原生运动误差与 ACT 运行时长"><table><tr><th>后端</th><th>机械臂</th><th>位移 / m</th><th>末端误差</th><th>状态</th></tr>'''+rows+'''</table>
<p>cuRobo 使用官方 V2 GPU 求解；搜索使用稀疏候选几何，返回路径再经过原生 MuJoCo 离散碰撞复核。以执行器控制，未使用物体焊接或实时 qpos 跳转。空夹爪、单臂位姿能力已验证；携物规划与任意物体视觉抓取未开放。</p>
<p>ACT 使用只读复制并严格核验的真机权重、官方 LeRobot 和原归一化器，32 步预测、16 步执行、30 Hz。严格接口两次因夹爪越界停止；诊断对照才将夹爪裁到记录动作范围，均运行 900 步/30 s，仍未完成抓取。对照与真机部署不等价，正式接口没有放宽。π0.5 当前未配置兼容后端。</p>
<p>桌内目标及不可达目标都拒绝，0 次动作、实时状态不变。五个关节姿态、共十次 TCP 对比通过，只证明软件坐标一致。</p>
<p>保留的固定 TAMP 演示已在页面重新跑通抓取、交接、入篮；原生仿真评估确认瓶子在篮内、由篮子支撑并稳定超过 1 秒。这仍是候选场景下的仿真几何基线。<a href="tamp-demo.json">演示记录</a> · <a href="../20261007-232132-tamp-9e3e4c/summary.json">完整评估</a></p>
<a href="motion-tests.json">运动记录</a> · <a href="rejection-tests.json">拒绝记录</a> · <a href="act-rollouts.json">ACT 严格记录</a> · <a href="act-rollouts-clipped.json">限幅诊断</a> · <a href="curobo-fk.json">坐标一致性</a></section>
<section><h2>页面与记忆</h2><img src="skills.jpg" alt="可点击的后端技能窗口"><p>默认记忆呈现稳定偏好和关键经验；同类执行合并为一条，保留次数、最近代表结果与原始前后相机证据。原始行没有删除。不同失败原因、后端、参数、版本或证据来源分别保留。SQL 在检索数量限制前合并，160 条重复失败不会挤掉另一种关键失败。</p>
<p>两轮真实 deepseek-flash 调用 cuRobo，自行制定阶段、移动 1 cm、读取新反馈并提交 uncertain；记忆显示一条经验、重复计数 2。模型 API 记录不是成功率测评。</p>
<a href="live-curobo.json">第一轮 API</a> · <a href="live-curobo-repeat.json">第二轮及合并证据</a> · <a href="memory.jpg">记忆页面</a> · <a href="view-check.json">视角隔离与方向检查</a>
<p>自由视角向左、向上分别使 azimuth +10°、elevation +6°；操作者画面改变，而三路模型图像哈希、观测编号不变。主界面已移除重复的手动工具入口。</p></section>
<section><h2>复验与边界</h2><p>73 项相关回归检查通过。检查包括记忆去重、不同失败原因分离、角色与版本隔离、ACT/π0.5 契约不混用、cuRobo 坐标错误拒绝、原生移动与时序。<a href="checks.log">检查输出</a> · <a href="audit.json">最终审计</a></p>
<p><code>./run</code> 按需启动主页面与本机 ACT/cuRobo 服务；<code>./run stop</code> 一并停止。没有定时或开机启用。无硬件访问；所有真机标定、动力学和视觉域匹配结论仍需独立测量验证。</p>
<p>记忆借鉴 <a href="https://code.claude.com/docs/en/memory">Claude Code 的索引与按需明细</a>、<a href="https://github.com/LZY-1021/RoboHarness/blob/main/src/soma_memory.py">RoboHarness 的结果记忆库</a>；标签交互参考 <a href="https://www.radix-ui.com/primitives/docs/components/tabs">Radix Tabs</a>。这是结构化检索与设计借鉴，不宣称完整复现或向量 RAG。</p></section></main></html>'''
(OUT/'index.html').write_text(page)
print(OUT/'index.html')
