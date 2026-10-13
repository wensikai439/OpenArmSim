"""A concise native-hold evidence page; all failures remain visible."""
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-hold'


def read(name):return json.loads((OUT/name).read_text())


def main():
    targets=read('summary.json');solvers=read('solver/summary.json');long=read('long/result.json')
    series=read('long/series.json');review=read('api-review.json')
    drift=lambda r:r['grasp_stability']['arms']['right']['max_window_relative_drift_m']*1000
    with tempfile.TemporaryDirectory(prefix='openarm-hold-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,
                             'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
        for ax,labels,values,title in ((axes[0],['0.05\n原闭合','0.07\n原生最大','0.03\n较小闭合'],[drift(r) for r in targets],'同一抓取后的闭合目标'),
            (axes[1],['0','1','20'],[drift(r) for r in solvers],'NoSlip 隔离数值候选')):
            ax.bar(labels,values,color=['#b18467','#718b83','#b1a394'],width=.6)
            ax.axhline(1,color='#926e50',ls='--',lw=1,label='原标准1mm')
            ax.set_ylim(0,7.5);ax.set_ylabel('最大2秒窗口位置范围 / mm');ax.set_title(title,loc='left')
            for i,v in enumerate(values):ax.text(i,v+.15,f'{v:.2f}',ha='center')
        axes[1].set_xlabel('NoSlip迭代数 · 未部署')
        complete=[r for r in series if r['elapsed_s']>=2-1e-6]
        x=[r['elapsed_s'] for r in complete];y=[r['recent_position_range_m']*1000 for r in complete]
        axes[2].plot(x,y,color='#718b83',lw=1.2);axes[2].axhline(1,color='#926e50',ls='--',lw=1)
        axes[2].set_yscale('log');axes[2].set_ylim(.1,1000);axes[2].set_xlabel('保持原目标 / 秒')
        axes[2].set_ylabel('当前2秒窗口位置范围 / mm · 对数');axes[2].set_title('长保持：51.17秒物理失效',loc='left')
        axes[2].axvline(long['actual_seconds'],color='#b18467',lw=1,ls=':')
        fig.savefig(OUT/'hold.png',dpi=180);plt.close(fig)
    rows=''.join(f'<tr><td>{r["motor_target_rad"]:.2f} rad</td><td>{drift(r):.2f} mm</td><td>{r["grasp_stability"]["arms"]["right"]["status"]}</td><td><a href="{r["case"]}/result.json">回执</a></td></tr>' for r in targets)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹持反馈与失稳</title>
<style>body{margin:0;background:#f8f7f3;color:#332f28;font:15px/1.7 system-ui,sans-serif}main{max-width:1080px;margin:32px auto;padding:0 24px}h1{font-size:30px;line-height:1.35}h2{font-size:19px}section{background:white;border:1px solid #e7e1d7;border-radius:15px;padding:24px;margin:18px 0}img{width:100%;height:auto}.muted,small{color:#797367}a{color:#806146;text-underline-offset:3px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #eee8df}.scroll{overflow:auto}code{font-size:13px;background:#f7f5ef;padding:2px 4px}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 原生夹持与只读反馈</p><h1>接触、抬升与稳定夹持分别验证</h1><p>前一轮连续搬运已完成，但持续夹持仍未通过。本轮用同一原生抓取状态比较闭合目标和数值候选，保留全部失败；增加模型可以读取的实时夹持质量，主工作台布局保持不变。</p>
<section><h2>受控保持测试</h2><img src="hold.png" alt="闭合目标、NoSlip候选和原生长保持的位置范围"><div class="scroll"><table><thead><tr><th>闭合目标</th><th>最大2秒范围</th><th>原保持结论</th><th>证据</th></tr></thead><tbody>'''+rows+'''</tbody></table></div><p>每组在相同qpos、qvel和执行器目标的原生抓取后开始，保持10秒。原标准：每个2秒窗口位置范围≤1mm、旋转范围≤0.02rad，且连续双指接触不中断。0.07属于现有TAMP原生指关节范围；记录／VLA的最大指令仍为0.05，没有修改其范围。电机目标按300ms斜坡改变，审计从斜坡之前开始。</p><p>三个闭合目标都失败。最大闭合仅将接触法向力中位数从约13.14N增加到13.71N，位置范围从6.26mm降到5.91mm，仍未过门槛。没有用这些结果替换生产抓取。</p><p>NoSlip的0／1／20迭代对照只在隔离模型、相同抓取之后改变数值选项，最大窗口分别6.26／5.70／4.89mm，全部失败。普通生产记录仍会拒绝参数改变；这些结果具有明确研究标签，不产生正常技能成功回执。</p><a href="protocol.json">闭合协议</a> · <a href="summary.json">全部闭合结果</a> · <a href="solver/protocol.json">数值对照协议</a> · <a href="solver/summary.json">全部数值结果</a></section>
<section><h2>延长等待也未获得稳定阶段</h2><p>声明保持60秒，固定原0.05目标、当前默认物理。约51.17秒时瓶子滑落并触发物理无效，停止于原生终止条件。此前没有满足同一位置／朝向／连续接触标准的完整近期窗口；不能把等待更久或末端停止运动当作稳定夹持。</p><p>整段原始保持仍判定失败，没有截取较好的窗口重写整段结论。滑动在摩擦锥内和接触持续时也存在，表明提高夹爪目标不能独自解决问题；几何、软接触和控制器仍需进一步区分与独立物理标定。</p><a href="long/protocol.json">长保持协议</a> · <a href="long/result.json">终止与原评分</a> · <a href="long/series.json">完整时序</a></section>
<section><h2>反馈已接入Harness</h2><p><code>inspect_manipulation.grasp_quality</code>现在返回每只手的最近2秒相对位姿范围、连续双指接触时长和证据段身份。只有物理线程推进后才采样，重复观察不累计时长。重置清除历史，新任务保留相同物理场景的历史；接触中断后重新积累证据。</p><p>模型可区分未持握、时长不足、正在滑动和近期窗口稳定。最后一项只覆盖已测窗口，不证明长期夹持或真机迁移。接口明确标记为仿真oracle，读取会沿用已有的学习策略隔离。VLA仍只有原图像与本体状态，任务评分、动作、物理参数及模型固定相机没有改动；这些诊断默认不堆到操作者页面。</p><p>187项回归通过，包含6项新反馈检查：接触与时长不足、滑出再回来的位姿范围、旋转、区间内接触中断、重复观察、实际原生滑动及任务保持／重置边界。</p><a href="feedback-checks.log">6项针对检查</a> · <a href="regression.log">187项回归</a> · <a href="audit.json">版本与部署</a></section>
<section><h2>实际API读取与记忆保留</h2><p>'''+f'原生准备一次右手抓取后，实际{review["actual_api_calls"]}次API完成只读质量检查，接口验收{review["status"]}。模型收到实时质量字段，检查过程中没有运动工具、重置或状态推进。这是反馈送达和只读行为验收，不是独立的模型判断准确率或夹持通过。'+'''</p><p>被动反馈更新触发了环境代码版本隔离。对上一轮4份完整冻结模型回执重新执行共3944条归档控制指令，逐步机器人状态、每项原生评分和终止标志一致，瓶子终点误差为零；在只允许这次已审查环境代码差异、模型／参数完全相同的条件下，恢复这4份回执的兼容。当前仍为3条路线方法、1条偏好、共同规则1份；原始反馈／时间和用户遗忘规则保留。没有迁移学习策略、场景笔记或缺少模型快照的旧档案。</p><p>首次迁移检查误读嵌套指纹字段，失败日志保留；修正字段读取后完成轨迹验证再写库。工作区数据库备份位于私有运行目录，不通过报告静态接口公开。这些一致性是软件轨迹证据，不是实物标定。</p><a href="api-review.json">完整原始请求过程与反馈</a> · <a href="before-review.json">检查前原生状态</a> · <a href="memory/proofs.json">3944步一致性证据</a> · <a href="memory/migration.json">受限迁移审计</a></section>
<small>参考<a href="https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip">MuJoCo官方接触滑动诊断</a>和<a href="https://github.com/NVlabs/GraspDataGen/blob/main/docs/components/grasp-sim.md">GraspDataGen的物理抓取验证</a>；本轮没有复现后者的Isaac Lab扰动力测试，也不把数值候选当作实物标定。<a href="../20261008-relocate/index.html">上一轮实际搬运与记忆验收</a>保留。本轮无SSH修改、真机操作、模型训练或参数部署。</small></main></html>'''
    (OUT/'index.html').write_text(html)


if __name__=='__main__':main()
