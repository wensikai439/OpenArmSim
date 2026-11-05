"""Readable planner-clock evidence with all paired conditions retained."""
import html
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-planning'


def main():
    study=json.loads((OUT/'native.json').read_text());rows=study['cases']
    if not study['complete']:raise ValueError('all six declared conditions must finish')
    names={'forward':'右手前放8cm','transfer':'左交右后前放6cm','curobo':'cuRobo携物后前放'}
    with tempfile.TemporaryDirectory(prefix='openarm-planning-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,
            'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
        fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
        cases=['forward','transfer','curobo']
        labels=['右手前放8cm','左交右后前放6cm','cuRobo携物后前放']
        for ax,field,title in [(axes[0],'planner_wall_seconds','实际底层计算耗时'),
                               (axes[1],'integrated_hold_seconds','实际积分的保持时间')]:
            maximum=max(r['metrics']['planning_timing'][field] for r in rows)
            for mode,offset,color,label in [('paused',-.18,'#b7a897','暂停对照'),
                                           ('measured_delay',.18,'#738c83','计入规划等待')]:
                values=[next(r['metrics']['planning_timing'][field] for r in rows if r['case']==case and r['mode']==mode) for case in cases]
                positions=[i+offset for i in range(3)]
                ax.bar(positions,values,color=color,width=.33,label=label)
                for x,v in zip(positions,values):ax.text(x,v+maximum*.025,f'{v:.2f}',ha='center',fontsize=9)
            ax.set_xticks(range(3),labels,fontsize=9);ax.set_ylabel('秒');ax.set_title(title,loc='left')
            ax.legend(frameon=False,fontsize=9,loc='upper left')
            ax.margins(y=.2)
        fig.savefig(OUT/'planning.png',dpi=180);plt.close(fig)
    table=''
    for r in rows:
        m=r['metrics']['planning_timing'];t=r['metrics']['tamp']
        result='完成原生放置' if r['status'] in ('primitive_completed','success') and t.get('primitive_success') else t.get('failure_code',r['status'])
        table+=f'<tr><td>{names[r["case"]]}</td><td>{r["mode"]}</td><td>{m["planner_wall_seconds"]:.3f}s</td><td>{m["integrated_hold_seconds"]:.3f}s / {m["hold_transitions"]}步</td><td>{html.escape(result)}</td><td><a href="../{r["run_id"]}/summary.json">原回执</a></td></tr>'
    api=''
    if (OUT/'api.json').exists():
        a=json.loads((OUT/'api.json').read_text())
        api=f'<section><h2>实际API检验</h2><p>{html.escape(a["description"])}</p><p>实际API请求：{a["actual_api_calls"]}；结果：{html.escape(a["status"])}。原始计划、执行与终止证据全部保留，不以模型回答替代原生评分。</p><a href="api.json">完整记录</a></section>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>规划等待验收</title>
<style>body{margin:0;background:#f8f7f3;color:#332f28;font:15px/1.7 system-ui,sans-serif}main{max-width:1080px;margin:32px auto;padding:0 24px}h1{font-size:30px;line-height:1.35}h2{font-size:19px}section{background:white;border:1px solid #e7e1d7;border-radius:15px;padding:24px;margin:18px 0}img{width:100%;height:auto}.muted,small{color:#797367}a{color:#806146;text-underline-offset:3px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;min-width:820px}td,th{text-align:left;padding:10px;border-bottom:1px solid #eee8df}code{font-size:13px;background:#f7f5ef;padding:2px 4px}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 单一物理线程 · 原门槛</p><h1>等待规划时，物体也会继续运动</h1><p>原 measured_delay 只包含声明之后的模型请求等待，底层规划计算被排除。现在将 TAMP / cuRobo 的构造、IK、碰撞搜索与局部规划器请求耗时计入同一任务预算；保留 paused 对照，原物理参数、控制器、碰撞及任务成功条件不变。</p>
<section><h2>相同场景的六项开发对照</h2><p>seed 0、episode 0；每项从相同重置状态开始，提前锁定120秒任务预算，不在失败后加时。全部结果展示，不作为统计泛化率。两种条件实际搜索路径及修正次数可能因等待导致的状态变化而不同。cuRobo含等待项在首次路径返回后已被拒绝，较短总计算耗时不代表提速。</p><img src="planning.png" alt="六项条件实际规划计算耗时及被积分的保持时间"><div class="scroll"><table><thead><tr><th>任务</th><th>计时</th><th>计算耗时</th><th>保持积分</th><th>结果</th><th>证据</th></tr></thead><tbody>'''+table+'''</tbody></table></div><a href="protocol.json">预先声明协议</a> · <a href="native.json">全部结构化结果</a></section>
<section><h2>计算结束后，先补物理时间，再检查路径是否过期</h2><p>使用单一物理线程的离散事件模型：计算期间状态保持只供规划查询；返回后在下发任何新目标之前，按累计耗时积分上一条实际执行器目标。等价范围仅限这段时间没有其他控制事件且模型参数不变；这不是实时控制器，也不提供阻塞计算期间的连续视频流。</p><p>短暂的生成器恢复耗时累计到已有的1毫秒物理网格，避免微秒计算被放大成33毫秒等待；结束／错误时补最后不足1毫秒的部分。普通目标指令仍保持30Hz间隔。保持指令不计为新运动指令，但实际物理时间消耗原任务预算。取消、预算或物理失效后，迟到动作不执行。</p><p>若机器人／瓶子位姿或夹持状态变化超过现有观测阈值与路径起点检查，旧提案拒绝并返回可核对的漂移。控制器不自动重置或换后端，也不放宽碰撞／成功门槛。需要模型用新观测重新规划，原物体目标不变。</p></section>
<section><h2>原始量化对照保留</h2><p>首版按30Hz量化累计等待，cuRobo路线在抓取抬升阶段因额外一整周期的漂移被拒绝，尚未进入GPU携物规划。六项完整结果保留。随后明确改为已有1ms物理网格，并重新运行全部六项；没有修改原有漂移门槛。实际GPU的延迟是否触发持瓶失稳，以新结果为准。</p><a href="native-grid30.json">首版六项结果</a> · <a href="protocol-grid30.json">首版协议</a></section><section><h2>旧记忆的经验条件保留</h2><p>重新执行4份冻结回执的3944条指令，逐步状态、评分与瓶子终点一致后恢复兼容，保留迁移链和私有数据库备份。当前默认索引仍为3条方法、1条偏好及一份共同规则。旧结果明确标记为 legacy_unmodeled，不能计入新 planning-timing/1 的通过统计。方法和共同操作协议仍可检索，默认上下文不重复注入原始轨迹；具体性能须核对模型等待与底层规划等待两项条件。没有把本轮隔离开发对照自动升级为真实API经验。</p><a href="memory/proofs.json">冻结轨迹核验</a> · <a href="memory/migration.json">记忆迁移审计</a> · <a href="checks-current.log">新增检查</a> · <a href="regression.log">完整回归</a> · <a href="audit.json">部署及参数审计</a></section>'''+api+'''
<small>参考<a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey的观测／工具／新观测闭环</a>及<a href="https://github.com/NVlabs/curobo">官方cuRobo运动生成能力</a>，本平台接口不等于完整复现。初次模型计划声明前的等待、真机相机／CAN／看门狗仍未复现；长期夹持、独立物理标定和真实迁移仍未通过。<a href="../20261008-hold/index.html">夹持失稳证据</a>完整保留。</small></main></html>'''
    (OUT/'index.html').write_text(page)


if __name__=='__main__':main()
