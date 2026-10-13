"""Evidence page for parameterized relocation; preserve original failures."""
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-relocate'


def read(name):return json.loads((OUT/name).read_text())


def main():
    api=[('右手前放8cm','api-forward-verified.json'),('左手交给右手，再前放6cm','api-transfer.json'),
         ('cuRobo持瓶前移3cm，再放到前8cm','api-curobo.json')]
    rows=[];results=[]
    for title,file in api:
        r=read(file);plan=r['terminal']['agent']['plan'];goal=plan['goal_assessment']
        feedback=r['feedbacks'][0]['feedback'];run=r['feedbacks'][0]['evidence']['run_id']
        manifest=json.loads((ROOT/'runs'/run/'manifest.json').read_text());summary=json.loads((ROOT/'runs'/run/'summary.json').read_text())
        lo=manifest['started_at_unix'];hi=lo+summary['wall_seconds']
        calls=[e for t in r['turns'] for e in t['trace'] if e['kind']=='api']
        overlap=sum(e['started']<hi and (e['ended'] or hi)>lo for e in calls)
        row={'title':title,'file':file,'status':r['status'],'api_calls':len(calls),
            'model_verdict':r['model_verdict'],'goal_error_mm':goal['position_error_m']*1000,
            'stable_seconds':goal['stable_seconds'],'sim_seconds':r['terminal']['observation']['sim_time'],
            'native_execution':run,'API_requests_overlapping_native_operation':overlap,
            'timing_mode':plan['timing_mode'],'target_unchanged':r['target_unchanged'],
            'substeps':[{'skill':s['skill'],'backend':s['backend'],'status':s['status'],
                        'native_commands':s['end_tick']-s['start_tick']} for s in feedback['substeps']]}
        results.append(row)
        rows.append(f'<tr><td>{title}</td><td>{r["status"]}</td><td>{len(calls)}</td><td>{row["goal_error_mm"]:.2f} mm</td><td>{r["model_verdict"]}</td><td><a href="{file}">记录</a></td></tr>')
    chain=read('native-chain/summary.json');recovery=read('native-recovery/result.json')
    language=read('api-natural-transfer-verified.json')
    language_plan=language['terminal']['agent']['plan']
    language_result={'status':language['status'],'api_calls':language['actual_api_calls'],
        'model_verdict':language['model_verdict'],'timing_mode':language_plan['timing_mode'],
        'budget_s':language_plan['time_budget_s'],'target_unchanged':language['target_unchanged'],
        'goal_error_mm':language_plan['goal_assessment']['position_error_m']*1000,
        'file':'api-natural-transfer-verified.json','collection':language['collection']}
    with tempfile.TemporaryDirectory(prefix='openarm-relocation-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,
                             'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        for ax,labels,errors in [(axes[0],['右手前放','左→右前放','cuRobo组合'],[r['goal_error_mm'] for r in results]),
                                (axes[1],['前8cm','左5cm','后6cm','右5cm'],[r['feedback'].get('position_error_m',0)*1000 for r in chain['results']])]:
            ax.bar(labels,errors,color=['#b18467','#8b9d93','#718b83','#c1a891'][:len(labels)],width=.58)
            ax.axhline(10,color='#b18467',ls='--',lw=1,label='本组原目标容差10mm');ax.set_ylabel('瓶子目标位置误差 / mm');ax.set_ylim(0,11.5);ax.grid(axis='y',alpha=.15)
            for i,v in enumerate(errors):ax.text(i,v+.3,f'{v:.2f}',ha='center')
        axes[0].set_title('三项实际 API 开发案例（含模型等待）',loc='left')
        axes[1].set_title('一个连续场景的四个方向（原生，无 API）',loc='left')
        fig.savefig(OUT/'errors.png',dpi=180);plt.close(fig)
    dump={'real_API_cases':results,'plain_language_API':language_result,'native_chain':chain['results'],
        'native_recovery':{k:recovery[k] for k in ('status','feedback','sim_seconds')},
        'scope':'declared development cases, not independent statistical generalization or transfer',
        'long_duration_grasp_validated':False,'hardware_access':False,'transfer_ready':False}
    (OUT/'results.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2)+'\n')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续搬运验收</title>
<style>body{margin:0;background:#f8f7f3;color:#332f28;font:15px/1.7 system-ui,sans-serif}main{max-width:1060px;margin:32px auto;padding:0 24px}h1{font-size:30px;line-height:1.35}h2{font-size:19px}section{background:white;border:1px solid #e7e1d7;border-radius:15px;padding:24px;margin:18px 0}img{width:100%;height:auto}.muted,small{color:#797367}a{color:#806146;text-underline-offset:3px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #eee8df}.scroll{overflow:auto}.flow{padding:18px;background:#f6f5ef;border-radius:10px;text-align:center;color:#675647}details{margin:12px 0}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 参数化动作与原生验证</p><h1>模型指定目标，底层连续执行并逐步验证</h1><p>新增连续搬运工具，按当前场景求解抓取、可选交接／携物路径、放置和撤离。原来的单项工具与固定演示保留；物理、碰撞、精度、支撑和稳定门槛不变。</p>
<section><h2>完整动作怎样发生</h2><div class="flow">LLM锁定物体目标与路线 → 原生抓取 → 可选交接／明确选择的路径后端 → 放置、松手与撤离 → 原生验收 → 新相机复查</div><p>每个子动作使用它开始时的实际机器人／瓶子状态，确认后才进入下一步。相对物体目标在动作前固定，换手和恢复不会重解成新的目标。cuRobo可作为显式携物子路径，抓取和接触放置仍由TAMP承担；服务未就绪在抓取前拒绝，不替换后端。</p><p>只在明确允许时，放置不可达才尝试一次换手；失去夹持、物理失效、取消或超时立即停止。没有重置、瞬移、真实瓶子焊接、质量／摩擦／力修改。模型阶段仍由模型给出，只有整次搬运的真实目标、支撑、松手和稳定回执能完成该阶段；抬手或TCP到位不能代替。</p><p>三次API记录均未在原生操作区间发起新模型请求。measured_delay仍推进首个计划之后的外部模型等待；底层规划壁钟延迟仍未建模，完整硬件实时控制尚未验证。</p></section>
<section><h2>实际API与位置结果</h2><img src="errors.png" alt="实际API和连续原生场景的物体目标误差"><div class="scroll"><table><thead><tr><th>任务</th><th>原生任务</th><th>API次数</th><th>位置误差</th><th>模型复查</th><th>证据</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><p>均为episode 0、seed 0、首次计划锁定120秒、measured_delay，任务中不重置、不加时、不切回paused。模型输出和请求延迟不同，这三项是固定开发验收，不能当作泛化成功率或与旧方案的受控增益比较。原生目标和模型视觉判断分别保留。</p><p>cuRobo组合实际执行110条携物原生控制指令，GPU附件碰撞与原生路径复核通过；不是用TAMP替代该子路径。首个报告的API计数器误读字段得到0；原trace确有5次请求。原报告保留，修正计数另存并绑定原报告摘要。</p><a href="api-forward.json">首版原始报告</a> · <a href="native-curobo/result.json">无API的真实GPU组合</a></section>
<section><h2>方向变化与局部恢复</h2><p>在一个连续原生场景中，瓶子依次前8cm、左5cm、后6cm、右5cm，四次均验证支撑与松手。后续任务只开启新时间窗，机器人和瓶子位置、速度完全保留；不是每次回到同一个初始姿态，也不是四份独立随机episode。</p><p>另一项原生恢复案例：左手抓取 → 左手放置逆解不可达 → 左交右 → 右手完成原前6cm目标，误差2.96mm。失败子动作保留，恢复只执行一次。此恢复案例未调用LLM。</p><a href="native-chain/protocol.json">连续场景协议</a> · <a href="native-chain/summary.json">逐目标完整回执</a> · <a href="native-recovery/result.json">含失败的恢复记录</a></section>
<section><h2>已部署与边界</h2><p>181项最终回归通过；新增原生搬运、左交右、取消、无效目标、cuRobo未就绪和阶段不能冒领的检查。技能窗口加入「连续搬到目标」，主布局仍是左对话、右上场景、右下模型阶段；子动作留帧和详细回执按需查看。</p><p>长期记忆的共同搬运规则只加载一份，三个条件条目分别描述实际完成路线，不再重复同一模板。新目标坐标合并为同一方法，左右手／交接方向／后端区别保留；失败子动作不冒充完成路线，缺少原生放置证据不能提炼成功方法。原始回执和相机证据留在档案，默认上下文和页面没有原始轨迹。旧档案没有采集时模型快照，不能强行认作新模型证据。当前三条方法与一条偏好兼容本模型。</p><p><strong>长时间悬空夹持仍未通过</strong>，这一组合通过不替代持续夹持验收。当前只支持声明的直立瓶子与候选工作区，尚无开放物体视觉抓取、独立标定或真机迁移结论。ACT完整任务和匹配OpenArmπ0.5仍未完成，本轮未访问SSH或操作真机。</p><a href="regression-final.log">181项最终回归</a> · <a href="regression.log">记忆精修前180项</a> · <a href="audit.json">版本与部署审计</a> · <a href="results.json">结构化结果</a><details><summary>查看实际API完成页面</summary><img src="platform.jpg" alt="原布局、模型生成阶段和真实执行结果"></details><details><summary>查看精炼记忆</summary><img src="memory.jpg" alt="共同规则一份、按实际路线区分的三条记忆"></details></section>
<small>参考 <a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey的分层工具、抓取验证和新观测复查</a>、<a href="https://github.com/NVlabs/curobo">官方cuRobo</a>和<a href="https://code.claude.com/docs/en/memory#how-it-works">Claude Code的小索引、按需读主题记忆</a>。本实现是原生参数化组合，并非这些系统的完整复现。</small></main></html>'''
    language_html=f'''<section><h2>普通中文，无工具名提示</h2><p>任务：「用左手拿起桌上的瓶子，交给右手，再用右手把瓶子放到起始瓶位向前6厘米的桌面上。放好后松开并退开，完成后复查。不要在动作中重置场景。」</p><p>模型自行拟定3个阶段，并选择连续搬运工具和左→右路线。实际{language_result['api_calls']}次API，原生目标{language_result['status']}、模型复查{language_result['model_verdict']}，物体误差{language_result['goal_error_mm']:.2f}mm。这次由模型选择paused计时和90秒预算，单独列出，不能混为前述含模型等待的实验。</p><p>旁路收集进程曾以143退出，平台后端继续执行并正常完成。依据同一turn的原始trace、动作前瓶位和原生回执只读补齐报告；没有重跑、重置或新的模型请求。日志和收集异常保留。</p><a href="api-natural-transfer-verified.json">原任务补齐的完整证据</a> · <a href="api-natural-transfer.log">收集日志</a><details><summary>查看普通中文任务的实际完成页面</summary><img src="language.jpg" alt="普通中文目标、模型自行选择路线、三阶段完成"></details></section>'''
    html=html.replace('<section><h2>方向变化与局部恢复</h2>',language_html+'<section><h2>方向变化与局部恢复</h2>')
    html=html.replace('src="memory.jpg"','src="memory-final.jpg"')
    (OUT/'index.html').write_text(html)


if __name__=='__main__':main()
