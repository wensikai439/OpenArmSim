"""Rebuild the grip/child-receipt report from all retained declared outcomes."""
import html
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-grip'
def load(name):return json.loads((OUT/name).read_text())
def esc(value):return html.escape(str(value))
def link(name,label):return f'<a href="{esc(name)}">{esc(label)}</a>'


def main():
    numbers=load('numerics/summary.json')['cases'];torsion=load('torsion/summary.json')['cases']
    with tempfile.TemporaryDirectory(prefix='openarm-grip-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,
            'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
        for ax,field,unit,limit in [(axes[0],'max_window_relative_drift_m','毫米',1),
                                  (axes[1],'max_window_relative_rotation_rad','毫弧度',20)]:
            for rows,offset,color,label in [(numbers,-.16,'#b39e8c','原接触模型 · 维数3'),
                                           (torsion,.16,'#758e83','有限面候选 · 维数4')]:
                x=[[0,1,3,20].index(r['noslip_iterations'])+offset for r in rows]
                y=[r['stability'][field]*1000 for r in rows]
                ax.bar(x,y,width=.29,color=color,label=label)
                for p,v in zip(x,y):ax.text(p,v*1.18,f'{v:.3g}',ha='center',fontsize=8)
            ax.set_yscale('log');ax.set_ylim(.001 if unit=='毫米' else .04,30 if unit=='毫米' else 160)
            ax.axhline(limit,color='#a56647',linestyle='--',linewidth=1,label='原验收阈值')
            ax.set_xticks(range(4),['0','1','3','20']);ax.set_xlabel('NoSlip 迭代上限');ax.set_ylabel(unit)
            ax.set_title('完整10秒中最差2秒相对漂移' if unit=='毫米' else '完整10秒中最差2秒相对旋转',loc='left')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.1),
                   ncol=3,frameon=False,fontsize=9)
        fig.savefig(OUT/'grip.png',dpi=180,bbox_inches='tight');plt.close(fig)

    table=''
    for folder,title in [('direct-right','右手直接抓取'),('direct-left','左手直接抓取'),('receiver-right','左抓后右接')]:
        d=load(folder+'/result.json');s=d['stability']
        table+=f'<tr><td>{title}</td><td>未通过</td><td>{s["max_window_relative_drift_m"]*1000:.2f}mm</td><td>{s["max_window_relative_rotation_rad"]:.4f}rad</td><td>{link(folder+"/result.json",s["status"])}</td></tr>'
    candidate=''
    for folder,title in [('right-long','右手持续保持60秒'),('left-episode1','轨迹02初始姿态 · 左手保持20秒'),
                         ('right-episode2-move','轨迹03初始姿态与新瓶位 · 右手cuRobo前放8cm'),
                         ('left-to-right-move','新瓶位 · 左手抓取后右接')]:
        d=load('validation/'+folder+'/result.json');s=d.get('stability');f=d.get('native_feedback') or {}
        detail=(f'{s["arms"][d["case"]["arm"]]["max_window_relative_drift_m"]*1000:.5f}mm / 2秒' if s else
                f'落点误差 {f["position_error_m"]*1000:.2f}mm' if d['passed'] else d['failure']['code'])
        candidate+=f'<tr><td>{title}</td><td>{"通过" if d["passed"] else "未通过"}</td><td>{esc(detail)}</td><td>{link("validation/"+folder+"/result.json","全部结果")}</td></tr>'
    api_table=''
    for name,title,reason in [('api-right.json','首轮连续搬运','子动作阶段绑定被拒绝；未运动'),
        ('api-right-composite.json','加入子动作绑定后的首轮','preserve 的场景编号与目标锚点不匹配；未运动'),
        ('api-right-current.json','修正锚点 · 右手直取前放8cm',''),
        ('api-transfer-current.json','修正锚点 · 左抓右接前放6cm',''),
        ('workbench-empty.json','浏览器首轮 · 默认供应商模式','空内容、无工具；未运动，终止原因未保存'),
        ('workbench-current.json','浏览器 · 显式非思考模式 · 逐个调用动作','')]:
        d=load(name);good=d.get('status')=='passed'
        receipts=d.get('feedbacks',[])
        feedback=d.get('native_feedback') or (receipts[-1]['feedback'] if receipts else {})
        result=f'完成 · {feedback["position_error_m"]*1000:.2f}mm' if good else reason
        api_table+=f'<tr><td>{title}</td><td>{esc(result)}</td><td>{d["actual_api_calls"]}</td><td>{esc(d.get("model_verdict") or "未提交")}</td><td>{link(name,"完整过程")}</td></tr>'
    browser=load('workbench-current.json');gallery=''
    for f in browser['terminal']['agent']['goal_keyframes']:
        url=f['url'];gallery+=f'<a href="{esc(url)}" target="_blank" rel="noopener"><img src="{esc(url)}" alt="{esc(f["label"])}"><span>{esc(f["label"])} · {f["sim_time"]:.1f}s</span></a>'
    live=''
    history=load('api-transfer-current.json')['stage_history']
    transfer=load('api-transfer-current.json')
    titles={s['id']:s['title'] for s in transfer['terminal']['agent']['plan']['steps']}
    archived=len(transfer['feedbacks'][0]['feedback']['substeps'])
    for h in history:
        done='、'.join(titles[i] for i,status in h['steps'] if status=='done') or '尚无已完成阶段'
        receipt=f'{len(h["completed_substeps"])}个实时子回执' if h['outer_running'] else f'{archived}个已归档子回执'
        live+=f'<tr><td>{"搬运仍运行" if h["outer_running"] else "搬运已结束"}</td><td>{esc(done)}</td><td>{receipt}</td></tr>'
    audit=load('audit.json');private=load('candidate-memory.json')['memory'];lessons=private['knowledge']['lessons']
    memories=''.join(f'<li>{esc(x["title"])} · {esc(x["support"])}</li>' for x in lessons)
    note=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹持稳定与连续搬运</title>
<style>body{{margin:0;background:#f8f7f3;color:#332f28;font:15px/1.7 system-ui,sans-serif}}main{{max-width:1080px;margin:32px auto;padding:0 24px}}h1{{font-size:30px;line-height:1.35}}h2{{font-size:19px}}section{{background:white;border:1px solid #e7e1d7;border-radius:15px;padding:24px;margin:18px 0}}img{{width:100%;height:auto}}.muted,small{{color:#797367}}a{{color:#806146;text-underline-offset:3px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:690px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #eee8df}}.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.gallery a{{border:1px solid #e7e1d7;border-radius:10px;overflow:hidden;text-decoration:none}}.gallery span{{display:block;font-size:12px;padding:6px 10px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{flex:1;min-width:180px;background:#f7f6f1;border:1px solid #eee9df;border-radius:12px;padding:16px}}.card strong{{display:block;font-size:23px}}code{{font-size:13px;background:#f7f5ef;padding:2px 4px}}ul{{padding-left:22px}}@media(max-width:640px){{.gallery{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:360px){{.gallery{{grid-template-columns:1fr}}}}</style>
<main><a href="http://127.0.0.1:8766/">← 独立候选工作台</a><p class="muted">2026-10-08 · 原生 MuJoCo / 实际 CUDA / 实际 LLM API · 接触候选尚未标定</p><h1>夹持稳定与连续搬运</h1><p>有限接触面候选解决了本轮保持测试中的滑移，并通过真实模型的右手直取及左→右搬运。harness 现在能按原生子动作回执更新阶段；保留原目标、碰撞检查、夹持漂移与落点阈值。全部失败和无动作退出一并列出。主平台仍保留原接触模型与已有场景。</p>
<p><a href="#api">实际模型结果 ↓</a> · <a href="#memory">记忆与边界 ↓</a></p><section><div class="cards"><div class="card"><strong>60秒</strong>候选右手完整保持通过；未丢弃初始窗口</div><div class="card"><strong>1.79毫米</strong>浏览器左抓→右接→cuRobo→桌面落点误差</div><div class="card"><strong>69项通过</strong>目标、记忆、参数身份及15项新增对抗检查</div></div></section>
<section><h2>两种滑移机制，分别验证</h2><p>原维数3接触下，提高 NoSlip 迭代上限降低了接触切向速度，却仍出现绕接触线的旋转和相对位置漂移。NoSlip3 的多数采样呈松弛接触扳手秩5；该诊断不包含摩擦或力界，不能据此宣称力闭合。候选维数4启用有限接触面的扭转摩擦，再配合 NoSlip3，完整10秒的原1mm / 0.02rad窗口检查通过。</p><img src="grip.png" alt="原维数3与候选维数4全部NoSlip条件的最差两秒漂移和旋转，纵轴为对数"><p>维数4只测试迭代上限0、3、20，迭代1未测试。图中的数值是全段最差窗口；接触丢失同样导致失败。原接触模型的 NoSlip20 后段出现稳定窗口，仍因早期漂移失败。图不是成功率，也不是实物精度。</p>{link('numerics/protocol.json','数值比较协议')} · {link('numerics/summary.json','全部原接触结果')} · {link('torsion/protocol.json','有限面比较协议')} · {link('torsion/summary.json','全部有限面结果')}</section>
<section><h2>原接触模型 · 三种完整10秒保持均未通过</h2><div class="scroll"><table><tr><th>抓取来源</th><th>验收</th><th>最差相对漂移</th><th>最差相对旋转</th><th>依据</th></tr>{table}</table></div><p>抓起与接收的原生原语本身完成，但持续保持不满足原稳定标准。未把抬起瓶子的单帧当成稳定携物能力。</p>{link('protocol.json','事前协议')} · {link('summary.json','完整比较')}</section>
<section><h2>候选接触 · 新保持时长、初始姿态与瓶位</h2><div class="scroll"><table><tr><th>声明条件</th><th>验收</th><th>依据</th><th>完整数据</th></tr>{candidate}</table></div><p>这些是候选选择后的开发测试，并非独立统计评测。保持测试的规划暂停、物理保持持续积分；两项手工搬运计入生成器中的规划等待，但构造器在计时器之外，不能称完整延迟评测。后面的实际 API / 浏览器路线使用正常 SimulationService，构造和生成器都计入等待。</p>{link('validation/protocol.json','验证协议')} · {link('validation/timing-scope.json','计时覆盖限制')}</section>
<section><h2>明确的模型派生与普通记录器验收</h2><p>工作台使用公开 MjSpec 在编译前设置接触维数4与 NoSlip3，随后冻结普通模型证据。质量、惯量、摩擦系数、控制增益、力限、关节范围和网格数组与原模型逐项相同。已有扭转系数因接触模型改变而开始生效；瓶子与两侧手指的其他碰撞对也受影响，不能称只修复一个手指接触对。编译后的模型身份与原模型明确不同。</p><p>普通 Recorder 接受预先冻结的候选，之后再改变 NoSlip 则正常拒绝。没有重置参数审计器、焊接、瞬移、在线改姿态或改任务成功条件。当前主平台编译身份和物理场景均保留，候选参数未设为生产默认。</p>{link('cold-composite/result.json','编译前派生与后改拒绝')} · {link('cold-composite/manifest.json','原模型／候选模型身份')} · {link('audit.json','当前审计')}</section>
<section id="api"><h2>真实 API 与页面 · 所有条件</h2><div class="scroll"><table><tr><th>条件</th><th>原生结果或未执行原因</th><th>API次数</th><th>模型判定</th><th>依据</th></tr>{api_table}</table></div><p>右手直取用一个 relocate，左交右的独立测试也用一个 relocate；浏览器中的模型选择分别调用 pick、handover、cuRobo move_relative、place，再复查。每项都保留锁定目标和120秒 measured_delay预算，未在失败后重置、改目标、加时或改成暂停等待。完成仅说明这些声明任务在此候选模型中通过。</p><p>浏览器首轮空输出的结束原因未保存，因此不能断言为 token 截断；已保存的 completion_tokens=1600 与旧输出上限相同，只是排查线索。当前实验适配器对官方 DeepSeek 显式采用非思考模式，仍保留1600输出上限；全部12次新响应为 tool_calls。空输出或不完整响应现在会在工具下发前被拒绝。适配器只在独立工作台启用，不修改主平台密钥、保存配置或供应商地址。</p><p>浏览器首版收集器在原生抓取结束与异步后续回合之间提前退出，错误地给出3次请求的失败记录。修正后只恢复只读收集，没有重跑机器人任务；最终保存全12次API和4个原生动作。遗漏的中间实时采样不重建。首版结果、日志与源码保留。</p>{link('workbench-early.json','提前退出的收集记录')} · {link('workbench-collector-current.protocol.json','收集修正与限制')} · {link('provider-checks.log','供应商边界检查')}<div class="gallery">{gallery}</div></section>
<section><h2>阶段依照实际子回执更新</h2><div class="scroll"><table><tr><th>连续搬运状态</th><th>此时已完成阶段</th><th>当前子回执</th></tr>{live}</table></div><p>同一次 relocate 的抓取、交接、运动和放置阶段共享外层执行编号；按子技能与次数选择回执。只有正确原生来源、接收手身份、运动误差和完成状态才通过。父动作成功不能替代缺失的子回执，重复或恢复动作必须明确次数；旧执行和旧场景不能完成当前阶段。</p><p>prepare(preserve) 只有在服务证明保留状态、前后机器人与物体快照相同且任务身份匹配时，才转移原锚点的场景编号，原初始瓶位不会重算。reset 仍使原目标失效，任务中再次 prepare 仍禁止。界面已处理“模型等待有物理积分但暂时没有机器人job”的状态。</p>{link('composite-final.log','12项子回执／锚点对抗检查')} · {link('api-transfer-current.json','完整实时里程碑')} · {link('regression.log','69项相关检查')}</section>
<section id="memory"><h2>记忆保持小索引，轨迹归入实验</h2><p>主页面目前{audit['main']['knowledge_methods']}条方法／教训与1条偏好；独立候选页面为{audit['candidate']['knowledge_methods']}条方法与1条偏好。页面默认原始经历和 episode 均为0。模型默认也读取有限方法索引，匹配详情按需展开；原姿态不是当前感知。完整回执按执行ID留在实验档案，未删除故障依据，也未将原物理模型的方法当成候选模型能力。</p><ul>{memories}</ul><p>用户偏好只在候选数据库初次建立时复制一次；候选对话、遗忘和新记录独立持久化。保留左对话、右上相机、右下阶段与关键帧的布局；供应商检查及详细力学数据折叠在过程或记录中。</p>{link('candidate-memory.json','候选默认记忆核验')} · {link('main-memory.json','主页面默认记忆核验')} · {link('memory.png','实际记忆窗口')}</section>
<section><h2>可以试用，迁移结论仍待验证</h2><p>现在运行的独立工作台：<a href="http://127.0.0.1:8766/">127.0.0.1:8766</a>。下次在项目目录运行 <code>./run contact</code> 即可开启，Ctrl+C停止；没有添加定时器或自动启动。它只用于原生执行验证，训练仍在原主平台；候选模型尚无匹配且经过验证的VLA评测契约。</p><p>开放物体感知、抓取候选覆盖、不可达恢复、独立基座／TCP／相机／力学／控制器标定及跨种子评测未完成。当前TAMP仍使用已声明瓶子与仿真几何，真实ACT权重和OpenArmπ0.5任务能力也尚未验证。不能据本轮结果宣称任意动作或 sim-to-real 就绪。本轮无SSH和硬件操作。</p></section>
<small>接触与滑移依据 <a href="https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip">MuJoCo 官方建模文档</a>；编译前派生依据 <a href="https://mujoco.readthedocs.io/en/stable/python.html#model-editing">公开 MjSpec</a>；工具闭环参考 <a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey</a> 和 <a href="https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html">MoveIt Task Constructor 的可验证阶段</a>；供应商适配依据 <a href="https://api-docs.deepseek.com/guides/thinking_mode/">DeepSeek 官方思考模式与工具协议</a>。这些是结构参考，不代表完成了对应系统的复现。</small></main></html>'''
    (OUT/'index.html').write_text(note)
    print(json.dumps({'report':str((OUT/'index.html').relative_to(ROOT)),'all_api_conditions':6,'candidate_deployed_as_default':False},ensure_ascii=False))


if __name__=='__main__':main()
