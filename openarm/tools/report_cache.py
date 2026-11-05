"""Cache latency, successful transfers and retained grasp failures."""
import html
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/20261008-cache'


def load(name):return json.loads((OUT/name).read_text())
def esc(value):return html.escape(str(value))


def main():
    first,fast=load('gpu.json'),load('gpu-fast.json')
    native=load('native.json');right=load('api.json')
    transfer=load('api-transfer-isolated.json');matched=load('api-transfer-current.json')
    interrupted=load('api-transfer-memory-interrupted.json')
    collision=load('collision-final.json');memory=load('memory.json')
    assert native['complete'] and len(native['cases'])==2
    with tempfile.TemporaryDirectory(prefix='openarm-cache-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,
            'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
        fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
        labels=['抓取后','锁臂 +0.001rad','锁臂 +0.15rad','回到原锁臂值']
        for values,offset,color,label in [(first['cases'],-.17,'#baa997','首版：重建碰撞对表'),
                                         (fast['cases'],.17,'#758e83','优化：复用原几何并精确更新')]:
            positions=[i+offset for i in range(4)]
            times=[x['wall_seconds'] for x in values]
            axes[0].bar(positions,times,width=.31,color=color,label=label)
            for x,t in zip(positions,times):axes[0].text(x,t+.07,f'{t:.2f}',ha='center',fontsize=9)
        axes[0].set_xticks(range(4),labels,fontsize=8);axes[0].set_ylabel('秒')
        axes[0].set_title('同一组冻结输入 · GPU 提案延迟',loc='left');axes[0].legend(frameon=False,fontsize=8)
        axes[0].set_ylim(0,5.4)
        times=[r['metrics']['tamp']['backend_preparation']['seconds'] for r in native['cases']]
        axes[1].bar(['冷准备 · 右手路线','已准备 · 左交右路线'],times,color=['#baa997','#758e83'],width=.5)
        for i,t in enumerate(times):axes[1].text(i,t+.25,f'{t:.2f}s',ha='center')
        axes[1].set_ylim(0,max(times)*1.25);axes[1].set_ylabel('秒')
        axes[1].set_title('准备耗时在抓取前实际积分',loc='left')
        fig.savefig(OUT/'cache.png',dpi=180);plt.close(fig)
    table=''
    for row in native['cases']:
        t=row['metrics']['tamp'];clock=row['metrics']['planning_timing']
        good=row['status']=='primitive_completed' and t.get('primitive_success')
        title='右手直接抓取 → cuRobo → 前放8cm' if row['case']=='curobo' else '左手抓取 → 右手接收 → cuRobo → 前放6cm'
        result=f'完成 · 误差 {t["position_error_m"]*1000:.2f}mm' if good else '失败 · 瓶子漂移6.64mm，旧提案拒绝'
        table+=f'<tr><td>{title}</td><td>{esc(result)}</td><td>{clock["planner_wall_seconds"]:.3f}s</td><td>{clock["integrated_hold_seconds"]:.3f}s</td><td><a href="../{row["run_id"]}/summary.json">原生回执</a></td></tr>'
    api_rows=''
    for name,data,label in [('api.json',right,'右手直取 · 主平台'),
                           ('api-transfer-isolated.json',transfer,'左交右 · 独立实例，策略配置未载入'),
                           ('api-transfer-current.json',matched,'左交右 · 独立实例，策略与记忆契约匹配')]:
        receipts=[r for r in data['feedbacks'] if r['execution']['skill']=='relocate']
        f=receipts[0]['feedback'] if len(receipts)==1 else {}
        good=data['status']=='passed'
        result=(f'完成 · 误差 {f["position_error_m"]*1000:.2f}mm' if good else
                f'原生完成 · 误差 {f["position_error_m"]*1000:.2f}mm · 复查未闭环' if f.get('primitive_success') else
                '失败 · 未完成放置')
        api_rows+=f'<tr><td>{label}</td><td>{esc(result)}</td><td>{data["actual_api_calls"]}</td><td>{esc(data["model_verdict"])}</td><td><a href="{name}">完整过程</a></td></tr>'
    api_rows+=f'<tr><td>匹配记忆契约首轮 · 收集器等待终止判定</td><td>原生完成 · {interrupted["position_error_m"]*1000:.2f}mm · 模型复查 UNKNOWN</td><td>UNKNOWN</td><td>UNKNOWN</td><td><a href="api-transfer-memory-interrupted.json">部分依据与限制</a></td></tr>'
    gallery=''
    run=matched['terminal']['last_run']['run_id']
    events=[json.loads(line) for line in (ROOT/'runs'/run/'events.jsonl').read_text().splitlines()]
    frames=[e for e in events if e.get('type')=='keyframe' and e.get('camera')=='overview']
    selected=[frames[0]]+[e for e in frames[1:-1] if e['label'] in ('左手抓取','双手交接','右手接收','cuRobo携物运动','桌面放置')]+[frames[-1]]
    if len(selected)<4:selected=[frames[0],*frames[1:-1:2],frames[-1]]
    for e in selected[:8]:
        gallery+=f'<a href="{e["url"]}" target="_blank"><img src="{e["url"]}" alt="{esc(e["label"])}"><span>{esc(e["label"])} · {e["sim_time"]:.1f}s</span></a>'
    lesson_rows=''.join(f'<li>{esc(x["title"])} · {"失败教训" if x["outcome"]=="failure" else "已验方法"} · {x["repeat_count"]}份依据</li>' for x in memory['methods'])
    fk=max(r['response']['cache']['geometry_guard']['max_sphere_coordinate_error_m'] for r in fast['cases'])*1000
    note='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>cuRobo规划与交接验收</title>
<style>body{margin:0;background:#f8f7f3;color:#332f28;font:15px/1.7 system-ui,sans-serif}main{max-width:1080px;margin:32px auto;padding:0 24px}h1{font-size:30px;line-height:1.35}h2{font-size:19px}section{background:white;border:1px solid #e7e1d7;border-radius:15px;padding:24px;margin:18px 0}img{width:100%;height:auto}.muted,small{color:#797367}a{color:#806146;text-underline-offset:3px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;min-width:790px}td,th{text-align:left;padding:10px;border-bottom:1px solid #eee8df}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.gallery a{border:1px solid #e7e1d7;border-radius:10px;overflow:hidden;text-decoration:none}.gallery span{display:block;font-size:12px;padding:6px 10px}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{flex:1;min-width:180px;background:#f7f6f1;border:1px solid #eee9df;border-radius:12px;padding:16px}.card strong{display:block;font-size:23px}code{font-size:13px;background:#f7f5ef;padding:2px 4px}ul{padding-left:22px}@media(max-width:640px){.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:360px){.gallery{grid-template-columns:1fr}}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 原生 MuJoCo / 实际 CUDA / 实际 LLM API</p><h1>规划已经提速，夹持仍是瓶颈</h1><p>锁臂数值变化现在通过原位参数更新进入全部规划内部模型，保留 CUDA 图、碰撞几何和碰撞对表。真实模型完成了左抓 → 右接 → cuRobo 携瓶 → 放置；右手直接抓取仍出现滑动，失败完整保留。没有因此放宽原漂移、碰撞或任务评分。</p>
<p><a href="#api">查看模型实验 ↓</a></p><section><div class="cards"><div class="card"><strong>0.62–0.77秒</strong>本轮冻结输入的优化后GPU提案</div><div class="card"><strong>205项通过</strong>接口、物理、记忆与计时回归</div><div class="card"><strong>两条路线有差异</strong>左交右完成；右手直接抓取失稳</div></div></section>
<section><h2>精确更新，避免重新生成碰撞对表</h2><p>首版已尝试复用规划器，但每次创建新运动学配置仍用约3.35秒生成碰撞对表，单次提案约4.13秒。改为缓存完整FK、按官方加载器的父坐标系逆变换更新锁定关节，只更新数值，不舍入关节值、不复用旧几何。当前10个内部模型全部核对；捕获缓冲区指针和碰撞对表不变。世界几何、机械臂或携物拓扑改变仍重新构建。</p><img src="cache.png" alt="冻结输入的两种GPU更新实现及原生冷暖准备时间"><p>这两组是软件诊断，不下发机器人动作；不能把延迟降低当作任务成功率。完整FK球坐标差异最大 '''+f'{fk:.6f}mm'+'''，仅证明软件一致，不能证明实物标定。真实CUDA在缓存复用时拒绝了与前臂重叠的假设携物球。携物表示只用于规划，原生仿真没有焊接或瞬移瓶子。</p><a href="requests.json">冻结输入</a> · <a href="gpu.json">首版全部提案</a> · <a href="profile.json">耗时剖析</a> · <a href="gpu-fast.json">优化后全部提案</a> · <a href="collision-final.json">当前版本碰撞拒绝</a></section>
<section><h2>两项原生完整路线 · 全部结果</h2><p>seed0、episode0，每项开始前重置，120秒预算提前声明，measured_delay。冷准备实际计入物理时间，随后按原生状态逐步抓取、交接、移动和放置。两个条件没有组成统计成功率。</p><div class="scroll"><table><thead><tr><th>任务</th><th>结果</th><th>底层计算</th><th>实际保持积分</th><th>依据</th></tr></thead><tbody>'''+table+'''</tbody></table></div><p>冷准备约11.86秒在抓取前积分；右手路线后续GPU提案可行，但瓶子相对位置改变6.64mm，超过原3mm路径新鲜度门槛而中止。较短等待也可能使当前抓取滑动，提速不能替代夹持稳定验收。</p><a href="native.protocol.json">事前协议</a> · <a href="native.json">两项完整结果</a> · <a href="../20261008-hold/index.html">长期夹持失败证据</a></section>
<section id="api"><h2>模型调用完整工具链</h2><div class="scroll"><table><thead><tr><th>条件</th><th>原生结果</th><th>API请求</th><th>模型判定</th><th>过程</th></tr></thead><tbody>'''+api_rows+'''</tbody></table></div><p>右手路线的真实API失败保留，未把可行GPU路径当作任务完成。你随后正在主页面使用左手抓取，因此后续组合验证改用独立物理实例和内存数据库，不重置你的页面场景，也不写入生产长期记忆。第一项独立组合未加载策略配置，兼容过滤使方法索引为空；匹配生产策略配置后另立协议再测。匹配首轮原生完成，但收集器未处理没有finish判定的终止，原生结束后中断收集器，保留部分日志，完整API轨迹及调用数为UNKNOWN，不计完整通过。修正后的收集器持续保存过程，并识别模型复查结束，无判定也保存失败。全部条件分别保留。实际API均未在目标内重置、改目标、加时或切回暂停计时。</p><div class="gallery">'''+gallery+'''</div><p><a href="api.protocol.json">右手API协议</a> · <a href="api-transfer-isolated.protocol.json">首项独立协议</a> · <a href="api-transfer-memory.protocol.json">匹配首轮协议</a> · <a href="api-transfer-current.protocol.json">修正收集器协议</a> · <a href="../'''+run+'''/summary.json">组合原生回执</a></p></section>
<section><h2>记忆核验</h2><p>主页面目前 '''+str(len(memory['methods']))+''' 条方法／教训、1条用户偏好。一份公共搬运规则按需加载。两次“左手抓取”归成同一方法，保留2份依据；旧暂停条件的成功和新等待条件的失败分别标注，不合并为当前能力宣称。默认页面和模型上下文均不加载原始经历／原始episode；完整回执只在实验档案中，按ID审计。</p><ul>'''+lesson_rows+'''</ul><a href="memory.json">只读记忆核验</a> · <a href="boundaries.log">7项新增边界检查</a> · <a href="regression.log">205项完整检查</a> · <a href="audit.json">部署与参数审计</a></section>
<small>依据<a href="https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/robot/kinematics/kinematics.py">固定版本cuRobo的原位参数更新</a>和<a href="https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/robot/loader/kinematics_loader.py">锁关节加载算法</a>；保留<a href="https://github.com/lianegalanti/Pigey/blob/main/real/agent.ts">Pigey的观测、工具与反馈闭环</a>作为harness参考。仍只覆盖声明瓶子和仿真几何；开放物体感知抓取、持续稳定夹持、独立相机／TCP／力学标定和真实迁移尚未通过。本轮无SSH、远端改写或硬件操作。</small></main></html>'''
    (OUT/'index.html').write_text(note)


if __name__=='__main__':main()
