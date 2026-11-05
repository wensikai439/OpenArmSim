"""Build a source-labelled report from immutable multi-object task evidence."""
import html
import json
import os
import re
import tempfile

from sim.environment import ROOT

OUT = ROOT / 'runs/20261009-tasks'


def read(path): return json.loads((OUT / path).read_text())
def esc(value): return html.escape(str(value))
def link(path, title): return f'<a href="{esc(path)}">{esc(title)}</a>'


def main():
    labels = {'native-3': '原生脚本', 'api-3': '独立真实 API', 'workbench': '页面真实 API'}
    records = {name: read(name + '/result.json') for name in labels if (OUT / name / 'result.json').exists()}
    with tempfile.TemporaryDirectory(prefix='openarm-task-plot-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False,
            'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), layout='constrained', facecolor='#fffefa')
        names = list(records); x = np.arange(len(names))
        chart_labels = [labels[name] + ('\n复查不确定' if name == 'workbench' and not records[name]['passed'] else '') for name in names]
        for i, ident in enumerate(('can', 'block')):
            values = [next((goal['position_error_m'] * 1000 for goal in record['terminal']['agent'].get('plan', {}).get('goal_assessment', {}).get('objects', []) if goal['object_id'] == ident), 0.) for record in records.values()]
            bars = axes[0].bar(x + (i-.5)*.3, values, width=.3, color=('#819488', '#b49378')[i], label=('圆柱', '长方体')[i])
            axes[0].bar_label(bars, fmt='%.2f', padding=3)
        axes[0].set(xticks=x, xticklabels=chart_labels, ylabel='原目标最终 XY 误差 / mm', title='同时复查两个原目标')
        axes[0].legend(frameon=False); axes[0].set_ylim(0, max(3., axes[0].get_ylim()[1]))
        times = [record['terminal']['observation']['sim_time'] for record in records.values()]
        bars = axes[1].bar(x, times, width=.5, color='#819488'); axes[1].bar_label(bars, fmt='%.2f', padding=3)
        axes[1].axhline(120, linestyle='--', color='#ae7862', linewidth=1, label='原 120 秒预算')
        axes[1].set(xticks=x, xticklabels=chart_labels, ylabel='同一任务累计仿真时间 / s', ylim=(0, 140), title='计入模型与底层规划等待')
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.set_facecolor('#fffefa'); axis.grid(axis='y', color='#e6e0d5', linewidth=.6); axis.set_axisbelow(True)
        fig.savefig(OUT / 'results.png', dpi=180); plt.close(fig)
    rows = []
    reasons = {'native-1': '收集器启动失败；未创建任务。原异常未捕获，根因仅作源码诊断。',
        'native-2': '离线驱动遗漏原任务标识，首项放置被新任务门禁拒绝。',
        'api-1': '前置检查顺序与重复调用限制冲突，零运动。',
        'api-2': '圆柱已放好；反馈误标下一动作运行中，模型反复轮询后停止。',
        'workbench': '五次运动、两个原目标通过；模型视觉复查uncertain，保留5/6阶段及9/10检查。'}
    for name in ('native-1', 'native-2', 'native-3', 'api-1', 'api-2', 'api-3', 'workbench'):
        path = OUT / name / 'result.json'
        if path.exists():
            record = read(name + '/result.json'); plan = record['terminal']['agent'].get('plan') or {}
            spec = read(name + '/protocol.json'); version = spec['source_sha256']['harness/task_queue.py'][:12]
            status = '通过' if record['passed'] else '未通过'
            detail = reasons.get(name, '所有原目标、阶段、原生回执与时间检查通过。' if record['passed'] else '见原始检查项；未改判。')
            rows.append(f'<tr><td>{name}</td><td>{status}</td><td>{record["actual_api_calls"]}</td><td>{record["terminal"]["observation"]["sim_time"]:.3f}</td><td><code>{version}</code></td><td>{esc(detail)}</td><td>{link(name+"/result.json", "回执")}</td></tr>')
        elif (OUT / name / 'collector-error.json').exists():
            rows.append(f'<tr><td>{name}</td><td>启动失败</td><td>0</td><td>0</td><td>原快照保留</td><td>{esc(reasons[name])}</td><td>{link(name+"/collector-error.json", "诊断")}</td></tr>')
    gallery = []
    record = records.get('workbench') or records['api-3']
    for receipt in record['receipts']:
        frames = receipt['evidence'].get('camera_keyframes', [])
        if not frames: continue
        frame = frames[-1]; ident = receipt['feedback'].get('object_id'); skill = receipt['execution']['skill']
        title = {'can': '圆柱', 'block': '长方体'}.get(ident, ident) + ' · ' + {'pick': '抓起', 'handover': '交接', 'place': '放置'}.get(skill, skill)
        url = frame.get('url')
        if url and re.fullmatch(r'/runs/[A-Za-z0-9_./-]+', url):
            gallery.append(f'<a href="{esc(url)}" target="_blank" rel="noopener"><img src="{esc(url)}" alt="{esc(title)}"><span>{esc(title)} · {frame["sim_time"]:.2f}s</span></a>')
    memory = record['memory']['knowledge']['lessons']
    memory_rows = ''.join(f'<tr><td>{esc(item["title"])}</td><td>{item["repeat_count"]}</td><td>{esc(item["summary"])}</td></tr>' for item in memory)
    calibration = json.loads((ROOT / 'sim/calibration.json').read_text())
    calibration_rows = ''.join(f'<tr><td>{esc(item["label"])}</td><td>{esc(item["status"])}</td><td>{esc(item["needed"])}</td></tr>' for item in calibration['parameters'])
    style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261009-scene/index.html').read_text(), re.S)[1]
    style += '.gallery{grid-template-columns:repeat(3,1fr)}table{display:block;overflow:auto}.stats b{font-size:24px}details{margin:18px 0}code{font-size:12px} @media(max-width:650px){.gallery{grid-template-columns:1fr 1fr}}'
    browser = records.get('workbench'); browser_text = ('全部预声明检查通过' if browser['passed'] else '整项预声明检查未全部通过') if browser else '尚未结束'
    api = records['api-3']; native = records['native-3']
    def measure(value):
        goals = value['terminal']['agent']['plan']['goal_assessment']['objects']
        return '／'.join(f'{goal["position_error_m"]*1000:.2f}' for goal in goals)
    screenshot = '<a href="workbench.png"><img class="chart" src="workbench.png" alt="多物体任务实际页面结果"></a>' if (OUT / 'workbench.png').exists() else ''
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 多物体任务</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.09</div><h1>一项任务，多个物体，同一时钟</h1><p>先腾出位置，再交接另一个物体，最后复查每个原目标。</p></div><a class="pill" href="http://127.0.0.1:8767/">多物体工作台 ↗</a></header>
<div class="stats"><div><b>2 个原目标</b><span>起点锁定，不能随物体移动改写</span></div><div><b>5 次原生技能</b><span>模型自行选择并顺序执行</span></div><div><b>38 项检查</b><span>物理不变与任务绑定门禁通过</span></div><div><b>尚未标定</b><span>不是任意动作或真实迁移证明</span></div></div>
<section><h2>具体任务与不可替换的目标</h2><p>右手把圆柱从起始 world XY [0.503, −0.345] m 向机器人前方放 8 cm，最终 [0.583, −0.345] m；随后左手抓长方体，交给右手，放在圆柱<b>本任务最初的位置</b> [0.503, −0.345] m。瓶子不动。所有原生/API/页面尝试使用同一任务、120 秒 measured_delay 与原门槛，失败没有缩短目标、加时或任务内重置。</p>
<p>可选 <code>set_plan.object_goals</code> 声明有序原目标，每个机器人阶段绑定 object_id。声明内完成新观测下的原生条件检查与首对象选择记账，不启动运动。模型显式调用 pick/place/handover；物体脱离夹爪、落在原目标、稳定支撑并有本次放置回执后，才可选择下一声明对象。选择保持 qpos、qvel、控制指令与原时钟，不启动动作或 prepare。原单物体接口与瓶子演示保留。</p>
<div class="flow"><div>锁定各自原目标<span>对象、起点、终点、预算</span></div><div>模型逐步调用<span>抓取 · 放置 · 交接</span></div><div>松手后再换对象<span>原生回执与稳定门禁</span></div><div>整项复查<span>所有物体仍在各自原目标</span></div></div></section>
<section><h2>实际执行结果</h2><img class="chart" src="results.png" alt="原生、API和页面任务落点误差与同一预算时间"><p>原生脚本通过：{native['terminal']['observation']['sim_time']:.3f} s，圆柱／长方体误差 {measure(native)} mm。它只验证底层工具与任务门禁，不是模型成绩。独立真实 DeepSeek API 通过：{api['actual_api_calls']} 次调用，{api['terminal']['observation']['sim_time']:.3f} s，误差 {measure(api)} mm。该版本含反馈修正，与原生脚本源版本分别留档。</p><p>实际页面任务：<b>{browser_text}</b>。{'模型 '+str(browser['actual_api_calls'])+' 次 API，'+format(browser['terminal']['observation']['sim_time'],'.3f')+' s；误差 '+measure(browser)+' mm。全部正式条件与源码版本见对应回执。' if browser else '读观察器只读跟踪同一个任务，没有机器人命令。'}两种模型运行均由模型自行制定中文阶段。阶段只绑定实际派发的执行回执，模型思考不再把未派发的下一动作标成运行中。单个物体成功不代表整项成功。</p><p>页面五次实际运动均成功、两个物体当前原目标检查都通过。模型认为固定相机不足以清晰确认最终落位，提交 uncertain，复查阶段为 5/6，原预声明检查为 9/10。保留原失败标记、模型判定和相机，不强改阶段完成、不另跑一个更容易的目标，也不更换模型输入视角来制造完整通过。物理执行证据与模型视觉证据分别报告。</p><div class="gallery">{''.join(gallery)}</div>{screenshot}<div class="links">{link('workbench/protocol.json','页面预声明与源码')}{link('workbench/result.json','页面原始回执')}{link('api-3/result.json','独立 API 回执')}{link('final-checks.log','38 项检查')}</div></section>
<section><h2>所有尝试与失败保留</h2><p>结果不能汇总为泛化成功率：这是同一声明开发任务的迭代，版本不同。首版 API 没有运动，第二版仅完成圆柱；其余目标没完成，整项仍判失败。收集器错误和模型错误分开记录。</p><table><thead><tr><th>尝试</th><th>整项</th><th>API</th><th>仿真/s</th><th>任务适配器 SHA</th><th>结论</th><th>证据</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>反馈与记忆各自负责什么</h2><p><b>当前任务状态</b>保留原目标、已完成对象、真正运行中的 job 和下一声明对象；这些坐标只属于这项任务，不提炼成长期运动记忆。<b>长期记忆</b>按对象、空手/持物、手臂、交接方向、意图和契约合并方法与关键失败教训，先筛选本任务对象再排序；默认最多 8 条，索引与共用规则最多 8192 UTF-8 字节。历史坐标、相机链接和原始轨迹不进入默认索引，详细依据按 ID 读取。删除方法使其稳定 ID 失效，不删除实验事实。</p>
<p>原始回执是审计与诊断档案，并非另一套默认记忆。多物体执行反馈简化为实际 job、原生成功条件与整体目标进度；完整规划/接触细节仍能按需读取。每个自由物体的原生接触、速度、释放和稳定在原 1ms 子步连续监测：独立采样器不改旧接触评分、物理参数或控制指令，32 步对照的 qpos/qvel/ctrl/time 完全一致。</p><p>当前生产记忆数据库的只读备份复核：长方体索引 3 条/2973 字节，圆柱索引 5 条/4731 字节；两者均无原始经历、实验链接或旧坐标。这是同一适配器的离线重建检查，非 API 传输抓包。{link('memory.json','对象索引检查')}</p><details><summary>查看页面精炼方法与支持次数</summary><table><thead><tr><th>方法或教训</th><th>支持记录</th><th>精炼内容</th></tr></thead><tbody>{memory_rows}</tbody></table></details>{'<a href="memory.png"><img class="chart" src="memory.png" alt="精炼记忆面板"></a>' if (OUT/'memory.png').exists() else ''}</section>
<section><h2>仍未解决的边界</h2><p>这轮是已注册直立瓶子、长方体与圆柱的显式仿真几何工具验收，TAMP 实际执行上述任务。没有把 GPU 搬运或 VLA 冒充本次工具；真实 cuRobo 运输的独立证据在 <a href="../20261009-handover/index.html">上一轮报告</a>。ACT 原权重已接入但完整任务迁移未通过，新形状没有匹配的 ACT/π0.5 契约。未覆盖未知网格、视觉抓取、倒置/柔软物体、任意动作或通用符号 TAMP。</p>
<p>当前 12 类标定证据均未完成独立实物验收。软件 FK 一致、仿真落点误差和回放只验证相应软件条件，不能证明真实空间/接触/控制精度。候选接触参数不是真机实测；控制器语义、相机标定、时间同步、质量/摩擦等差距仍在。SSH 公司电脑本轮零操作，真机未连接，没有新增自动启动。</p><details><summary>展开当前标定缺口</summary><table><thead><tr><th>项目</th><th>证据级别</th><th>仍需独立测量</th></tr></thead><tbody>{calibration_rows}</tbody></table></details><div class="links">{link('audit.json','源代码与服务审计')}{link('before-workbench.json','升级前保留的页面终态')}{link('cleanup.json','仅清理自建重复文件')}</div></section>
<section><h2>参考</h2><p>参考 <a href="https://lianegalanti.github.io/Pigey/">Pigey</a> 的多物体、工具、复核闭环，以及 <a href="https://code.claude.com/docs/en/memory">Claude Code</a> 的短索引和按需主题读取。这是本平台工程实现，不宣称移植其完整系统或复现它们的评测。</p></section>
</main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print('Report rendered from archived evidence; zero API, hardware or remote commands.')


if __name__ == '__main__': main()
