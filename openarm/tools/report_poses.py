"""Report every declared pose and version; partial grasp is never full success."""
import html
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261009-poses'


def read(path): return json.loads(path.read_text())
def esc(value): return html.escape(str(value))


def main():
    versions = ['baseline', 'baseline-2'] + [f'try-{i}' for i in range(1, 10)]
    trials = {name: read(OUT / name / 'results.json') for name in versions
        if (OUT / name / 'results.json').exists()}
    complete = {name: value for name, value in trials.items() if value['complete']}
    latest = list(complete)[-1]; cases = complete[latest]['cases']
    labels = ['右侧倒卧圆柱', '左侧斜卧圆柱', '左侧倒卧长方体', '右侧倒卧瓶子']
    def grade(row):
        if row['passed']: return 2
        if row['checks']['actual_contact_lift'] and row['checks']['full10s_hold']: return 1
        return 0
    names = [name for name in complete if name != 'baseline']
    with tempfile.TemporaryDirectory(prefix='openarm-pose-chart-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 9, 'axes.unicode_minus': False})
        fig, axes = plt.subplots(1, 2, figsize=(11.5, max(3.8, len(names) * .43)), layout='constrained', facecolor='#fffefa')
        matrix = np.array([[grade(row) for row in complete[name]['cases']] for name in names])
        axes[0].imshow(matrix, vmin=0, vmax=2, cmap=ListedColormap(['#e7e0d7', '#d9b995', '#849b8a']), aspect='auto')
        axes[0].set(xticks=range(4), xticklabels=[label.replace('侧', '侧\n') for label in labels],
            yticks=range(len(names)), yticklabels=names, title='同一组目标：开发迭代，不能当作成功率')
        for y in range(len(names)):
            for x in range(4): axes[0].text(x, y, ['抓取未确认', '抓取＋持稳', '整项通过'][matrix[y,x]], ha='center', va='center', fontsize=8)
        for axis in axes:
            axis.set_facecolor('#fffefa'); axis.spines[['top','right']].set_visible(False)
        times = [row['sim_time'] for row in cases]
        bars = axes[1].barh(range(4), times, color=['#849b8a' if row['passed'] else '#b69b82' for row in cases], height=.5)
        axes[1].bar_label(bars, fmt='%.2f s', padding=3)
        axes[1].set(yticks=range(4), yticklabels=labels, xlim=(0,140), xlabel='累计仿真时间 / s', title=f'{latest}：原120秒预算，计入底层规划')
        axes[1].invert_yaxis(); axes[1].axvline(120, color='#a76d58', linestyle='--', linewidth=1)
        axes[1].grid(axis='x', color='#e8e1d7', linewidth=.5); axes[1].set_axisbelow(True)
        fig.savefig(OUT / 'results.png', dpi=180); plt.close(fig)
    rows = []
    for version, trial in trials.items():
        protocol = read(OUT / version / 'protocol.json')
        sha = protocol['source_sha256']['sim/oriented_grasp.py'][:12]
        for i, row in enumerate(trial['cases']):
            code = (row.get('error') or {}).get('code') or '完整通过'
            status = '收集器错误／零运动' if version == 'baseline' else ['未确认抓取', '仅抓取＋10秒持稳', '完整通过'][grade(row)]
            path = version + '/' + row['case']['name'] + '/result.json'
            rows.append(f'<tr><td>{esc(version)}</td><td>{labels[i]}</td><td>{status}</td><td>{row["sim_time"]:.3f}</td><td>{esc(code)}</td><td><code>{sha}</code></td><td><a href="{path}">原始回执</a></td></tr>')
    gallery = []
    for label, row in zip(labels, cases):
        folder = latest + '/' + row['case']['name']
        status = ['未确认抓取', '抓取与10秒持稳通过，放置未通过', '原目标完整通过'][grade(row)]
        images = ''.join(f'<a href="{folder}/{image}.jpg" target="_blank" rel="noopener"><img src="{folder}/{image}.jpg" alt="{label} · {title}"><span>{title}</span></a>'
            for image, title in [('start','原定初始姿态'),('final','实际终态')])
        gallery.append(f'<article><h3>{label}</h3><p>{status} · {esc((row.get("error") or {}).get("code", "PASS"))}</p><div class="pair">{images}</div></article>')
    calibration = read(ROOT / 'sim/calibration.json')
    calibration_rows = ''.join(f'<tr><td>{esc(item["label"])}</td><td>{esc(item["status"])}</td><td>{esc(item["needed"])}</td></tr>' for item in calibration['parameters'])
    api_rows = []
    for name in ('api-1', 'api-2'):
        path = OUT/name/'results.json'
        if not path.exists(): continue
        record = read(path)
        if not record['complete']: continue
        assessment_path = OUT/name/'assessment.json'
        corrected = {row['case']['name']:row for row in read(assessment_path)['cases']} if assessment_path.exists() else {}
        for row in record['cases']:
            ident = row['case']['name']; status = corrected.get(ident, row)['passed']
            place = next((r['feedback'] for r in row['receipts'] if r['execution']['skill']=='place'), {})
            error = f'{place["position_error_m"]*1000:.2f} mm' if place.get('primitive_success') else '未完成放置'
            reason = ('GPU字段重核通过；原False回执保留' if name=='api-1' and status else
                '首次计划缺少原生前置检查，零运动' if name=='api-1' and ident=='can-left-oblique' else
                '起点与篮子重叠，等待中被挤动；不改场景重跑' if ident=='can-left-oblique' else
                '完整原任务与正式复查通过' if status else '原目标联合抓取搜索失败')
            links = f'<a href="{name}/{ident}/result.json">原回执</a>'
            if ident in corrected: links += f' · <a href="{name}/{ident}/assessment.json">字段重核</a>'
            api_rows.append(f'<tr><td>{name}</td><td>{esc(ident)}</td><td>{"通过" if status else "未通过"}</td><td>{row["actual_api_calls"]}</td><td>{row["terminal"]["observation"]["sim_time"]:.3f}</td><td>{error}</td><td>{reason}</td><td>{links}</td></tr>')
    admission_text = '新起点检查已通过离线接口检查，真实API拒绝流程单独验证。'
    for name in ('admission-api', 'admission-2'):
        path = OUT/name/'results.json'
        if path.exists() and read(path)['complete']:
            row = read(path)['cases'][0]
            admission_text += f' {name}：原搬运目标仍失败；'+('模型正式报告起点无效，零机器人技能、没有伪造执行回执。' if row.get('admission_gate_verified') else '阻止了运动，但报告流程未通过；原失败保留。')
            admission_text += f' <a href="{name}/can-left-oblique/result.json">回执</a>。'
    source_style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261009-scene/index.html').read_text(), re.S)[1]
    style = source_style + '.pose-gallery{display:grid;grid-template-columns:1fr 1fr;gap:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.pair img{width:100%;border-radius:8px}.pair span{display:block;font-size:12px;padding:5px 0}table{display:block;overflow:auto}details{margin-top:18px}h3{font-size:15px}code{font-size:12px}@media(max-width:680px){.pose-gallery{grid-template-columns:1fr}}'
    passes = sum(row['passed'] for row in cases)
    counts = read(OUT/'checks.json') if (OUT/'checks.json').exists() else {'count': '待确认'}
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 姿态抓取</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.09</div><h1>抓取前，先看物体姿态</h1><p>解除直立假设，同时保留原目标、原生接触和真实执行门槛。</p></div><a class="pill" href="http://127.0.0.1:8767/">多物体工作台 ↗</a></header>
<div class="stats"><div><b>4 个固定起点</b><span>倒卧与倾斜，目标从未替换</span></div><div><b>{passes} 项完整通过</b><span>{latest}开发结果，不是泛化成功率</span></div><div><b>{counts['count']} 项检查</b><span>接口、状态保护与记忆适用范围</span></div><div><b>尚未标定</b><span>显式仿真几何，不是真实迁移</span></div></div>
<section><h2>新增的能力与明确边界</h2><p>原规划器拒绝全部四个非直立起点。现在可用注册盒子、圆柱与瓶体的当前碰撞几何，生成物体坐标系下的候选抓取点、方向和预抓取净空，再通过完整逆解、连接路径、原生执行器、双指接触和实际抬升检查。物体仍是自由刚体，没有绑定、传送或修改物理参数。直立对象保留已有交接规划器，原瓶子演示保留。</p><p>若计划已锁定桌面目标，抓取候选会在沙盒数据中预检查该原目标的直立放置逆解与降低段，并使用原放置求解精度。这仍是规划预测：实际抓取关系、运输、松手、支撑和稳定必须重新验证。数值优化只提供新逆解种子，原位置、角度与碰撞判据仍决定能否执行。</p><p>当前实现是几何候选提供器，不是通用符号TAMP、未知网格感知或GraspGenX模型推理。{latest}四个原任务保留所有失败；不能把“抓起了”或“能生成6D位姿”称为完整泛化。</p></section>
<section><h2>原定任务与完整验收</h2><p>每项都要求选定机械臂抓起原物体、原生持续夹持10秒、放到原桌面world XY目标并直立、松手、撤离、支撑稳定至少1秒；原10mm落点容差、120秒measured_delay及接触门槛不变。起点姿态仅在t=0、首条记录指令前初始化；任务内没有重置、移动物体或加时。当前四个起点是合成开发夹具，非真机测量。第一个baseline是收集器签名错误／零运动；baseline-2才是正确原规划器对照。</p><img class="chart" src="results.png" alt="四个固定姿态各版本完整通过和部分抓取结果"><div class="pose-gallery">{''.join(gallery)}</div><details><summary>展开所有版本、失败与完整回执</summary><table><thead><tr><th>版本</th><th>固定场景</th><th>原任务</th><th>仿真/s</th><th>原因</th><th>候选层SHA</th><th>证据</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details></section>
<section><h2>真实API闭环：持稳、计算运输、GPU运动、放置</h2><p>模型标签为deepseek-flash，使用已保存配置。每个版本都完整保留四个原定起点与目标，每项只有一次声明尝试，模型自行制定中文阶段与调用工具。长方体与瓶子完成原10秒持稳、真实GPU运输、原目标放置和正式success复查；两个圆柱没有完成，不能称为开放物体泛化。pick可选hold_seconds在原控制指令下核验2–10秒实际持稳；它是原生验证，不是等待暂停的墙钟时间。</p><p>抓取的原目标朝向预检查通过已有resolve_placement_transit读取，使用当前实际夹持变换，模型显式调用返回的运动请求，最后仍需原生place。不会自动运动。api-1的收集器误读了不存在的顶层backend字段；用原请求、原生路径核验、实际规划器、无回退标记和冻结工作器指纹重核，所有其它原检查不变，原False回执完整保留。api-2已修正字段读取与首次几何计划的检查顺序；单物体声明内复用原生新观测/特权门禁，不用模型猜一套仪式。</p><table><thead><tr><th>版本</th><th>原任务</th><th>结果</th><th>API</th><th>仿真/s</th><th>最终误差</th><th>说明</th><th>依据</th></tr></thead><tbody>{''.join(api_rows)}</tbody></table></section>
<section><h2>失败如何帮助定位</h2><p>规划前的物体点不能在开爪、接近后继续盲用。曾观察到夹爪提前碰动右侧圆柱，使原抓取点过期；现增加预接近净空，并在实际接近后重新读取同一几何抓取带。实验性的微步跟踪伺服使原本能完成的长方体失败，已撤回，失败源码和轨迹保留。增加候选数量也可能耗尽规划预算；所有迭代分别列出，不取各版本最好结果拼成成功。</p><p>倒卧瓶子的直接TAMP路线仍在精确放置失败；加入本次抓取预检查指向的真实GPU运输后，才完整通过。所有原生/API版本分别报告，没有把路径预测当成原生成功。右侧圆柱的原目标联合搜索仍失败。左侧圆柱的原夹具在t=0就与篮底相交8.83mm、篮壁相交24.44mm：这是无效起点，不能拿它当作有效泛化样本，也未把它挪到更容易的位置。<a href="initial-contacts.json">独立初始接触检查</a>。</p><p>{admission_text}当前增加原生起点检查，保存原任务的拒绝原因，禁止在该任务继续运动，并允许用新观测正式提交failure/uncertain；success仍拒绝。额外检查不改变物理参数或原验收条件。</p></section>
<section><h2>方法合并，详细依据按需读取</h2><p>参考Claude Code的短索引与主题按需读取，以及LangMem的提取、合并、更新思路。本平台按对象、手臂、空手／持物、交接方向、抓取意图与原生起始姿态合并方法；支持次数增长不会新增同一张卡。新增非直立条件在默认索引和详细记忆使用同一稳定ID，删除后不会重新出现；模型猜测不能升级为原生成功记忆。</p><p>默认最多8条、索引和公共规则最多8192 UTF-8字节，不包含原始轨迹、历史坐标或相机链接。原始记录是可核查的实验凭据，按ID查阅，区别于长期记忆。页面的“适用范围与实验依据”按需显示条件，长面板标题和关闭按钮始终可见，布局不变。当前已存方法与新增条件管线均经过检查；不把离线重建当成API传输抓包。</p><a href="memory.png"><img class="chart" src="memory.png" alt="实际页面精炼方法及按需展开的适用条件"></a><div class="links"><a href="checks.json">检查结果</a><a href="final-checks.log">完整检查日志</a><a href="audit.json">运行服务与源码审计</a></div></section>
<section><h2>标定与真实迁移仍不成立</h2><p>12类物理标定没有完成独立实物验收。软件FK一致、仿真接触或本次落点误差不证明真实基座、TCP、镜头、控制器、摩擦和同步准确。ACT真机权重已接入的历史结果仍是完整仿真任务未通过；π0.5没有匹配的OpenArm权重契约。本轮没有SSH操作、硬件访问、自动启动或训练新模型。</p><details><summary>查看仍需独立测量的参数</summary><table><thead><tr><th>类别</th><th>证据</th><th>需要什么</th></tr></thead><tbody>{calibration_rows}</tbody></table></details></section>
<section><h2>参考依据</h2><p><a href="https://github.com/NVlabs/GraspGenX">GraspGenX</a>公开实现把6D候选、cuRobo规划与Newton/MuJoCo接触仿真分层；支持新夹爪还需要URDF、开合配置和扫掠体积。本轮参考这一接口划分，没有安装或调用该模型，也未验证其作者声明在OpenArm上成立。</p><p><a href="https://code.claude.com/docs/en/memory">Claude Code记忆文档</a>说明短索引启动加载、主题文件按需读取；<a href="https://github.com/langchain-ai/langmem">LangMem</a>提供记忆提取、合并和更新。这里使用可追溯的原生回执提炼规则，没有让模型随意覆盖物理事实。<a href="https://lianegalanti.github.io/Pigey/">Pigey</a>是多次观察、工具执行、验证与恢复的闭环参考，不宣称本平台已达到其报告性能。</p></section>
</main></body></html>'''
    (OUT/'index.html').write_text(page)
    print(json.dumps({'latest':latest,'full_passes':passes,'declared_cases':len(cases),'not_a_benchmark':True}))


if __name__ == '__main__': main()
