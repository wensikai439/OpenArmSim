"""Render and audit retained GPU-object evidence; no actor/model command."""
import hashlib
import html
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs/20261009-object-gpu'


def read(name): return json.loads((OUT / name).read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def esc(value): return html.escape(str(value))
def link(path, label): return f'<a href="{esc(path)}">{esc(label)}</a>'
def get(port, endpoint):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{endpoint}', timeout=10) as response: return json.load(response)


def audit(api, ui):
    from tools.check_object_gpu_ui import score
    verified = {**score(ui, read('workbench.protocol.json')), 'input_sha256': digest(OUT / 'workbench.json'),
        'scorer_sha256': digest(ROOT / 'tools/check_object_gpu_ui.py'), 'new_API_calls': 0, 'new_physics_actions': 0,
        'physical_thresholds_changed': False, 'scope': 'read-only source/field verification of the same retained UI task'}
    assert verified['passed']; dump('workbench-verified.json', verified)
    sources = []
    for name, folder in [('native.protocol.json', 'native'), ('api.protocol.json', 'api'), ('workbench.protocol.json', 'workbench')]:
        for path, expected in read(name)['source_sha256'].items():
            frozen = OUT / 'source' / folder / path
            assert digest(frozen) == expected
            sources.append({'protocol': name, 'path': path, 'frozen_matches': True,
                'current_matches': digest(ROOT / path) == expected})
    qa_files = ['sim/object_curobo.py', 'harness/object_gpu.py', 'tools/object_workbench.py', 'tools/object_gpu.py',
        'tools/check_object_gpu_contract.py', 'tools/check_object_gpu_score.py', 'tools/check_object_gpu_ui.py',
        'tools/report_object_gpu.py', 'ui/agent.js', 'README.md']
    for name in qa_files:
        frozen = OUT / 'source/final' / name; frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes((ROOT / name).read_bytes())
        sources.append({'path': name, 'sha256': digest(frozen), 'capture': 'final QA/metadata revision, after original tasks'})
    states = {}
    for port, expected in [(8765, '94cbe546b265:319'), (8766, '4fc9e7c32b75:640'), (8767, None)]:
        state = get(port, '/api/status'); manifest = get(port, '/api/manifest')
        assert not state.get('job') and not state['agent']['busy']
        assert manifest['source_status']['matches_disk'] and not manifest['model_evidence']['runtime_parameters']['changed_fields']
        if expected: assert state['observation_id'] == expected
        states[str(port)] = {'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
            'idle': True, 'manifest': manifest, 'calibration_gaps': state['calibration_gaps'],
            'scope': 'old-process guards cover imported sources only; new adapter code is separately frozen'}
    workers = {str(port): get(port, '/metadata') for port in [8002, 8004]}
    assert workers['8002']['protocol'] == 'openarm-curobo/2' and workers['8004']['protocol'] == 'openarm-object-curobo/1'
    assert workers['8004']['worker_source_sha256'] == digest(ROOT / 'tools/object_gpu.py')
    assert workers['8002']['worker_source_sha256'] == digest(ROOT / 'sim/curobo_server.py')
    models = [states['8767']['manifest']['model_evidence']['compiled']['sha256']]
    for case in read('native.json')['cases']:
        assert case['checks']['model_evidence_unchanged']
        models.append(read(case['case']['name'] + '/initial.json')['manifest']['model_evidence']['compiled']['sha256'])
    for record in [api, ui]:
        for receipt in record['receipts']:
            manifest = json.loads((ROOT / 'runs' / receipt['evidence']['run_id'] / 'manifest.json').read_text())
            models.append(manifest['model_evidence']['compiled']['sha256'])
    assert len(set(models)) == 1
    cleanup = read('audit.json').get('own_progress_cleanup', []) if (OUT / 'audit.json').exists() else []
    for name, record in [('api.progress.json', api), ('workbench.progress.json', ui)]:
        path = OUT / name
        if not path.exists(): continue
        progress = json.loads(path.read_text()); final = {t['id']: t for t in record['turns']}
        assert progress['turn_id'] == record['turn_id']
        assert record['stage_history'][:len(progress['stage_history'])] == progress['stage_history']
        for turn in progress['turns']:
            events = {e['id']: e for e in final[turn['id']]['trace']}
            for event in turn['trace']:
                assert event['id'] in events
                if event['status'] not in ('running', 'waiting'): assert events[event['id']] == event
        cleanup.append({'path': name, 'sha256': digest(path), 'reason': 'own progress cache fully covered by final completed events/history'})
    for case in read('native.json')['cases']:
        path = OUT / case['case']['name'] / 'progress.json'
        if not path.exists(): continue
        progress = json.loads(path.read_text())
        assert case['passed'] and progress['steps'] == case['steps']
        assert abs(progress['sim_time'] - case['sim_time']) <= 1e-9
        cleanup.append({'path': str(path.relative_to(OUT)), 'sha256': digest(path),
            'reason': 'own native progress cache; all exact skill receipts and final time retained in completed result'})
    allowed_caches = {'api.progress.json', 'workbench.progress.json', 'block-left/progress.json', 'can-right/progress.json'}
    assert all(item['path'] in allowed_caches for item in cleanup)
    cleanup = list({item['path']: item for item in cleanup}.values())
    result = {'source_audit': sources, 'services': states, 'workers': workers,
        'compiled_model_identical_native_API_UI_final': True, 'compiled_sha256': models[0],
        'goal_status': 'active; arbitrary object manipulation and independent calibration remain unproven',
        'late_metadata_fix': 'remove inherited unavailable tooltip reason after backend becomes ready; idle restart after completed UI task, no action rerun',
        'same_thresholds': True, 'own_progress_cleanup': cleanup, 'company_SSH_commands': 0,
        'hardware_access': False, 'transfer_ready': False, 'new_automations': 0}
    dump('audit.json', result)
    for item in cleanup: (OUT / item['path']).unlink(missing_ok=True)
    return result


def dump(name, value): (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    native = read('native.json'); api = read('api.json'); ui = read('workbench.json')
    assert native['complete'] and all(c['passed'] for c in native['cases']) and api['passed'] and ui['passed']
    checked = audit(api, ui)
    batches = [(c['case']['name'], c['steps'][1]['feedback'], c['steps'][-1]['feedback']['position_error_m'] * 1000,
        c['sim_time'], c['case']['object_id']) for c in native['cases']]
    for name, record, ident in [('独立 API', api, 'block'), ('页面 API', ui, 'can')]:
        gpu = next(r['feedback'] for r in record['receipts'] if r['execution']['skill'] == 'move_relative')
        batches.append((name, gpu, record['terminal']['agent']['plan']['goal_assessment']['position_error_m'] * 1000,
            record['terminal']['observation']['sim_time'], ident))
    labels = ['原生 · 长方体', '原生 · 圆柱', 'API · 长方体', '页面 · 圆柱']
    with tempfile.TemporaryDirectory(prefix='openarm-object-gpu-plot-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 9, 'axes.unicode_minus': False,
            'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained', facecolor='#fffefa')
        for ax, values, title, ylabel in [(axes[0], [b[1]['position_error_m'] * 1000 for b in batches],
            'GPU 携物运动的实际 TCP 误差', 'TCP 误差 / mm'), (axes[1], [b[2] for b in batches],
            '最终物体放置的实际落点误差', '物体误差 / mm')]:
            bars = ax.bar(range(4), values, color=['#7a8f81', '#b29378'] * 2, width=.6)
            ax.bar_label(bars, fmt='%.2f', padding=3); ax.set(xticks=range(4), xticklabels=labels, title=title, ylabel=ylabel)
            ax.set_facecolor('#fffefa'); ax.grid(axis='y', color='#e6e0d5'); ax.set_axisbelow(True)
        axes[0].axhline(12, color='#b29378', linestyle='--', linewidth=1); axes[0].set_ylim(0, 14)
        axes[1].set_ylim(0, 2.1)
        fig.savefig(OUT / 'results.png', dpi=180); plt.close(fig)
    rows = ''.join(f'<tr><td>{esc(labels[i])}</td><td>{b[1]["position_error_m"] * 1000:.2f} mm</td>'
        f'<td>{b[2]:.2f} mm</td><td>{b[1]["planning_timing"]["planner_wall_seconds"]:.2f} s</td><td>{b[3]:.2f} s</td></tr>' for i,b in enumerate(batches))
    frames = ''.join(f'<a href="{esc(f["url"])}" target="_blank" rel="noopener"><img src="{esc(f["url"])}" alt="{esc(f["label"])}">'
        f'<span>{esc(f["label"])} · {f["sim_time"]:.1f}s</span></a>' for f in ui['terminal']['agent']['goal_keyframes'])
    gaps = ''.join(f'<li>{esc(g)}</li>' for g in checked['services']['8767']['calibration_gaps'])
    style = re.search(r'<style>(.*?)</style>', (ROOT / 'runs/20261009-objects/index.html').read_text(), re.S)[1]
    style += '.gallery{grid-template-columns:repeat(4,1fr)}@media(max-width:700px){.gallery{grid-template-columns:1fr 1fr}}'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm · 多物体GPU携物</title><style>{style}</style></head><body><main>
<header><div><div class="eyebrow">OPENARM LAB / 2026.10.09</div><h1>让 cuRobo 携带实际选中的对象</h1><p>形状进入 GPU 碰撞表示，路径回到原生接触执行，再验证原物体目标。</p></div><a class="pill" href="http://127.0.0.1:8767/">多物体工作台 ↗</a></header>
<div class="stats"><div><b>2 个形状</b><span>长方体与圆柱实际 GPU 携物</span></div><div><b>4 / 4</b><span>页面模型阶段与回执确认</span></div><div><b>{ui['actual_api_calls']} 次</b><span>页面真实模型 API 请求</span></div><div><b>34 项</b><span>相关边界与回归检查通过</span></div></div>
<section><h2>补齐瓶子之外的携物表示</h2><p>旧适配器只接受 bottle 和三块固定瓶体几何。新增对象适配器按实际所选的原生 body／free joint／collision geoms，为圆柱和长方体完整体积构造保守球覆盖，表达在当前 TCP 坐标中。携物对象从静态世界障碍中移除，原瓶子和其他物体仍是障碍；GPU 回执必须确认相同对象及几何名称。</p><div class="flow"><div>当前实际夹持<span>重新读取接触与手—物体变换</span></div><div>完整体积球覆盖<span>仍使用原有 32 个球的上限</span></div><div>官方 GPU 规划<span>同一机器人配置、锁臂 FK 与碰撞对</span></div><div>原生复核与执行<span>所选自由关节、路径、夹持与目标反馈</span></div></div>
<p>球覆盖只用于规划预测，物体在实际 MuJoCo 中始终为自由刚体。GPU 拒绝、返回错误对象、旧起点、越限路径、原生碰撞或夹持丢失时停止，没有 TAMP 替代 GPU。模型可调用有参数的末端运动工具；抓取和放置继续使用 TAMP 的原生接触闭环。</p><p class="scope">两种新形状仍是 0.20 kg 合成开发资产，原瓶子为 0.55 kg。机器人、物体参数、控制器、相机和验收门槛未为本轮改动。接触候选仍为未实测标定的 dim4／NoSlip3。</p><div class="links">{link('source/native/sim/object_curobo.py','原生GPU适配源码')}{link('source/native/tools/object_gpu.py','实际GPU工作器')}{link('source/final/harness/object_gpu.py','模型工具与对象绑定')}{link('native.protocol.json','原生预声明协议')}</div></section>
<section><h2>原生、独立 API、页面 API 分别核验</h2><p>两项原生任务完成 pick → cuRobo 前移 3 cm → place。独立真实 API 用左手把长方体放到任务起点前方 8 cm；页面普通中文任务用右手把圆柱放到起点前方 6 cm。两项模型任务都恰好执行三个原生技能，并由模型自行制定四阶段和复查。</p><img src="results.png" alt="实际GPU末端运动和最终物体落点误差，明确区分两种度量"><table><thead><tr><th>验证层</th><th>GPU TCP误差</th><th>最终物体误差</th><th>GPU计算等待</th><th>仿真总时间</th></tr></thead><tbody>{rows}</tbody></table>
<p class="provenance">长方体携物 TCP 误差约 11.66 mm，接近原有 12 mm 门槛，不能称为精密运动。最终放置另走物体反馈，落点约 1.55–1.66 mm；这是仿真状态度量，不是外部实物测量。四项是声明开发案例，不是随机泛化成功率。</p><p>首份计划均为 120 秒、measured_delay。模型和底层规划等待计入保持上一指令的物理积分；目标、对象与预算没有在失败恢复中改变，也没有重置。冷构造和反馈重规划产生约 9–20 秒计算等待，仍需继续优化缓存及控制跟踪；没有把计算从成绩中剔除。</p><div class="gallery">{frames}</div><div class="links">{link('native.json','两项原生完整记录')}{link('api.json','独立API · 11次请求')}{link('api.protocol.json','独立API原协议')}{link('workbench.json','页面API原始记录')}{link('workbench-verified.json','源码与字段只读核验')}{link('workbench.protocol.json','页面原协议')}</div></section>
<section><h2>页面保持简洁，方法记忆保留条件</h2><p>技能仍按 TAMP／cuRobo／π0.5／ACT 切换；cuRobo 页面列出末端位姿与相对运动，场景工具折叠显示。主结构仍为左对话、右上视频、右下阶段。完整 GPU 球覆盖、碰撞对、缓存时间及原生回执留在实验记录中。</p><p>当前只有两条圆柱方法，抓取与放置；同类历史与本轮记录合并。一次普通末端运动不会自动变成持久知识。达到可归并条件后，GPU 方法仍区分对象身份和空手／持物状态，不把长方体携物、圆柱携物与空手移动混为一种已验证经验。原始轨迹不作为默认记忆注入。</p><div class="geometry"><a href="skills.png"><img src="skills.png" alt="cuRobo按后端切换的精简技能目录"></a><a href="memory.png"><img src="memory.png" alt="两条归并方法，保留证据与适用范围"></a></div>
<p>最后清除了能力状态从“未接入”变为“已接入”后残留的旧悬停说明；任务完成留档后才重启加载该元数据修复，未重跑动作。新启动是新场景，旧任务与回执作为历史保留。<code>./run objects</code> 现在按需启动自己的 GPU 工作器，Ctrl+C 关闭它拥有的子进程；已有监听器不会被替换，原 8002 工作器保持。</p>
<details><summary>检查结果、版本与缓存清理</summary><p>最终 34 项检查包括对象身份、完整体积覆盖、未选物体障碍保留、对象／持物条件的记忆分组、原目标锁定、错误执行编号、GPU 来源与禁止替代，以及组合动作的既有边界。首个回归命令引用了不存在的模块名，入口笔误保留在日志；修正命令后同一动作源码通过。页面评分从实际字段重算，不依赖缓存绿勾。仅在确认最终记录完整覆盖回执、事件和阶段历史后，删除本轮自建的原生、API与页面中间进度缓存。</p><div class="links">{link('final-regression.log','34项最终检查')}{link('regression.log','首次入口错误日志')}{link('contracts-final.log','5项对象GPU契约')}{link('audit.json','源码、模型、工作器与主平台审计')}{link('previous-workbench.json','升级前旧任务状态')}{link('workbench.png','实际GPU任务页面')}</div></details></section>
<section><h2>距离 Pigey 和 Real → Sim → Real 还有哪些缺项</h2><p>参照 <a href="https://lianegalanti.github.io/Pigey/">Pigey</a> 的观察、工具反馈、重新决策与验证闭环；携物表示使用 <a href="https://github.com/NVlabs/curobo/blob/main/curobo/_src/collision/attachment_manager.py">官方 cuRobo AttachmentManager</a> 接口。它们提供结构与实现依据，本平台没有复现 Pigey 的视觉感知、策略互补和评测成绩。</p><div class="limits"><p><b>还不能做到“什么动作都能做”。</b>当前只核验注册直立形状、单臂空手或独立持物的有界位姿目标。新形状双臂交接、另一只手持物时的携物碰撞、任意网格／倒卧物体、复杂接触任务、视觉定位和 VLA 对象契约仍待补齐。原固定瓶子演示保留。</p><p><b>当前仿真未达到精确标定或真机迁移验收。</b>当前运行时明确列出这些缺项：</p><ul>{gaps}</ul><p>下一步优先补齐双臂接收抓取与对象携物碰撞，加入未知位姿、障碍和负载的留出任务；物理标定另需独立测量及留出轨迹。软件 FK 一致、原生接触通过或回放吻合都不能替代物理标定。</p></div><p class="small">主平台 8765 和原动作工作台 8766 的物理状态与观测编号保持。公司 SSH／真机操作均为 0；没有新增定时启动。总目标继续进行。</p></section></main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print('GPU object report, retained source audit and model identity verified', flush=True)


if __name__ == '__main__': main()
