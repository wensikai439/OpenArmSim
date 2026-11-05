"""Build a reviewable report from native recovery and real-API receipts."""
import hashlib
import html
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261008-recovery'


def read(name):
    return json.loads((OUT / name).read_text())


def signature(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    changed = [name for name, digest in manifest['hashes'].items()
               if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    return {'run_id': manifest['run_id'], 'manifest': str((path / 'manifest.json').relative_to(ROOT)),
            'matches_current_sources': not changed, 'changed': changed}


def main():
    cases = read('source-final/summary.json')
    routed = read('source-routed/summary.json')
    short, long = read('recovery40-final/summary.json'), read('recovery60-final/summary.json')
    api = read('api-ready.json')
    all_api = [read(name) for name in ('api-recovery.json', 'api-final.json', 'api-ready.json')]
    viewer = read('viewer-check.json')
    checks = (OUT / 'checks-ready.log').read_text()
    assert 'Ran 146 tests' in checks and checks.rstrip().endswith('OK')
    assert len(cases) == 6 and sum(c['status'] == 'passed' for c in cases) == 4
    assert len(routed) == 6 and sum(c['status'] == 'passed' for c in routed) == 5
    assert not read('source-final/protocol.json')['handover_recovery']
    assert read('source-routed/protocol.json')['handover_recovery']
    assert cases[-1]['failure']['code'] == 'ik_unreachable'
    assert short['status'] == 'failed' and short['time_budget_s'] == 40
    assert long['status'] == 'passed' and long['time_budget_s'] == 60
    assert long['reproduction_residual_m'] < 1e-8 and short['reproduction_residual_m'] < 1e-8
    assert api['status'] == 'passed' and api['actual_api_calls'] > 0
    assert api['required_sequence_matched'] and api['goal_unchanged'] and api['plan']['time_budget_s'] == 60
    assert viewer['passed'] and viewer['manual_frame_visible']
    signatures = [signature(ROOT / c['run_path']) for c in cases + routed + [short, long]]
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/agent.sqlite3') + '?mode=ro', uri=True) as db:
        for execution in api['plan']['execution_tools']:
            row = db.execute('SELECT run_id FROM episodes WHERE id=?', (execution['id'],)).fetchone()
            assert row and row[0], 'missing frozen native API execution receipt'
            signatures.append(signature(ROOT / 'runs' / row[0]))
    for item in signatures:
        item['matches_native_control_sources'] = not (set(item['changed']) - {'server.py'})
    assert all(s['matches_native_control_sources'] for s in signatures), 'native control changed; retest required'
    audit = {
        'status': 'verified_progress', 'goal_status': 'active', 'checks_passed': 146,
        'actual_api_calls': api['actual_api_calls'], 'turn_id': api['turn_id'],
        'round_api_calls_total': sum(a['actual_api_calls'] for a in all_api),
        'retained_api_runs': [{'file': name, 'calls': a['actual_api_calls'], 'status': a['status']}
                              for name, a in zip(('api-recovery.json', 'api-final.json', 'api-ready.json'), all_api)],
        'required_sequence_matched': api['required_sequence_matched'], 'goal_unchanged': api['goal_unchanged'],
        'native_direct_40s': {'passed': 4, 'total': 6, 'seed': 2718, 'handover_recovery': False},
        'native_routed_40s': {'passed': 5, 'total': 6, 'seed': 2718, 'handover_recovery': True},
        'recovery_40s': {'status': short['status'], 'sim_seconds': short['sim_seconds']},
        'recovery_60s': {'status': long['status'], 'sim_seconds': long['sim_seconds'],
                         'budget_declared_before_prefix_replay': long['budget_declared_before_prefix_replay'],
                         'position_error_m': long['steps'][-1]['feedback']['position_error_m']},
        'source_signatures': signatures, 'viewer_check': viewer,
        'source_audit_note': 'Native probes precede server.py capture-camera whitelist fix only; control sources match. API and UI verified after that fix.',
        'scope': 'declared upright bottle; simulator oracle; development checks, not a benchmark',
        'decision_wait_physics': 'paused between tools; simulation budget is not continuous wall time',
        'physical_calibration_verified': False, 'transfer_ready': False, 'hardware_access': False,
        'rejected_controller': 'candidate/tamp.py; not deployed',
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    rows = ''.join(
        f'<tr><td>{html.escape(c["case"]["id"])} · 初态 {c["case"]["episode"]}</td>'
        f'<td>{"完成" if c["status"] == "passed" else html.escape(c["failure"]["code"])}</td>'
        f'<td>{c["sim_seconds"]:.2f} s</td><td><a href="source-final/{c["case"]["id"]}.json">直接 ↗</a> · '
        f'<a href="source-routed/{c["case"]["id"]}.json">允许换手 ↗</a></td></tr>'
        for c in cases)
    placement = api['plan']['goal_assessment']
    frozen = api['terminal']['agent']['goal_keyframes']
    trail = ''.join(
        f'<figure><a href="{html.escape("../" + f["url"].removeprefix("/runs/"))}">'
        f'<img src="{html.escape("../" + f["url"].removeprefix("/runs/"))}" alt="{html.escape(f["label"])}"></a>'
        f'<figcaption>{html.escape(f["label"])} · {f["sim_time"]:.2f} s'
        f'{" · 未完成" if f.get("outcome") == "failure" else ""}</figcaption></figure>' for f in frozen)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenArm · 同场景恢复验收</title><style>
*{box-sizing:border-box}body{margin:0;background:#f7f5ef;color:#302f2b;font:15px/1.75 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:36px 28px 60px}a{color:#906849;text-underline-offset:4px}header{display:flex;justify-content:space-between;gap:18px;align-items:center;border-bottom:1px solid #ded8cd;padding-bottom:20px}.eyebrow{font-size:12px;letter-spacing:2px;color:#908574}h1{font-size:29px;letter-spacing:-.5px;margin:12px 0 8px}h2{font-size:19px;margin:0 0 12px}p{color:#6f675b;margin:9px 0}section{background:#fffefa;border:1px solid #e3ddd1;border-radius:16px;padding:24px;margin:22px 0}.pill{color:#59694c;background:#eff2e8;border:1px solid #dce2d1;border-radius:30px;padding:4px 12px;white-space:nowrap;font-size:13px}.stats{display:flex;gap:34px;flex-wrap:wrap;margin:22px 0}.stats b{display:block;font-size:27px;line-height:1.4}.stats span{color:#8a7d6b;font-size:13px}table{border-collapse:collapse;text-align:left;width:100%;font-size:14px}td,th{padding:11px 9px;border-bottom:1px solid #e9e3d8}th{font-weight:500;color:#8c7d67}.scroll{overflow:auto}img{display:block;width:100%;border-radius:9px}.trail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption,.small{font-size:13px;color:#857867;margin-top:6px}details{margin-top:17px}summary{cursor:pointer;color:#78614b}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:700px){main{padding:20px 14px}section{padding:18px}header{align-items:flex-start}h1{font-size:24px}.trail,.pair{grid-template-columns:1fr}td,th{white-space:nowrap}}
</style><main><header><div><div class="eyebrow">OPENARM LAB / 2026.10.08</div><h1>失败后，保持目标继续完成</h1><p>同一用户任务内保留物体状态、失败回执与目标，模型依据新反馈换手恢复。</p></div><a href="/" class="pill">返回工作台 ↗</a></header>
<div class="stats"><div><b>146</b><span>回归检查通过</span></div><div><b>CALLS</b><span>本次真实 API 请求</span></div><div><b>API_ERROR mm</b><span>API 任务的原生物体误差</span></div><div><b>5 / 6</b><span>40 秒直接放置开发检查</span></div></div>
<section><h2>真实模型：左取 → 放置失败 → 右接 → 放置完成</h2><p>已保存的 deepseek-flash API 提交首个计划时锁定 60 秒仿真预算。模型在左手放置失败后读取当前反馈，交给右手，仍放到本次任务起始瓶位前方 6 cm。全程未重置、未修改原目标、未失败后加时。</p><p>原生验收：物体误差 API_ERROR mm，允许 TOL mm；松手、直立支撑并稳定 STABLE 秒。模型阶段与工具回执绑定，不允许模型自行勾选完成。</p><div class="trail">TRAIL</div><p class="small">关键帧来自整个用户任务；包括失败尝试。此处为调试全景，不冒充模型的三路固定相机。点击可看原图。</p><div class="links"><a href="api-final.json">逐次 API、工具与反馈 ↗</a><a href="api-final.protocol.json">事先声明的任务 ↗</a><a href="audit.json">源码与验收审计 ↗</a></div></section>
<section><h2>40 秒与 60 秒，分别报告</h2><p>种子 2718 的六组初态／瓶位仍有一组左手直接放置逆解失败。按原生指令重现该失败状态，物体残差小于 10⁻⁸ m；从该状态先上移 6 cm，再左→右交接和放置。</p><div class="scroll"><table><thead><tr><th>事先锁定预算</th><th>结果</th><th>仿真用时</th><th>物体误差</th><th>回执</th></tr></thead><tbody><tr><td>40 s</td><td>未完成：到达预算</td><td>SHORT s</td><td>尚未完成稳定验收</td><td><a href="recovery40-final/summary.json">全部结果 ↗</a></td></tr><tr><td>60 s</td><td>恢复完成</td><td>LONG s</td><td>LONG_ERROR mm</td><td><a href="recovery60-final/summary.json">全部结果 ↗</a></td></tr></tbody></table></div><p class="small">预算在重放任何动作之前固定。不同预算属于不同开发条件，不能合并成同一个成功率。原有固定演示和学习策略的 40 秒契约未放宽。</p><details><summary>展开六组直接放置结果</summary><div class="scroll"><table><thead><tr><th>案例</th><th>结果</th><th>仿真用时</th><th>证据</th></tr></thead><tbody>ROWS</tbody></table></div><a href="source-final/protocol.json">全部案例在开始前生成 ↗</a></details></section>
<section><h2>工作台与反馈</h2><a href="platform.jpg"><img src="platform.jpg" alt="工作台：左侧聊天、上方相机、下方模型阶段及整个任务关键帧"></a><p>保留左侧对话、右上场景、右下阶段。失败帧用轻量提示区分；手动留帧保存当前观察视角，并加入本次任务帧列。观测编号、本体状态和三路模型相机在留帧前后保持一致。</p><div class="links"><a href="viewer-check.json">页面与相机隔离证据 ↗</a><a href="checks-deployed.log">146 项检查 ↗</a></div></section>
<section><h2>结论的适用范围</h2><p>这些是已知直立瓶子、仿真真值工具与有限初态下的工程检查。泛化还不足；未形成开放物体、视觉定位和任意动作能力，也未达到与 Pigey 同条件可比较的评测。</p><p>模型调用工具之间物理仍暂停，以上秒数是仿真动作时间，不能当作真机连续运行的时延结论。控制器、几何和成功阈值未因这次恢复实验而替换或放宽。</p><p>基座与 TCP 的独立实测、相机内外参／畸变／曝光时钟、夹爪传动与力、物体质量／惯量／摩擦、采集时控制器版本及状态仍需独立核验。软件 FK 一致、毫米级仿真目标误差和回放吻合均不等于物理标定完成。real-to-sim-to-real 尚未通过。</p><p>cuRobo 当前仅验证空手单臂；带物路径未接入。ACT 完整任务未通过，OpenArm π0.5 无匹配权重。连续决策等待、通用物体／抓取契约与带物规划是下一批具体缺口。</p><details><summary>保留的中间实验与设计来源</summary><p>支持面释放和近目标伺服候选出现回归，已拒绝部署；恢复生产中此前通过精度检查的控制器。失败点瓶底接近台面，但无实际支撑接触，不能只按高度推断可以松手。</p><div class="links"><a href="candidate/scope.json">未部署候选 ↗</a><a href="settling-probe.json">真实接触与静置探针 ↗</a><a href="supported/summary.json">支持面候选结果 ↗</a><a href="object-recovery/summary.json">伺服候选结果 ↗</a></div><p><a href="https://lianegalanti.github.io/Pigey/">Pigey</a> 的新观测、单工具、反馈、验证／恢复循环作为设计参考；<a href="https://code.claude.com/docs/en/memory">Claude Code memory</a> 的小索引与按需详情用于记忆设计。不是对应项目的完整复现。</p></details><p class="small">未访问硬件，未写远程公司电脑，未创建定时启动。原始证据与旧实验保留；目标继续 active。</p></section></main></html>'''
    values = {
        'CALLS': api['actual_api_calls'], 'API_ERROR': f'{placement["position_error_m"] * 1000:.2f}',
        'TOL': f'{placement["tolerance_m"] * 1000:.1f}', 'STABLE': f'{placement["stable_seconds"]:.2f}',
        'SHORT': f'{short["sim_seconds"]:.2f}', 'LONG_ERROR': f'{long["steps"][-1]["feedback"]["position_error_m"] * 1000:.2f}',
        'LONG': f'{long["sim_seconds"]:.2f}', 'TRAIL': trail, 'ROWS': rows,
    }
    page = page.replace('api-final.', 'api-ready.').replace('checks-deployed.log', 'checks-ready.log')
    page = page.replace('<b>5 / 6</b><span>40 秒直接放置开发检查</span>',
                        '<b>4 / 6</b><span>40 秒直接放置；允许换手为 5 / 6</span>')
    page = page.replace('种子 2718 的六组初态／瓶位仍有一组左手直接放置逆解失败。',
                        '种子 2718 的六组初态／瓶位，直接路径完成 4/6；事先允许执行前不可达时换手，完成 5/6。剩余一组左手放置在运动后逆解失败。')
    page = page.replace('点击可看原图。', f'点击可看原图。本轮三次验证共 {sum(a["actual_api_calls"] for a in all_api)} 次真实 API 请求；主流程最后一次为 {api["actual_api_calls"]} 次，早前调用记录也保留。')
    page = page.replace('观测编号、本体状态和三路模型相机在留帧前后保持一致。',
                        '观测编号、本体状态和三路模型相机在留帧前后保持一致。原生探针录制后只更新了 HTTP 留帧参数校验；其控制源码一致，API 与页面在该接口更新后重新验收。完整哈希差异列在审计中。')
    for marker in sorted(values, key=len, reverse=True):
        page = page.replace(marker, str(values[marker]))
    (OUT / 'index.html').write_text(page)
    print(json.dumps({'status': audit['status'], 'native_signatures': len(signatures),
                      'actual_api_calls': api['actual_api_calls']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
