"""Build the small local demonstration record from immutable case results."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261014-demo'
CASES = [('ui-arms-3', '页面输入：双臂定向移动'), ('point-2', '瓶子放到指定桌面坐标'),
         ('basket-5', '自主恢复并放入篮子')]


def main():
    cases = []
    for folder, label in CASES:
        path = OUT / folder / 'result.json'
        if not path.exists():
            cases.append({'folder': folder, 'label': label, 'status': 'incomplete'})
            continue
        data = json.loads(path.read_text())
        terminal = data['terminal']
        budget_ok = terminal['observation'].get('task_window', {}).get('remaining_seconds', 0) > 0
        errors = [r['summary']['metrics']['tamp'].get('position_error_m')
                  for r in data['native_receipts'] if r['passed']]
        cases.append({'folder': folder, 'label': label,
            'status': 'passed' if data['status'] == 'passed' and budget_ok else 'failed',
            'turn_id': data['turn_id'], 'source_revision': data['source_revision'],
            'checks': {**data['checks'], 'task_window_not_exhausted': budget_ok},
            'api_calls': data['actual_api_calls'], 'model_verdict': data['plan'].get('verdict'),
            'position_error_mm': round(max((x for x in errors if x is not None), default=0) * 1000, 3),
            'receipts': [{'skill': r['tool'], 'run_id': r['run_id'], 'passed': r['passed'],
                          'planner': (r.get('arguments') or {}).get('planner', 'tamp')}
                         for r in data['native_receipts']]})
    audit = json.loads((OUT / 'audit.json').read_text()) if (OUT / 'audit.json').exists() else {}
    report = {'scope': 'Fixed development demonstrations through the saved real API; not a generalization benchmark or sim-to-real proof',
        'cases': cases, 'runtime': audit,
        'complete': all(c['status'] == 'passed' for c in cases) and audit.get('passed') is True}
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    rows = []
    for case in cases:
        receipts = ' · '.join(f'<a href="../{html.escape(r["run_id"])}/summary.json">{html.escape(r["skill"])} / {html.escape(r["planner"])}{" ✓" if r["passed"] else " 未通过"}</a>'
                              for r in case.get('receipts', []))
        rows.append(f'<tr><td><a href="{case["folder"]}/result.json">{case["label"]}</a></td>'
                    f'<td>{case["status"]}</td><td>{case.get("api_calls", "—")}</td>'
                    f'<td>{case.get("position_error_mm", "—")}</td><td>{receipts}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>演示验收 · OpenArm Lab</title><style>body{font:15px/1.7 system-ui,sans-serif;color:#393b35;background:#faf8f4;margin:0}main{max-width:1100px;margin:auto;padding:42px 26px}h1{font-size:28px}a{color:#9b674f}p{max-width:90ch}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:14px;text-align:left;border-bottom:1px solid #e7e0d7}img{width:100%;border-radius:12px;border:1px solid #e7e0d7}details{margin:24px 0}code{font-size:13px}.quiet{color:#7b776d}</style><main>
<p class="quiet">OPENARM LAB · 2026-10-14—15</p><h1>API 自主动作演示验收</h1>
<p>默认桌面只有黑瓶子和篮子。真实模型API自行规划、调用工具、读取回执与复查；原生MuJoCo通过执行器与接触运动，没有粘连或瞬移。目标受可达工作区与碰撞约束。</p>
<table><thead><tr><th>任务与完整记录</th><th>验收</th><th>实际API次数</th><th>最大位置误差 / mm</th><th>原生回执</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table>
<p class="quiet">这些是固定开发案例，不作为泛化成功率；不同案例列出各自源码哈希。失败的中间尝试保留，自主恢复后仍须实现原目标。页面任务的收集时间包含等待人工式页面提交，不用作推理速度排名。</p>
<h2>页面实际验收</h2><img src="demo.png" alt="通过页面发送双臂任务后的实际界面">
<p><a href="audit.json">当前服务审计</a> · <a href="summary.json">机器可读汇总</a> · <a href="checks-final-release.log">完整回归日志</a> · <a href="/">返回平台</a></p>
<details><summary>保留的问题与边界</summary><p>初期执行合同转发、阶段误判、交接撤离与物体身份问题均保留原失败记录。basket-4因API HTTP503失败，ui-arms原始记录只完成物理动作、最终复查超时，不能按旧收集器的passed宣称完整通过；各自结果和flow-audit仍在本目录。较早front-2、point、arms-2不与当前记录合并成成功率。</p>
<p>ACT/cuRobo连接就绪与任务性能分开；π0.5未接入。当前TAMP使用显式仿真几何；独立实物标定、视觉定位泛化与sim-to-real尚未验收。</p></details></main>'''
    (OUT / 'index.html').write_text(page)
    print(json.dumps({'complete': report['complete'], 'cases': [(c['folder'], c['status']) for c in cases]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
