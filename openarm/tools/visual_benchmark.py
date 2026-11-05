"""Frozen RGB-only grasp evaluation and video gallery. Local simulation only."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def verify(protocol):
    for name, expected in protocol['sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    if 'legacy_manifest_sha256' in protocol:
        path=ROOT/'skills/bottle_recovery/manifest.json'
        if digest(path)!=protocol['legacy_manifest_sha256']:
            raise ValueError('legacy reference manifest changed')
        for name,expected in json.loads(path.read_text())['runtime_sources'].items():
            if digest(ROOT/name)!=expected:raise ValueError('native dependency changed: '+name)


def gallery(folder, protocol, rows):
    esc = html.escape
    passed = sum(r['right_grasp_lift_success'] for r in rows)
    cards, table = [], []
    for row in rows:
        name = row['name']; result = row['right_grasp_lift_success']
        badge = '通过' if result else '失败'
        video = folder / name / 'rollout.mp4'
        if video.exists():
            cards.append(f'''<article><div class="label">{esc(row['label'])}<b class="{'pass' if result else 'fail'}">{badge}</b></div>
              <video controls preload="metadata" playsinline src="{name}/rollout.mp4"></video>
              <p>{esc(row.get('error') or '双指持续夹持 · 下半瓶身接触 · 实际抬升')}</p>
              <details><summary>图像证据与内部记录</summary><p>控制器使用两路固定 RGB 和虚拟工具标记。</p>
              <a href="{name}/vision/target-0.png">初始目标</a> · <a href="{name}/vision/hold-check-0.jpg">抬升核验</a> ·
              <a href="{name}/result.json">独立物理评分</a> · <a href="{name}/tools.jsonl">工具记录</a></details></article>''')
        table.append(f'<tr><td>{esc(name)}</td><td>{esc(row["group"])}</td><td>{row["case"]["bottle_xy_m"]}</td><td>{badge}</td><td>{esc(row.get("error") or "—")}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>视觉闭环抓取</title><style>
    :root{font-family:Inter,"Noto Sans CJK SC",sans-serif;color:#302e2b;background:#f6f4ef;line-height:1.65}*{box-sizing:border-box}
    body{margin:0}main{max-width:1160px;margin:50px auto;padding:0 24px}header{margin-bottom:30px}h1{font-size:32px;margin:10px 0}p{color:#77716a;font-size:14px}
    .eyebrow{color:#9b775d;font-size:13px;letter-spacing:2px}.metrics{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}
    .metric{background:#fffefa;border:1px solid #e8e3da;border-radius:14px;padding:18px 24px;flex:1}.metric strong{font-size:27px;display:block}.metric span{color:#827b72;font-size:13px}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:#fffefa;border:1px solid #e8e3da;border-radius:15px;padding:18px;overflow:hidden}
    .label{display:flex;justify-content:space-between;margin-bottom:12px;font-size:15px}b{font-size:12px;border-radius:20px;padding:2px 10px}.pass{background:#edf2e9;color:#627356}.fail{background:#f7e9e2;color:#a26650}
    video{display:block;width:100%;border-radius:8px;background:#686f68}details{font-size:13px;color:#81766c}a{color:#996f53}section{margin:30px 0}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:12px;text-align:left;border-bottom:1px solid #e4dfd5;vertical-align:top}
    .note{border-left:3px solid #b89a81;padding:2px 18px}.table{overflow:auto}@media(max-width:760px){.grid{grid-template-columns:1fr}main{margin:24px auto}}
    </style><main><header><div class="eyebrow">OPENARM LAB / RGB FEEDBACK</div><h1>视觉闭环抓取</h1>
    <p>同一份 Haiku 修订程序，根据新图像重新确定接近位置。旧 TAMP 未参与执行。</p></header>'''
    page += f'<div class="metrics"><div class="metric"><strong>{passed} / {len(rows)}</strong><span>全部预先声明测试 · 含失败</span></div><div class="metric"><strong>0</strong><span>旧 TAMP 调用</span></div><div class="metric"><strong>RGB + 编码器</strong><span>不向程序提供物体三维真值</span></div></div>'
    page += '<div class="note"><p>本页验证右手抓下半瓶身并抬起；尚未验证新方法的交接、入篮或真机部署。实验依赖已知直立黑瓶、两路固定相机、虚拟平面工具标记。相机与瓶子在接近期间保持静止。通用避障和实物相机适配仍需补齐。</p></div>'
    page += '<section class="grid">' + ''.join(cards) + '</section>'
    page += '<section><h2>完整记录</h2><div class="table"><table><thead><tr><th>案例</th><th>分组</th><th>瓶位 / m</th><th>结果</th><th>失败原因</th></tr></thead><tbody>' + ''.join(table) + '</tbody></table></div></section>'
    page += '<p>程序、控制器与测试条件在运行前冻结；开发调参不计入留出测试。API 修订基于人为提供的控制器和开发反馈，没有训练神经网络。</p><a href="protocol.json">冻结协议</a> · <a href="summary.json">机器可读结果</a></main></html>'
    (folder / 'index.html').write_text(page)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); folder = a.output.resolve()
    if not folder.is_relative_to(ROOT / 'runs') or folder.exists():
        p.error('new output directory under runs/ required')
    protocol = json.loads(a.protocol.read_text()); verify(protocol)
    folder.mkdir(parents=True); save(folder / 'protocol.json', protocol)
    for name in protocol['sources']:
        target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    rows = []
    for spec in protocol['cases']:
        verify(protocol)
        name = spec['name']
        case_file = folder / (name + '.json'); save(case_file, spec['case'])
        start = time.monotonic()
        try:
            proc = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', '-m', 'harness.aspire.visual',
                '--case', str(case_file), '--output', str(folder / name), '--program',
                str(ROOT / protocol['program'])], cwd=ROOT, env={**os.environ, 'MUJOCO_GL': 'egl'},
                capture_output=True, text=True, timeout=150)
            output = folder / name
            output.mkdir(exist_ok=True)
            (output / 'process.txt').write_text(proc.stdout + proc.stderr)
            result = json.loads((output / 'result.json').read_text())
        except Exception as exc:
            result = {'right_grasp_lift_success': False, 'error': type(exc).__name__ + ': ' + str(exc)}
            output = folder / name; output.mkdir(exist_ok=True); save(output / 'result.json', result)
        rows.append({**spec, **result, 'wall_seconds': time.monotonic() - start})
        save(folder / 'summary.json', {'protocol_sources_verified': True, 'cases': rows,
            'passed': sum(r['right_grasp_lift_success'] for r in rows), 'total': len(rows)})
        gallery(folder, protocol, rows)
        print(json.dumps({'case': name, 'pass': result['right_grasp_lift_success'],
            'error': result.get('error')}, ensure_ascii=False), flush=True)
    verify(protocol)


if __name__ == '__main__':
    main()
