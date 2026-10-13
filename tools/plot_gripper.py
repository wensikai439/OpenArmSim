"""Render the fixed native gripper study; no fitting or source access."""
from pathlib import Path
import html
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
report = json.loads((root/'summary.json').read_text())
data = np.load(root/'series.npz', allow_pickle=False)
colors = {'legacy': '#c56b55', 'shared_coordinates': '#be961a', 'shared_motor_gain': '#287e9b'}
labels = {'legacy': 'Old split coordinates', 'shared_coordinates': 'Shared coordinates only',
          'shared_motor_gain': 'Shared coordinates + motor gain'}
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharey=True, constrained_layout=True)
for episode in report['episodes']:
    ep = episode['episode']
    for side in (0, 1):
        ax = axes[ep, side]
        n = max(v['available_samples'] for v in episode['variants'].values())
        longest = max(episode['variants'], key=lambda v: episode['variants'][v]['available_samples'])
        t = data[f'ep{ep}_{longest}_time']
        ax.plot(t, data[f'ep{ep}_target'][:n, side], color='#222b31', lw=1.6, label='Real measured motor angle')
        ax.plot(t, data[f'ep{ep}_command'][:n, side], color='#777d80', lw=1, ls=':', label='Recorded target motor angle')
        for variant in colors:
            ax.plot(data[f'ep{ep}_{variant}_time'], data[f'ep{ep}_{variant}_state'][:, [7,15][side]],
                    color=colors[variant], lw=1.1, alpha=.9, label=labels[variant])
        ax.axvline(episode['scored_seconds'], color='#888', lw=.8, ls='--')
        ax.set(title=f"Episode {ep} / {'Left' if side == 0 else 'Right'} gripper", xlabel='Recorded-clock replay (s)', ylabel='Motor angle (rad)')
        ax.grid(alpha=.16)
axes[0,0].legend(fontsize=7, loc='upper right')
fig.suptitle('Gripper coordinate correction — development replay; transmission uncalibrated', fontsize=13)
fig.savefig(root/'gripper.png', dpi=160)
plt.close(fig)
rows = ''.join(f'<tr><td>{html.escape(labels[v])}</td><td>{score[0]:.4f}</td><td>{score[1]:.4f}</td></tr>'
               for v,score in report['aggregate_rmse_rad'].items())
coverage = ''.join(f"<li>episode {e['episode']}：共同计分 {e['common_samples']} 帧 / {e['scored_seconds']:.2f} 秒；"
    + '，'.join(f"{html.escape(v)} {r['available_samples']}/{r['reference_samples']} 帧（{html.escape(r['status'])}）" for v,r in e['variants'].items())
    + '</li>' for e in report['episodes'])
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>夹爪坐标对照 · OpenArm Lab</title><style>
body{{margin:0;background:#f5f3ee;color:#243039;font:15px/1.7 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 24px}}
header,p,section{{margin:0 0 22px}}h1{{font-size:28px;margin:8px 0}}.muted{{color:#6c756f}}section{{padding:24px;background:#fff;border:1px solid #e4e5df;border-radius:14px}}
img{{width:100%;height:auto}}a{{color:#226980}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #eee}}.badge{{display:inline-block;padding:3px 10px;background:#fff0d5;border-radius:99px;color:#865c14;font-size:13px}}
</style><main><header><a href="/">← 返回平台</a><h1>夹爪坐标修正</h1><span class="badge">开发对照 · 物理标定未完成</span></header>
<section><p>旧实现把开爪目标 −0.90 rad 映射为反馈 −1.26 rad，即使完全跟随也有 0.36 rad 偏差。修正后，目标与反馈共用电机角度坐标；增益按候选传动比换算到指关节。</p>
<p class="muted">目标与实测仍可因运动、接触和负载而不同。端点取自数据极值，真实传动比、开口宽度和限矩仍待测。本次无参数拟合，不根据误差挑选实现。</p>
<table><thead><tr><th>实现</th><th>左夹爪 RMSE（rad）</th><th>右夹爪 RMSE（rad）</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">用三条轨迹的共同有效时间段，对误差平方做梯形时间积分；均为已使用过的开发数据。</p></section>
<section><a href="gripper.png" target="_blank"><img src="gripper.png" alt="三个 episode 左右夹爪的实测、指令及三种仿真实现对照"></a><p class="muted">点击图片查看原尺寸。竖虚线为该 episode 的共同计分终点；提前终止会缩短比较范围。</p><ul>{coverage}</ul></section>
<p><a href="summary.json">数值与终止原因</a> · <a href="manifest.json">来源和参数</a> · <a href="series.npz">轨迹数组</a></p>
</main></html>'''
(root/'index.html').write_text(page)
