"""Read-only recording inventory and pose-diversity screening. No devices."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.offline_calibration import Kinematics, REMOTE_ROOT, remote, write


def collect(out):
    if (out / 'inventory.json').exists():
        raise ValueError('inventory exists; use a new study directory')
    program = '''from pathlib import Path
from datetime import datetime
import json,hashlib,math,base64
root=Path(ROOT_VALUE)/'record_data'
paths=sorted(root.glob('*/trajectory.json'))
rows=[];protected={};images=[]
for number,path in enumerate(paths):
 raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();protected[str(path)]=sha
 frames=json.loads(raw);qs=[];depth=0;keys=set();timing_errors=0;last=None
 for frame in frames:
  obs=frame['observations'];q=obs['qpos']
  if len(q)!=14 or not all(math.isfinite(v) for v in q):raise ValueError('invalid q')
  qs.append(q);depth+=bool(obs.get('depth_paths'));keys.update(obs)
  stamp=datetime.fromisoformat(frame['timestamp'])
  if last is not None and stamp<=last:timing_errors+=1
  last=stamp
 pool=list(range(0,len(qs),8));chosen=[0]
 # Joint-angle distance only: no image labels or candidate camera fit involved.
 for unused in range(min(15,len(pool)-1)):
  best=max(pool,key=lambda i:min(sum((qs[i][j]-qs[k][j])**2 for j in range(14)) for k in chosen))
  if best in chosen:break
  chosen.append(best)
 rows.append({'session':path.parent.name,'source':str(path),'sha256':sha,
  'frames':len(frames),'depth_frames':depth,'observation_keys':sorted(keys),
  'timestamp_order_errors':timing_errors,'first_timestamp':frames[0]['timestamp'],
  'last_timestamp':frames[-1]['timestamp'],
  'q_min':[min(q[j] for q in qs) for j in range(14)],
  'q_max':[max(q[j] for q in qs) for j in range(14)],
  'pose_samples':[{'frame':i,'q':qs[i]} for i in chosen],
  'first_rgb_paths':frames[0]['observations'].get('rgb_paths',[])})
 if number%10==0 or number==len(paths)-1:
  name=frames[0]['observations']['rgb_paths'][2];image=(path.parent/name).resolve()
  assert image.is_relative_to(path.parent)
  raw=image.read_bytes();h=hashlib.sha256(raw).hexdigest();protected[str(image)]=h
  images.append({'session':path.parent.name,'source':str(image),'sha256':h,
   'data':base64.b64encode(raw).decode()})
after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in protected}
print(json.dumps({'episodes':rows,'images':images,'protected_before':protected,
 'protected_after':after,'unchanged':protected==after,'remote_writes':False,'hardware_access':False}))
'''.replace('ROOT_VALUE', repr(REMOTE_ROOT))
    data = remote(program)
    if not data['unchanged']:
        raise ValueError('source recordings changed while reading')
    (out / 'images').mkdir(parents=True, exist_ok=True)
    for im in data['images']:
        raw = base64.b64decode(im.pop('data'), validate=True)
        if hashlib.sha256(raw).hexdigest() != im['sha256']:
            raise ValueError('image digest mismatch')
        im['image'] = 'images/' + im['session'] + '.jpg'
        (out / im['image']).write_bytes(raw)
    kin = Kinematics(ROOT / 'reference/source/inputs/robot.urdf')
    for ep in data['episodes']:
        for sample in ep['pose_samples']:
            q = {j['name']: 0. for j in kin.joints.values()}
            q.update({f'openarm_{side}_joint{i+1}': sample['q'][offset+i]
                      for side, offset in [('left', 0), ('right', 7)] for i in range(7)})
            sample['wrists'] = {s: kin.pose(f'openarm_{s}_link7', q).tolist()
                                for s in ('left', 'right')}
        ep['diversity'] = {}
        for side in ('left', 'right'):
            poses = np.array([s['wrists'][side] for s in ep['pose_samples']])
            delta = poses[0, :3, :3].T @ poses[:, :3, :3]
            angles = Rotation.from_matrix(delta).as_rotvec()
            centred = angles - angles.mean(axis=0)
            # These are coverage diagnostics, not a calibration accuracy score.
            ep['diversity'][side] = {
                'sampled_translation_span_m': np.ptp(poses[:, :3, 3], axis=0).tolist(),
                'sampled_rotation_singular_values_rad': np.linalg.svd(centred, compute_uv=False).tolist()}
    data.update(schema='openarm-recording-inventory/1',
                candidate_urdf_sha256=hashlib.sha256((ROOT / 'reference/source/inputs/robot.urdf').read_bytes()).hexdigest(),
                source_revision_at_capture_verified=False,
                timestamp_semantics='logger entry; exposure/feedback timestamps not established',
                pose_sample_selection='farthest joint angles; every eighth frame pool',
                physical_trial_passed=False)
    write(out / 'inventory.json', data)
    print(json.dumps({'episodes': len(data['episodes']),
        'frames': sum(e['frames'] for e in data['episodes']),
        'depth_frames': sum(e['depth_frames'] for e in data['episodes']),
        'source_files_unchanged': data['unchanged'], 'images': len(data['images'])}))


def analyze(out):
    """Optional image consistency screen, never a physical camera-pose test."""
    import cv2
    from PIL import Image, ImageDraw

    data = json.loads((out / 'inventory.json').read_text())
    if data['protected_before'] != data['protected_after']:
        raise ValueError('source mutation recorded')
    reference = None
    stability = []
    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher()
    for row in data['images']:
        path = out / row['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('original image changed')
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (960, 1280):
            raise ValueError('unexpected head image resolution')
        mask = np.zeros_like(im)
        mask[70:350, 350:990] = 255
        kp, des = sift.detectAndCompute(im, mask)
        if des is None:
            raise ValueError('insufficient upper-room features')
        if reference is None:
            reference = (kp, des)
        good = [p[0] for p in matcher.knnMatch(reference[1], des, k=2)
                if len(p) == 2 and p[0].distance < .65*p[1].distance]
        if len(good) < 8:
            raise ValueError('insufficient matched room features')
        a = np.array([reference[0][m.queryIdx].pt for m in good])
        b = np.array([kp[m.trainIdx].pt for m in good])
        h, inlier = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
        if h is None or inlier is None or inlier.sum() < 8:
            raise ValueError('room image registration failed')
        delta = (b-a)[inlier[:, 0] > 0]
        stability.append({'session': row['session'], 'image': row['image'],
            'shape': list(im.shape), 'matches': len(good), 'inliers': len(delta),
            'median_flow_px': np.median(delta, axis=0).tolist(),
            'p95_flow_px': float(np.percentile(np.linalg.norm(delta, axis=1), 95))})
    write(out / 'stability.json', {'reference': data['images'][0]['session'],
        'method': 'SIFT upper-room ROI; RANSAC filtered. Image consistency only; not measured camera motion.',
        'rows': stability})
    sheet = Image.new('RGB', (1280, 250*((len(data['images'])+3)//4)), '#efeae2')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(data['images']):
        x, y = i % 4*320, i // 4*250
        with Image.open(out / row['image']) as im:
            sheet.paste(im.resize((320, 240)), (x, y))
        draw.text((x+8, y+4), row['session'].split('_')[1], fill='#d74c3e')
    sheet.save(out / 'sessions.jpg')
    candidates = {}
    for side in ('left', 'right'):
        ranked = sorted(data['episodes'], key=lambda e:
            e['diversity'][side]['sampled_rotation_singular_values_rad'][-1], reverse=True)
        candidates[side] = [{'session': e['session'], **e['diversity'][side]} for e in ranked[:5]]
    summary = {'episodes': len(data['episodes']),
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidate_urdf_sha256': hashlib.sha256((ROOT / 'reference/source/inputs/robot.urdf').read_bytes()).hexdigest(),
        'frames': sum(e['frames'] for e in data['episodes']),
        'depth_frames': sum(e['depth_frames'] for e in data['episodes']),
        'observation_key_sets': sorted({tuple(e['observation_keys']) for e in data['episodes']}),
        'timestamp_order_errors': sum(e['timestamp_order_errors'] for e in data['episodes']),
        'hashed_source_files': len(data['protected_before']),
        'source_files_unchanged': data['unchanged'],
        'ranked_pose_candidates': candidates,
        'ranking_scope': '16 sampled joint-diverse poses per recording, candidate URDF; not calibration identifiability or accuracy',
        'historical_pose_grouping_needed': True,
        'remote_writes': False, 'hardware_access': False, 'physical_trial_passed': False}
    write(out / 'summary.json', summary)
    cards = ''.join(f'<div><strong>{value}</strong><span>{label}</span></div>' for value, label in [
        (summary['episodes'], '段旧记录'), (f"{summary['frames']:,}", '帧状态 / 动作'),
        (len(stability), '张跨时段原图'), (0, '深度 / 力矩记录')])
    ranks = ''.join('<tr><td>'+side+'</td><td>'+html.escape(candidates[side][0]['session'])+
        '</td><td>'+', '.join(f'{v:.2f}' for v in candidates[side][0]['sampled_rotation_singular_values_rad'])+
        '</td></tr>' for side in ('left', 'right'))
    # The absence claim is deliberately limited to observation fields, not the whole machine.
    if any(any(k for k in e['observation_keys'] if 'torque' in k.lower() or 'effort' in k.lower())
           for e in data['episodes']):
        raise ValueError('report torque absence text requires review')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>旧数据标定筛选</title>
<style>body{margin:0;background:#f6f4ef;color:#35332e;font:16px/1.7 system-ui,sans-serif}main{max-width:1000px;margin:48px auto;padding:0 24px}h1{font-size:32px;margin-bottom:8px}h2{font-size:21px}p{color:#67635b}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cards div,section{background:#fffdf9;border:1px solid #e5dfd4;border-radius:14px;padding:22px}.cards strong{display:block;font-size:29px}.cards span{color:#80776b;font-size:14px}section{margin-top:22px}img{width:100%;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #eee7dc}a{color:#a55e43}.tag{color:#a55e43}.step{margin:14px 0}small{color:#80776b}@media(max-width:650px){.cards{grid-template-columns:repeat(2,1fr)}table{font-size:11px}}</style>
<main><small>OpenArm Lab · 只读离线检查</small><h1>旧数据还能怎样用于标定</h1>
<p>已扩展至全部记录。当前结论：旧数据有进一步利用的空间，尚无新增的标定通过结果。</p>
<div class="cards">CARDS</div>
<section><h2>更有效的路线</h2>
<div class="step"><b>① 先分组、再选帧</b><br>按相机画面一致性筛分采集时段；从不同方向转动充分的轨迹取样，避免重复姿态占满拟合数据。</div>
<div class="step"><b>② 先约束镜头，再估计安装位置</b><br>旧画面中的自然直线用于镜头候选；整臂 CAD 轮廓和关节状态用于相机外参。未知内参与外参不应毫无约束地一起拟合。</div>
<div class="step"><b>③ 复核对应点与时间</b><br>复查原始像素特征；用动作变化检查记录偏移。历史记录偏移不能自动视作当前设备的真实延迟。</div>
<div class="step"><b>④ 新轨迹独立验证</b><br>当前看过的 ep0–ep3 都列为开发数据；后续在预先冻结、未参与调整的轨迹上评分，保留失败分母。</div>
<p>参考 <a href="https://github.com/ootts/EasyHeC">EasyHeC</a> 的 CAD 轮廓与姿态覆盖思路。
<a href="https://github.com/StoneT2000/simple-easyhec">Simple EasyHeC</a> 仍要求内参及外参初值，不能直接替代缺失的镜头标定。</p></section>
<section><h2>姿态筛选结果</h2><table><tr><th>机械臂</th><th>候选轨迹</th><th>旋转变化奇异值 / rad</th></tr>RANKS</table>
<p>这里只使用关节数据筛选覆盖范围；并非标定通过率，也没有用这些候选的像素误差挑选“容易的测试”。</p></section>
<section><h2>跨时段画面抽查</h2><img src="sessions.jpg"><p>16 张原图的上方房间特征出现约 1–5 像素的差异。
这提示需要检查采集时段的一致性；匹配误差、光照和场景变化也会影响结果，不能据此判定相机真的移动了多少毫米。</p></section>
<section><h2>已有记录的边界</h2><p>148 段记录没有深度帧；观测字段没有夹爪力矩。它们可以帮助估计几何和时序候选，但不能独立恢复真实夹持力、防滑余量或当前安装状态。</p>
<p class="tag">没有打开相机 / CAN，没有运动或写入公司电脑，没有修改运行标定。</p>
<details><summary>内部证据</summary><p><a href="inventory.json">逐轨迹清单与摘要</a> · <a href="summary.json">筛选摘要</a> · <a href="stability.json">画面一致性</a> · <a href="../20261025-offline/index.html">前两轮离线试验</a></p></details></section></main></html>'''
    (out / 'index.html').write_text(page.replace('CARDS', cards).replace('RANKS', ranks))
    print(json.dumps({k: summary[k] for k in ['episodes', 'frames', 'depth_frames', 'hashed_source_files', 'source_files_unchanged']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyze', action='store_true', help='analyze existing inventory locally; requires OpenCV and Pillow')
    args = p.parse_args()
    (analyze if args.analyze else collect)(args.output.resolve())
