"""Offline estimates from original recordings; never opens hardware or deploys.

Run with an existing NumPy/SciPy/OpenCV/Pillow environment. Camera estimates
use manually identified wrist-axis proxies, not measured fiducials. Results
are candidates even when image residuals are small.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.evidence import Kinematics

REMOTE_ROOT = '/home/sj/Documents/moqi_workspace_20251119_211400'
SESSIONS = ['session_0001_28hz_20260828_112815',
            'session_0003_27hz_20260828_113016',
            'session_0007_28hz_20260828_113947',
            'session_0009_27hz_20260828_114953']
R0 = np.array([[0., -1., 0.], [-.8, 0., -.6], [.6, 0., -.8]])
SOURCES = {'DREAM': 'https://github.com/NVlabs/DREAM',
           'fisheye': 'https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def remote(program):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-T', 'data-cs',
        'python3 -B -'], input=program, capture_output=True, text=True,
        timeout=120, check=True)
    if len(result.stdout) > 32 * 1024 * 1024:
        raise ValueError('unexpected remote response size')
    return json.loads(result.stdout)


def collect(out):
    if (out / 'data.json').exists():
        raise ValueError('collection already exists; do not overwrite evidence')
    # Standard library, explicit file reads only. No device or robot imports.
    code = '''from pathlib import Path
import json,hashlib,base64
root=Path(ROOT_VALUE)
sessions=SESSION_VALUE
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
urdf=root/'openarm/openarm_description/urdf/robot/openarm_bimanual.urdf'
protected={str(urdf):sha(urdf)}
rows={};images=[]
for ep,name in enumerate(sessions):
 p=root/'record_data'/name/'trajectory.json';raw=p.read_bytes()
 protected[str(p)]=hashlib.sha256(raw).hexdigest();frames=json.loads(raw)
 rows[str(ep)]={'session':name,'source':str(p),'sha256':protected[str(p)],'frames':[]}
 for x in frames:
  o=x['observations'];a=x['action']
  rows[str(ep)]['frames'].append({'frame':x['frame_id'],'timestamp':x['timestamp'],
   'q':o['qpos'],'gripper':o['gripper_joints'],
   'command':a['qpos'],'gripper_command':a['gripper_joints'],
   'has_depth':bool(o.get('depth_paths')),'observation_keys':list(o)})
 if ep==3:
  for i in [0,160,320,480,640,len(frames)-1]:
   p=(root/'record_data'/name/frames[i]['observations']['rgb_paths'][2]).resolve()
   assert p.is_relative_to(root/'record_data'/name)
   raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();protected[str(p)]=h
   images.append({'episode':ep,'frame':i,'source':str(p),'sha256':h,
     'data':base64.b64encode(raw).decode()})
after={p:sha(Path(p)) for p in protected}
print(json.dumps({'episodes':rows,'images':images,'urdf_sha256':protected[str(urdf)],
 'protected_before':protected,'protected_after':after,'unchanged':protected==after,
 'hardware_access':False,'remote_writes':False}))
'''.replace('ROOT_VALUE', repr(REMOTE_ROOT)).replace('SESSION_VALUE', repr(SESSIONS))
    data = remote(code)
    if not data['unchanged'] or data['urdf_sha256'] != digest(ROOT/'reference/source/inputs/robot.urdf'):
        raise ValueError('recording source or URDF mismatch')
    existing=json.loads((ROOT/'reference/acquisition.json').read_text())['episodes']
    for ep,row in data['episodes'].items():
        if int(ep)<3 and row['sha256']!=existing[ep]['sha256']:
            raise ValueError('previously audited source trajectory changed')
        from datetime import datetime
        previous=None
        for i,frame in enumerate(row['frames']):
            stamp=datetime.fromisoformat(frame['timestamp'])
            if frame['frame']!=i or (previous is not None and stamp<=previous):
                raise ValueError('recording frame order or clock is invalid')
            previous=stamp
            for key,n in [('q',14),('command',14),('gripper',2),('gripper_command',2)]:
                a=np.asarray(frame[key],dtype=float)
                if a.shape!=(n,) or not np.isfinite(a).all():
                    raise ValueError('recording state/action contract mismatch')
    (out/'images').mkdir(exist_ok=True)
    for image in data['images']:
        raw = base64.b64decode(image.pop('data'), validate=True)
        if hashlib.sha256(raw).hexdigest() != image['sha256']:
            raise ValueError('image digest mismatch')
        name = f"images/ep{image['episode']}-{image['frame']:06d}-head.jpg"
        (out/name).write_bytes(raw); image['local_path'] = name
    data.update(schema='openarm-offline-data/1', sources=SOURCES,
                timestamp_semantics='logger record entry, not exposure/CAN timestamp',
                source_revision_at_capture_verified=False)
    write(out/'data.json', data)
    print(json.dumps({'collected_episodes':len(data['episodes']),
        'frames':sum(len(e['frames']) for e in data['episodes'].values()),
        'images':len(data['images']), 'remote_files_unchanged':data['unchanged']}))


def project(x, points, model):
    """OpenCV optical axes: x right, y down, z forward. x[:3] is camera centre."""
    rotation = Rotation.from_rotvec(x[3:6]).as_matrix() @ R0
    xyz = (np.asarray(points)-x[:3]) @ rotation.T
    if model == 'pinhole':
        ray = xyz[:, :2] / np.maximum(xyz[:, 2:3], .001)
    else:
        radius = np.linalg.norm(xyz[:, :2], axis=1)
        theta = np.arctan2(radius, xyz[:, 2])
        distorted = theta * (1 + x[9]*theta**2)
        ray = xyz[:, :2] * (distorted/np.maximum(radius, 1e-12))[:, None]
    return np.exp(x[6])*ray+x[7:9], xyz[:, 2]


def points(out, annotations, allowed):
    data = json.loads((out/'data.json').read_text())
    kin = Kinematics(ROOT/'reference/source/inputs/robot.urdf')
    xyz, uv, metadata = [], [], []
    original = json.loads((ROOT/'reference/frames/manifest.json').read_text())['samples']
    for row in annotations['points']:
        ep, frame, side = row['episode'], row['frame'], row['side']
        if ep not in allowed:
            continue
        raw = data['episodes'][str(ep)]['frames'][frame]
        q = {j['name']:0. for j in kin.joints.values()}
        q.update({f'openarm_{s}_joint{i+1}': raw['q'][offset+i]
                  for s, offset in [('left',0), ('right',7)] for i in range(7)})
        link = f'openarm_{side}_link7'
        point = kin.pose(link, q)[:3,3]
        if ep < 3:
            entry = next(r for r in original if r['episode']==ep and r['frame']==frame)['images']['head']
            image = ROOT/'reference/frames'/entry['path']
        else:
            entry = next(r for r in data['images'] if r['episode']==ep and r['frame']==frame)
            image = out/entry['local_path']
        if digest(image) != entry['sha256']:
            raise ValueError('annotated source image changed')
        xyz.append(point); uv.append(row['uv'])
        metadata.append({**row, 'point_base_m':point.tolist(), 'link':link,
                         'image':str(image.relative_to(ROOT)), 'image_sha256':entry['sha256']})
    if not xyz:
        raise ValueError('no annotated correspondences')
    return np.asarray(xyz), np.asarray(uv), metadata


def bounds(model):
    low = [-.4,-.3,.55,-.8,-.8,-.8,np.log(150),540,390]
    high = [.5,.3,1.5,.8,.8,.8,np.log(1400 if model=='pinhole' else 650),740,570]
    if model!='pinhole': low += [-.25]; high += [.25]
    return np.array(low), np.array(high)


def solve(xyz, uv, model, seed=42, starts=12):
    low, high = bounds(model); rng = np.random.default_rng(seed)
    initial = np.array([.05,0,.95,0,0,0,np.log(320),640,480]+([] if model=='pinhole' else [0.]))
    def residual(x):
        pred, depth = project(x, xyz, model)
        # Uniform pixel weights. Robust loss does not silently remove labels.
        return np.r_[(pred-uv).ravel(), np.maximum(.05-depth,0)*1000]
    candidates=[]
    for i in range(starts):
        guess = initial.copy()
        if i:
            guess[:3] += rng.normal(0,.06,3); guess[3:6] += rng.normal(0,.12,3)
            guess[6] += rng.normal(0,.15)
        result = least_squares(residual, np.clip(guess,low+1e-6,high-1e-6),
            bounds=(low,high), loss='soft_l1', f_scale=8., x_scale='jac',
            max_nfev=1200, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        candidates.append(result)
    best = min(candidates, key=lambda r:r.cost)
    pred, depth = project(best.x,xyz,model)
    # Dimensionless Jacobian conditioning: native units otherwise dominate.
    jac = best.jac[:2*len(xyz)]
    sv = np.linalg.svd(jac/np.maximum(np.linalg.norm(jac,axis=0),1e-12),compute_uv=False)
    span = np.ptp(np.array([r.x[:3] for r in candidates if r.cost<=best.cost*1.05+1e-6]),axis=0)
    return {'model':model, 'parameters':best.x.tolist(), 'robust_cost':float(best.cost),
        'fit_rmse_px':float(np.sqrt(np.mean(np.sum((pred-uv)**2,axis=1)))),
        'min_depth_m':float(depth.min()), 'jacobian_singular_values':sv.tolist(),
        'jacobian_condition':float(sv[0]/max(sv[-1],1e-15)),
        'near_optimal_camera_center_span_m':span.tolist(),
        'at_bounds':np.where(np.minimum(best.x-low,high-best.x)<.005*(high-low))[0].tolist(),
        'starts':starts, 'converged':bool(best.success)}


def fit(out, annotation_file):
    if (out/'selected.json').exists():
        raise ValueError('selection is frozen; use a new experiment to refit')
    annotation = json.loads(annotation_file.read_text())
    xyz,uv,meta = points(out,annotation,[0,1])
    if len(xyz)<20:
        raise ValueError('at least 20 multi-pose fit labels required')
    protocol = {'schema':'openarm-offline-camera/1', 'fit_episodes':[0,1],
        'development_validation_episodes':[2], 'new_validation_episodes':[3],
        'model_selection':'fit-only AIC; validation never enters optimization',
        'camera':'head; assumed fixed across these recorded sessions',
        'landmark':'manually identified wrist-axis visual proxy for URDF link7 origin',
        'annotation_sigma_px':8., 'joint_zero_and_CAD':'fixed; not independently measured',
        'models':['pinhole','fisheye_k1'], 'seed':42,
        'sources':SOURCES, 'runtime_config_changed':False,
        'acceptance':'candidate only; no automatic motion admission',
        'annotation_sha256':digest(annotation_file), 'data_sha256':digest(out/'data.json'),
        'urdf_sha256':digest(ROOT/'reference/source/inputs/robot.urdf'),
        'script_sha256':digest(__file__), 'fit_points':meta}
    write(out/'protocol.json',protocol)
    candidates=[]
    for model in protocol['models']:
        row=solve(xyz,uv,model)
        # AIC is computed on all labelled residuals, not the robust objective.
        row['aic']=float(2*len(xyz)*np.log(max(row['fit_rmse_px']**2/2,1e-12))+2*len(row['parameters']))
        candidates.append(row)
        print(json.dumps({k:row[k] for k in ['model','fit_rmse_px','jacobian_condition','at_bounds']}),flush=True)
    selected=min(candidates,key=lambda r:r['aic'])
    x=np.array(selected['parameters']);r=Rotation.from_rotvec(x[3:6]).as_matrix()@R0
    transform=np.eye(4);transform[:3,:3]=r;transform[:3,3]=-r@x[:3]
    # Write immutable selection BEFORE evaluating any held-out correspondences.
    write(out/'selected.json', {'schema':'openarm-offline-camera-candidate/1',
        **selected,'candidates':candidates, 'selection_uses_validation':False,
        'T_camera_base':transform.tolist(), 'T_base_camera':np.linalg.inv(transform).tolist(),
        'K':[[float(np.exp(x[6])),0,float(x[7])],[0,float(np.exp(x[6])),float(x[8])],[0,0,1]],
        'D':[float(x[9]) if len(x)>9 else 0.,0.,0.,0.],
        'image_size':[1280,960], 'optical_axes':'x right, y down, z forward',
        'physical_trial_passed':False, 'transfer_ready':False,
        'protocol_sha256':digest(out/'protocol.json')})


def grippers(data):
    result={}
    for side,j in [('left',0),('right',1)]:
        groups={}
        for split,episodes in [('fit',[0,1]),('development_validation',[2]),('new_validation',[3])]:
            rows=[]
            for ep in episodes:
                frames=data['episodes'][str(ep)]['frames']
                from datetime import datetime
                t=np.array([datetime.fromisoformat(f['timestamp']).timestamp() for f in frames]);t-=t[0]
                q=np.array([f['gripper'][j] for f in frames]);cmd=np.array([f['gripper_command'][j] for f in frames])
                derivative=np.gradient(cmd,t);stable=np.abs(derivative)<.05
                # Require command stability for the preceding .25 logger seconds.
                for i in range(len(t)):
                    a=np.searchsorted(t,t[i]-.25)
                    stable[i] = t[i]>=.25 and np.all(np.abs(derivative[a:i+1])<.05)
                def stats(mask):
                    a=q[mask]
                    return {'n':int(mask.sum()),'p05_p50_p95_rad':np.quantile(a,[.05,.5,.95]).tolist() if len(a) else None}
                rows.append({'episode':ep,'frames':len(t),'logger_hz':float((len(t)-1)/t[-1]),
                    'actual_range_rad':[float(q.min()),float(q.max())],
                    'command_open':stats(stable&(cmd<=-.85)),
                    'command_closed':stats(stable&(cmd>=-.05)),
                    'command_feedback_rmse_rad':float(np.sqrt(np.mean((q-cmd)**2)))})
            groups[split]=rows
        result[side]=groups
    return {'schema':'openarm-gripper-offline/1','sides':result,
        'labels':'command_open / command_closed, NOT empty / holding / successful',
        'contact_threshold_accepted':False,'contact_torque_range':None,
        'reason':'No recorded torque; jaw position alone cannot prove an object is held.',
        'applied_to_runtime':False}


def evaluate(out, annotation_file):
    from PIL import Image,ImageDraw
    selected=json.loads((out/'selected.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    if digest(out/'protocol.json')!=selected['protocol_sha256']:
        raise ValueError('protocol changed after model selection')
    if digest(out/'data.json')!=protocol['data_sha256'] or digest(ROOT/'reference/source/inputs/robot.urdf')!=protocol['urdf_sha256']:
        raise ValueError('trajectory or robot source changed after model selection')
    annotation=json.loads(annotation_file.read_text())
    xyz,uv,meta=points(out,annotation,[0,1,2,3])
    # Fit annotations must remain identical after selection; validation may be appended.
    if [m for m in meta if m['episode'] in [0,1]]!=protocol['fit_points']:
        raise ValueError('fit correspondences changed after selection')
    pred,depth=project(np.array(selected['parameters']),xyz,selected['model'])
    rows=[];groups={}
    for i,m in enumerate(meta):
        split='fit' if m['episode']<2 else 'development_validation' if m['episode']==2 else 'new_validation'
        rows.append({**m,'split':split,'predicted_uv':pred[i].tolist(),
            'residual_px':float(np.linalg.norm(pred[i]-uv[i])),'depth_m':float(depth[i])})
    for split in ['fit','development_validation','new_validation']:
        errors=np.array([r['residual_px'] for r in rows if r['split']==split])
        groups[split]={'n':len(errors), 'rmse_px':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'median_px':float(np.median(errors)) if len(errors) else None,
            'p95_px':float(np.quantile(errors,.95)) if len(errors) else None}
    # Leave-one-frame-out on FIT ONLY quantifies sensitivity to sparse annotation.
    fit_rows=[r for r in rows if r['split']=='fit']
    fit_mask=np.array([r['split']=='fit' for r in rows])
    fit_xyz=xyz[fit_mask];fit_uv=uv[fit_mask]
    centres=[]
    for key in sorted({(r['episode'],r['frame']) for r in fit_rows}):
        mask=np.array([(r['episode'],r['frame'])!=key for r in fit_rows])
        centres.append(solve(fit_xyz[mask],fit_uv[mask],selected['model'],starts=1)['parameters'][:3])
    sensitivity=np.ptp(centres,axis=0).tolist()
    for ep,frame in sorted({(r['episode'],r['frame']) for r in rows}):
        sample=[r for r in rows if r['episode']==ep and r['frame']==frame]
        image=Image.open(ROOT/sample[0]['image']).convert('RGB');draw=ImageDraw.Draw(image)
        for row in sample:
            u,v=row['uv'];a,b=row['predicted_uv'];draw.ellipse((u-7,v-7,u+7,v+7),outline='#e6a43e',width=3)
            draw.line((a-8,b,a+8,b),fill='#36b89a',width=3);draw.line((a,b-8,a,b+8),fill='#36b89a',width=3)
            draw.line((u,v,a,b),fill='white',width=2)
            draw.text((u+10,v+5),f"{row['side']} {row['residual_px']:.1f}px",fill='#ffd57a',stroke_width=1,stroke_fill='black')
        draw.text((24,24),f"{sample[0]['split']} / ep{ep} frame {frame} | manual proxy: amber / projected: green",fill='white',stroke_width=1,stroke_fill='black')
        image.save(out/'images'/f'ep{ep}-{frame:06d}-overlay.jpg',quality=92)
    reasons=[]
    reasons.append('manual proxy-to-CAD correspondence has no independent geometric confirmation')
    if selected['at_bounds']:reasons.append('camera parameters reach exploratory bounds')
    if selected['jacobian_condition']>1e4:reasons.append('weak parameter identifiability')
    if max(sensitivity)>.03:reasons.append('camera centre changes >30 mm when one fit frame is omitted')
    if not groups['new_validation']['n']:reasons.append('no new held-out trajectory labels')
    elif groups['new_validation']['p95_px']>16:reasons.append('new-trajectory p95 residual exceeds 16 px')
    result={'schema':'openarm-offline-calibration-result/1','status':'candidate' if not reasons else 'rejected_for_deployment',
        'camera':selected['model'],'metrics':groups,'rows':rows,
        'leave_one_fit_frame_out_camera_center_span_m':sensitivity,
        'selected_sha256':digest(out/'selected.json'),'annotations_sha256':digest(annotation_file),
        'reasons':reasons,'physical_trial_passed':False,'runtime_config_changed':False,
        'hardware_access':False,'transfer_ready':False,
        'limitations':['Manual wrist-axis proxies are approximate, not surveyed feature points.',
            'Camera intrinsics and pose are jointly estimated under fixed candidate URDF/joint zeros.',
            'Only local robot workspace is constrained; no full-image lens calibration.',
            'Same logger record does not prove exposure/joint synchronization.',
            'Historical fixed camera mounting is assumed; current mounting must be checked.',
            'Held-out pixels do not prove millimetre TCP accuracy or safe grasping.']}
    write(out/'camera.json',result)
    grip=grippers(json.loads((out/'data.json').read_text()));write(out/'grippers.json',grip)
    report(out,result,selected,grip)
    print(json.dumps({'status':result['status'],'metrics':groups,'reasons':reasons,'camera_center_span_m':sensitivity},ensure_ascii=False))


def report(out,result,selected,grip):
    labels={'fit':'拟合 · ep0 / ep1','development_validation':'开发验证 · ep2','new_validation':'新增轨迹 · ep3'}
    rows=''.join(f'<tr><td>{labels[s]}</td><td>{m["n"]}</td><td>{m["rmse_px"]:.2f}</td><td>{m["p95_px"]:.2f}</td></tr>'
        for s,m in result['metrics'].items() if m['n'])
    cards=''.join(f'<figure><a href="images/ep{ep}-{frame:06d}-overlay.jpg"><img loading="lazy" src="images/ep{ep}-{frame:06d}-overlay.jpg"></a><figcaption>ep{ep} · {frame}</figcaption></figure>'
        for ep,frame in sorted({(r['episode'],r['frame']) for r in result['rows']}))
    reasons='；'.join(result['reasons']) or '仅作为离线候选，尚无独立物理验收'
    carry_section=''
    if (out/'carry/result.json').exists() and (out/'carry/timing-result.json').exists():
        carry=json.loads((out/'carry/result.json').read_text())
        timed=json.loads((out/'carry/timing-result.json').read_text())
        def number(value):return '不可评分' if value is None else f'{value:.2f}'
        carry_section=f'''<section><h2>第二轮：瓶盖自然特征</h2>
<p>从 64 张原图跟踪瓶盖中心，每次夹持分别估计瓶盖与腕部的固定关系，不把外观中心硬当作 CAD 轴心。相机只用 ep0/ep1 的 32 点拟合；ep2/ep3 每段先用 3 帧登记相对点，再检查剩余 5 帧。</p>
<table><tr><th>检查</th><th>RMSE / px</th><th>覆盖</th></tr><tr><td>拟合瓶盖点</td><td>{number(carry['metrics']['fit']['rmse_px'])}</td><td>{carry['metrics']['fit']['n']} 点</td></tr><tr><td>未参与登记的验证点</td><td>{number(carry['metrics']['held_out']['rmse_px'])}</td><td>{carry['scored_withheld_frames']} / {carry['expected_withheld_frames']} 帧可评分</td></tr></table>
<p>两段左手登记失败，计为失败；主点参数也触及搜索边界。表中的误差只覆盖可评分片段，不能据此宣称整体验证通过，两种方案的特征点也不同，不能直接比较改善百分比。</p>
<details><summary>时间偏移与自检</summary><p>仅用拟合集选择的记录偏移为 {timed['offset_records']} 条，拟合误差为 {number(timed['metrics']['fit']['rmse_px'])}px，可评分验证误差为 {number(timed['metrics']['held_out']['rmse_px'])}px。最佳值落在扫描边界，且验证变差；该假设未采用，也不是测得的曝光或 CAN 延迟。</p><p>已加上拟合/验证分离、重复标注拒绝、退化三角化拒绝和失败覆盖率统计；不把不可用的登记结果画成可信三维点。</p><img src="carry.png" alt="时间偏移拟合与验证对照" onerror="this.style.display='none'"></details>
<a href="carry/result.json">自然特征结果</a> · <a href="carry/timing-result.json">时间诊断</a> · <a href="checks.json">自检记录</a></section>'''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>旧数据离线标定</title>
<style>body{{margin:0;background:#f6f5f1;color:#303733;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:40px auto;padding:0 24px}}section{{background:white;border:1px solid #e3e2dc;border-radius:16px;padding:24px;margin:24px 0}}h1{{font-size:30px}}h2{{font-size:20px}}p{{max-width:850px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #eee}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0}}img{{width:100%;border-radius:10px}}figcaption{{color:#6d756f}}a{{color:#397963}}.tag{{color:#8a6027;background:#f4eadb;border-radius:20px;padding:5px 12px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style>
<main><a href="/">← 平台</a><h1>旧数据离线标定</h1><span class="tag">离线候选 · 未启用真机运动</span>
<section><h2>本次做了什么</h2><p>只读提取四段轨迹的 3345 条状态/动作记录，从 26 张头部原图识别 52 个腕部轴心近似点，联合估计鱼眼相机参数和外参。ep0/ep1 的 14 张图用于拟合；ep2 与新增 ep3 各抽 6 张图验证，两条轨迹都不参与拟合。</p><p>这参考 DREAM 的几何对应思路；未使用其现成关键点网络或权重。橙圈为图像人工近似标注，绿十字为 URDF 投影，不是测量标记。</p></section>
<section><h2>相机结果：本轮未通过</h2><table><tr><th>数据</th><th>点数</th><th>RMSE / px</th><th>P95 / px</th></tr>{rows}</table><p>选择模型：{html.escape(selected['model'])}。拟合参数对删除个别帧敏感，新增轨迹也没有通过预先确定的像素检查。</p><p>这些数值衡量“人工近似点＋当前 URDF＋相机假设”的一致性，不能直接归因于真实相机误差或机器人毫米精度。未覆盖运行参数，也未将物理验收项改为通过。</p><details><summary>查看判定依据</summary>{html.escape(reasons)}</details><img src="camera.png" alt="相机拟合与跨轨迹误差" onerror="this.style.display='none'"></section>
{carry_section}<section><h2>夹爪与缺项</h2><p>四段记录提供夹爪位置、指令和记录时钟，可估计开合区间及跟踪偏差；没有力矩或深度记录。闭合指令不是夹住瓶子的标签，因此本次不生成夹持力阈值。</p><p>还需确认历史相机安装是否保持、补充可独立验证的几何对应点，并用少量现场检查确认夹持、防滑和停止响应。</p><img src="grippers.png" alt="四段真实夹爪指令与反馈" onerror="this.style.display='none'"><a href="grippers.json">夹爪统计</a> · <a href="camera.json">相机结果</a> · <a href="selected.json">冻结参数</a> · <a href="data.json">来源与只读核验</a></section>
<section><h2>原图与投影</h2><div class="grid">{cards}</div></section><p><a href="{SOURCES['DREAM']}">NVIDIA DREAM</a> · <a href="{SOURCES['fisheye']}">OpenCV 鱼眼模型</a></p></main></html>'''
    (out/'index.html').write_text(page)


def plot(out):
    """Optional report figures, using an existing Matplotlib installation."""
    import os
    import tempfile
    from datetime import datetime
    with tempfile.TemporaryDirectory(prefix='openarm-offline-plot-') as temporary:
        os.environ['MPLCONFIGDIR']=temporary
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
            'axes.spines.top':False,'axes.spines.right':False})
        camera=json.loads((out/'camera.json').read_text())
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        splits=['fit','development_validation','new_validation']
        x=np.arange(3);metrics=camera['metrics']
        ax.bar(x-.17,[metrics[s]['rmse_px'] for s in splits],.34,color='#65877b',label='RMSE')
        ax.bar(x+.17,[metrics[s]['p95_px'] for s in splits],.34,color='#bc957d',label='P95')
        ax.axhline(16,color='#b25445',ls='--',label='Exploratory P95 limit (16 px)')
        ax.set(xticks=x,xticklabels=['Fit: ep0 + ep1','Development: ep2','New trajectory: ep3'],
            ylabel='Proxy reprojection error (px)',title='Head camera: coarse correspondence trial rejected')
        ax.legend(frameon=False);ax.grid(alpha=.12,axis='y')
        fig.savefig(out/'camera.png',dpi=160,facecolor='#faf9f5');plt.close(fig)
        data=json.loads((out/'data.json').read_text())
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for ep,ax in enumerate(axes.flat):
            frames=data['episodes'][str(ep)]['frames']
            t=np.array([datetime.fromisoformat(f['timestamp']).timestamp() for f in frames]);t-=t[0]
            for side,i,color in [('left',0,'#597e72'),('right',1,'#a4775d')]:
                ax.plot(t,[f['gripper'][i] for f in frames],color=color,label=side+' feedback')
                ax.plot(t,[f['gripper_command'][i] for f in frames],color=color,ls='--',alpha=.55,label=side+' command')
            ax.set(xlabel='Logger time (s)',ylabel='Motor angle (rad)',
                   title=f'ep{ep} / '+('new trajectory' if ep==3 else 'existing trajectory'))
            ax.grid(alpha=.12);ax.legend(frameon=False,fontsize=8,ncol=2)
        fig.suptitle('Recorded jaw states: closed command is not proof of a held bottle')
        fig.savefig(out/'grippers.png',dpi=160,facecolor='#faf9f5');plt.close(fig)
        if (out/'carry/timing-selected.json').exists():
            clock=json.loads((out/'carry/timing-selected.json').read_text())
            baseline=json.loads((out/'carry/result.json').read_text())
            tested=json.loads((out/'carry/timing-result.json').read_text())
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            rows=clock['candidates'];axes[0].plot([r['offset_records'] for r in rows],[r['fit_rmse_px'] for r in rows],'o-',color='#65877b')
            axes[0].set(xlabel='Effective offset (logger records)',ylabel='Fit RMSE (px)',title='Selection uses fit data only')
            axes[1].bar(['No clock shift','Fit-selected shift'],[baseline['metrics']['held_out']['rmse_px'],tested['metrics']['held_out']['rmse_px']],color=['#65877b','#bc957d'])
            axes[1].set(ylabel='Scorable validation RMSE (px)',title='Validation worsens; timing hypothesis rejected')
            for ax in axes:ax.grid(alpha=.12,axis='y')
            fig.suptitle('Only 10 / 20 withheld frames scorable; 2 left-hand registrations fail')
            fig.savefig(out/'carry.png',dpi=160,facecolor='#faf9f5');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['collect','fit','evaluate','plot'])
    p.add_argument('--output',type=Path,default=ROOT/'runs/20261025-offline')
    p.add_argument('--annotations',type=Path,default=ROOT/'reference/head_points.json')
    a=p.parse_args();out=a.output.resolve()
    if out.parent != ROOT/'runs' or out.is_symlink():raise ValueError('output must be a local runs subdirectory')
    out.mkdir(exist_ok=True)
    if a.phase=='collect':collect(out)
    elif a.phase=='fit':fit(out,a.annotations)
    elif a.phase=='evaluate':evaluate(out,a.annotations)
    else:plot(out)


if __name__=='__main__':
    main()
