"""Raw-frame visual audit: recorded-pose geometry versus commanded dynamics.

The kinematic model is a separate diagnostic instance. It never steps physics,
produces policy data, scores tasks, or writes poses into the running simulator.
"""
from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import time
import uuid

import mujoco
import numpy as np
from PIL import Image

from .environment import ROOT, OpenArmEnv
from .recording import dump
from .reference_clock import load as load_clock
from .replay import ReferenceReplay
from .reference_phases import landmarks

CAMERAS = ('head', 'left', 'right')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_samples(root=ROOT):
    folder = root / 'reference/frames'
    data = json.loads((folder / 'manifest.json').read_text())
    if (data.get('schema') != 'openarm-reference-images/1'
            or data['acquisition_sha256'] != sha(root / 'reference/acquisition.json')
            or data['reference_sha256'] != sha(root / 'reference/real_trajectory.json')):
        raise ValueError('reference image provenance differs from the audited trajectories')
    times = json.loads((root / 'reference/acquisition.json').read_text())['episodes']
    reference = json.loads((root / 'reference/real_trajectory.json').read_text())['episodes']
    plans = {ep: landmarks(row) for ep,row in reference.items()}
    seen = set()
    for sample in data['samples']:
        episode, frame = str(sample['episode']), sample['frame']
        pair = (episode, frame)
        if pair in seen or type(frame) is not int or not 0 <= frame < times[episode]['frames']:
            raise ValueError('invalid or duplicate image sample')
        seen.add(pair)
        if abs(sample['recorded_seconds']-times[episode]['recorded_seconds'][frame]) > 1e-6:
            raise ValueError('reference image timestamp mismatch')
        if 'command_landmarks' in sample and sample['command_landmarks'] != plans[episode]['samples'].get(frame,[]):
            raise ValueError('reference command landmark mismatch')
        state, action = [], []
        for side in ('left', 'right'):
            e = reference[episode]
            state += e[f'{side}_joints'][frame]+[e[f'{side}_gripper'][frame]]
            action += e[f'{side}_action_joints'][frame]+[e[f'{side}_action_gripper'][frame]]
        if (np.asarray(sample['state']).shape != (16,) or np.asarray(sample['action']).shape != (16,)
                or not np.allclose(sample['state'], state, rtol=0, atol=1e-6)
                or not np.allclose(sample['action'], action, rtol=0, atol=1e-6)):
            raise ValueError('reference image encoders or commands mismatch')
        if set(sample['images']) != set(CAMERAS):
            raise ValueError('all three camera views are required')
        for camera, item in sample['images'].items():
            path = folder / f'ep{episode}/{frame:06d}-{camera}.jpg'
            if (item['path'] != str(path.relative_to(folder)) or sha(path) != item['sha256']):
                raise ValueError('reference image path or content mismatch')
    if not seen:
        raise ValueError('no reference samples')
    return data


def recorded_pose(env, state):
    """Set a diagnostic-only model to measured encoders, with candidate fingers."""
    state = np.asarray(state, float)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError('recorded state must be a finite 16-vector')
    for i, side in enumerate(('left', 'right')):
        env.data.qpos[env.qids[side]] = state[8*i:8*i+7]
        for joint in env.fingers[side]:
            env.data.qpos[joint.qposadr[0]] = np.clip(env.grippers[side].to_joint(state[8*i+7]), *joint.range)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def build():
    samples = load_samples()
    load_clock()  # Validate monotonicity and complete trajectory bindings too.
    output = ROOT / 'runs' / (time.strftime('%Y%m%d-%H%M%S')+'-alignment-'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    physics = geometry = None
    manifest = {'schema': 'openarm-camera-alignment/1', 'status': 'running', 'kind': 'alignment',
                'run_id': output.name, 'source_sha256': sha(__file__),
                'images_manifest_sha256': sha(ROOT / 'reference/frames/manifest.json'),
                'reference_sha256': sha(ROOT / 'reference/real_trajectory.json'),
                'acquisition_sha256': sha(ROOT / 'reference/acquisition.json'),
                'profile_sha256': sha(ROOT / 'sim/profile.json'),
                'task_sha256': sha(ROOT / 'task/bottle_to_box.xml'),
                'calibrate_sha256': sha(ROOT / 'calibrate.json'),
                'environment_sha256': sha(ROOT / 'sim/environment.py'),
                'gripper_sha256': sha(ROOT / 'sim/gripper.py'),
                'phases_sha256': sha(ROOT / 'sim/reference_phases.py'),
                'replay_sha256': sha(ROOT / 'sim/replay.py'),
                'scope': 'development visual alignment; no policy or training data', 'transfer_ready': False}
    dump(output / 'manifest.json', manifest)
    started = time.monotonic()
    report = {**manifest, 'samples': [], 'episodes': [], 'cameras': {},
              'limitations': [
                  '同一记录行不等于相机同时曝光；曝光与状态采样偏移未知。',
                  '头部原图呈明显鱼眼畸变；当前模型仍为未标定的透视相机。',
                  '关节对照仅用于投影检查，隐藏未重建的瓶子与篮子，不参与任务评分。',
                  '动力学列从初态积分真实指令，不覆盖关节位置；物体初始位姿仍是候选。',
                  '关节与末端差是两个模型状态之间的误差，不是真机空间标定精度。',
                  '尚无独立标注的 2D/3D 对应点，不能报告真实重投影误差。']}
    try:
        physics, geometry = OpenArmEnv(images=False), OpenArmEnv(images=False)
        # Render-only masking in an independent model, not removal of collisions.
        for body in (geometry.bottle_id, geometry.box_id):
            geometry.model.geom_group[geometry.model.geom_bodyid == body] = 5
        manifest['model'] = physics.manifest()
        report['model'] = manifest['model']
        for camera in CAMERAS:
            cid = physics.model.camera(physics.profile['camera_map'][camera]).id
            width, height = physics.profile['camera_sizes'][camera]
            fovy = float(physics.model.cam_fovy[cid])
            focal = height / (2*np.tan(np.deg2rad(fovy)/2))
            report['cameras'][camera] = {'model': 'candidate_pinhole_unvalidated', 'size': [width, height],
                'fovy_degrees': fovy, 'derived_focal_px': float(focal),
                'note': 'Derived from the simulation fovy, not measured real intrinsics.'}
        for episode in sorted({s['episode'] for s in samples['samples']}):
            selected = sorted((s for s in samples['samples'] if s['episode'] == episode), key=lambda s:s['frame'])
            physics.reset(seed=0, options={'episode': episode})
            geometry.reset(seed=0, options={'episode': episode})
            replay = ReferenceReplay(physics, episode)
            current = 0
            for sample in selected:
                frame = sample['frame']
                while current < frame and not physics._done:
                    replay.advance()
                    current += 1
                live = current == frame
                recorded_pose(geometry, sample['state'])
                row = {'episode': episode, 'frame': frame, 'recorded_seconds': sample['recorded_seconds'],
                       'command_landmarks': sample.get('command_landmarks', []),
                       'timestamp': sample['timestamp'], 'images': {}, 'dynamics_available': live,
                       'dynamics_unavailable_reason': None if live else '回放已因物理终止；不展示旧帧冒充当前状态。',
                       'sim_seconds': float(physics.data.time) if live else None}
                for camera in CAMERAS:
                    paths = {'real': '/reference/frames/'+sample['images'][camera]['path']}
                    hashes = {'real': sample['images'][camera]['sha256']}
                    for label, env in (('geometry', geometry), ('dynamics', physics)):
                        if label == 'dynamics' and not live:
                            continue
                        path = output / 'images' / f'ep{episode}-{frame:06d}-{camera}-{label}.png'
                        path.parent.mkdir(exist_ok=True)
                        Image.fromarray(env.render(camera)).save(path)
                        paths[label] = '/runs/'+str(path.relative_to(ROOT/'runs'))
                        hashes[label] = sha(path)
                    row['images'][camera] = {'urls': paths, 'sha256': hashes, 'size': sample['images'][camera]['size']}
                if live:
                    evaluation = physics.evaluation
                    row['simulation'] = {
                        'evidence_source': 'privileged simulator diagnostics; not real camera measurements',
                        'reached_stages': [event['name'] for event in physics.events],
                        'held': evaluation['held'], 'touching': evaluation['touching'],
                        'inside_basket': evaluation['inside_basket'],
                        'supported_by_basket': evaluation['supported_by_basket'],
                        'success': evaluation['success'], 'invalid': evaluation['invalid'],
                        'failure_reasons': evaluation['failure_reasons'],
                        'bottle_world': physics.data.xpos[physics.bottle_id].tolist(),
                        'gripper_motor_rad': physics.state()[[7,15]].tolist(),
                        'recorded_gripper_motor_rad': [sample['state'][7], sample['state'][15]],
                        'max_penetration_m': evaluation['max_penetration_m']}
                    indices = [*range(7), *range(8, 15)]
                    delta = physics.state()[indices]-np.asarray(sample['state'])[indices]
                    row['joint_rmse_rad'] = float(np.sqrt(np.mean(delta**2)))
                    row['joint_max_abs_rad'] = float(np.max(np.abs(delta)))
                    row['clock_error_seconds'] = float(physics.data.time)-sample['recorded_seconds']
                    row['ee_body_model_delta_m'] = {side: float(np.linalg.norm(
                        physics.data.xpos[physics.ee_ids[side]]-
                        geometry.data.xpos[geometry.ee_ids[side]])) for side in ('left','right')}
                report['samples'].append(row)
            report['episodes'].append({'episode': episode, 'last_observation': current,
                'physical_terminal': bool(physics._done), 'metrics': replay.metrics(),
                'record_end_reached': current == physics.reference[str(episode)]['frames']-1,
                'events': physics.events, 'final_evaluation': physics.evaluation,
                'scope': 'development command replay through the last requested image; no policy called'})
            print(json.dumps({'episode': episode, 'samples': len(selected), 'last_observation': current}), flush=True)
        report.update({'status': 'completed', 'wall_seconds': time.monotonic()-started})
        dump(output / 'alignment.json', report)
        dump(output / 'summary.json', {'status': 'completed', 'scope': manifest['scope'],
            'samples': len(report['samples']), 'task_success_evaluated': False, 'transfer_ready': False,
            'report_url': '/runs/'+output.name+'/alignment.json'})
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        dump(output / 'manifest.json', manifest)
        if physics is not None:
            physics.close()
        if geometry is not None:
            geometry.close()
    print(json.dumps({'run': str(output), 'samples': len(report['samples'])}, ensure_ascii=False))
    return output


def latest():
    """Operator/research-only artifacts; never exposed to the execution agent."""
    for path in sorted((ROOT/'runs').glob('*-alignment-*/alignment.json'), reverse=True):
        data = json.loads(path.read_text())
        if data.get('status') != 'completed':
            continue
        checks = {'profile_sha256': 'sim/profile.json', 'task_sha256': 'task/bottle_to_box.xml',
                  'calibrate_sha256': 'calibrate.json', 'environment_sha256': 'sim/environment.py',
                  'gripper_sha256': 'sim/gripper.py',
                  'phases_sha256': 'sim/reference_phases.py',
                  'replay_sha256': 'sim/replay.py', 'images_manifest_sha256': 'reference/frames/manifest.json',
                  'acquisition_sha256': 'reference/acquisition.json', 'reference_sha256': 'reference/real_trajectory.json',
                  'source_sha256': 'sim/alignment.py'}
        data['matches_current_model'] = all(data.get(k) == sha(ROOT/v) for k,v in checks.items())
        assets = data.get('model', {}).get('hashes', {})
        model_paths = ('task/cell-v1.xml', 'openarm_mujoco/v1/openarm_bimanual.xml')
        data['matches_current_model'] = data['matches_current_model'] and all(
            assets.get(name) == sha(ROOT/name) for name in model_paths)
        data['report_url'] = '/runs/'+path.parent.name+'/alignment.json'
        return data
    return {'status': 'unavailable', 'samples': [], 'transfer_ready': False}


def inspect(episode=0, frame=0, camera='head'):
    data = latest()
    row = next((r for r in data['samples'] if r['episode'] == episode and r['frame'] == frame), None)
    if row is None or camera not in CAMERAS:
        raise ValueError('没有该帧的对照；先读取第 0 帧返回的 available_samples 与 available_landmarks')
    images = row['images'][camera]
    result = {k:v for k,v in row.items() if k != 'images'}
    result.update({'evidence_id': f"{data['run_id']}:{episode}:{frame}:{camera}",
                   'evidence_kind': 'recorded_alignment_not_live_observation',
                   'matches_current_model': data['matches_current_model'],
                   'camera': camera, 'camera_model': data['cameras'][camera],
                   'limitations': data['limitations'], 'report_url': data['report_url'],
                   'available_samples': [[r['episode'],r['frame']] for r in data['samples']],
                   'available_landmarks': [{'episode': r['episode'], 'frame': r['frame'],
                        'commands': r.get('command_landmarks', [])} for r in data['samples'] if r.get('command_landmarks')],
                   'image_sha256': images['sha256'], 'images': {}, 'transfer_ready': False})
    for label, url in images['urls'].items():
        path = (ROOT/url.lstrip('/')).resolve()
        allowed = ROOT/'reference/frames' if label == 'real' else ROOT/'runs'/data['run_id']/'images'
        if not path.is_relative_to(allowed) or sha(path) != images['sha256'][label]:
            raise ValueError('alignment image path or content mismatch')
        mime = 'image/jpeg' if label == 'real' else 'image/png'
        result['images'][label] = f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()
    return result


if __name__ == '__main__':
    build()
