"""Import a small, audited image sample; SSH only reads existing source files.

Uses each trajectory record's rgb_paths, not video seeking or filename seconds.
No remote archive, cache, camera, hardware or model process is created.
"""
import hashlib
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.reference_phases import landmarks
FRAMES = [0, 100, 300, 500]
CAMERAS = {"cam_0_rgb": "left", "cam_1_rgb": "right", "cam_2_rgb": "head"}
REMOTE = r'''
from pathlib import Path
import datetime,hashlib,io,json,sys,tarfile
request=json.loads(sys.argv[1])
base=Path('/home/sj/Documents/moqi_workspace_20251119_211400')
converter=base/'scripts/data_transfer/moqi_to_lerobot.py'
assert hashlib.sha256(converter.read_bytes()).hexdigest()==request['converter_sha256']
rows=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 def add(name,raw):
  item=tarfile.TarInfo(name);item.size=len(raw);out.addfile(item,io.BytesIO(raw))
 for ep,source in request['episodes'].items():
  path=Path(source['path']);assert path.is_relative_to(base/'record_data')
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256']
  records=json.loads(raw);start=datetime.datetime.fromisoformat(records[0]['timestamp'])
  for index in request['frames'][ep]:
   record=records[index];assert record['frame_id']==index
   images={}
   for relative in record['observations']['rgb_paths']:
    image_path=(path.parent/relative).resolve();assert image_path.is_relative_to(path.parent)
    camera=request['cameras'][image_path.parent.name]
    name=f'ep{ep}/{index:06d}-{camera}.jpg';raw=image_path.read_bytes();add(name,raw)
    images[camera]={'path':name,'source_path':str(image_path),'sha256':hashlib.sha256(raw).hexdigest()}
   assert set(images)=={'head','left','right'}
   o=record['observations'];q=o['qpos'];g=o['gripper_joints']
   a=record['action'];u=a['qpos'];ug=a['gripper_joints']
   rows.append({'episode':int(ep),'frame':index,'timestamp':record['timestamp'],
    'recorded_seconds':(datetime.datetime.fromisoformat(record['timestamp'])-start).total_seconds(),
    'state':q[:7]+[g[0]]+q[7:]+[g[1]],'action':u[:7]+[ug[0]]+u[7:]+[ug[1]],'images':images})
 add('manifest.json',json.dumps({'samples':rows},ensure_ascii=False).encode())
'''


def main():
    importer_source = Path(__file__).read_bytes()
    selector_source = Path(landmarks.__code__.co_filename).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-stages', action='store_true', help='append command-landmark images; retain every existing JPEG')
    args = parser.parse_args()
    destination = ROOT / 'reference/frames'
    if destination.exists() and not args.task_stages:
        raise ValueError('reference/frames already exists; never overwrite audited images')
    clock_path = ROOT / 'reference/acquisition.json'
    clock = json.loads(clock_path.read_text())
    reference = json.loads((ROOT / 'reference/real_trajectory.json').read_text())['episodes']
    existing_bytes = (destination/'manifest.json').read_bytes() if destination.exists() else None
    existing = json.loads(existing_bytes) if existing_bytes else None
    if existing:
        from sim.alignment import load_samples
        load_samples()  # Verify the old image, encoder and timestamp associations before extending.
    plans = {ep: landmarks(row) for ep,row in reference.items()} if args.task_stages else {}
    already = {(str(s['episode']),s['frame']) for s in (existing or {}).get('samples',[])}
    frames = {ep: sorted((set(FRAMES) | set(plans.get(ep,{}).get('samples',{}))) -
                        {frame for key,frame in already if key == ep}) for ep in reference}
    frames = {ep:indices for ep,indices in frames.items() if indices}
    if not frames:
        print(json.dumps({'status':'already_imported','path':str(destination)}))
        return
    converter = '/home/sj/Documents/moqi_workspace_20251119_211400/scripts/data_transfer/moqi_to_lerobot.py'
    request = {'frames': frames, 'cameras': CAMERAS,
               'episodes': {k: {f: v[f] for f in ('path', 'sha256')} for k, v in clock['episodes'].items() if k in frames},
               'converter_sha256': clock['sources'][converter]}
    command = 'python3 -B - ' + shlex.quote(json.dumps(request))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                             '-o', 'UpdateHostKeys=no', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
                             '-T', 'data-cs', command], input=REMOTE.encode(), capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[:2000])
    if len(result.stdout) > 32 * 1024 * 1024:
        raise ValueError('unexpected reference sample size')
    expected = {f'ep{ep}/{f:06d}-{c}.jpg' for ep,indices in frames.items() for f in indices for c in CAMERAS.values()}
    with tempfile.TemporaryDirectory(prefix='.frames-', dir=ROOT / 'reference') as temporary:
        stage = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
            names = archive.getnames()
            if set(names) != expected | {'manifest.json'} or len(names) != len(set(names)):
                raise ValueError('archive does not contain exactly the requested samples')
            for item in archive:
                if not item.isfile() or item.size > 4 * 1024 * 1024:
                    raise ValueError('unexpected archive entry')
                p = stage / item.name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(archive.extractfile(item).read())
        manifest = json.loads((stage / 'manifest.json').read_text())
        pairs = [(str(s['episode']),s['frame']) for s in manifest['samples']]
        if len(pairs) != len(set(pairs)) or set(pairs) != {(ep,f) for ep,indices in frames.items() for f in indices}:
            raise ValueError('unexpected or duplicated image sample records')
        for sample in manifest['samples']:
            e, f = str(sample['episode']), sample['frame']
            src = reference[e]
            state, action = [], []
            for side in ('left', 'right'):
                state += src[f'{side}_joints'][f] + [src[f'{side}_gripper'][f]]
                action += src[f'{side}_action_joints'][f] + [src[f'{side}_action_gripper'][f]]
            if (not np.allclose(state, sample['state'], rtol=0, atol=1e-6)
                    or not np.allclose(action, sample['action'], rtol=0, atol=1e-6)
                    or abs(sample['recorded_seconds']-clock['episodes'][e]['recorded_seconds'][f]) > 1e-6):
                raise ValueError('raw record does not align with the local reference')
            for camera, item in sample['images'].items():
                raw = (stage / item['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise ValueError('reference image hash mismatch')
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    item['size'] = list(im.size)
                    if im.size != ((1280, 960) if camera == 'head' else (640, 480)):
                        raise ValueError('unexpected camera resolution')
        if existing:
            manifest['samples'] += existing['samples']
        manifest['samples'].sort(key=lambda s:(s['episode'],s['frame']))
        if plans:
            manifest['command_landmark_selection'] = plans
            for sample in manifest['samples']:
                sample['command_landmarks'] = plans[str(sample['episode'])]['samples'].get(sample['frame'],[])
        manifest.update({'schema': 'openarm-reference-images/1', 'host': 'data-cs',
                         'acquisition_sha256': hashlib.sha256(clock_path.read_bytes()).hexdigest(),
                         'reference_sha256': clock['reference_sha256'],
                         'sources': {k: {f: v[f] for f in ('path','sha256')} for k,v in clock['episodes'].items()},
                         'camera_map': CAMERAS, 'camera_map_source': converter,
                         'camera_map_source_sha256': request['converter_sha256'],
                         'importer_sha256': hashlib.sha256(importer_source).hexdigest(),
                         'selector_sha256': hashlib.sha256(selector_source).hexdigest(),
                         'image_transform': 'none; original JPEG bytes, no crop, resize or video re-encode',
                         'alignment': 'same logger record; camera exposure synchronization remains unverified',
                         'purpose': 'development visual calibration; not a fresh held-out test set'})
        source_history = (destination if existing else stage)/'history'
        source_history.mkdir(exist_ok=True)
        manifest['local_source_snapshots'] = {}
        for label,raw in (('importer',importer_source),('selector',selector_source)):
            target = source_history/(label+'-'+hashlib.sha256(raw).hexdigest()[:16]+'.py')
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('local source snapshot mismatch')
            if not target.exists():
                target.write_bytes(raw)
            manifest['local_source_snapshots'][label] = 'history/'+target.name
        if existing:
            previous_hash = hashlib.sha256(existing_bytes).hexdigest()
            manifest['previous_manifest_sha256'] = previous_hash
            history = destination/'history'
            history.mkdir(exist_ok=True)
            snapshot = history/(previous_hash[:16]+'.json')
            if snapshot.exists() and snapshot.read_bytes() != existing_bytes:
                raise ValueError('previous manifest snapshot mismatch')
            if not snapshot.exists():
                snapshot.write_bytes(existing_bytes)
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        if existing:
            moved = []
            try:
                for name in sorted(expected):
                    target = destination/name
                    if target.exists():
                        raise ValueError('refusing to overwrite an existing image')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    (stage/name).rename(target)
                    moved.append(target)
                (stage/'manifest.json').replace(destination/'manifest.json')
            except BaseException:
                for target in moved:
                    target.unlink()  # Only files created by this invocation.
                raise
        else:
            stage.rename(destination)
    print(json.dumps({'path': str(destination), 'samples': len(manifest['samples']), 'added_images': len(expected)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
