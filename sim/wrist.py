"""Bounded wrist projection studies and read-only research evidence."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from .environment import ROOT
from .alignment import sha
from .recording import dump

NUMERICS = Path('/home/wsk/xunlian/LeberoPai0.5/openpi/.venv/bin/python')
PROTOCOL = {
    'id':'wrist_silhouette_v2', 'fit':[[0,0]], 'development_validation':[[1,0],[2,100]],
    'review_only':[[0,100],[0,300],[0,500],[1,100],[1,300],[1,500],[2,0],[2,300],[2,500]],
    'selection':'convex-hull proposal then native MuJoCo visibility refinement; fit frame only; freeze selected before validation',
    'selection_note':'ep2 frame100 chosen from raw visibility, before validation scoring; ep1 right retained with explicit mask rejection for bottle occlusion',
    'scope':'visual camera candidate under fixed unverified finger mapping; not physical calibration',
    'camera_model':'pinhole, square pixels, centered principal point, zero distortion',
    'mask':'grayscale < 75, y>=200; largest bottom-connected component in each half; reject truncated or ambiguous masks',
    'native_optimizer':{'seed':42,'population':64,'iterations':18,'wall_limit_seconds_per_wrist':90},
    'seed':42, 'transfer_ready':False,
}


def build(output=None):
    if not NUMERICS.is_file():
        raise ValueError('existing SciPy/OpenCV runtime unavailable; no packages installed')
    path=Path(output).resolve() if output else ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-wrist-'+uuid.uuid4().hex[:6])
    if path.parent != ROOT/'runs' or path.is_symlink():
        raise ValueError('study must stay directly inside local runs')
    if path.exists() and (not (path/'job.json').is_file() or any(p.name not in {'job.json','worker.log'} for p in path.iterdir())):
        raise ValueError('refusing to overwrite an existing study')
    path.mkdir(exist_ok=True);(path/'images').mkdir()
    dump(path/'protocol.json',PROTOCOL)
    env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH') if k in os.environ}
    env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl')
    started=time.monotonic()
    try:
        with (path/'study.log').open('w') as log:
            for phase in ('prepare','fit','refine','masks','verify'):
                python=NUMERICS if phase in {'fit','masks'} else ROOT/'.venv/bin/python'
                dump(path/'progress.json',{'phase':phase,'elapsed_seconds':time.monotonic()-started})
                print(json.dumps({'phase':phase,'run_id':path.name}),flush=True)
                subprocess.run([str(python),'-B',str(ROOT/'tools/fit_wrist.py'),phase,str(path)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240 if phase=='refine' else 60)
        result=json.loads((path/'wrist.json').read_text())
        if result['status']!='completed' or len(result['samples'])!=24 or result['applied']:
            raise ValueError('incomplete or unexpectedly applied camera study')
        dump(path/'summary.json',{'status':'completed','scope':PROTOCOL['scope'],'samples':24,
            'elapsed_seconds':time.monotonic()-started,'calibrated':False,'transfer_ready':False})
        print(json.dumps({'status':'completed','report':str(path/'index.html'),'calibrated':False}),flush=True)
        return path
    except BaseException as exc:
        manifest=json.loads((path/'manifest.json').read_text()) if (path/'manifest.json').exists() else {}
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc))
        dump(path/'manifest.json',manifest)
        raise


def latest(root=ROOT):
    for path in sorted((root/'runs').glob('*-wrist-*/wrist.json'),reverse=True):
        result=json.loads(path.read_text())
        if result.get('status')!='completed':
            continue
        directory=path.parent
        manifest=json.loads((directory/'manifest.json').read_text())
        selected=json.loads((directory/'selected.json').read_text())
        if sha(directory/'selected.json')!=result['selected_sha256']:
            raise ValueError('wrist candidate changed after verification')
        matching=all(sha(root/name)==digest for name,digest in manifest['hashes'].items())
        matching=matching and sha(directory/'protocol.json')==manifest['protocol_sha256']
        stats={}
        for side in ('left','right'):
            rows=[r for r in result['samples'] if r['side']==side and r['split']=='development_validation' and r['metrics']]
            stats[side]={'scored_frames':len(rows),
                **{label+'_iou':sum(r['metrics'][label]['iou'] for r in rows)/len(rows) if rows else None for label in ('baseline','candidate')},
                'fovy':selected['cameras'][side]['fovy'],'at_parameter_bound':bool(selected['cameras'][side]['at_bounds'])}
        result.update(matches_current_model=matching,development=stats,
            report_url='/runs/'+directory.name+'/index.html',data_url='/runs/'+directory.name+'/wrist.json',
            excluded=[{'frame':r['key'],'side':r['side'],'reason':r['score_exclusion']} for r in result['samples'] if r.get('score_exclusion')])
        return result
    return {'status':'unavailable','samples':[],'calibrated':False,'transfer_ready':False}


def inspect(episode=0,frame=0,camera='right'):
    data=latest()
    row=next((r for r in data['samples'] if r['episode']==episode and r['frame']==frame and r['side']==camera),None)
    if row is None:
        raise ValueError('no wrist sample; use episode 0..2, frames 0/100/300/500, left/right')
    path=ROOT/'runs'/data['run_id']
    manifest=json.loads((path/'manifest.json').read_text())
    source=next(r for r in manifest['samples'] if r['episode']==episode and r['frame']==frame)['images'][camera]
    real=(ROOT/'reference/frames'/source['path']).resolve()
    if not real.is_relative_to(ROOT/'reference/frames'):
        raise ValueError('invalid raw wrist image path')
    images={'real':(real,source['sha256'],'image/jpeg')}
    for key in ('baseline','candidate','overlay'):
        entry=row['images'][key]
        file=(path/entry['path']).resolve()
        if not file.is_relative_to(path/'images'):
            raise ValueError('invalid wrist image path')
        images[key]=(file,entry['sha256'],'image/png')
    result={'evidence_id':f"{data['run_id']}:{episode}:{frame}:{camera}",
        'evidence_kind':'archived_wrist_candidate_not_live_observation','camera':camera,
        'split':row['split'],'metrics':row['metrics'],'score_exclusion':row['score_exclusion'],
        'matches_current_model':data['matches_current_model'],'limitations':data['limitations'],
        'report_url':data['report_url'],'applied':False,'calibrated':False,'transfer_ready':False,
        'images':{},'image_sha256':{}}
    for key,(file,digest,mime) in images.items():
        if sha(file)!=digest:
            raise ValueError('wrist image changed')
        result['images'][key]='data:'+mime+';base64,'+base64.b64encode(file.read_bytes()).decode()
        result['image_sha256'][key]=digest
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    build(parser.parse_args().output)
