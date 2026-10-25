"""Offline hand-eye self-calibration with a carried natural point.

The bottle-cap point is estimated in each grasp's wrist frame. It is never
assumed to be a robot joint, and need not have a known metric bottle offset.
Images and robot recordings are read from disk only; no device code is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.offline_calibration import Kinematics,remote,digest,write,project,bounds,R0
from scipy.spatial.transform import Rotation


def collect(out):
    if (out/'data.json').exists():raise ValueError('existing carry study is immutable')
    original=json.loads((out.parent/'data.json').read_text())
    requests=[];groups=[];kin=Kinematics(ROOT/'reference/source/inputs/robot.urdf')
    for ep,e in original['episodes'].items():
        frames=e['frames'];commands=np.array([r['gripper_command'] for r in frames])
        left=np.where(commands[:,0]>-.3)[0];right=np.where(commands[:,1]>-.3)[0]
        for side,start,stop in [('right',right[0]+20,min(right[-1]-10,left[0]-10)),
                                ('left',max(left[0]+20,right[-1]+15),left[-1]-10)]:
            ids=np.rint(np.linspace(start,stop,8)).astype(int).tolist()
            key=f'ep{ep}-{side}';group={'key':key,'episode':int(ep),'side':side,'frames':ids}
            groups.append(group)
            for i,f in enumerate(ids):
                q={j['name']:0. for j in kin.joints.values()}
                q.update({f'openarm_{s}_joint{n+1}':frames[f]['q'][off+n]
                    for s,off in [('left',0),('right',7)] for n in range(7)})
                requests.append({'group':key,'episode':int(ep),'frame':f,'order':i,
                    'trajectory':e['source'],'trajectory_sha256':e['sha256'],
                    'T_base_wrist':kin.pose(f'openarm_{side}_link7',q).tolist()})
    program='''from pathlib import Path
import json,hashlib,base64
requests=REQUESTS
rows=[];protected={};trajectories={}
for r in requests:
 p=Path(r['trajectory']).resolve()
 assert p.is_relative_to(Path('/home/sj/Documents/moqi_workspace_20251119_211400/record_data')) and p.name=='trajectory.json'
 if str(p) not in trajectories:
  raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==r['trajectory_sha256']
  protected[str(p)]=h;trajectories[str(p)]=json.loads(raw)
 frame=trajectories[str(p)][r['frame']]
 image=(p.parent/frame['observations']['rgb_paths'][2]).resolve()
 assert image.is_relative_to(p.parent)
 raw=image.read_bytes();h=hashlib.sha256(raw).hexdigest();protected[str(image)]=h
 rows.append({**r,'source':str(image),'sha256':h,'bytes':base64.b64encode(raw).decode()})
after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in protected}
print(json.dumps({'rows':rows,'protected_before':protected,'protected_after':after}))
'''.replace('REQUESTS',repr(requests))
    result=remote(program)
    if result['protected_before']!=result['protected_after']:raise ValueError('source changed')
    (out/'images').mkdir(exist_ok=True)
    import base64,hashlib
    for row in result['rows']:
        raw=base64.b64decode(row.pop('bytes'),validate=True)
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('image changed')
        row['image']=f"images/{row['group']}-{row['frame']:06d}.jpg"
        (out/row['image']).write_bytes(raw)
    result.update(groups=groups,urdf_sha256=digest(ROOT/'reference/source/inputs/robot.urdf'),
                  schema='openarm-carried-point-data/1',hardware_access=False,remote_writes=False)
    write(out/'data.json',result)
    if (ROOT/'reference/cap_points.json').is_file():
        (out/'points.json').write_bytes((ROOT/'reference/cap_points.json').read_bytes())
    print(json.dumps({'images':len(result['rows']),'protected':len(result['protected_before'])}))


def samples(out,episodes=None):
    data=json.loads((out/'data.json').read_text());labels=json.loads((out/'points.json').read_text())
    if data['urdf_sha256']!=digest(ROOT/'reference/source/inputs/robot.urdf'):
        raise ValueError('URDF changed')
    lookup={(r['group'],r['frame']):r for r in labels['points']};rows=[]
    if len(lookup)!=len(labels['points']):raise ValueError('duplicate point annotations')
    for r in data['rows']:
        if episodes is not None and r['episode'] not in episodes:continue
        if digest(out/r['image'])!=r['sha256']:raise ValueError('original image changed')
        label=lookup[(r['group'],r['frame'])];uv=np.asarray(label['uv'],float)
        if label.get('image_sha256',r['sha256'])!=r['sha256']:raise ValueError('point labels refer to a different image')
        if uv.shape!=(2,) or not np.isfinite(uv).all() or not np.all((uv>0)&(uv<[1280,960])):
            raise ValueError('invalid original-image coordinates')
        rows.append({**r,'uv':uv.tolist()})
    return rows


def local_point(camera,rows):
    """Triangulate one fixed wrist-local point from known camera bearings."""
    f=np.exp(camera[6]);xy=(np.array([r['uv'] for r in rows])-camera[7:9])/f
    rd=np.linalg.norm(xy,axis=1);theta=rd.copy()
    for _ in range(12):theta-=(theta*(1+camera[9]*theta**2)-rd)/(1+3*camera[9]*theta**2)
    ray=np.c_[xy*np.divide(np.sin(theta),rd,out=np.ones_like(rd),where=rd>1e-12)[:,None],np.cos(theta)]
    rc=Rotation.from_rotvec(camera[3:6]).as_matrix()@R0;a=[];b=[]
    for d,r in zip(ray,rows):
        t=np.array(r['T_base_wrist']);n=np.eye(3)-np.outer(d,d)
        a.append(n@rc@t[:3,:3]);b.append(-n@rc@(t[:3,3]-camera[:3]))
    a=np.vstack(a);b=np.concatenate(b);point=np.linalg.lstsq(a,b,rcond=None)[0]
    return point,float(np.linalg.cond(a))


def predicted(camera,attachment,rows):
    poses=np.array([r['T_base_wrist'] for r in rows]);world=np.einsum('nij,nj->ni',poses[:,:3,:3],attachment)+poses[:,:3,3]
    return project(camera,world,'fisheye_k1')


def registration_valid(point,condition):
    point=np.asarray(point,float)
    return bool(point.shape==(3,) and np.isfinite(point).all() and np.isfinite(condition)
        and condition<=100 and np.all(np.abs(point)<=.35) and np.linalg.norm(point)<=.45)


def fit(out):
    if (out/'selected.json').exists():raise ValueError('selection is already frozen')
    rows=samples(out,[0,1])
    groups=sorted({r['group'] for r in rows});indices=np.array([groups.index(r['group']) for r in rows])
    uv=np.array([r['uv'] for r in rows]);lo,hi=bounds('fisheye_k1');lo[9]=-.10
    low=np.r_[lo,np.full(3*len(groups),-.35)];high=np.r_[hi,np.full(3*len(groups),.35)]
    protocol={'schema':'openarm-carried-point-protocol/1','fit_episodes':[0,1],
        'validation_episodes':[2,3], 'registration_frames_per_validation_grasp':3,
        'validation':'camera frozen; first 3 frames register a new wrist-local cap point; remaining 5 only score',
        'scope':'historical markerless self-calibration candidate; not object pose service or real acceptance',
        'assumption':'cap centre fixed relative to link7 within each selected single-hand carry segment',
        'optimizer':'bounded robust least squares, 12 fixed-seed starts, fit pixels only',
        'limits':{'validation_p95_px':12,'camera_loo_span_m':.03,'dimensionless_condition':10000},
        'pixel_annotations':'manual cap centres; not surveyed fiducials',
        'data_sha256':digest(out/'data.json'),'points_sha256':digest(out/'points.json'),
        'script_sha256':digest(__file__),'fit_rows':rows,'hardware_access':False,'runtime_config_changed':False}
    write(out/'protocol.json',protocol)
    def residual(x):
        attachment=x[10:].reshape(-1,3)[indices];p,z=predicted(x[:10],attachment,rows)
        return np.r_[(p-uv).ravel(),np.maximum(.04-z,0)*1000]
    initial=np.array(json.loads((out.parent/'selected.json').read_text())['parameters'])
    rng=np.random.default_rng(142);solutions=[]
    for i in range(12):
        camera=initial.copy()
        if i:
            camera[:3]+=rng.normal(0,.04,3);camera[3:6]+=rng.normal(0,.08,3);camera[6]+=rng.normal(0,.08)
        camera=np.clip(camera,lo+1e-6,hi-1e-6)
        attachment=np.array([local_point(camera,[r for r in rows if r['group']==g])[0] for g in groups]).ravel()
        start=np.clip(np.r_[camera,attachment],low+1e-6,high-1e-6)
        solution=least_squares(residual,start,bounds=(low,high),loss='soft_l1',f_scale=4.,x_scale='jac',max_nfev=1500,
            ftol=1e-10,xtol=1e-10,gtol=1e-8)
        solutions.append(solution)
    best=min(solutions,key=lambda r:r.cost);p,z=predicted(best.x[:10],best.x[10:].reshape(-1,3)[indices],rows)
    jac=best.jac[:2*len(rows)];sv=np.linalg.svd(jac/np.maximum(np.linalg.norm(jac,axis=0),1e-12),compute_uv=False)
    write(out/'selected.json',{'schema':'openarm-carried-point-candidate/1','parameters':best.x[:10].tolist(),
        'attachments':dict(zip(groups,best.x[10:].reshape(-1,3).tolist())),
        'fit_rmse_px':float(np.sqrt(np.mean(np.sum((p-uv)**2,axis=1)))),
        'fit_max_px':float(np.linalg.norm(p-uv,axis=1).max()),
        'jacobian_singular_values':sv.tolist(),'jacobian_condition':float(sv[0]/max(sv[-1],1e-15)),
        'at_bounds':np.where(np.minimum(best.x-low,high-best.x)<.005*(high-low))[0].tolist(),
        'selection_uses_validation':False,'protocol_sha256':digest(out/'protocol.json'),
        'physical_trial_passed':False,'transfer_ready':False,'applied_to_runtime':False})
    print(json.dumps({k:json.loads((out/'selected.json').read_text())[k] for k in ['fit_rmse_px','jacobian_condition','at_bounds']}),flush=True)


def evaluate(out):
    from PIL import Image,ImageDraw
    selected=json.loads((out/'selected.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    if digest(out/'protocol.json')!=selected['protocol_sha256'] or digest(out/'data.json')!=protocol['data_sha256']:
        raise ValueError('source/protocol changed after selection')
    rows=samples(out)
    if [r for r in rows if r['episode']<2]!=protocol['fit_rows']:raise ValueError('fit points changed')
    camera=np.array(selected['parameters']);groups=[];results=[]
    for g in sorted({r['group'] for r in rows}):
        group=sorted([r for r in rows if r['group']==g],key=lambda r:r['order']);ep=group[0]['episode']
        if ep<2:
            point=np.array(selected['attachments'][g]);condition=None;split=['fit']*len(group)
        else:
            point,condition=local_point(camera,group[:3]);split=['registration']*3+['held_out']*5
            if not registration_valid(point,condition):
                # Do not render or score a physically implausible triangulation
                # as though it were a valid calibrated 3D feature.
                groups.append({'group':g,'attachment_wrist_m':None,'registration_condition':condition,
                    'registration_failed':True,'reason':'ill-conditioned or implausible wrist-local point',
                    'withheld_frames_unvalidated':5,'scored_rmse_px':None,'scored_p95_px':None})
                results.extend({**r,'split':'registration_failed','predicted_uv':None,
                    'residual_px':None,'depth_m':None} for r in group)
                # Replace any earlier diagnostic overlay too. Leaving an old
                # green projection on disk would contradict the rejected result.
                for r in group:
                    im=Image.open(out/r['image']);draw=ImageDraw.Draw(im);u,v=r['uv']
                    draw.ellipse((u-6,v-6,u+6,v+6),outline='#e9ae4c',width=3)
                    draw.text((25,25),f"{g} / frame {r['frame']} / registration REJECTED; no 3D estimate",
                        fill='#ffbc85',stroke_width=1,stroke_fill='black')
                    im.save(out/'images'/f"{g}-{r['frame']:06d}-overlay.jpg",quality=90)
                continue
        pred,z=predicted(camera,np.repeat(point[None],len(group),axis=0),group)
        errors=np.linalg.norm(pred-np.array([r['uv'] for r in group]),axis=1)
        groups.append({'group':g,'attachment_wrist_m':point.tolist(),'registration_condition':condition,
            'scored_rmse_px':float(np.sqrt(np.mean(errors[0 if ep<2 else 3:]**2))),
            'scored_p95_px':float(np.quantile(errors[0 if ep<2 else 3:],.95)),
            'registration_failed':False})
        for i,row in enumerate(group):
            record={**row,'split':split[i],'predicted_uv':pred[i].tolist(),'residual_px':float(errors[i]),'depth_m':float(z[i])}
            results.append(record)
            im=Image.open(out/row['image']);draw=ImageDraw.Draw(im);u,v=row['uv'];a,b=pred[i]
            draw.ellipse((u-6,v-6,u+6,v+6),outline='#e9ae4c',width=3)
            draw.line((a-7,b,a+7,b),fill='#3bd1a4',width=3);draw.line((a,b-7,a,b+7),fill='#3bd1a4',width=3)
            draw.text((25,25),f"{g} / frame {row['frame']} / {split[i]} / {errors[i]:.1f} px",fill='white',stroke_width=1,stroke_fill='black')
            im.save(out/'images'/f"{g}-{row['frame']:06d}-overlay.jpg",quality=90)
    metrics={}
    for name in ['fit','registration','held_out']:
        e=np.array([r['residual_px'] for r in results if r['split']==name])
        metrics[name]={'n':len(e),'rmse_px':float(np.sqrt(np.mean(e**2))) if len(e) else None,
            'p95_px':float(np.quantile(e,.95)) if len(e) else None}
    reasons=['Rigid cap attachment and current physical mounting have no independent acceptance.']
    if selected['at_bounds']:reasons.append('Joint fit reaches exploratory parameter bounds.')
    if selected['jacobian_condition']>10000:reasons.append('Camera/attachment parameters are weakly identifiable.')
    if metrics['held_out']['p95_px'] is None or metrics['held_out']['p95_px']>12:reasons.append('Withheld pixels fail the 12 px exploratory limit.')
    failed=[g['group'] for g in groups if g.get('registration_failed')]
    if failed:reasons.append('Failed point registrations remain failures; they are not counted as passed validation.')
    if any(g['registration_condition'] is not None and g['registration_condition']>100 for g in groups):
        reasons.append('Three-frame point registration is poorly conditioned in at least one grasp.')
    result={'schema':'openarm-carried-point-result/1','status':'not_admitted','metrics':metrics,'groups':groups,
        'rows':results,'reasons':reasons,'selected_sha256':digest(out/'selected.json'),
        'expected_withheld_frames':20,'scored_withheld_frames':metrics['held_out']['n'],
        'failed_registration_groups':failed,'all_validation_groups_accepted':False,
        'pixel_consistency_only':True,'object_pose_service_implemented':False,
        'applied_to_runtime':False,'physical_trial_passed':False,'hardware_access':False}
    write(out/'result.json',result)
    print(json.dumps({'metrics':metrics,'reasons':reasons},ensure_ascii=False))


def timing(out):
    """Fit-only clock hypotheses; no inference of measured exposure latency."""
    import tempfile,os,copy,shutil
    source=json.loads((out/'data.json').read_text())
    original=json.loads((out.parent/'data.json').read_text())
    kin=Kinematics(ROOT/'reference/source/inputs/robot.urdf')
    candidates=[];best=None
    # Short signed scan; fixed before optimization. Labels never change.
    for offset in [-4,-2,0,2,4]:
        data=copy.deepcopy(source)
        data['alignment_hypothesis']={'offset_logger_records':offset,
            'qualification':'effective pairing hypothesis, not measured exposure or CAN latency',
            'source_data_sha256':digest(out/'data.json')}
        for row in data['rows']:
            frames=original['episodes'][str(row['episode'])]['frames'];i=row['frame']+offset
            if not 0<=i<len(frames):raise ValueError('offset leaves the recorded trajectory')
            q={j['name']:0. for j in kin.joints.values()}
            q.update({f'openarm_{s}_joint{n+1}':frames[i]['q'][j+n]
                for s,j in [('left',0),('right',7)] for n in range(7)})
            side=row['group'].split('-')[1];row['T_base_wrist']=kin.pose(f'openarm_{side}_link7',q).tolist()
            row['joint_frame']=i
        with tempfile.TemporaryDirectory(prefix='carry-clock-',dir=out.parent) as temporary:
            tmp=Path(temporary);(tmp/'images').mkdir()
            for row in source['rows']:os.link(out/row['image'],tmp/row['image'])
            write(tmp/'data.json',data);shutil.copyfile(out/'points.json',tmp/'points.json')
            fit(tmp)
            selected=json.loads((tmp/'selected.json').read_text());protocol=json.loads((tmp/'protocol.json').read_text())
            rank=selected['fit_rmse_px'];candidates.append({'offset_records':offset,
                'fit_rmse_px':rank,'at_bounds':selected['at_bounds'],'condition':selected['jacobian_condition']})
            if best is None or rank<best['selected']['fit_rmse_px']:
                best={'selected':selected,'protocol':protocol,'data':data,'offset_records':offset}
    # Freeze the fit-only winner on disk before any hypothesis sees validation.
    write(out/'timing-selected.json',{'offset_records':best['offset_records'],
        'selected':best['selected'],'candidates':candidates,'selection_uses_validation':False,
        'all_validation_was_previously_inspected':True,'physical_trial_passed':False})
    write(out/'timing-protocol.json',best['protocol'])
    with tempfile.TemporaryDirectory(prefix='carry-clock-check-',dir=out.parent) as temporary:
        tmp=Path(temporary);(tmp/'images').mkdir()
        for row in source['rows']:os.link(out/row['image'],tmp/row['image'])
        write(tmp/'data.json',best['data']);write(tmp/'selected.json',best['selected']);write(tmp/'protocol.json',best['protocol'])
        shutil.copyfile(out/'points.json',tmp/'points.json');evaluate(tmp)
        result=json.loads((tmp/'result.json').read_text());result.update(offset_records=best['offset_records'],
            qualification='development timing diagnostic; no independent latency measurement',
            alignment_data=best['data']['alignment_hypothesis'])
        write(out/'timing-result.json',result)
    print(json.dumps({'clock_candidates':candidates,'selected_offset':best['offset_records'],
        'validation':result['metrics'],'failed_groups':result['failed_registration_groups']},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['collect','fit','evaluate','timing'])
    p.add_argument('--output',type=Path,default=ROOT/'runs/20261025-offline/carry')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.is_symlink():raise ValueError('local runs only')
    out.mkdir(exist_ok=True)
    if a.phase=='collect':collect(out)
    elif a.phase=='fit':fit(out)
    elif a.phase=='evaluate':evaluate(out)
    else:timing(out)


if __name__=='__main__':main()
