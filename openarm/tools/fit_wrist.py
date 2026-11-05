"""Offline wrist-camera silhouette experiment; never applies runtime settings.

prepare/verify use the project's MuJoCo 3.14 environment. fit uses an existing
SciPy/OpenCV environment and the exported EE-local mesh only, not its MuJoCo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('left', 'right')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def folder(value):
    path = Path(value).resolve()
    if path.parent != ROOT/'runs' or not (path/'protocol.json').is_file():
        raise ValueError('use an existing study directly inside local runs')
    return path


def geom_names(side, version='v2'):
    if version == 'v1':
        return [f'openarm_{side}_{direction}_finger_visual_{index}' for direction in ('left','right') for index in (0,1)]
    return [f'finger_{part}_{side}_{index:02d}' for part in ('inner','outer') for index in (0,1)]


def camera_parameters(x):
    """Camera-to-EE rotation: Rz(-90 degrees) followed by a local rotvec."""
    v=np.asarray(x[3:6]);angle=np.linalg.norm(v)
    if angle<1e-12:
        rotation=np.eye(3)
    else:
        a=v/angle
        cross=np.array([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
        rotation=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    base=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    return np.asarray(x[:3]),base@rotation,float(x[6])


def prepare(path):
    import mujoco
    from sim.alignment import load_samples, recorded_pose
    from sim.environment import OpenArmEnv
    if (path/'inputs.npz').exists():
        raise ValueError('prepared evidence is immutable; use a new study')
    data = load_samples()
    protocol = json.loads((path/'protocol.json').read_text())
    manifest = {'schema':'openarm-wrist-projection/1', 'run_id':path.name,
        'status':'prepared', 'protocol':protocol, 'mujoco':mujoco.__version__,
        'hashes':{name:sha(ROOT/name) for name in ('sim/profile.json','calibrate.json',
            'task/bottle_to_box.xml','sim/environment.py','sim/gripper.py','sim/alignment.py',
            'reference/frames/manifest.json','tools/fit_wrist.py')},
        'protocol_sha256':sha(path/'protocol.json'), 'applied':False, 'transfer_ready':False,
        'samples':[], 'cameras':{}}
    env = OpenArmEnv(images=False)
    manifest['robot_version'] = env.robot_version
    for name in ('task/cell-v1.xml', f'openarm_mujoco/{env.robot_version}/openarm_bimanual.xml'):
        manifest['hashes'][name] = sha(ROOT/name)
    arrays = {}
    try:
        env.reset(seed=0)
        selected_pairs = {tuple(pair) for group in ('fit','development_validation','review_only') for pair in protocol[group]}
        available_pairs = {(row['episode'],row['frame']) for row in data['samples']}
        if not selected_pairs <= available_pairs:
            raise ValueError('fixed wrist protocol is missing required reference frames')
        for sample in data['samples']:
            pair = [sample['episode'], sample['frame']]
            if tuple(pair) not in selected_pairs:
                continue
            split = ('fit' if pair in protocol['fit'] else 'development_validation'
                     if pair in protocol['development_validation'] else 'review_only')
            recorded_pose(env,sample['state'])
            key = f"ep{sample['episode']}-{sample['frame']:06d}"
            row = {'key':key, 'episode':sample['episode'], 'frame':sample['frame'],
                   'split':split, 'state':sample['state'], 'images':sample['images']}
            manifest['samples'].append(row)
            for side in SIDES:
                ee = env.ee_ids[side]
                for i,name in enumerate(geom_names(side, env.robot_version)):
                    geom = env.model.geom(name)
                    mesh = int(geom.dataid[0]); start = env.model.mesh_vertadr[mesh]
                    vertices = env.model.mesh_vert[start:start+env.model.mesh_vertnum[mesh]]
                    world = vertices@env.data.geom_xmat[geom.id].reshape(3,3).T+env.data.geom_xpos[geom.id]
                    arrays[f'{key}-{side}-{i}'] = (world-env.data.xpos[ee])@env.data.xmat[ee].reshape(3,3)
        for side in SIDES:
            cid = env.model.camera(env.profile['camera_map'][side]).id
            ee = env.ee_ids[side]
            if env.model.cam_bodyid[cid] != ee:
                raise ValueError('camera must be attached to the exported EE frame')
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix,env.model.cam_quat[cid])
            manifest['cameras'][side] = {'position':env.model.cam_pos[cid].tolist(),
                'rotation':matrix.reshape(3,3).tolist(), 'quaternion_wxyz':env.model.cam_quat[cid].tolist(),
                'fovy':float(env.model.cam_fovy[cid]), 'size':env.profile['camera_sizes'][side],
                'frame':env.model.body(ee).name, 'convention':'OpenGL camera: -Z forward, +Y up'}
        manifest['upstream'] = env.manifest()['source']
        np.savez_compressed(path/'inputs.npz',**arrays)
        manifest['inputs_sha256'] = sha(path/'inputs.npz')
        (path/'source.py').write_bytes(Path(__file__).read_bytes())
        write(path/'manifest.json',manifest)
    finally:
        env.close()
    print(json.dumps({'status':'prepared','samples':len(manifest['samples']),'run':str(path)}),flush=True)


def real_mask(image):
    """Declared image-only rule; not conditioned on candidate camera geometry."""
    import cv2
    dark = (cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)<75).astype(np.uint8)
    dark[:200] = 0
    mask = np.zeros_like(dark)
    components = []
    for x0,x1 in ((0,320),(320,640)):
        n,labels,stats,_ = cv2.connectedComponentsWithStats(dark[:,x0:x1],8)
        ids = [i for i in range(1,n) if (labels[-1]==i).any()]
        if not ids:
            raise ValueError('no bottom-connected finger candidate in image half')
        selected = max(ids,key=lambda i:stats[i,cv2.CC_STAT_AREA])
        part = labels==selected
        area = int(part.sum())
        y = int(np.nonzero(part)[0].min())
        if area<2000 or area>50000 or y<=201:
            raise ValueError('ambiguous or truncated finger mask')
        mask[:,x0:x1] = part
        components.append({'area':area,'top_y':y})
    return mask.astype(bool),components


def validate_inputs(path,manifest):
    if sha(path/'inputs.npz') != manifest['inputs_sha256'] or sha(path/'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('study evidence changed')
    for relative,digest in manifest['hashes'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('study source changed: '+relative)
    for row in manifest['samples']:
        for side in SIDES:
            item=row['images'][side]
            if sha(ROOT/'reference/frames'/item['path']) != item['sha256']:
                raise ValueError('raw image changed')


def fit(path):
    import cv2
    import scipy
    from scipy.optimize import differential_evolution, minimize
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from PIL import Image
    manifest=json.loads((path/'manifest.json').read_text())
    validate_inputs(path,manifest)
    if (path/'proposal.json').exists():
        raise ValueError('proposal is immutable; use a new study')
    inputs=np.load(path/'inputs.npz',allow_pickle=False)
    rows=[r for r in manifest['samples'] if r['split']=='fit']
    if len(rows)!=1:
        raise ValueError('v1 fits exactly one recorded aperture per wrist')
    row=rows[0]
    result={'schema':'openarm-wrist-candidate/1','selection':'fit frame only; validation not read by optimizer',
            'versions':{'scipy':scipy.__version__,'opencv':cv2.__version__,'numpy':np.__version__},
            'cameras':{},'applied':False,'transfer_ready':False}
    for side in SIDES:
        started=time.monotonic()
        raw=np.asarray(Image.open(ROOT/'reference/frames'/row['images'][side]['path']).convert('RGB'))
        target,components=real_mask(raw)
        Image.fromarray(target.astype(np.uint8)*255).save(path/'images'/f'{side}-fit-mask.png')
        target_small=cv2.resize(target.astype(np.uint8),(160,120),interpolation=cv2.INTER_NEAREST).astype(bool)
        distance=cv2.distanceTransform((~target_small).astype(np.uint8),cv2.DIST_L2,3)
        meshes=[]
        for i in range(4):
            vertices=inputs[f"{row['key']}-{side}-{i}"]
            h=ConvexHull(vertices)
            # Keep edge intersections for correct clipping at the near plane.
            edges=np.unique(np.sort(np.concatenate([h.simplices[:,[0,1]],h.simplices[:,[1,2]],h.simplices[:,[0,2]]]),axis=1),axis=0)
            meshes.append((vertices,h.vertices,edges))
        calls=0
        def project(x):
            position,rotation,fovy=camera_parameters(x)
            focal=60/np.tan(np.deg2rad(fovy)/2)
            mask=np.zeros((120,160),np.uint8)
            for vertices,hull,edges in meshes:
                cam=(vertices-position)@rotation
                valid=cam[:,2]<-.001
                points=cam[hull[valid[hull]]]
                a,b=cam[edges[:,0]],cam[edges[:,1]]
                cross=(a[:,2]<-.001)!=(b[:,2]<-.001)
                if cross.any():
                    a,b=a[cross],b[cross]
                    points=np.vstack([points,a+(b-a)*((-.001-a[:,2])/(b[:,2]-a[:,2]))[:,None]])
                if len(points)<3:
                    continue
                pixel=np.column_stack([80+focal*points[:,0]/-points[:,2],60-focal*points[:,1]/-points[:,2]])
                poly=cv2.convexHull(np.clip(pixel,-20000,20000).astype(np.float32))
                cv2.fillConvexPoly(mask,np.rint(poly).astype(np.int32),1)
            return mask.astype(bool)
        def objective(x):
            nonlocal calls
            calls+=1
            if time.monotonic()-started>180:
                raise TimeoutError('per-wrist 180 s fit budget exceeded')
            mask=project(x)
            union=np.count_nonzero(mask|target_small)
            iou=np.count_nonzero(mask&target_small)/max(1,union)
            outside=float(distance[mask].mean()) if mask.any() else 160.
            # IoU prevents collapse; distance guides initially disjoint silhouettes.
            return 1-iou+.008*outside
        bounds=[(-.025,.060),(-.040,.040),(-.130,-.020)]+[(-.8,.8)]*3+[(40.,100.)]
        de=differential_evolution(objective,bounds,seed=manifest['protocol']['seed'],
            maxiter=65,popsize=12,tol=.001,polish=False,workers=1)
        local=minimize(objective,de.x,method='Powell',bounds=bounds,
            options={'maxiter':30,'maxfev':2500,'xtol':1e-5,'ftol':1e-5})
        selected=local if local.fun<de.fun else de
        position,rotation,fovy=camera_parameters(selected.x)
        result['cameras'][side]={'position':position.tolist(),'rotation':rotation.tolist(),'fovy':fovy,
            'parameters':selected.x.tolist(),'bounds':bounds,'fit_loss':float(selected.fun),
            'fit_calls':calls,'fit_seconds':time.monotonic()-started,'mask_components':components,
            'fit_mask_sha256':sha(path/'images'/f'{side}-fit-mask.png'),
            'at_bounds':[i for i,(v,(lo,hi)) in enumerate(zip(selected.x,bounds)) if min(v-lo,hi-v)<.01*(hi-lo)]}
        print(json.dumps({'side':side,**result['cameras'][side]},ensure_ascii=False),flush=True)
    write(path/'proposal.json',result)
    print(json.dumps({'status':'proposal','run':str(path)}),flush=True)


def refine(path):
    """Use actual native visibility/near clipping to select on FIT data only."""
    import mujoco
    from PIL import Image
    from sim.alignment import recorded_pose
    from sim.environment import OpenArmEnv
    manifest=json.loads((path/'manifest.json').read_text())
    validate_inputs(path,manifest)
    proposal=json.loads((path/'proposal.json').read_text())
    if (path/'selected.json').exists():
        raise ValueError('selected parameters already frozen')
    rows=[r for r in manifest['samples'] if r['split']=='fit']
    if len(rows)!=1:
        raise ValueError('v2 fits exactly one aperture per wrist')
    env=OpenArmEnv(images=False);renderer=None
    selected={'schema':'openarm-wrist-candidate/2','selection':'native segmentation on fit frame only',
              'proposal_sha256':sha(path/'proposal.json'),'cameras':{},'applied':False,'transfer_ready':False}
    try:
        env.reset(seed=0);recorded_pose(env,rows[0]['state'])
        for body in (env.bottle_id,env.box_id):
            env.model.geom_group[env.model.geom_bodyid==body]=5
        renderer=mujoco.Renderer(env.model,height=120,width=160)
        renderer.enable_segmentation_rendering()
        for side in SIDES:
            start=time.monotonic();calls=0
            prior=proposal['cameras'][side]
            mask_path=path/'images'/f'{side}-fit-mask.png'
            if sha(mask_path)!=prior['fit_mask_sha256']:
                raise ValueError('fit mask changed')
            target=np.asarray(Image.open(mask_path).resize((160,120),Image.Resampling.NEAREST))>0
            cid=env.model.camera(env.profile['camera_map'][side]).id
            ids=[env.model.geom(name).id for name in geom_names(side, env.robot_version)]
            def loss(x):
                nonlocal calls
                calls+=1
                if time.monotonic()-start>90:
                    raise TimeoutError('native fit exceeded 90 seconds per wrist')
                pos,rot,fovy=camera_parameters(x)
                env.model.cam_pos[cid]=pos;env.model.cam_fovy[cid]=fovy
                quat=np.empty(4);mujoco.mju_mat2Quat(quat,rot.ravel());env.model.cam_quat[cid]=quat
                mujoco.mj_forward(env.model,env.data)
                renderer.update_scene(env.data,camera=cid,scene_option=env._option)
                seg=renderer.render()
                mask=np.isin(seg[:,:,0],ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                return 1-np.count_nonzero(mask&target)/max(1,np.count_nonzero(mask|target))
            bounds=np.asarray(prior['bounds']);lo,hi=bounds.T
            rng=np.random.default_rng(manifest['protocol']['seed'])
            best=np.asarray(prior['parameters']);best_loss=initial_loss=loss(best)
            mean=best.copy();sigma=np.array([.012,.010,.015,.14,.14,.10,8.])
            floor=np.array([.0004,.0004,.0004,.005,.005,.005,.25])
            trace=[]
            for iteration in range(18):
                candidates=np.clip(rng.normal(mean,sigma,(64,7)),lo,hi)
                candidates[0]=best
                scores=np.array([loss(x) for x in candidates])
                elite=np.argsort(scores)[:10]
                if scores[elite[0]]<best_loss:
                    best=candidates[elite[0]].copy();best_loss=float(scores[elite[0]])
                mean=.35*mean+.65*candidates[elite].mean(axis=0)
                sigma=np.maximum(floor,.3*sigma+.7*candidates[elite].std(axis=0))
                trace.append({'iteration':iteration,'best_native_loss':best_loss})
            # Finish with a deterministic local coordinate search on native pixels.
            for scale in (1.,.5,.25):
                step=np.maximum(floor,sigma)*scale
                for _ in range(4):
                    changed=False
                    for axis in range(7):
                        for direction in (-1,1):
                            x=best.copy();x[axis]=np.clip(x[axis]+direction*step[axis],lo[axis],hi[axis])
                            score=loss(x)
                            if score<best_loss:
                                best,best_loss=x,score;changed=True
                    if not changed:break
            pos,rot,fovy=camera_parameters(best)
            selected['cameras'][side]={'position':pos.tolist(),'rotation':rot.tolist(),'fovy':fovy,
                'parameters':best.tolist(),'bounds':bounds.tolist(),'fit_loss':best_loss,
                'proposal_native_loss':initial_loss,'native_fit_calls':calls,'fit_seconds':time.monotonic()-start,
                'at_bounds':[i for i,(v,(low,high)) in enumerate(zip(best,bounds)) if min(v-low,high-v)<.01*(high-low)],
                'native_znear_m':float(env.model.vis.map.znear*env.model.stat.extent),'trace':trace}
            print(json.dumps({'side':side,'native_fit_iou':1-best_loss,'proposal_native_iou':1-initial_loss,'seconds':time.monotonic()-start}),flush=True)
        # The next phase may read validation masks only after this write.
        write(path/'selected.json',selected)
    finally:
        if renderer:renderer.close()
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','fit','refine','masks','verify'))
    parser.add_argument('study',type=folder)
    args=parser.parse_args()
    if args.phase=='verify':
        verify(args.study)
    else:
        {'prepare':prepare,'fit':fit,'refine':refine,'masks':make_masks}[args.phase](args.study)


def make_masks(path):
    from PIL import Image
    manifest=json.loads((path/'manifest.json').read_text())
    validate_inputs(path,manifest)
    if not (path/'selected.json').exists() or (path/'masks.json').exists():
        raise ValueError('masks require a locked candidate and no previous mask report')
    report={'selected_sha256':sha(path/'selected.json'),'images':{},'excluded':{}}
    for row in manifest['samples']:
        if row['split']=='review_only':
            continue
        for side in SIDES:
            raw=np.asarray(Image.open(ROOT/'reference/frames'/row['images'][side]['path']).convert('RGB'))
            png=path/'images'/f"{row['key']}-{side}-real-mask.png"
            try:
                mask,components=real_mask(raw)
            except ValueError as exc:
                if row['split']=='fit':
                    raise
                report['excluded'][png.name]=str(exc)
                continue
            if png.exists():
                raise ValueError('mask already exists')
            Image.fromarray(mask.astype(np.uint8)*255).save(png)
            report['images'][png.name]={'sha256':sha(png),'components':components,'split':row['split']}
    write(path/'masks.json',report)
    print(json.dumps({'status':'masks_frozen','samples':len(report['images']),'excluded':report['excluded']}),flush=True)


def verify(path):
    import mujoco
    from PIL import Image,ImageDraw
    from sim.alignment import recorded_pose
    from sim.environment import OpenArmEnv
    manifest=json.loads((path/'manifest.json').read_text())
    selected=json.loads((path/'selected.json').read_text())
    validate_inputs(path,manifest)
    selected_hash=sha(path/'selected.json')
    frozen_masks=json.loads((path/'masks.json').read_text())
    if frozen_masks['selected_sha256']!=selected_hash:
        raise ValueError('masks were not generated after this selection')
    if (path/'wrist.json').exists():
        raise ValueError('verification already exists; use a new study')
    env=OpenArmEnv(images=False)
    report={'schema':'openarm-wrist-verification/1','run_id':path.name,'status':'running',
        'selected_sha256':selected_hash,'samples':[],'applied':False,'calibrated':False,'transfer_ready':False,
        'limitations':['Only one aperture per wrist was fitted; camera depth, focal length and finger geometry are confounded.',
            'Gripper encoder-to-opening mapping is a fixed uncalibrated assumption.',
            'Centered pinhole intrinsics, zero distortion; no independent camera or hand-eye measurement.',
            'Held-out episodes are development validation, previously inspected, not a new final test set.',
            'Silhouette scores concern visible dark finger parts, not physical TCP accuracy or task success.',
            'Review-only frames can have object occlusion and are not scored.']}
    renderer=None
    try:
        env.reset(seed=0)
        for body in (env.bottle_id,env.box_id):
            env.model.geom_group[env.model.geom_bodyid==body]=5
        renderer=mujoco.Renderer(env.model,height=480,width=640)
        for row in manifest['samples']:
            recorded_pose(env,row['state'])
            for side in SIDES:
                cid=env.model.camera(env.profile['camera_map'][side]).id
                ids=[env.model.geom(name).id for name in geom_names(side, env.robot_version)]
                raw=np.asarray(Image.open(ROOT/'reference/frames'/row['images'][side]['path']).convert('RGB'))
                record={k:row[k] for k in ('key','episode','frame','split')}
                record.update(side=side,images={},metrics={},score_exclusion=None)
                target=None
                # Fit masks are frozen. Development masks are generated by a
                # separate numerical phase before scoring, never edited here.
                mask_path=path/'images'/f"{row['key']}-{side}-real-mask.png"
                if row['split']!='review_only':
                    if mask_path.name in frozen_masks['excluded']:
                        record['score_exclusion']=frozen_masks['excluded'][mask_path.name]
                    elif not mask_path.is_file():
                        raise ValueError('run mask verification in numerical environment first')
                    else:
                        if sha(mask_path)!=frozen_masks['images'][mask_path.name]['sha256']:
                            raise ValueError('frozen mask changed')
                        target=np.asarray(Image.open(mask_path))>0
                masks={}
                for label,params in (('baseline',manifest['cameras'][side]),('candidate',selected['cameras'][side])):
                    env.model.cam_pos[cid]=params['position']
                    env.model.cam_fovy[cid]=params['fovy']
                    quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.asarray(params['rotation']).ravel())
                    env.model.cam_quat[cid]=quat
                    mujoco.mj_forward(env.model,env.data)
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(env.data,camera=cid,scene_option=env._option)
                    rgb=renderer.render().copy()
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(env.data,camera=cid,scene_option=env._option)
                    seg=renderer.render().copy()
                    mask=np.isin(seg[:,:,0],ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                    masks[label]=mask
                    png=path/'images'/f"{row['key']}-{side}-{label}.png"
                    Image.fromarray(rgb).save(png)
                    record['images'][label]={'path':str(png.relative_to(path)),'sha256':sha(png)}
                    if target is not None:
                        union=np.count_nonzero(mask|target)
                        record['metrics'][label]={'iou':float(np.count_nonzero(mask&target)/max(1,union)),
                            'target_pixels':int(target.sum()),'predicted_pixels':int(mask.sum())}
                # Overlay native segmentation, not an optimizer's hull proxy.
                overlay=raw.copy()
                for label,color in (('baseline',np.array([220,100,55])),('candidate',np.array([30,180,135]))):
                    mask=masks[label]
                    edge=mask.copy()
                    edge[1:-1,1:-1]&=~(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
                    overlay[edge]=color
                png=path/'images'/f"{row['key']}-{side}-overlay.png"
                Image.fromarray(overlay).save(png)
                record['images']['overlay']={'path':str(png.relative_to(path)),'sha256':sha(png)}
                report['samples'].append(record)
        if sha(path/'selected.json')!=selected_hash:
            raise ValueError('selected parameters changed during validation')
        for side in SIDES:
            print(json.dumps({'side':side,'scores':[{'frame':r['key'],'split':r['split'],**r['metrics']} for r in report['samples'] if r['side']==side and r['metrics']]}),flush=True)
        report['status']='completed'
        write(path/'wrist.json',report)
        write(path/'summary.json',{'status':'completed','scope':'development wrist projection; no task score',
            'samples':len(report['samples']),'applied':False,'calibrated':False,'transfer_ready':False,
            'report_url':'/runs/'+path.name+'/index.html'})
        # A single browsable report keeps the raw data and candidates reviewable
        # without applying settings or adding another application.
        cards=[]
        for row in report['samples']:
            score=row['metrics']
            desc=(f"轮廓重合度 IoU {score['baseline']['iou']:.3f} → {score['candidate']['iou']:.3f}"
                  if score else '仅画面复查：未计分'+('，遮挡或分割歧义' if row['score_exclusion'] else ''))
            cards.append(f'<article><h2>{row["side"]} · {row["key"]} · {row["split"]}</h2><p>{desc}</p>'+
                '<div class="images">'+''.join(f'<figure><a href="{row["images"][k]["path"]}" target="_blank"><img src="{row["images"][k]["path"]}" alt="{k}"></a><figcaption>{label}</figcaption></figure>'
                for k,label in (('overlay','原始图叠加：橙色当前，绿色候选'),('baseline','当前模型'),('candidate','候选投影')))+'</div></article>')
        html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>腕部相机对齐</title><style>
        *{box-sizing:border-box}body{margin:0;background:#f6f5f1;color:#26332e;font:15px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:28px;margin:0}h2{font-size:17px;font-weight:600}p{color:#5d6963}article{background:white;padding:18px;border:1px solid #dfdfd7;border-radius:14px;margin:20px 0}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}img{display:block;width:100%;border-radius:8px}figcaption{font-size:13px;margin-top:8px;color:#5d6963}a{color:#27634c}@media(max-width:800px){.images{grid-template-columns:1fr}main{padding:16px}}
        </style><main><h1>腕部相机 · 候选投影</h1><p>固定夹爪映射，仅拟合相机位姿与视场角。候选未应用；轮廓相似不等于物理标定通过。</p><p>分组由 protocol.json 预先指定；有遮挡或分割歧义的验证帧明确排除。其余画面仅复查。点击画面查看原尺寸。<a href="wrist.json">完整指标</a> · <a href="selected.json">候选参数</a></p>'''+''.join(cards)+'</main></html>'
        (path/'index.html').write_text(html)
        manifest['status']='completed';write(path/'manifest.json',manifest)
    finally:
        if renderer:renderer.close()
        env.close()


if __name__=='__main__':
    main()
