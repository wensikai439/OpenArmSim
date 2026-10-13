"""Build CAD-contained finger proposal spheres; native mesh checks stay required."""
import hashlib
import json
import argparse
import re

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from sim.environment import ROOT, OpenArmEnv

OUT=ROOT/'runs/20261008-carry/geometry'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',default='geometry');args=parser.parse_args()
    if not re.fullmatch('[a-z0-9-]{1,40}',args.group):raise ValueError('simple geometry study name required')
    output=ROOT/'runs/20261008-carry'/args.group
    if output.exists():raise ValueError('preserve earlier geometry study')
    output.mkdir()
    protocol={'grid_m':.008,'minimum_radius_m':.0005,'sphere_containment_margin_m':1e-7,
              'algorithm':'maximal inscribed balls at a fixed grid of convex CAD hull interior points; greedy interior coverage',
              'scope':'GPU proposal representation only, not a conservative surface enclosure; mandatory unchanged native mesh recheck',
              'changes':'four finger representations only; no collision-ignore changes, native geometry or physics changes',
              'hardware_access':False,'transfer_ready':False}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    config=json.loads((ROOT/'task/planners/openarm.json').read_text());kin=config['robot_cfg']['kinematics']
    env=OpenArmEnv(images=False);results=[]
    try:
        model=env.model
        for side in ('left','right'):
            for finger in ('right','left'):
                name=f'openarm_{side}_{finger}_finger';body=model.body(name).id;points=[]
                for g in np.flatnonzero(model.geom_bodyid==body):
                    if not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
                    if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('mesh finger required')
                    mesh=model.geom_dataid[g];verts=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.geom_quat[g])
                    points.extend(verts@rotation.reshape(3,3).T+model.geom_pos[g])
                points=np.asarray(points);hull=ConvexHull(points)
                normals=hull.equations[:,:3];offset=hull.equations[:,3]
                lo,hi=points.min(0),points.max(0)
                axes=[np.arange(a+.004,b,.008) for a,b in zip(lo,hi)]
                centers=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
                radii=-(centers@normals.T+offset).max(axis=1)-protocol['sphere_containment_margin_m']
                valid=radii>=protocol['minimum_radius_m'];centers,radii=centers[valid],radii[valid]
                if not len(centers):raise ValueError('no CAD-contained spheres; do not omit a finger')
                selected=[];remaining=np.ones(len(centers),dtype=bool)
                while remaining.any():
                    index=int(np.argmax(np.where(remaining,radii,-1)))
                    selected.append({'center':centers[index].tolist(),'radius':float(radii[index])})
                    remaining &= np.linalg.norm(centers-centers[index],axis=1)>radii[index]
                fitted=np.array([s['center'] for s in selected]);radius=np.array([s['radius'] for s in selected])
                contained=float((fitted@normals.T+offset+radius[:,None]).max())
                old=kin['collision_spheres'][name]
                old_bad=int(sum(np.max(np.array(s['center'])@normals.T+offset+s['radius'])>1e-7 for s in old))
                # Independent deterministic face samples measure uncovered surface;
                # never claim interior sphere fitting covers the complete mesh.
                samples=[]
                for tri in points[hull.simplices]:
                    for u in np.linspace(0,1,9):
                        for v in np.linspace(0,1-u,9):samples.append((1-u-v)*tri[0]+u*tri[1]+v*tri[2])
                samples=np.array(samples)
                distance=(np.linalg.norm(samples[:,None]-fitted[None],axis=-1)-radius[None]).min(axis=1)
                result={'link':name,'old_count':len(old),'old_spheres_extending_outside_CAD':old_bad,
                        'new_count':len(selected),'max_containment_violation_m':contained,
                        'surface_distance_p95_m':float(np.percentile(distance,95)),
                        'max_sampled_surface_distance_m':float(distance.max()),'native_recheck_required':True}
                assert contained<=1e-7
                kin['collision_spheres'][name]=selected;results.append(result)
        (output/'openarm.json').write_text(json.dumps(config,indent=2)+'\n')
        (output/'summary.json').write_text(json.dumps({'protocol':protocol,'results':results,
            'old_config_sha256':hashlib.sha256((ROOT/'task/planners/openarm.json').read_bytes()).hexdigest()},indent=2)+'\n')
        print(json.dumps(results),flush=True)
    finally:env.close()


if __name__=='__main__':main()
