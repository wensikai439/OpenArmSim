"""Bounded pose IK proposals in scratch MjData; native collision checks follow.

Uses trust-region reflective least squares instead of clipping an unconstrained
Jacobian step. This is a numerical proposal solver, not a robot controller or
a proof of reachability, grasp, physical calibration or transfer.
"""
import time
import mujoco
import numpy as np
import scipy
from scipy.optimize import least_squares

CONTRACT={'schema':'bounded-pose-ik/1','algorithm':'SciPy trust-region reflective least squares',
    'scipy_version':scipy.__version__,'orientation_weight_m':.15,'max_nfev_per_seed':220,
    'kinematics':'MuJoCo mj_kinematics only, scratch MjData',
    'acceptance':'caller original position/orientation tolerance plus full native collision guard',
    'scope':'finite seeded proposal search, exhaustion is not proof of geometric impossibility',
    'hardware_access':False,'transfer_ready':False,
    'source':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html'}


def pose(model,data,body,offset):
    mujoco.mj_kinematics(model,data)
    rotation=data.xmat[body].reshape(3,3)
    return data.xpos[body]+rotation@offset,rotation


def solve(model,data,qids,body,offset,bounds,target,rotation,seeds,*,check=lambda:None,
          position_tolerance=.0015,orientation_tolerance=.018,valid=lambda q:True):
    bounds=np.asarray(bounds);target=np.asarray(target);offset=np.asarray(offset)
    best=None;trials=[];started=time.monotonic()
    def residual(q):
        check();data.qpos[qids]=q
        point,actual=pose(model,data,body,offset)
        angular=np.zeros(3)
        if rotation is not None:
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,(actual@rotation.T).ravel())
            mujoco.mju_quat2Vel(angular,quat,1.)
        return np.r_[point-target,.15*angular]
    for index,seed in enumerate(seeds):
        check()
        initial=np.clip(seed,bounds[:,0]+1e-10,bounds[:,1]-1e-10)
        result=least_squares(residual,initial,bounds=(bounds[:,0],bounds[:,1]),method='trf',
            max_nfev=220,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        values=residual(result.x);p=float(np.linalg.norm(values[:3]));r=float(np.linalg.norm(values[3:])/.15)
        item={'seed_index':index,'position_error_m':p,'rotation_error_rad':r,'nfev':int(result.nfev),
              'optimality':float(result.optimality),'optimizer_status':int(result.status),
              'bound_active':result.active_mask.tolist(),'native_collision_free':None}
        if best is None or p+.15*r<best['cost']:
            best={'cost':p+.15*r,'position_error_m':p,'rotation_error_rad':r,'q':result.x.tolist()}
        feasible=p<position_tolerance and r<orientation_tolerance
        if feasible:item['native_collision_free']=bool(valid(result.x))
        trials.append(item)
        if feasible and item['native_collision_free']:
            return result.x.copy(),{'contract':CONTRACT,'status':'proposal_verified','trials':trials,
                'best':best,'wall_seconds':time.monotonic()-started}
    return None,{'contract':CONTRACT,'status':'seed_search_exhausted','trials':trials,
        'best':best,'wall_seconds':time.monotonic()-started}


def seeds(initial,reference,bounds,rng,count=12):
    proposed=[np.array(initial,copy=True)]+[np.clip(q,bounds[:,0],bounds[:,1]) for q in reference]
    while len(proposed)<count:
        proposed.append(np.clip(initial+rng.normal(0,.65,len(initial)),bounds[:,0],bounds[:,1]))
    return proposed[:count]
