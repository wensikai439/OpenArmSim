"""RGB-estimated coarse reach, precise local correction, and audited relay.

The v3 controller remains unchanged. No legacy task macro, object truth, camera
matrix, physical parameter, native score, or control clock is changed here.
"""
import argparse
import json
from pathlib import Path
import uuid

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .runtime import ROOT, dump
from .visual_geometry import GeometryRig
from .visual_relay import RelayWorld


class CachedRig(GeometryRig):
    def __init__(self,*args,**kwargs):
        self.cache={};self.cache_tick=None
        super().__init__(*args,**kwargs)

    def image(self,view,*,dots=True):
        if self.cache_tick!=self.env.tick:
            self.cache.clear();self.cache_tick=self.env.tick
        key=(view,dots)
        if key not in self.cache:
            rgb=super().image(view,dots=dots)
            exposure=getattr(self,'exposure',1.)
            self.cache[key]=np.clip(rgb.astype(float)*exposure,0,255).astype(np.uint8) if exposure!=1. else rgb
        return self.cache[key]


class AdaptiveWorld(RelayWorld):
    PUBLIC=(*RelayWorld.PUBLIC,'calibrate_views')

    def __init__(self,case,output):
        super().__init__(case,output)
        # The base constructor starts an encoder. Development trials retain RGB
        # evidence and measured traces, and may skip only the overview video.
        if not case.get('record_video',True):
            self.video.stdin.close();self.video.wait(timeout=20);self.video.stderr.close();self.video=None
            (self.output/'rollout.mp4').unlink(missing_ok=True)  # This instance's empty encoder file only.
        self.rig.renderer.close()
        self.rig=CachedRig(self.env,self.output/'vision',yaw_offset=case.get('camera_yaw_deg',0),
            pixel_noise=case.get('pixel_noise',0),seed=case['seed'])
        for i,view in enumerate(self.rig.views):
            view['azimuth']+=case.get('camera_delta_deg',[0,0])[i]
            view['distance']*=case.get('camera_distance_scale',1.)
            view['lookat'][2]+=case.get('camera_lookat_z_m',0.)
        self.rig.exposure=case.get('exposure',1.)
        self.rig.lock_target();self.adaptive_trace=[];self.approach_checks=[]

    def approach_options(self,arm,target_m,yaws_deg,pitch_deg=-15,standoff_m=.07):
        self._guard();target=np.asarray(target_m,float)
        if target.shape!=(3,) or not np.isfinite(target).all() or not .03<=standoff_m<=.12 or len(yaws_deg)>8:
            raise ValueError('bounded target and approach candidates required')
        rejected=[]
        for yaw in yaws_deg:
            theta=np.deg2rad(yaw);up=np.array([0.,0.,1.]);direction=np.array([np.cos(theta),np.sin(theta),0.])
            rotation=np.column_stack([up,np.cross(direction,up),direction])@Rotation.from_euler('y',pitch_deg,degrees=True).as_matrix()
            pre=target-standoff_m*rotation[:,2]
            try:
                q=self._solve(arm,pre,rotation)
                # Endpoint feasibility does not imply a feasible insertion
                # corridor. Search the entire local approach before moving.
                for f in np.linspace(0,1,max(2,int(np.ceil(standoff_m/.006)))+1)[1:]:
                    self.planning_seed[arm]=q
                    q=self._solve(arm,pre+(target-pre)*f,rotation)
                result={'ok':True,'target_m':target.tolist(),'pre_m':pre.tolist(),'yaw_deg':yaw,
                    'pitch_deg':pitch_deg,'rejected':rejected,'corridor_checked':True}
                self.approach_checks.append(result);dump(self.output/'approach-checks.json',self.approach_checks)
                return result
            except ValueError as exc:rejected.append({'yaw_deg':yaw,'reason':str(exc)})
            finally:self.planning_seed.clear()
        result={'ok':False,'rejected':rejected,'corridor_checked':True}
        self.approach_checks.append(result);dump(self.output/'approach-checks.json',self.approach_checks)
        return result

    def calibrate_views(self):
        self._guard();self.phase='view_probe'
        origin,rotation=self.pose('right');origin=origin.copy()
        self.rig.read()
        for delta in ([0,.020,0],[.020,.020,0],[.020,.020,-.020],[0,0,0]):
            self._move('right',origin+delta,rotation);self.rig.read()
        self.rig.calibrate()
        result={'ok':True,'quality':self.rig.quality,'source':'RGB tool markers plus measured robot FK'}
        dump(self.output/'view-probe.json',result)
        return result

    def locate_pixels(self,pixels):
        self._guard()
        self.rig.read();self.rig.calibrate()
        position,error=self.rig.locate(pixels)
        result={'estimated_position_m':position.tolist(),'reprojection_error_px':error,
            'source':'fresh RGB tool marks and measured FK projection estimate; no object truth'}
        self.adaptive_trace.append({'tick':self.env.tick,'kind':'locate',**result})
        dump(self.output/'adaptive-trace.json',self.adaptive_trace)
        return result

    def _solve(self,arm,target,rotation):
        # Tighter continuation avoids waypoint residual noise dominating the
        # path's jerk estimate. Native arm gains and actual-arrival gates stay.
        env=self.env;data=self.scratch;ids=env.qids[arm];dofs=env.dids[arm];body=env.ee_ids[arm]
        data.qpos[:]=env.data.qpos;data.qvel[:]=0
        q0=self.planning_seed.get(arm,env.data.qpos[ids]).copy()
        off=0 if arm=='left' else 8;lo=env.action_space.low[off:off+7];hi=env.action_space.high[off:off+7]
        rng=np.random.default_rng(17);best=1e9;best_errors=None
        for attempt in range(10):
            data.qpos[ids]=q0 if not attempt else np.clip(q0+rng.normal(0,.65,7),lo,hi)
            for _ in range(240):
                mujoco.mj_forward(env.model,data);p,r=self.pose(arm,data)
                dp=target-p;dr=Rotation.from_matrix(rotation@r.T).as_rotvec()
                score=float(np.linalg.norm(dp)+.2*np.linalg.norm(dr))
                if score<best:best=score;best_errors=[float(np.linalg.norm(dp)),float(np.linalg.norm(dr))]
                if np.linalg.norm(dp)<.0002 and np.linalg.norm(dr)<.006:return data.qpos[ids].copy()
                jp=np.zeros((3,env.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(env.model,data,jp,jr,p,body)
                jac=np.vstack([jp[:,dofs],.2*jr[:,dofs]]);error=np.r_[dp,.2*dr]
                dq=jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(6),error)
                dq*=min(1.,.10/max(np.max(abs(dq)),1e-9));data.qpos[ids]=np.clip(data.qpos[ids]+dq,lo,hi)
        dump(self.output/'ik-failure.json',{'arm':arm,'phase':self.phase,'target_m':np.asarray(target).tolist(),
            'position_error_m':best_errors[0],'rotation_error_rad':best_errors[1]})
        raise ValueError('local IK did not converge; best weighted residual '+str(round(best,5)))

    def visual_align(self,max_steps=30,yaw_deg=0,pitch_deg=-15,target_pixels=None,offset_m=(0,0,0),step_m=.006,tolerance_px=4.):
        self._guard();self.phase='visual_alignment';offset=np.asarray(offset_m,float)
        if offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.13:
            raise ValueError('bounded visual offset required')
        if not .002<=step_m<=.025 or not 1<=tolerance_px<=15:raise ValueError('bounded visual parameters required')
        if target_pixels is not None:
            target=np.asarray(target_pixels,float)
            if target.shape!=(4,) or not np.isfinite(target).all():raise ValueError('two RGB points required')
            self.rig.goals=target.copy()
        if self.rig.cameras is None:raise ValueError('calibrate_views required before local alignment')
        self.rig.read();self.rig.calibrate()
        from .visual_geometry import project
        estimated,_=self.rig.locate(self.rig.goals)
        image_goal=self.rig.goals+np.concatenate([project(p,[estimated+offset])[0]-project(p,[estimated])[0] for p in self.rig.cameras])
        for step in range(min(int(max_steps),40)):
            f,row=self.rig.read();error=image_goal-f
            row.update(max_view_error_px=float(np.linalg.norm(error.reshape(2,2),axis=1).max()),phase=self.phase)
            self.last_visual=row;self.vision_trace.append(row)
            if row['max_view_error_px']<tolerance_px:return {'ok':True,'steps':step,'error_px':row['max_view_error_px']}
            p=self.pose('right')[0];rows=[]
            for camera in self.rig.cameras:
                q=camera@np.r_[p,1.]
                rows.extend([(camera[i,:3]*q[2]-q[i]*camera[2,:3])/q[2]**2 for i in range(2)])
            jac=np.array(rows);sv=np.linalg.svd(jac,compute_uv=False)
            if sv[-1]<8 or sv[0]/sv[-1]>40:raise ValueError('unobservable visual geometry')
            delta=np.linalg.solve(jac.T@jac+25*np.eye(3),jac.T@error)*.7
            delta*=min(1.,step_m/max(np.linalg.norm(delta),1e-9))
            row['requested_delta_m']=delta.tolist()
            self._move('right',p+delta,self.rotation_reference['right'])
        return {'ok':False,'error_code':'alignment_budget'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/('adaptive-'+uuid.uuid4().hex[:8]))
    p.add_argument('--program',type=Path,default=ROOT/'skills/bottle_visual/adaptive/program.py')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():p.error('new output under runs/ required')
    case=json.loads(a.case.read_text()) if a.case else {'seed':0,'bottle_xy_m':[.5,-.21]}
    world=None
    try:
        world=AdaptiveWorld(case,out);code=a.program.read_text();(out/'program.py').write_text(code)
        r=world.result(world.execute(code))
        print(json.dumps({'output':str(out),**{k:r[k] for k in ('program_ok','error','full_task_success','sim_seconds')}}),flush=True)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True)
        dump(out/'result.json',{'full_task_success':False,'program_ok':False,'error':type(exc).__name__+': '+str(exc),'hardware_access':False})
        print(json.dumps({'error':str(exc),'full_task_success':False}),flush=True)
    finally:
        if world is not None:world.close()


if __name__=='__main__':main()
