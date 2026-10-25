"""Smooth independent RGB/proprioceptive relay primitives; simulation only."""
import argparse
import json
from pathlib import Path
import uuid

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from .runtime import ROOT, dump
from .visual_control import RobotWorld


class RelayWorld(RobotWorld):
    PUBLIC = (*RobotWorld.PUBLIC, 'move_tcp', 'grip_evidence', 'label', 'locate_pixels', 'locate_basket','approach_options')

    def __init__(self, case, output):
        super().__init__(case, output)
        from .visual_geometry import GeometryRig
        self.rig.renderer.close()
        self.rig=GeometryRig(self.env,self.output/'vision',yaw_offset=case.get('camera_yaw_deg',0),pixel_noise=case.get('pixel_noise',0),seed=case['seed'])
        try:self.rig.lock_target()
        except Exception:
            self.close();raise
        self.lower_contacts = []; self.evidence = []; self.jacobians = {}
        self.stages = []; self.done = False
        self.planning_seed = {}
        self.closed_flags = {'left':False,'right':False}

    def locate_pixels(self,pixels):
        self._guard();position,error=self.rig.locate(pixels)
        return {'estimated_position_m':position.tolist(),'reprojection_error_px':error,
                'source':'RGB + projection estimated from robot motion; no simulator object pose'}

    def locate_basket(self):
        self._guard();return self.rig.basket()

    def approach_options(self,arm,target_m,yaws_deg,pitch_deg=-15,standoff_m=.07):
        self._guard();target=np.asarray(target_m,float)
        if target.shape!=(3,) or not np.isfinite(target).all() or not .03<=standoff_m<=.12 or len(yaws_deg)>8:
            raise ValueError('bounded target and approach candidates required')
        rejected=[]
        for yaw in yaws_deg:
            theta=np.deg2rad(yaw);up=np.array([0.,0.,1.]);a=np.array([np.cos(theta),np.sin(theta),0.])
            rotation=np.column_stack([up,np.cross(a,up),a])@Rotation.from_euler('y',pitch_deg,degrees=True).as_matrix()
            pre=target-standoff_m*rotation[:,2]
            try:
                self._solve(arm,target,rotation);self._solve(arm,pre,rotation)
                return {'ok':True,'target_m':target.tolist(),'pre_m':pre.tolist(),'yaw_deg':yaw,'pitch_deg':pitch_deg,'rejected':rejected}
            except ValueError as ex:rejected.append({'yaw_deg':yaw,'reason':str(ex)})
        return {'ok':False,'rejected':rejected}

    def label(self, name):
        self._guard()
        if not isinstance(name,str) or len(name)>60:raise ValueError('short stage label required')
        self.phase=name
        self.stages.append({'name':name,'tick':self.env.tick,'time_s':float(self.env.data.time)})
        for view in range(2):
            Image.fromarray(self.rig.image(view,dots=False)).save(self.output/'vision'/f'stage-{len(self.stages):02d}-{view}.jpg',quality=90)
        return {'ok':True}

    def _guard(self):
        self.calls += 1
        if self.calls>120 or self.env.evaluation['invalid'] or self.done:
            raise ValueError('episode terminal or tool budget exhausted')

    def _tick(self,q):
        if self.done:return
        env=self.env;measured=env.state().astype(float)
        for offset in (0,8):
            error=q[offset:offset+7]-measured[offset:offset+7]
            self.integral[offset:offset+7]=np.clip(self.integral[offset:offset+7]+error*.04,-.14,.14)
        action=q+self.integral
        for offset in (0,8):
            action[offset:offset+7]=np.clip(action[offset:offset+7],env.action_space.low[offset:offset+7],env.action_space.high[offset:offset+7])
        _,_,terminal,truncated,info=env.step_policy(action,native_fingers=True)
        # Native object/contact state below is independent grading/diagnostics.
        height=env._bottle_bounds()[0][2]-self.initial_bottom
        self.peak_lift=max(self.peak_lift,float(height))
        if env.evaluation['held']['right'] and height>.035:
            self.held_consecutive+=1;self.grasp_samples.append({'tick':env.tick,'height_m':float(height)})
            if not self.lower_contacts:self.lower_contacts=self.contact_fractions('right')
        else:self.held_consecutive=0
        if env.tick%2==0 and self.video:
            self.video.stdin.write(env.render('operator',view={'mode':'free','azimuth':180.,
                'elevation':-24.,'distance':1.2,'lookat':[.43,0.,1.19]}).tobytes())
        self.trace.append({'tick':env.tick,'time_s':float(env.data.time),'phase':self.phase,
            'requested':q.tolist(),'applied':action.tolist(),'measured':env.state().astype(float).tolist(),
            'tcp':{s:self.pose(s)[0].tolist() for s in ('left','right')},
            'evaluation_only':{'bottle':env.data.xpos[env.bottle_id].tolist(),
                 'bottle_axis':env.data.xmat[env.bottle_id].reshape(3,3)[:,2].tolist(),
                 'held':dict(env.evaluation['held']),'touching':dict(env.evaluation['touching']),
                 'invalid':env.evaluation['invalid']}})
        if terminal or truncated:
            if env.evaluation['success']:self.done=True
            else:raise ValueError('native termination: '+str(info['evaluation'].get('failure_reasons')))

    def contact_fractions(self,side):
        env=self.env;body=env.model.geom('bottle_body').id;fractions=[]
        for i in range(env.data.ncon):
            c=env.data.contact[i]
            if body not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==body else c.geom1
            name=env.model.body(env.model.geom_bodyid[other]).name
            if 'openarm_'+side in name and 'finger' in name:
                axis=env.data.geom_xmat[body].reshape(3,3)[:,2]
                fractions.append(float(.5+np.dot(c.pos-env.data.geom_xpos[body],axis)/(2*env.model.geom_size[body,1])))
        return fractions

    def _move(self,arm,target,rotation):
        target=np.asarray(target);rotation=np.asarray(rotation);p0,r0=self.pose(arm)
        distance=np.linalg.norm(target-p0);angle=np.linalg.norm(Rotation.from_matrix(rotation@r0.T).as_rotvec())
        off=0 if arm=='left' else 8;start=self.target.copy()
        table=[start.copy()]
        cartesian=self.closed_flags[arm] or (angle<.15 and distance>.02)
        count_points=max(1,int(np.ceil(max(distance/.012,angle/.10)))) if cartesian else 1
        slerp=Slerp([0,1],Rotation.from_matrix(np.stack([r0,rotation])))
        try:
            for f in np.linspace(0,1,count_points+1)[1:]:
                self.planning_seed[arm]=table[-1][off:off+7].copy()
                q=self._solve(arm,p0+(target-p0)*f,slerp(f).as_matrix())
                if count_points>1 and np.max(abs(q-table[-1][off:off+7]))>.6:raise ValueError('IK branch discontinuity')
                row=start.copy();row[off:off+7]=q;table.append(row)
        finally:self.planning_seed.clear()
        table=np.array(table);goal=table[-1]
        # Quintic timing: zero velocity and acceleration at segment boundaries.
        arc=np.sum(np.max(abs(np.diff(table,axis=0)),axis=1))
        count=max(6,int(np.ceil(1.875*arc/.024)))
        for t in np.linspace(0,1,count+1)[1:]:
            f=t*t*t*(10+t*(-15+6*t));position=f*(len(table)-1)
            index=min(int(position),len(table)-2);fraction=position-index
            self._tick(table[index]*(1-fraction)+table[index+1]*fraction)
        self.target=goal
        for _ in range(4):self._tick(self.target)
        for _ in range(8):
            if np.linalg.norm(self.pose(arm)[0]-target)<.0025:break
            self._tick(self.target)

    def _solve(self,arm,target,rotation):
        # Local continuation along the Cartesian path; live state is untouched.
        env=self.env;data=self.scratch;ids=env.qids[arm];dofs=env.dids[arm];body=env.ee_ids[arm]
        data.qpos[:]=env.data.qpos;data.qvel[:]=0
        q0=getattr(self,'planning_seed',{}).get(arm,env.data.qpos[ids]).copy()
        off=0 if arm=='left' else 8;lo=env.action_space.low[off:off+7];hi=env.action_space.high[off:off+7]
        rng=np.random.default_rng(17);best=1e9
        for attempt in range(10):
            data.qpos[ids]=q0 if not attempt else np.clip(q0+rng.normal(0,.65,7),lo,hi)
            for _ in range(180):
                mujoco.mj_forward(env.model,data);p,r=self.pose(arm,data)
                dp=target-p;dr=Rotation.from_matrix(rotation@r.T).as_rotvec()
                best=min(best,float(np.linalg.norm(dp)+.2*np.linalg.norm(dr)))
                if np.linalg.norm(dp)<.0007 and np.linalg.norm(dr)<.02:return data.qpos[ids].copy()
                jp=np.zeros((3,env.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(env.model,data,jp,jr,p,body)
                jac=np.vstack([jp[:,dofs],.2*jr[:,dofs]]);error=np.r_[dp,.2*dr]
                dq=jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(6),error)
                dq*=min(1.,.10/max(np.max(abs(dq)),1e-9));data.qpos[ids]=np.clip(data.qpos[ids]+dq,lo,hi)
        raise ValueError('local IK did not converge; best weighted residual '+str(round(best,5)))

    def move_tcp(self, arm, position_m, yaw_deg=None, pitch_deg=-15):
        self._guard();p=np.asarray(position_m,float)
        if p.shape!=(3,) or not np.isfinite(p).all() or np.linalg.norm(p-self.pose(arm)[0])>.55:
            raise ValueError('finite TCP target within 550mm required')
        rotation=self.rotation_reference[arm]
        if yaw_deg is not None:
            theta=np.deg2rad(float(yaw_deg));up=np.array([0.,0.,1.]);a=np.array([np.cos(theta),np.sin(theta),0.])
            rotation=np.column_stack([up,np.cross(a,up),a])@Rotation.from_euler('y',pitch_deg,degrees=True).as_matrix()
        self._move(arm,p,rotation);self.rotation_reference[arm]=rotation.copy()
        for _ in range(24):
            if np.linalg.norm(self.pose(arm)[0]-p)<.008:break
            self._tick(self.target)
        error=float(np.linalg.norm(self.pose(arm)[0]-p))
        if error>.008:raise ValueError('TCP tracking error exceeds 8mm: '+str(round(error,4)))
        return {'ok':True,'actual_tcp_m':self.pose(arm)[0].tolist(),'tracking_error_m':error}

    def move_relative(self,arm='right',delta_m=(0,0,0)):
        delta=np.asarray(delta_m,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.18:
            raise ValueError('bounded relative motion <=180mm required')
        return self.move_tcp(arm,self.pose(arm)[0]+delta)

    def set_gripper(self,arm='right',opened=True):
        self._guard();goal=self.target.copy();idx=7 if arm=='left' else 15
        goal[idx]=self.env.grippers[arm].open_motor if opened else self.env.grippers[arm].closed_motor
        if arm=='right' and not opened and self.case.get('close_drop_once') and not self.close_loss_used:
            goal[idx]=self.env.grippers[arm].open_motor;self.close_loss_used=True
        start=self.target.copy();count=14 if opened else 22
        for t in np.linspace(0,1,count+1)[1:]:
            f=t*t*t*(10+t*(-15+6*t));self._tick(start+(goal-start)*f)
        self.target=goal
        for _ in range(8):self._tick(goal)
        self.closed_flags[arm]=not opened
        if not opened:self.grip_reference=self.pose(arm)[0].copy()
        return {'ok':True,'motor_rad':float(self.env.state()[idx])}

    def grip_evidence(self,arm):
        self._guard();m=float(self.env.state()[7 if arm=='left' else 15]);mapping=self.env.grippers[arm]
        blocked=bool(mapping.open_motor+.12<m<mapping.closed_motor-.12)
        result={'arm':arm,'blocked_closure':blocked,'measured_motor_rad':m,
            'source':'proprioceptive jaw obstruction proxy; independent contact grade remains private'}
        self.evidence.append(result);return result

    def visual_align(self,max_steps=30,yaw_deg=0,pitch_deg=-15,target_pixels=None,offset_m=(0,0,0),step_m=.018,tolerance_px=4.):
        self._guard();self.phase='visual_alignment';offset=np.asarray(offset_m,float)
        if offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.13:
            raise ValueError('bounded visual offset required')
        if not .002<=step_m<=.025 or not 1<=tolerance_px<=15:raise ValueError('bounded visual parameters required')
        if target_pixels is not None:
            target=np.asarray(target_pixels,float)
            if target.shape!=(4,) or not np.isfinite(target).all():raise ValueError('two RGB target points required')
            self.rig.goals=target.copy()
        f,_=self.rig.read();origin=self.pose('right')[0].copy()
        if 'right' not in self.jacobians:
            ds=[];df=[]
            for axis in range(3):
                delta=np.zeros(3);delta[axis]=.006
                self._move('right',origin+delta,self.rotation_reference['right']);new,_=self.rig.read()
                ds.append(self.pose('right')[0]-origin);df.append(new-f)
                self._move('right',origin,self.rotation_reference['right'])
            ds=np.column_stack(ds)
            if np.linalg.cond(ds)>30:raise ValueError('probe motion not observable')
            jac=np.column_stack(df)@np.linalg.inv(ds)
        else:
            jac=self.jacobians['right'].copy()
            self.rig.calibrate()
        image_goal=self.rig.goals+jac@offset
        for step in range(min(int(max_steps),40)):
            f,row=self.rig.read();error=image_goal-f
            row['max_view_error_px']=float(np.linalg.norm(error.reshape(2,2),axis=1).max())
            row['phase']=self.phase;self.last_visual=row;self.vision_trace.append(row)
            if row['max_view_error_px']<tolerance_px:
                self.jacobians['right']=jac;return {'ok':True,'steps':step,'error_px':row['max_view_error_px']}
            sv=np.linalg.svd(jac,compute_uv=False)
            if sv[-1]<8 or sv[0]/sv[-1]>40:raise ValueError('unobservable visual geometry')
            delta=np.linalg.solve(jac.T@jac+25*np.eye(3),jac.T@error)*.7
            delta*=min(1.,step_m/max(np.linalg.norm(delta),1e-9))
            p=self.pose('right')[0].copy();self._move('right',p+delta,self.rotation_reference['right'])
            new,_=self.rig.read();actual=self.pose('right')[0]-p
            if np.linalg.norm(actual)<.0001:raise ValueError('motion stalled')
            if self.rig.cameras is None:
                jac+=np.outer(new-f-jac@actual,actual)/(actual@actual)
            else:
                p=self.pose('right')[0];rows=[]
                for camera in self.rig.cameras:
                    q=camera@np.r_[p,1.]
                    rows.extend([(camera[i,:3]*q[2]-q[i]*camera[2,:3])/q[2]**2 for i in range(2)])
                jac=np.array(rows)
        return {'ok':False,'error_code':'alignment_budget'}

    def result(self,execution):
        lower=bool(self.lower_contacts) and min(self.lower_contacts)>=0 and max(self.lower_contacts)<=.5
        result={'program_ok':execution.ok,'error':str(execution.error) if execution.error else None,
            'full_task_success':bool(execution.ok and self.env.evaluation['success'] and lower),
            'right_lower_contacts':self.lower_contacts,'lower_body_passed':lower,
            'evaluation':self.env.evaluation,'events':self.env.events,'stages':self.stages,
            'sim_seconds':float(self.env.data.time),'peak_lift_m':self.peak_lift,
            'legacy_tamp_calls':0,'hardware_access':False,'transfer_ready':False,
            'object_xyz_exposed':False,'simulator_camera_matrices_read':False,
            'estimated_camera_model_used':self.rig.cameras is not None,'projection_quality':self.rig.quality,
            'gripper_range':'explicit native model range; not learned-policy range or physical certification'}
        dump(self.output/'result.json',result);dump(self.output/'visual-trace.json',self.vision_trace)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/('relay-'+uuid.uuid4().hex[:8]))
    p.add_argument('--program',type=Path,default=ROOT/'skills/bottle_visual/relay/program.py')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():p.error('new output under runs/ required')
    case=json.loads(a.case.read_text()) if a.case else {'seed':0,'bottle_xy_m':[.5,-.21]}
    try:world=RelayWorld(case,out)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True);dump(out/'result.json',{'program_ok':False,
            'full_task_success':False,'error':'initialization_failed: '+str(exc),'hardware_access':False})
        print(json.dumps({'output':str(out),'error':str(exc),'full_task_success':False}),flush=True);return
    try:
        code=a.program.read_text();(out/'program.py').write_text(code)
        r=world.result(world.execute(code));print(json.dumps({'output':str(out),**{k:r[k] for k in ('program_ok','error','full_task_success','sim_seconds')}}),flush=True)
    finally:world.close()


if __name__=='__main__':main()
