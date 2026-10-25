"""Independent image-guided skill runtime: no legacy TAMP/manipulation macros.

The image controller receives RGB features and robot encoder-derived TCPs only.
Native task state is read separately by the evaluator, never exposed as XYZ.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image

from .runtime import ROOT, ProgramExecutor, ToolRegistry, PublicTrace, dump, fixture_environment


class RobotWorld:
    PUBLIC = ('observe', 'orient', 'move_relative', 'set_gripper', 'visual_align', 'wait','check_grasp')

    def __init__(self, case, output):
        self.case, self.output = dict(case), Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        from sim.scene_library import LibraryEnv
        from .visual import RGBRig
        fixture = {k:case[k] for k in ('seed','bottle_xy_m')}
        self.env = fixture_environment(LibraryEnv, fixture)(images=False, selection=[])
        self.env.reset(seed=case['seed']); self.env.set_task_budget(120)
        self.scratch = mujoco.MjData(self.env.model)
        self.target = self.env.state().astype(float)
        self.rotation_reference={arm:self.pose(arm)[1].copy() for arm in ('left','right')}
        self.integral = np.zeros(16)
        self.calls = 0; self.trace = []; self.video = None
        self.last_visual = None; self.vision_trace = []; self.phase = 'initial'
        self.initial_bottom = self.env._bottle_bounds()[0][2]  # evaluator only
        self.peak_lift = 0.; self.grasp_samples = []
        self.held_consecutive=0;self.hold_checked=False;self.grip_reference=None;self.close_loss_used=False
        self.rig = RGBRig(self.env, self.output/'vision',
            yaw_offset=case.get('camera_yaw_deg',0), pixel_noise=case.get('pixel_noise',0), seed=case['seed'])
        try:self.rig.lock_target()
        except Exception:
            self.rig.renderer.close();self.env.close();raise
        encoder = ROOT/'.runtime/video-encoder'
        sys.path.insert(0,str(encoder))
        import imageio_ffmpeg
        self.video = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel','error',
            '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','15',
            '-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(self.output/'rollout.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        dump(self.output/'case.json',case)

    def _guard(self):
        self.calls += 1
        if self.calls > 80 or self.env._done or self.env.evaluation['invalid']:
            raise ValueError('native episode terminal or tool budget exhausted')

    def pose(self, arm, data=None):
        data = self.env.data if data is None else data
        body = self.env.ee_ids[arm]; r = data.xmat[body].reshape(3,3)
        return data.xpos[body]+r@np.array([0., .006, .075]), r

    def _tick(self, q):
        env=self.env
        measured=env.state().astype(float)
        # Bounded generic position integral from encoder tracking error. No
        # object mass, true hold state or contact wrench enters this controller.
        for offset in (0,8):
            error=q[offset:offset+7]-measured[offset:offset+7]
            self.integral[offset:offset+7]=np.clip(self.integral[offset:offset+7]+error*.04,-.14,.14)
        action=q+self.integral
        action=np.clip(action,env.action_space.low,env.action_space.high)
        _,_,terminal,truncated,info=env.step_policy(action)
        # Independent physical evaluator; these values are not policy inputs.
        height=env._bottle_bounds()[0][2]-self.initial_bottom
        self.peak_lift=max(self.peak_lift,float(height))
        if env.evaluation['held']['right'] and height>.035:
            self.grasp_samples.append({'tick':env.tick,'height_m':float(height)})
            self.held_consecutive+=1
        else:self.held_consecutive=0
        if env.tick%2==0 and self.video:
            self.video.stdin.write(env.render('operator',view={'mode':'free','azimuth':165.,
                'elevation':-24.,'distance':1.02,'lookat':[.43,0.,1.15]}).tobytes())
        self.trace.append({'tick':env.tick,'phase':self.phase,'requested':q.tolist(),
            'applied':action.tolist(),'measured':measured.tolist(),
            'tcp':{s:self.pose(s)[0].tolist() for s in ('left','right')}})
        if terminal or truncated:
            raise ValueError('native physics terminated: '+str(info['evaluation'].get('invalid_reason')))

    def _solve(self, arm, target, rotation):
        env=self.env;data=self.scratch
        data.qpos[:]=env.data.qpos; data.qvel[:]=0
        ids=env.qids[arm]; dofs=env.dids[arm];body=env.ee_ids[arm]
        q0=env.data.qpos[ids].copy(); rng=np.random.default_rng(17)
        best=1e9
        for attempt in range(12):
            data.qpos[ids]=q0 if not attempt else np.clip(q0+rng.normal(0,.65,7),
                env.action_space.low[0 if arm=='left' else 8:7 if arm=='left' else 15],
                env.action_space.high[0 if arm=='left' else 8:7 if arm=='left' else 15])
            for iteration in range(180):
                mujoco.mj_forward(env.model,data);p,r=self.pose(arm,data)
                dp=target-p;dr=Rotation.from_matrix(rotation@r.T).as_rotvec()
                best=min(best,float(np.linalg.norm(dp)+.2*np.linalg.norm(dr)))
                if np.linalg.norm(dp)<.0007 and np.linalg.norm(dr)<.02:
                    return data.qpos[ids].copy()
                jp=np.zeros((3,env.model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jac(env.model,data,jp,jr,p,body)
                jac=np.vstack([jp[:,dofs],.20*jr[:,dofs]])
                error=np.r_[dp,.20*dr]
                dq=jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(6),error)
                dq*=min(1.,.10/max(np.max(abs(dq)),1e-9))
                off=0 if arm=='left' else 8
                data.qpos[ids]=np.clip(data.qpos[ids]+dq,env.action_space.low[off:off+7],env.action_space.high[off:off+7])
        raise ValueError('local IK did not converge; best weighted residual '+str(round(best,5)))

    def _move(self, arm, target, rotation):
        q=self._solve(arm,np.asarray(target),np.asarray(rotation))
        start=self.target.copy();goal=start.copy();off=0 if arm=='left' else 8;goal[off:off+7]=q
        count=max(12,int(np.ceil(np.max(abs(goal-start))/.012)))
        for fraction in np.linspace(0,1,count+1)[1:]:
            self._tick(start+(goal-start)*fraction)
        self.target=goal
        for _ in range(12):self._tick(self.target)

    def observe(self):
        self._guard()
        robot={s:{'tcp_m':self.pose(s)[0].tolist(),'rotation':self.pose(s)[1].tolist(),
                   'gripper_motor_rad':float(self.env.state()[7 if s=='left' else 15]),
                   'gripper_open_rad':self.env.grippers[s].open_motor} for s in ('left','right')}
        return {'tick':self.env.tick,'robot':robot,'vision':self.last_visual,
                'image_targets':{'bottle_lower':self.rig.goals.tolist()},
                'object_xyz_available':False,'legacy_tamp_available':False}

    def orient(self, arm='right', yaw_deg=0, pitch_deg=-15):
        self._guard();self.phase='orient_'+arm
        theta=np.deg2rad(float(yaw_deg));up=np.array([0.,0.,1.]);approach=np.array([np.cos(theta),np.sin(theta),0.])
        r=np.column_stack([up,np.cross(approach,up),approach])
        r=r@Rotation.from_euler('y',float(pitch_deg),degrees=True).as_matrix()
        self._move(arm,self.pose(arm)[0],r)
        self.rotation_reference[arm]=r.copy()
        return {'ok':True}

    def move_relative(self, arm='right', delta_m=(0,0,0)):
        self._guard();delta=np.array(delta_m,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.181:
            raise ValueError('bounded relative movement <=180mm required')
        self.phase='relative_'+arm
        self._move(arm,self.pose(arm)[0]+delta,self.rotation_reference[arm]);return {'ok':True}

    def set_gripper(self, arm='right', opened=True):
        self._guard();self.phase=('open_' if opened else 'close_')+arm
        goal=self.target.copy();goal[7 if arm=='left' else 15]=self.env.grippers[arm].open_motor if opened else self.env.grippers[arm].closed_motor
        if arm=='right' and not opened and self.case.get('close_drop_once') and not self.close_loss_used:
            goal[15]=self.env.grippers['right'].open_motor;self.close_loss_used=True
        start=self.target.copy()
        for f in np.linspace(0,1,35)[1:]:self._tick(start+(goal-start)*f)
        self.target=goal
        for _ in range(12):self._tick(goal)
        if not opened:self.grip_reference=self.pose(arm)[0].copy()
        return {'ok':True,'motor_rad':float(self.env.state()[7 if arm=='left' else 15])}

    def check_grasp(self):
        self._guard()
        shifts=self.rig.object_motion()
        lift=float(self.pose('right')[0][2]-self.grip_reference[2]) if self.grip_reference is not None else 0.
        self.hold_checked=bool(min(shifts)>12 and lift>.035)
        return {'ok':self.hold_checked,'source':'fresh RGB bottle-bottom displacement and encoder TCP motion',
                'bottom_up_px':shifts,'actual_tcp_lift_m':lift}

    def wait(self, seconds=.5):
        self._guard()
        if not 0<=seconds<=2:raise ValueError('bounded wait required')
        self.phase='hold'
        for _ in range(int(seconds*30)):self._tick(self.target)
        return {'ok':True}

    def visual_align(self, max_steps=20, yaw_deg=0, pitch_deg=-15, target_pixels=None, offset_m=(0,0,0), step_m=.018, tolerance_px=3.):
        self._guard();self.phase='visual_alignment'
        offset=np.asarray(offset_m,float)
        if offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.13:
            raise ValueError('visual target offset must be bounded by 130mm')
        if not .002<=step_m<=.018:raise ValueError('visual step must be 2 to 18mm')
        if not 1.<=tolerance_px<=15.:raise ValueError('pixel tolerance must be 1 to 15px')
        if target_pixels is not None:
            target=np.asarray(target_pixels,float)
            if target.shape!=(4,) or not np.isfinite(target).all() or not np.all((target>0)&(target<[1280,960,1280,960])):
                raise ValueError('two bounded RGB target points required')
            self.rig.goals=target.copy()
        f,_=self.rig.read();origin=self.pose('right')[0].copy();f0=f.copy();ds=[];df=[]
        for axis in range(3):
            delta=np.zeros(3);delta[axis]=.006
            self._move('right',origin+delta,self.rotation_reference['right']);new,_=self.rig.read()
            ds.append(self.pose('right')[0]-origin);df.append(new-f0)
            self._move('right',origin,self.rotation_reference['right'])
        ds=np.column_stack(ds)
        if np.linalg.cond(ds)>20:raise ValueError('insufficient probing directions')
        jac=np.column_stack(df)@np.linalg.inv(ds)
        image_goal=self.rig.goals+jac@offset
        theta=np.deg2rad(float(yaw_deg));up=np.array([0.,0.,1.]);approach=np.array([np.cos(theta),np.sin(theta),0.])
        desired_rotation=np.column_stack([up,np.cross(approach,up),approach])
        desired_rotation=desired_rotation@Rotation.from_euler('y',float(pitch_deg),degrees=True).as_matrix()
        for step in range(min(int(max_steps),24)):
            f,row=self.rig.read();error=image_goal-f
            row['max_view_error_px']=float(np.linalg.norm(error.reshape(2,2),axis=1).max())
            self.last_visual=row;self.vision_trace.append(row)
            dump(self.output/'visual-trace.json',self.vision_trace)
            if row['max_view_error_px']<tolerance_px:
                for _ in range(4):self._tick(self.target)
                fresh_f,fresh=self.rig.read()
                fresh['max_view_error_px']=float(np.linalg.norm((image_goal-fresh_f).reshape(2,2),axis=1).max())
                if fresh['max_view_error_px']<tolerance_px:
                    self.last_visual=fresh
                    return {'ok':True,'steps':step,'error_px':fresh['max_view_error_px']}
            sv=np.linalg.svd(jac,compute_uv=False)
            if sv[-1]<8 or sv[0]/sv[-1]>30:raise ValueError('view geometry is not observable')
            delta=np.linalg.solve(jac.T@jac+25*np.eye(3),jac.T@error)*.7
            delta*=min(1.,step_m/max(np.linalg.norm(delta),1e-9))
            row.update(robot_tcp_m=self.pose('right')[0].tolist(),requested_delta_m=delta.tolist(),
                       singular_values=sv.tolist())
            dump(self.output/'visual-trace.json',self.vision_trace)
            p=self.pose('right')[0].copy();present=self.rotation_reference['right']
            rotation=Slerp([0,1],Rotation.from_matrix(np.stack([present,desired_rotation])))(.30).as_matrix()
            self._move('right',p+delta,rotation)
            self.rotation_reference['right']=rotation.copy()
            new,_=self.rig.read();actual=self.pose('right')[0]-p
            if np.linalg.norm(actual)<.0004:raise ValueError('motion stalled')
            jac+=np.outer(new-f-jac@actual,actual)/(actual@actual)
        return {'ok':False,'error_code':'alignment_budget'}

    def execute(self, program):
        registry=ToolRegistry(shared_public_names=self.PUBLIC,public_extension_allowlist=())
        for name in self.PUBLIC:
            registry.register(getattr(self,name),name=name,public=True,layer='shared_atomic',capability='bounded robot motion and RGB feedback; no task macros')
        result=ProgramExecutor(registry,recorder=PublicTrace(self.output/'tools.jsonl')).execute_program(program)
        return result

    def result(self, execution):
        # Evaluation-only geometry/contact access. No evaluator outcome is fed
        # back to the generated program during execution.
        stable=self.held_consecutive>=20
        body=self.env.model.geom('bottle_body').id
        contacts=[];fractions=[]
        for i in range(self.env.data.ncon):
            c=self.env.data.contact[i]
            if body in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==body else c.geom1
                name=self.env.model.body(self.env.model.geom_bodyid[other]).name
                if 'openarm_right' in name and 'finger' in name:
                    axis=self.env.data.geom_xmat[body].reshape(3,3)[:,2]
                    centre=self.env.data.geom_xpos[body]
                    fraction=.5+np.dot(c.pos-centre,axis)/(2*self.env.model.geom_size[body,1])
                    fractions.append(float(fraction));contacts.append(c.pos.tolist())
        lower=bool(fractions) and min(fractions)>=0 and max(fractions)<=.5
        hold=[row['tcp'] for row in self.trace if row['phase']=='hold']
        hold_drift={side:float(np.ptp([row[side][2] for row in hold])) if hold else None for side in ('left','right')}
        r={'program_ok':execution.ok,'error':str(execution.error) if execution.error else None,
           'right_grasp_lift_success':bool(execution.ok and stable and lower and self.hold_checked and not self.env.evaluation['invalid']),
           'peak_lift_m':self.peak_lift,'hold_samples':len(self.grasp_samples),
           'final_consecutive_hold_samples':self.held_consecutive,'lower_body_contact_fractions':fractions,
           'lower_body_contact_passed':lower,'visual_hold_confirmed':self.hold_checked,
           'hold_vertical_range_m':hold_drift,'close_drop_injected':self.close_loss_used,
           'evaluation':self.env.evaluation,'vision_iterations':len(self.vision_trace),
           'legacy_tamp_calls':0,'object_xyz_exposed':False,'camera_matrices_used':False,
           'physical_world_resets_after_start':0,'hardware_access':False,'transfer_ready':False,
           'sensor_assumptions':'two synthetic fixed RGB views, visible rigid tool dots; known upright black bottle, stationary until grasp',
           'scope':'right visual grasp/lift, not yet full relay/place or real robot validation'}
        dump(self.output/'result.json',r);return r

    def close(self):
        dump(self.output/'controller.json',self.trace)
        if self.video:
            self.video.stdin.close();self.video.wait(timeout=40)
            error=self.video.stderr.read().decode();self.video.stderr.close()
            if self.video.returncode:raise RuntimeError('video encoding failed: '+error[:200])
        self.rig.renderer.close()
        self.env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--program',type=Path,default=ROOT/'skills/bottle_visual/program.py')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('output must stay in runs/')
    if out.exists():p.error('refusing to overwrite an existing experiment')
    try:world=RobotWorld(json.loads(a.case.read_text()),out)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True)
        dump(out/'result.json',{'program_ok':False,'right_grasp_lift_success':False,
            'error':'initialization_failed: '+type(exc).__name__+': '+str(exc),
            'hardware_access':False,'transfer_ready':False})
        print(json.dumps({'error':str(exc),'right_grasp_lift_success':False}),flush=True)
        return
    try:
        code=a.program.read_text();(out/'program.py').write_text(code)
        result=world.result(world.execute(code))
        print(json.dumps({k:result[k] for k in ['program_ok','error','right_grasp_lift_success','peak_lift_m','vision_iterations']}),flush=True)
    finally:world.close()


if __name__=='__main__':main()
