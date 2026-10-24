"""Body-only relay with explicit payload gravity compensation in joint targets.

The controller applies bounded reference offsets, never edits live qpos, gains,
mass, friction, contacts or task scoring. Payload mass/Jacobians are simulator
truth: this is an opt-in simulation controller, not calibrated hardware control.
"""
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from .precision import PrecisionWorld
from .recovery import SegmentPlanner
from .runtime import dump
from sim.grasp import body_bands
from sim.tamp import PlanningFailure


class SupportedRelease(SegmentPlanner):
    def actions(self):
        env=self.env
        support=env.contacts.read(env.data)
        gap=env._bottle_bounds()[0][2]-env.placement_surface('table')['top_z']
        dof=env.registry['bottle']['dofadr']
        if (not support['table_supported'] or abs(gap)>.003 or
                np.linalg.norm(env.data.qvel[dof:dof+3])>.01 or env.evaluation['held']['left']):
            raise PlanningFailure('supported_release_required','Opening a rejected grasp requires measured stationary table support')
        self._phase('gripper');yield from self._grip('right',env.grippers['right'].open_motor)
        if env.evaluation['held']['right']:
            raise PlanningFailure('supported_release_failed','Right contact remained after supported opening')
        self.result.update(primitive_success=True,status='completed',support_verified=True)


class BodyWorld(PrecisionWorld):
    PUBLIC=(*PrecisionWorld.PUBLIC,'release_on_support')
    def __init__(self,case,output):
        super().__init__({**case,'grasp_region':'body'},output)
        self.nominal=self.env.last_action.copy()
        self.payload_offset=np.zeros(7)
        self.payload_rows=[]
        self.load_reference=None
        self.grasp_contacts={}
        self.source_hash=hashlib.sha256(Path(__file__).read_bytes()+
            Path(__file__).with_name('precision.py').read_bytes()+
            Path(__file__).with_name('recovery.py').read_bytes()).hexdigest()

    def _grasp_bands(self,planner):
        return body_bands(self.env.model,planner.data,self.env.bottle_id,'lower_body')

    def start_transfer(self,placement_aware=True):
        self._guard()
        planner=self.planner_factory(self.env,'handover_bottle',parameters={
            'object_id':'bottle','from_arm':'right','to_arm':'left','hold_seconds':2,
            'placement_aware':placement_aware,'receiver_grasp_region':'upper_body'})
        self._begin(planner,'handover');return {'ok':True,'done':False}

    def close_gripper(self,arm='right'):
        result=super().close_gripper(arm)
        if not result['ok'] or arm!='right':return result
        contact=self.grasp_contacts.get('right',{})
        if contact and contact['mean_body_fraction']<=.60 and all(
                c['geom']=='bottle_body' for c in contact['contacts']):return result
        row=self.rows[-1];row.update(ok=False,native_contact_ok=True,
            failure_code='grasp_region_mismatch',message='Actual contact is outside the requested lower bottle body')
        self.last_failure={'code':'grasp_region_mismatch','tool':'close_right','tick':self.env.tick}
        self.capture('wrong-region')
        # Same declared one-shot SENSOR fault convention: discard stale pose
        # hypotheses after a measured failed grasp; never edit the live bottle.
        if self.pose_bias_active:self.pose_bias_active=False;self.pose_revision+=1
        dump(self.output/'stages.json',self.rows)
        return {**result,'ok':False,'error_code':'grasp_region_mismatch'}

    def release_on_support(self):
        self._guard()
        result=self._once(SupportedRelease(self.env,mode='supported_release'),'release_on_support')
        if result['ok']:self.grasp_contacts.pop('right',None)
        return result

    def _begin(self,planner,kind):
        if hasattr(self,'nominal'):
            planner.base_action=self.nominal.copy()
        return super()._begin(planner,kind)

    def _step(self,action,diagnostic):
        env=self.env;nominal=np.array(action,copy=True)
        # Native dwell/audit wrappers repeat the LAST APPLIED command. Remove
        # its existing feedforward so those wrappers cannot accumulate offsets.
        if np.array_equal(nominal,env.last_action):nominal[:7]-=self.payload_offset
        self.nominal=nominal.copy();target=np.zeros(7)
        held=env.evaluation['held'];phase=diagnostic.get('phase')
        if phase=='release_right' and self.load_reference is None:
            hand=env.ee_ids['left']
            self.load_reference=(env.data.xpos[hand]+env.data.xmat[hand].reshape(3,3)@np.array([0.,.006,.075])).copy()
        if (self.case.get('payload_compensation',True) and held['left'] and not held['right']
                and not env.evaluation['supported_by_basket']):
            jac=np.zeros((3,env.model.nv));rot=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,rot,
                env.data.xipos[env.bottle_id],env.ee_ids['left'])
            force=-float(env.model.body_subtreemass[env.bottle_id])*env.model.opt.gravity
            torque=jac[:,env.dids['left']].T@force
            gain=env.model.actuator_gainprm[env.aids['left'],0]
            target=torque/gain
            if np.max(abs(target))>.20:
                raise PlanningFailure('payload_compensation_limit','Required reference offset exceeds0.20rad')
        self.payload_offset+=np.clip(target-self.payload_offset,-.05,.05)
        bounds=env.action_space
        requested_offset=self.payload_offset.copy()
        # Anti-windup at the ACTUAL reference limits; the Cartesian planner's
        # original measured tracking checks still reject unreachable targets.
        self.payload_offset=np.clip(self.payload_offset,bounds.low[:7]-nominal[:7],bounds.high[:7]-nominal[:7])
        applied=nominal.copy();applied[:7]+=self.payload_offset
        done=super()._step(applied,{**diagnostic,'payload_joint_reference_offset_rad':self.payload_offset.tolist(),
            'payload_saturation_rad':(requested_offset-self.payload_offset).tolist(),
            'nominal_left_target_rad':nominal[:7].tolist(),'payload_source':'candidate simulator mass and hand Jacobian'})
        self.nominal=env.last_action.copy();self.nominal[:7]-=self.payload_offset
        body=env.ee_ids['left'];position=env.data.xpos[body]+env.data.xmat[body].reshape(3,3)@np.array([0.,.006,.075])
        self.payload_rows.append({'tick':env.tick,'phase':phase,'kind':self.active['kind'],
            'left_hand_m':position.tolist(),'left_held':bool(env.evaluation['held']['left']),
            'left_offset_rad':self.payload_offset.tolist(),
            'drop_from_release_m':None if self.load_reference is None else float(self.load_reference[2]-position[2])})
        for side in ('right','left'):
            if not env.evaluation['held'][side] or side in self.grasp_contacts:continue
            band=body_bands(env.model,env.data,env.bottle_id,'lower_body')[0]
            geom=env.model.geom(band['geom']).id;axis=env.data.geom_xmat[geom].reshape(3,3)[:,2]
            bottom=env.data.geom_xpos[geom]-axis*band['half_height_m'];samples=[]
            force=np.zeros(6)
            for index in range(env.data.ncon):
                contact=env.data.contact[index];a,b=int(contact.geom1),int(contact.geom2)
                if min(a,b)<0:continue
                object_geom=a if a in env.bottle_geoms else b if b in env.bottle_geoms else None
                if object_geom is None:continue
                other=b if object_geom==a else a;name=env.model.body(int(env.model.geom_bodyid[other])).name
                if f'openarm_{side}_' not in name or 'finger' not in name:continue
                mujoco.mj_contactForce(env.model,env.data,index,force)
                if force[0]<=.05:continue
                samples.append({'geom':env.model.geom(object_geom).name,
                    'body_fraction':float(np.dot(contact.pos-bottom,axis)/(2*band['half_height_m']))})
            if samples:self.grasp_contacts[side]={'contacts':samples,'mean_body_fraction':float(np.mean([s['body_fraction'] for s in samples]))}
        return done

    def result(self):
        result=super().result()
        stationary=[r for r in self.payload_rows if r['phase'] in ('release_right','retreat','hold')
            and r['kind'] in ('handover','finish_transfer') and r['left_held']
            and r['drop_from_release_m'] is not None]
        drop=max([0.,*[r['drop_from_release_m'] for r in stationary]])
        zone_ok=all(side in self.grasp_contacts and
            all(c['geom']=='bottle_body' for c in self.grasp_contacts[side]['contacts']) and
            (self.grasp_contacts[side]['mean_body_fraction']<=.60 if side=='right' else
             .60<=self.grasp_contacts[side]['mean_body_fraction']<=1.) for side in ('right','left'))
        result['native_task_success']=result['task_success']
        result['task_success']=bool(result['task_success'] and zone_ok and drop<=.005)
        result['body_relay']={'right_grasp_region':'lower_body','left_grasp_region':'upper_body',
            'first_native_grasp_contacts':self.grasp_contacts,
            'body_contact_verified':zone_ok,'stationary_drop_limit_m':.005,
            'stationary_hold_passed':drop<=.005,
            'payload_compensation_enabled':self.case.get('payload_compensation',True),
            'payload_compensation':'bounded joint reference feedforward; candidate model mass/Jacobian',
            'max_left_handover_drop_m':drop,
            'physical_parameters_changed':False,'hardware_access':False}
        return result

    def close(self):
        dump(self.output/'payload.json',self.payload_rows)
        super().close()


def main():
    import argparse,json
    from .recovery_campaign import run_trial
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--program',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    from .runtime import ROOT
    if not out.is_relative_to(ROOT/'runs'):p.error('output must stay in runs/')
    run_trial(json.loads(a.case.read_text()),out,a.program,'frozen',world_type=BodyWorld)


if __name__=='__main__':main()
