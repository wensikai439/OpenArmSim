"""Optional loaded-motion correction; all legacy/demo routes stay unchanged."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from .runtime import ROOT, dump
from .recovery import RecoveryWorld, SegmentPlanner
from .variants import variant_environment
from sim.tamp import PlanningFailure
from sim.adaptive_grasp import KinematicIK
from sim.relay_place import RelayBasketPlacement


class CartesianBasket(KinematicIK,RelayBasketPlacement):
    """Interpolate the OBJECT pose, re-grounding its measured grasp each step."""
    def _move(self,target,allow_fingers=False,**kwargs):
        if self.carried is None or self.phase!='place':
            yield from super()._move(target,allow_fingers,**kwargs);return
        from scipy.spatial.transform import Rotation, Slerp
        env=self.env;target=np.asarray(target)
        desired_rotation=self.rotation.copy()
        goal=target+desired_rotation@(self.carried[0]-self.local_point)
        object_rotation=desired_rotation@self.carried[1]
        initial=env.data.xpos[env.bottle_id].copy()
        initial_rotation=env.data.xmat[env.bottle_id].reshape(3,3).copy()
        angle=Rotation.from_matrix(object_rotation@initial_rotation.T).magnitude()
        count=max(1,int(np.ceil(max(np.linalg.norm(goal-initial)/.04,angle/.18))))
        interpolator=Slerp([0.,1.],Rotation.from_matrix(np.stack([initial_rotation,object_rotation])))
        history=self.result.setdefault('object_cartesian_waypoints',[])
        # Transport originally allowed12mm TCP error; keep intermediate
        # segments tighter (6mm), while preserving the requested3mm descent.
        tolerance=min(kwargs.get('position_tolerance',.012),.006)
        for i in range(1,count+1):
            fraction=i/count;position=initial+(goal-initial)*fraction
            rotation=interpolator(fraction).as_matrix()
            hand=env.data.xmat[self.body].reshape(3,3)
            offset=hand.T@(env.data.xpos[env.bottle_id]-env.data.xpos[self.body])
            relative=hand.T@env.data.xmat[env.bottle_id].reshape(3,3)
            # The SAME measured grasp relation must ground target generation
            # AND scratch collision prediction. An old planning attachment
            # can invent collisions even after the target was re-grounded.
            self.carried=(offset.copy(),relative.copy())
            self.rotation=rotation@relative.T
            tcp=position+self.rotation@(self.local_point-offset)
            yield from super()._move(tcp,allow_fingers,position_tolerance=tolerance)
            history.append({'target_object_m':position.tolist(),
                'actual_object_m':env.data.xpos[env.bottle_id].tolist(),
                'position_error_m':float(np.linalg.norm(position-env.data.xpos[env.bottle_id])),
                'native_tick':env.tick,'actual_upright':float(env.data.xmat[env.bottle_id].reshape(3,3)[2,2])})


class ParkTransfer(SegmentPlanner):
    """Use the modern relay's checked donor parking instead of legacy exit."""
    def actions(self):
        env=self.env
        if not all(env.evaluation['held'].values()):
            raise PlanningFailure('dual_hold_required','Both actual holds required before donor release')
        self._phase('release_right');yield from self._grip('right',env.grippers['right'].open_motor)
        if not env.evaluation['held']['left']:
            raise PlanningFailure('receiver_grasp_lost','Receiver lost native hold after release')
        self._phase('retreat');self._arm('right');self.speed_scale=.5
        from sim.bottle_routing import BottleHandoverPlanner
        yield from BottleHandoverPlanner._park_donor(self)
        self._phase('hold')
        for _ in range(60):
            yield self.base_action.copy(),{'phase':'hold','scene_source':'simulator_ground_truth'}
            if not env.evaluation['held']['left'] or env.evaluation['touching']['right']:
                raise PlanningFailure('handover_not_confirmed','Independent receiver hold not maintained')
        self.result.update(primitive_success=True,status='completed')


class PrecisionWorld(RecoveryWorld):
    def __init__(self,case,output):
        self.profile=case.get('motion_profile','precision')
        if self.profile not in ('baseline','slow','precision'):raise ValueError('unknown motion profile')
        base={k:case[k] for k in ('seed','bottle_xy_m','fault')}
        if 'pose_bias_m' in case:base['pose_bias_m']=case['pose_bias_m']
        self.variant={'schema':'relay-fixture/2','seed':case['seed'],'bottle_xy_m':case['bottle_xy_m'],
            'bottle_yaw_rad':case.get('bottle_yaw_rad',0.),'basket_offset_m':case.get('basket_offset_m',[0.,0.]),
            'bottle_mass_scale':case.get('bottle_mass_scale',1.),'bottle_friction_scale':case.get('bottle_friction_scale',1.)}
        super().__init__(base,output,environment_factory=lambda env,_:variant_environment(env,self.variant))
        self.case=dict(case)
        self.source_hash=hashlib.sha256((Path(__file__).read_bytes()+Path(__file__).with_name('recovery.py').read_bytes())).hexdigest()
        dump(self.output/'case.json',case)

    def start_place(self):
        self._guard()
        planner_type=CartesianBasket if self.profile=='precision' else self.planner_factory
        planner=planner_type(self.env,'place_bottle',parameters={'object_id':'bottle',
            'arm':'left','receptacle_id':'basket','shape_clearance':True,
            'speed_scale':1. if self.profile=='baseline' else .5 if self.profile=='precision' else .35})
        self._begin(planner,'place');return {'ok':True,'done':False}

    def finish_transfer(self):
        if self.profile!='precision':return super().finish_transfer()
        self._guard();return self._once(ParkTransfer(self.env,mode='finish_transfer'),'finish_transfer')

    def result(self):
        r=super().result();r['motion_profile']=self.profile
        target=self.env.placement_target
        r['pose_error_m']=float(np.linalg.norm(self.env.data.xpos[self.env.bottle_id,:2]-target['xy_m'])) if target else None
        return r


def main():
    import json
    from .recovery_campaign import run_trial
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('output must stay in runs/')
    run_trial(json.loads(a.case.read_text()),out,a.program,'frozen',world_type=PrecisionWorld)


if __name__=='__main__':main()
