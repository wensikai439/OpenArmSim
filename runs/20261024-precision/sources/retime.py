"""Optional loaded-motion correction; all legacy/demo routes stay unchanged."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from .runtime import ROOT, dump
from .recovery import RecoveryWorld, SegmentPlanner
from .variants import variant_environment
from sim.tamp import PlanningFailure


class AxialTransfer(SegmentPlanner):
    """Withdraw an open donor along its measured approach before rotating."""
    def actions(self):
        env=self.env
        if not all(env.evaluation['held'].values()):
            raise PlanningFailure('dual_hold_required','Both actual holds required before donor release')
        self._phase('release_right');yield from self._grip('right',env.grippers['right'].open_motor)
        if not env.evaluation['held']['left']:
            raise PlanningFailure('receiver_grasp_lost','Receiver lost native hold after release')
        self._phase('retreat');self._arm('right');self._sync();origin,rotation=self._pose()
        self.rotation=rotation.copy();self.strict_orientation=True;self.speed_scale=.35
        self.result['axial_withdrawal']={'axis_world':(-rotation[:,2]).tolist(),
            'distance_m':.08,'step_m':.02,'speed_scale':self.speed_scale,
            'source':'measured donor TCP orientation, no world-axis waypoint or state restore'}
        for distance in (.02,.04,.06,.08):
            for action,diagnostic in self._move(origin-distance*rotation[:,2],position_tolerance=.004):
                yield action,{**diagnostic,'axial_withdrawal_m':distance}
                if not env.evaluation['held']['left']:
                    raise PlanningFailure('receiver_grasp_lost','Receiver hold lost during axial withdrawal')
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
        planner=self.planner_factory(self.env,'place_bottle',parameters={'object_id':'bottle',
            'arm':'left','receptacle_id':'basket','shape_clearance':True,
            'speed_scale':1. if self.profile=='baseline' else .35})
        self._begin(planner,'place');return {'ok':True,'done':False}

    def finish_transfer(self):
        if self.profile!='precision':return super().finish_transfer()
        self._guard();return self._once(AxialTransfer(self.env,mode='finish_transfer'),'finish_transfer')

    def result(self):
        r=super().result();r['motion_profile']=self.profile
        r['pose_error_m']=float(np.linalg.norm(self.env.data.xpos[self.env.bottle_id,:2]-self.env.placement_xy)) if hasattr(self.env,'placement_xy') else None
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
