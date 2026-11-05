"""Native object-centered yaw alternatives for a failed upright placement.

The task constrains object position and upright axis, not axial yaw. Every
alternative retains the original IK, collision, tracking and scoring gates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .precision import CartesianBasket
from .robust import RobustWorld
from .runtime import ROOT
from sim.tamp import PlanningFailure


def yaw_alternative(object_rotation,relative,offset,local_point,goal,angle):
    turn=Rotation.from_rotvec(object_rotation[:,2]*angle).as_matrix()
    object_target=turn@object_rotation
    hand_target=object_target@relative.T
    tcp=goal+hand_target@(local_point-offset)
    return hand_target,tcp,object_target


class UprightBasket(CartesianBasket):
    def _move(self,target,allow_fingers=False,**kwargs):
        if self.carried is None or self.phase!='place':
            yield from super()._move(target,allow_fingers,**kwargs);return
        desired=self.rotation.copy()
        goal=np.asarray(target)+desired@(self.carried[0]-self.local_point)
        object_rotation=desired@self.carried[1]
        try:
            yield from super()._move(target,allow_fingers,**kwargs)
            return
        except PlanningFailure as exc:
            if exc.code!='ik_unreachable':raise
            if (self.env._done or self.env.evaluation['invalid']
                or not self.env.evaluation['held']['left']):raise
            original=exc
        # Re-ground the hand-object relation after any partial native motion.
        env=self.env;hand=env.data.xmat[self.body].reshape(3,3)
        offset=hand.T@(env.data.xpos[env.bottle_id]-env.data.xpos[self.body])
        relative=hand.T@env.data.xmat[env.bottle_id].reshape(3,3)
        self.carried=(offset.copy(),relative.copy());self.planning_qpos=None
        audit=self.result.setdefault('free_yaw_replans',[])
        for degrees in (5,-5,10,-10,20,-20,30,-30):
            self._check()
            rotation,tcp,obj=yaw_alternative(object_rotation,relative,offset,self.local_point,goal,np.deg2rad(degrees))
            self.rotation=rotation;self._sync()
            try:
                q=self._ik(tcp,allow_fingers);self._path(q)
            except PlanningFailure as exc:
                if exc.code in ('planning_timeout','cancelled'):raise
                audit.append({'yaw_delta_degrees':degrees,'accepted':False,'code':exc.code})
                continue
            audit.append({'yaw_delta_degrees':degrees,'accepted':True,
                'object_target_m':goal.tolist(),'object_upright_cosine':float(obj[2,2]),
                'source':'fresh object-frame goal; full native collision path checked',
                'task_position_upright_release_stability_gates_unchanged':True})
            yield from super()._move(tcp,allow_fingers,**kwargs)
            return
        raise original


class PlacementWorld(RobustWorld):
    def __init__(self,case,output):
        super().__init__(case,output)
        self.source_hash=hashlib.sha256(Path(__file__).read_bytes()+
            Path(__file__).with_name('robust.py').read_bytes()+
            Path(__file__).with_name('body.py').read_bytes()+
            Path(__file__).with_name('precision.py').read_bytes()+
            Path(__file__).with_name('recovery.py').read_bytes()).hexdigest()

    def start_place(self):
        self._guard()
        planner=UprightBasket(self.env,'place_bottle',parameters={'object_id':'bottle',
            'arm':'left','receptacle_id':'basket','shape_clearance':True,'speed_scale':.5})
        self._begin(planner,'place');return {'ok':True,'done':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('local output belongs in runs/')
    from .recovery_campaign import run_trial
    run_trial(json.loads(a.case.read_text()),out,a.program,'frozen',world_type=PlacementWorld)
