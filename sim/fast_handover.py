"""Experimental scratch-only IK efficiency for the original bottle relay.

No new waypoint, solver seed, tolerance, speed or contact/clearance gate. Only
Jacobian iterations use the existing minimal kinematics pipeline; candidate
acceptance and every path edge run full native collision calculations.
"""
from .adaptive_grasp import KinematicIK
from .bottle_routing import BottleHandoverPlanner


class FastBottleHandover(KinematicIK,BottleHandoverPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result['ik_pipeline_contract']={
            'iterations':'mj_kinematics then mj_comPos, scratch MjData only',
            'acceptance_and_path_edges':'original full native forward/collision/clearance',
            'proposal_solver_tolerances_joint_bounds_unchanged':True,
            'scope':'original bottle handover; compute efficiency only',
            'live_state_written_by_solver':False,'hardware_access':False,'transfer_ready':False}
