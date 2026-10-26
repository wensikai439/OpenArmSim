"""Real-only relative-goal adapter. Import/construction never opens devices.

This ports the adaptive simulation skill's relative-target strategy, not its
synthetic cameras, virtual tool marks, dimensions, gains or joint trajectories.
All object/tool transforms below come from the admitted real perception/FK path.
"""
import copy
import numpy as np

try:
    from .evidence import finite,transform
    from .runtime import Runtime
    from .robot import RELATIVE_TOOLS
except ImportError:
    from evidence import finite,transform
    from runtime import Runtime
    from robot import RELATIVE_TOOLS


CONTRACT='openarm-relative-v1/1'


def bounded_target(current,target,step_m=.006,rotation_step_rad=.03):
    """Bound translation AND rotation from measured pose; no IK/result clipping."""
    from scipy.spatial.transform import Rotation
    a=transform(current);b=transform(target)
    if not 0<step_m<=.006 or not 0<rotation_step_rad<=.03:
        raise ValueError('relative correction bounds exceeded')
    delta=b[:3,3]-a[:3,3];rot=Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).as_rotvec()
    ratio=min(1.,step_m/max(float(np.linalg.norm(delta)),1e-12),
              rotation_step_rad/max(float(np.linalg.norm(rot)),1e-12))
    out=a.copy();out[:3,3]+=ratio*delta
    out[:3,:3]=a[:3,:3]@Rotation.from_rotvec(ratio*rot).as_matrix()
    return out


def goal_error(current,target):
    from scipy.spatial.transform import Rotation
    a=transform(current);b=transform(target)
    return float(np.linalg.norm(a[:3,3]-b[:3,3])),float(np.linalg.norm(
        Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).as_rotvec()))


def transfer_goal(bottle,start_tools,offset_m):
    """Common frame anchored to this session's measured ACT-start tool midpoint."""
    out=transform(bottle).copy()
    midpoint=.5*(transform(start_tools['left'])[:3,3]+transform(start_tools['right'])[:3,3])
    out[:3,3]=midpoint+finite(offset_m,(3,))
    return out


class RelativeRuntime(Runtime):
    tools=RELATIVE_TOOLS

    def __init__(self,root,record):
        super().__init__(root,record)
        self.relative=self.cal['motion']['relative']
        self.start_tools=None;self.transfer_target=None;self.donor_withdrawn=False

    def begin_session(self):
        super().begin_session()
        self.start_tools=copy.deepcopy(self.planner.tools(self.servo.state['q']))
        self.transfer_target=None;self.donor_withdrawn=False
        self.record('relative_session_anchor',{'source':'measured_FK_after_ACT_initialization',
            'tools':{s:t.tolist() for s,t in self.start_tools.items()}})

    def _checked_bottle(self,state,anchor):
        pose=transform(state['object_poses']['bottle'])
        distance,angle=goal_error(anchor,pose)
        if distance>self.relative['maximum_target_shift_m'] or angle>.15:
            raise RuntimeError('object moved outside relative refinement envelope; fresh proposal required')
        return pose

    def _relative_step(self,side,target,*,contact=()):
        # Feedback is measured q, never the previous requested endpoint.
        current=self.planner.tools(self.servo.state['q'])[side]
        distance,angle=goal_error(current,target)
        self.record('relative_goal',{'arm':side,'source':'fresh_real_object_pose_and_FK',
            'target':target.tolist(),'measured':current.tolist(),
            'position_error_m':distance,'rotation_error_rad':angle,
            'maximum_step_m':self.relative['correction_step_m']})
        if (distance<=self.relative['correction_tolerance_m'] and
            angle<=self.relative['rotation_tolerance_rad']):return True
        step=bounded_target(current,target,self.relative['correction_step_m'])
        self._move({side:step},contact=contact)
        # Require another fresh observation before declaring arrival.
        return False

    def align_relative(self,ident):
        state=self.observe()
        if state['held_by']:raise RuntimeError('relative grasp alignment requires empty hands')
        if ident!=self.current_grasp or ident not in self.candidates:raise ValueError('active grasp candidate required')
        row=next(r for r in self.cal['motion']['right_grasps'] if r['id']==ident)
        pose=self._checked_bottle(state,self.candidates[ident]['bottle'])
        target=pose@transform(row['T_object_tool'])
        done=self._relative_step('right',target,contact=('right',))
        return self._result(done=done)

    def start_transfer(self,placement_aware=True):
        result=super().start_transfer(placement_aware)
        if self.start_tools is None:raise RuntimeError('measured session anchor missing')
        self.transfer_target=transfer_goal(self.observe()['object_poses']['bottle'],
            self.start_tools,self.relative['handover_offset_m'])
        self.record('relative_transfer_anchor',{'T_base_bottle':self.transfer_target.tolist(),
            'source':'session_start_tool_midpoint_plus_measured_offset'})
        return result

    def close_gripper(self,arm='right'):
        if arm=='right':
            state=self.observe()
            if self.current_grasp not in self.candidates:raise RuntimeError('relative grasp candidate missing')
            row=next(r for r in self.cal['motion']['right_grasps'] if r['id']==self.current_grasp)
            bottle=self._checked_bottle(state,self.candidates[self.current_grasp]['bottle'])
            target=bottle@transform(row['T_object_tool'])
            distance,angle=goal_error(self.planner.tools(self.servo.state['q'])['right'],target)
            if distance>self.relative['correction_tolerance_m'] or angle>self.relative['rotation_tolerance_rad']:
                raise RuntimeError('fresh relative alignment required before right closure')
        return super().close_gripper(arm)

    def advance(self):
        if not self.operation:raise RuntimeError('no active operation')
        op=self.operation;phase=op['phase'];state=self.observe()
        tools=self.planner.tools(self.servo.state['q'])
        bottle=transform(state['object_poses']['bottle'])
        if op['kind']=='transfer':
            if phase=='position':
                if self.transfer_target is None:raise RuntimeError('relative transfer anchor missing')
                right=self.transfer_target@np.linalg.inv(bottle)@tools['right']
                left=self.transfer_target@transform(self.cal['motion']['T_bottle_left_tool'])
                left[:3,3]-=left[:3,2]*self.cal['motion']['pregrasp_m']
                # Entire dual-arm path, with carried object, must pass FCL.
                self._move({'right':right,'left':left},contact=('right',))
                op['phase']='receiver_approach'
                return self._result(done=False)
            if phase=='receiver_approach':
                target=bottle@transform(self.cal['motion']['T_bottle_left_tool'])
                if self._relative_step('left',target,contact=('left','right')):
                    op['phase']='receiver_close'
                return self._result(done=False)
            return super().advance()
        if not self.donor_withdrawn:raise RuntimeError('verified donor withdrawal required before placement')
        if phase=='lower':
            desired=transform(state['object_poses']['basket'])@transform(self.cal['motion']['T_basket_bottle'])
            target=desired@np.linalg.inv(bottle)@tools['left']
            if self._relative_step('left',target,contact=('left',)):op['phase']='release_left'
            return self._result(done=False)
        return super().advance()

    def withdraw_donor(self):
        state=self.observe()
        if state['held_by']!=['left'] or self.operation is not None:
            raise RuntimeError('left-only verified hold after transfer required')
        target=self.planner.tools(self.servo.state['q'])['right'].copy()
        target[:3,3]+=finite(self.relative['donor_retreat_base_m'],(3,))
        self._move({'right':target},contact=('left','right'))
        # Left hold is continuously monitored by the inherited provider.
        if self.observe()['held_by']!=['left']:raise RuntimeError('receiver hold lost during donor withdrawal')
        self.donor_withdrawn=True
        return self._result()


def description():
    return {'skill':'relative-bottle/1','robot_contract':CONTRACT,
        'strategy':['fresh object-relative grasp','bounded 6mm final correction',
                    'measured session-relative handover','verified receiver before donor release',
                    'basket-relative placement with measured tool-object transform'],
        'simulation_source':'adaptive visual relay v4',
        'identical_simulation_controller':False,'synthetic_cameras_included':False,
        'physical_validation_passed':False,'devices_started':False,'motion_commands':0}
