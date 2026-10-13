"""Object-centered cylindrical grasp proposals with connected native stages.

Only explicitly parameterized pick/reach uses this extension. The fixed TAMP
demonstration remains the original implementation. Shape/pose are simulator
geometry, not visual perception; all live motion stays actuator/contact driven.
"""
import copy
import numpy as np
import mujoco
from .tamp import TampPlanner,PlanningFailure

CONTRACT={'schema':'geometric-grasp-pool/1','source':'simulator_ground_truth',
    'orientation_sampling':'24 full-circle approach azimuths; pad pitch0,-15,-30,+15degrees',
    'shape_sampling':'current collision-cylinder centers, highest first; no fixed world grasp position',
    'composition':'grasp IK -> warm-started pregrasp IK -> native connecting path and approach edge',
    'native_acceptance':'original IK, collision, target tracking, sustained two-finger contact and lift',
    'scope':'known upright cylindrical object geometry on candidate OpenArm; no open-object perception',
    'hardware_access':False,'transfer_ready':False,
    'reference':'https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html'}


def cylinder_bands(model,data,body):
    rows=[]
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom])!=body or not model.geom_contype[geom]:continue
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_CYLINDER:continue
        axis=data.geom_xmat[geom].reshape(3,3)[:,2].copy()
        rows.append({'geom':model.geom(geom).name,'center_m':data.geom_xpos[geom].copy(),
            'axis':axis,'radius_m':float(model.geom_size[geom,0]),'half_height_m':float(model.geom_size[geom,1])})
    return sorted(rows,key=lambda b:-b['center_m'][2])


def frames(axis,theta,pitch):
    base=TampPlanner._grasp_frame(axis,theta)
    c,s=np.cos(pitch),np.sin(pitch)
    return base@np.array([[c,0,s],[0,1,0],[-s,0,c]])


def candidates(bands):
    # Prefer the naturally nearer approach quadrant first, but cover all360deg.
    angles=np.deg2rad([-45,-60,-30,-75,-15,-90,0,-105,15,-120,30,-135,45,-150,60,-165,75,180,90,165,105,150,120,135])
    for band in bands:
        for pitch in np.deg2rad([0,-15,-30,15]):
            for theta in angles:
                rotation=frames(band['axis'],theta,pitch)
                grasp=band['center_m'].copy();pre=grasp-.10*rotation[:,2]
                yield {'geom':band['geom'],'grasp_m':grasp,'pregrasp_m':pre,'rotation':rotation,
                    'yaw_rad':float(theta),'pitch_rad':float(pitch),'radius_m':band['radius_m']}


class GeometricPlanner(TampPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.parameterized_pick=bool(kwargs.get('parameters')) and self.task in ('pick_bottle','reach_bottle')
        if self.parameterized_pick:self.result['grasp_pool_contract']=CONTRACT

    def _choose_pick_pose(self,bottle):
        rejected=[];saved_seeds=copy.deepcopy(self.ik_seeds[self.side])
        live_q=self.env.data.qpos.copy();live_v=self.env.data.qvel.copy()
        try:
            self.planning_qpos=live_q.copy()
            for finger in self.env.fingers[self.side]:self.planning_qpos[finger.qposadr[0]]=max(finger.range)
            self._sync();bands=cylinder_bands(self.env.model,self.data,self.env.bottle_id)
            for index,candidate in enumerate(candidates(bands)):
                self._check();self.rotation=candidate['rotation'];self.ik_seeds[self.side]=copy.deepcopy(saved_seeds)
                try:
                    grasp_q=self._ik(candidate['grasp_m'],True)
                    self.ik_seeds[self.side].insert(0,grasp_q)
                    pre_q=self._ik(candidate['pregrasp_m'])
                    self._path(pre_q)
                    if not self._edge(pre_q,grasp_q,True):
                        raise PlanningFailure('approach_collision','预抓取到夹持位姿的连接路径存在碰撞')
                    self.result['grasp_search']={'contract':CONTRACT,'arm':self.side,'candidate_index':index,
                        'selected':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in candidate.items()},
                        'rejected':rejected,'pregrasp_q':pre_q.tolist(),'grasp_q':grasp_q.tolist()}
                    self.result.pop('planning_diagnostics',None)
                    # A proposal remains an IK seed. Live segments are still
                    # re-grounded/replanned after opening and each native move.
                    self.ik_seeds[self.side]=[grasp_q,pre_q,*saved_seeds]
                    return candidate['grasp_m'].copy(),candidate['pregrasp_m'].copy()
                except PlanningFailure as exc:
                    if exc.code in ('cancelled','planning_timeout'):raise
                    rejected.append({'index':index,'geom':candidate['geom'],'yaw_rad':candidate['yaw_rad'],
                        'pitch_rad':candidate['pitch_rad'],'code':exc.code,
                        'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['grasp_search']={'contract':CONTRACT,'arm':self.side,'rejected':rejected}
            raise PlanningFailure('grasp_search_exhausted','已声明几何候选未找到完整无碰撞抓取路线；这不是物理不可达的证明')
        finally:
            self.planning_qpos=None
            np.testing.assert_array_equal(live_q,self.env.data.qpos)
            np.testing.assert_array_equal(live_v,self.env.data.qvel)

    def _pick(self):
        if not self.parameterized_pick:
            yield from super()._pick();return
        side=self.pick_arm;self._arm(side)
        initial_bottom=self.env._bottle_bounds()[0][2]
        grasp,pre=self._choose_pick_pose(self.env.data.xpos[self.env.bottle_id].copy())
        self._phase('approach');yield from self._grip(side,self.env.grippers[side].open_motor)
        yield from self._move(pre)
        if self.task=='reach_bottle':return
        self._phase('grasp');yield from self._move(grasp)
        self._phase('close');yield from self._grip(side,.05)
        if not self.env.evaluation['held'][side]:raise PlanningFailure('grasp_not_confirmed','闭合后未形成持续双指接触，未开始抬升')
        hand=self.env.data.xmat[self.body].reshape(3,3)
        self.carried=(hand.T@(self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
                      hand.T@self.env.data.xmat[self.env.bottle_id].reshape(3,3))
        self._phase('lift');self._sync();point,_=self._pose()
        yield from self._move(point+np.array([0.,0.,.12]),True)
        self._phase('hold')
        for _ in range(30):
            yield self.base_action.copy(),{'phase':'hold','scene_source':'simulator_ground_truth'}
            if not self.env.evaluation['held'][side]:raise PlanningFailure('grasp_lost','抬升后未保持持续双指夹持')
        bottom=self.env._bottle_bounds()[0][2]
        if bottom<initial_bottom+.06:raise PlanningFailure('lift_not_confirmed','物体未实际抬离初始支撑面达到原验收高度')
        self.result.update(arm=side,verification_source='native_two_finger_contact_and_lift',
            held_by=side,lift_height_m=float(bottom-initial_bottom))


def install_isolated():
    import sim.service as service
    from harness.composite import ReportingRelocation
    service.TampPlanner=GeometricPlanner
    class GeometricRelocation(ReportingRelocation):
        def _skill(self,name,parameters,planner_type=TampPlanner):
            yield from super()._skill(name,parameters,
                GeometricPlanner if planner_type is TampPlanner else planner_type)
    service.RelocationPlanner=GeometricRelocation
