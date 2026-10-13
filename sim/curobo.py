"""cuRobo plans, native collision validation and actuator execution."""
import hashlib
import json
import time
import urllib.request
import urllib.error

import mujoco
import numpy as np

from .environment import ROOT
from .tamp import TampPlanner, PlanningFailure


def call(path, data=None, timeout=60):
    request=urllib.request.Request('http://127.0.0.1:8002'+path,
        None if data is None else json.dumps(data,allow_nan=False).encode(),
        {} if data is None else {'Content-Type':'application/json'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=timeout) as response:
            payload=response.read(1024*1024+1)
            if len(payload)>1024*1024:raise ValueError('cuRobo response exceeds limit')
            return json.loads(payload)
    except urllib.error.HTTPError as exc:
        raise PlanningFailure('curobo_rejected',exc.read(4096).decode()) from None
    except (OSError,ValueError) as exc:
        raise PlanningFailure('curobo_unavailable',str(exc)) from None


def metadata():
    try:
        result=call('/metadata',timeout=.5)
        expected=hashlib.sha256(b''.join((ROOT/'task/planners'/f).read_bytes()
                                       for f in ('openarm.json','openarm.urdf'))).hexdigest()
        if result.get('protocol')!='openarm-curobo/2' or result.get('geometry_sha256')!=expected:
            raise ValueError('cuRobo geometry/version mismatch')
        for field,file in [('worker_source_sha256','sim/curobo_server.py'),('cache_source_sha256','sim/curobo_cache.py')]:
            if result.get(field)!=hashlib.sha256((ROOT/file).read_bytes()).hexdigest():
                raise ValueError('cuRobo worker source changed; restart idle worker before using new interfaces')
        return result
    except (PlanningFailure,ValueError,OSError) as exc:
        return {'backend':'curobo','ready':False,'reason':str(exc)}


def joint_state(env, data=None):
    data=env.data if data is None else data
    names=json.loads((ROOT/'task/planners/openarm.json').read_text())['robot_cfg']['kinematics']['cspace']['joint_names']
    return {n:float(data.qpos[env.model.joint(n).qposadr[0]]) for n in names}


def scene(env, *, carrying=False):
    cuboid={}
    for g in range(env.model.ngeom):
        body=int(env.model.geom_bodyid[g]);name=env.model.body(body).name or ''
        if carrying and body==env.bottle_id:continue
        if name.startswith('openarm_') or not (env.model.geom_contype[g] or env.model.geom_conaffinity[g]):continue
        kind=env.model.geom_type[g];size=env.model.geom_size[g]
        if kind==mujoco.mjtGeom.mjGEOM_PLANE:
            dims=[4.,4.,.05];pos=[0.,0.,-.025]
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            dims=(2*size).tolist();pos=env.data.geom_xpos[g].tolist()
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            dims=[2*size[0],2*size[0],2*size[1]];pos=env.data.geom_xpos[g].tolist()
        else:raise PlanningFailure('unsupported_geometry','cuRobo obstacle adapter supports box, cylinder bounds and floor only')
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,env.data.geom_xmat[g])
        cuboid[env.model.geom(g).name or str(g)]={'dims':dims,'pose':pos+quat.tolist()}
    return {'cuboid':cuboid}


def attachment(env, side):
    """Conservative cylinder-volume covers, expressed in the measured TCP frame."""
    hand=env.data.xmat[env.ee_ids[side]].reshape(3,3)
    tcp=env.data.xpos[env.ee_ids[side]]+hand@np.array([0,.006,.075])
    spheres=[];names=[]
    for g in env.bottle_geoms:
        if env.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise PlanningFailure('curobo_attachment_geometry','携物接口只支持当前声明的圆柱瓶体组合')
        radius,half=env.model.geom_size[g,:2]
        count=max(1,int(np.ceil(2*half/.012)))
        length=2*half/count
        rotation=env.data.geom_xmat[g].reshape(3,3)
        for i in range(count):
            world=env.data.geom_xpos[g]+rotation@np.array([0,0,-half+(i+.5)*length])
            spheres.append([*(hand.T@(world-tcp)).tolist(),float(np.hypot(radius,length/2))])
        names.append(env.model.geom(g).name)
    if not 1<=len(spheres)<=32:
        raise PlanningFailure('curobo_attachment_geometry','携物碰撞表示超出已声明的球数量')
    return {'object_id':'bottle','spheres_tcp':spheres,'removed_world_geoms':names,
            'representation':'each axial cylinder slab enclosed by a sphere; native mesh path recheck remains mandatory'}


class CuRoboPlanner(TampPlanner):
    CONTRACT={**TampPlanner.CONTRACT,'planner':'official cuRobo V2 GPU IK/trajectory optimization',
        'scope':'reachable single-arm Cartesian goal, empty or selected arm holding declared bottle; candidate simulator geometry',
        'native_path_revalidation':True,'fallback':False}

    def __init__(self,env,task,**kwargs):
        if task not in ('move_to_pose','move_relative'):
            raise PlanningFailure('unsupported_curobo_skill','cuRobo 当前仅开放末端位姿和相对移动')
        selected=(kwargs.get('parameters') or {}).get('arm')
        if selected in ('left','right') and env.evaluation['held']['right' if selected=='left' else 'left']:
            raise PlanningFailure('curobo_other_arm_hold_not_supported','另一只手持物时的锁臂携物碰撞尚未验证；使用 TAMP 或先完成交接')
        super().__init__(env,task,**kwargs)
        self.deadline=time.monotonic()+180
        self.result['backend_metadata']=metadata()
        if not self.result['backend_metadata']['ready']:
            raise PlanningFailure('curobo_unavailable',self.result['backend_metadata']['reason'])
        self.result['plans']=[]

    def _move(self,target,allow_fingers=False):
        command=np.array(target,copy=True)
        command_rotation=self.rotation.copy()
        for correction in range(4):
            self._sync();self._check()
            carrying=self.carried is not None
            if carrying:
                if not self.env.evaluation['held'][self.side]:
                    raise PlanningFailure('grasp_lost','cuRobo 重新规划前未确认持物，未启动新路径')
                hand=self.env.data.xmat[self.body].reshape(3,3)
                self.carried=(hand.T@(self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
                              hand.T@self.env.data.xmat[self.env.bottle_id].reshape(3,3))
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,command_rotation.ravel())
            payload={'arm':self.side,'joints':joint_state(self.env),'scene':scene(self.env,carrying=carrying),
                     'position_m':command.tolist(),'quaternion_wxyz':quat.tolist()}
            if carrying:payload['attachment']=attachment(self.env,self.side)
            # Refuse a software frame mismatch before trusting any plan.
            fk=call('/fk',{'joints':payload['joints']})
            idx=fk['tool_frames'].index(self.side+'_candidate_tcp')
            point,rotation=self._pose()
            fk_error=float(np.linalg.norm(np.asarray(fk['position']).reshape(2,3)[idx]-point))
            self.result['fk_consistency_error_m']=fk_error
            if fk_error>1e-5:raise PlanningFailure('curobo_fk_mismatch','cuRobo 与 MuJoCo 坐标不一致')
            response=call('/plan',payload)
            self._check()
            if carrying and response.get('attachment',{}).get('object_id')!='bottle':
                raise PlanningFailure('curobo_attachment_missing','规划器没有确认瓶子的携物碰撞表示，路径未执行')
            self.result['plans'].append({k:v for k,v in response.items() if k!='path'})
            if not response['success']:
                raise PlanningFailure('curobo_no_path','cuRobo 未找到满足位姿、碰撞和限位的轨迹：'+response['status'])
            names=[f'openarm_{self.side}_joint{i}' for i in range(1,8)]
            if set(response['joint_names'])!=set(names):raise PlanningFailure('curobo_joint_mismatch','cuRobo 返回关节不匹配')
            path=np.asarray(response['path'],dtype=float)[:,[response['joint_names'].index(n) for n in names]]
            bounds=np.array([self.env.model.joint(n).range for n in names])
            if path.ndim!=2 or path.shape[1]!=7 or not 2<=len(path)<=1200 or not np.isfinite(path).all() or np.any(path<bounds[:,0]-1e-5) or np.any(path>bounds[:,1]+1e-5):
                raise PlanningFailure('curobo_invalid_path','cuRobo 返回轨迹不满足关节契约')
            measured=self.env.data.qpos[self.qids].copy()
            if np.max(abs(path[0]-measured))>.02:
                raise PlanningFailure('curobo_start_mismatch','cuRobo 轨迹起点与实测关节不一致')
            previous=measured
            for q in path:
                if not self._edge(previous,q,carrying):
                    self.result['native_blocked_pair']=self.last_collision
                    raise PlanningFailure('native_collision_rejected','MuJoCo 复核拒绝 cuRobo 轨迹；未执行')
                previous=q
            self._sync()
            for start,end in zip(path[:-1],path[1:]):
                count=max(1,int(np.ceil(1/self.speed_scale)),int(np.ceil(np.max(abs(end-start))/.018)))
                for i in range(1,count+1):
                    self._check()
                    action=self.base_action.copy();action[self.start:self.start+7]=start+(end-start)*i/count
                    yield action,{'phase':'move','planner':'curobo','path_points':len(path),'feedback_correction':correction,'scene_source':'simulator_ground_truth'}
                    yield from self._check_hold(action)
                self.base_action[self.start:self.start+7]=end
            for _ in range(15):
                yield self.base_action.copy(),{'phase':'move','settling':True,'planner':'curobo'}
                yield from self._check_hold(self.base_action)
            self._sync();point,rotation=self._pose();error=target-point
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,(self.rotation@rotation.T).ravel())
            angular=np.empty(3);mujoco.mju_quat2Vel(angular,quat,1.)
            self.result['feedback_position_error_m']=float(np.linalg.norm(error))
            self.result['feedback_rotation_error_rad']=float(np.linalg.norm(angular))
            if np.linalg.norm(error)<=.012 and np.linalg.norm(angular)<=.06:return
            command+=np.clip(error,-.04,.04)
            angle=float(np.linalg.norm(angular))
            if angle>1e-8:
                quat=np.empty(4);matrix=np.empty(9)
                mujoco.mju_axisAngle2Quat(quat,angular/angle,min(angle,.15))
                mujoco.mju_quat2Mat(matrix,quat)
                command_rotation=matrix.reshape(3,3)@command_rotation
        raise PlanningFailure('tracking_error','cuRobo 路径执行后位置或朝向误差仍过大')
