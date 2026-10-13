"""Local cuRobo V2 GPU worker. Returns plans only; never accesses hardware."""
from pathlib import Path
import copy
import hashlib
import json
import os
import logging
import faulthandler
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from .curobo_cache import robot_config,cache_key,kinematic_models,copy_kinematics,joint_links,refresh_from_full_fk

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'.runtime/cache'))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')


def validate_scene(scene):
    if type(scene) is not dict or set(scene)!={'cuboid'} or type(scene['cuboid']) is not dict or len(scene['cuboid'])>32:
        raise ValueError('unsupported scene')
    for item in scene['cuboid'].values():
        if (type(item) is not dict or set(item)!={'pose','dims'} or len(item['pose'])!=7
                or len(item['dims'])!=3 or not np.isfinite(item['pose']+item['dims']).all()
                or min(item['dims'])<=0):
            raise ValueError('invalid scene cuboid')


class Planner:
    def __init__(self):
        import torch
        import curobo.runtime as runtime
        from curobo.kinematics import Kinematics, KinematicsCfg
        runtime.cache_dir = str(ROOT/'.runtime/cache/curobo')
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU or TAMP substitution')
        self.torch = torch
        self.config = json.loads((ROOT/'task/planners/openarm.json').read_text())
        self.names = self.config['robot_cfg']['kinematics']['cspace']['joint_names']
        self.kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(self.config['robot_cfg']),
                             tool_frames=['left_candidate_tcp','right_candidate_tcp']))
        self.cached = None
        self.cached_locks=None
        self.cache_generation=0
        self.warmed_generation=0
        self.refresh_fk=None
        self.joint_links,self.all_links=joint_links(ROOT/'task/planners/openarm.urdf')
        self.metadata = {'backend':'curobo','protocol':'openarm-curobo/2','ready':True,
            'source_revision':'78fd485fa82d9b9a063fb4985e371814587e666a',
            'geometry_sha256':self.geometry_hash(), 'cuda_graph':True,
            'scope':'single-arm Cartesian planning, empty or selected arm carrying declared bottle; native collision recheck mandatory',
            'attachment_contract':'cylinder-volume sphere cover; planning only; no physics weld or payload dynamics model',
            'solver_update_contract':'exact-locked-joints/1',
            'worker_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cache_source_sha256':hashlib.sha256((ROOT/'sim/curobo_cache.py').read_bytes()).hexdigest(),
            'hardware_access':False, 'transfer_ready':False}

    def _solver(self,side,locked,scene,carrying):
        from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
        from curobo.kinematics import KinematicsCfg
        key=cache_key(side,locked,scene,carrying)
        info={'reused':bool(self.cached and self.cached[0]==key),'locked_changed':False,
              'build_seconds':0.,'update_seconds':0.}
        if not info['reused']:
            if self.cached:self.cached[1].destroy()
            self.cached=None;self.cached_locks=None;self.refresh_fk=None;self.warmed_generation=0
            self.torch.cuda.empty_cache()
            start=time.monotonic();config=robot_config(self.config['robot_cfg'],side,locked,carrying)
            planner=MotionPlanner(MotionPlannerCfg.create(robot=config,scene_model=scene,
                num_ik_seeds=32,num_trajopt_seeds=4,use_cuda_graph=True,
                self_collision_check=True,interpolation_dt=1/30))
            self.torch.cuda.synchronize();info['build_seconds']=time.monotonic()-start
            # Compile a full FK model once. All parent/child poses are retained
            # so lock refresh need not regenerate collision pair tables.
            full=robot_config(self.config['robot_cfg'],side,{},carrying)
            full['kinematics']['tool_frames']=self.all_links+list(full['kinematics'].get('extra_links',{}))
            from curobo.kinematics import Kinematics
            try:self.refresh_fk=Kinematics(KinematicsCfg.from_robot_yaml_file(full))
            except Exception:
                planner.destroy();raise
            self.cached=(key,planner);self.cached_locks=dict(locked);self.cache_generation+=1
        elif self.cached_locks!=locked:
            start=time.monotonic();planner=self.cached[1]
            from curobo.types import JointState
            incoming=JointState.from_position(self.torch.as_tensor([[locked.get(n,0.) for n in self.refresh_fk.joint_names]],
                device='cuda',dtype=self.torch.float32),joint_names=self.refresh_fk.joint_names)
            # Active joint values cancel in parent^-1 * child for a locked
            # joint. Using zero for them follows the pinned official loader.
            poses=self.refresh_fk.compute_kinematics(incoming).tool_poses
            models=kinematic_models(planner)
            fresh=refresh_from_full_fk(planner.kinematics,poses,self.joint_links,locked,self.torch)
            pair_hash=lambda model:hashlib.sha256(model.get_self_collision_config().collision_pairs.detach().cpu().numpy().tobytes()).hexdigest()
            before=[pair_hash(model) for model in models]
            try:
                info['kinematic_update']=copy_kinematics(models,fresh)
                if before!=[pair_hash(model) for model in models]:raise ValueError('collision pairs changed during lock refresh')
                info['kinematic_update']['collision_pairs_preserved']=True
                if planner.graph_planner is not None:planner.graph_planner.reset_buffer()
            except Exception:
                planner.destroy();self.cached=None;self.cached_locks=None
                raise
            self.torch.cuda.synchronize();info.update(locked_changed=True,update_seconds=time.monotonic()-start)
            self.cached_locks=dict(locked)
        info.update(generation=self.cache_generation,warmed=self.warmed_generation==self.cache_generation,
            locked_state_sha256=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest())
        return self.cached[1],info

    def _geometry_guard(self,planner,js,data):
        full=self.kin.compute_kinematics(self.state(data)).get_link_spheres().detach().cpu().numpy().reshape(-1,4)
        fullparams=self.kin.config.kinematics_config
        maximum=0.;models=kinematic_models(planner)
        for model in models:
            params=model.config.kinematics_config
            reduced=model.compute_kinematics(js).get_link_spheres().detach().cpu().numpy().reshape(-1,4)
            for link in self.config['robot_cfg']['kinematics']['collision_link_names']:
                a=fullparams.get_sphere_index_from_link_name(link).detach().cpu().numpy()
                b=params.get_sphere_index_from_link_name(link).detach().cpu().numpy()
                if len(a)!=len(b):raise ValueError('reduced robot collision sphere inventory mismatch')
                if len(a):maximum=max(maximum,float(np.max(abs(full[a]-reduced[b]))))
        if maximum>1e-5:raise ValueError('updated locked-arm collision geometry differs from full FK')
        return {'all_rollout_models':len(models),'max_sphere_coordinate_error_m':maximum,
            'scope':'current robot sphere geometry versus full official FK; software consistency, not physical calibration'}

    def prepare(self,data):
        if set(data)!={'arm','joints','scene','carrying'} or type(data['carrying']) is not bool:
            raise ValueError('bounded backend preparation fields required')
        self.state(data);side=data['arm']
        if side not in ('left','right'):raise ValueError('invalid preparation arm')
        validate_scene(data['scene'])
        active=[f'openarm_{side}_joint{i}' for i in range(1,8)]
        locked={n:float(data['joints'][n]) for n in self.names if n not in active}
        started=time.monotonic();planner,info=self._solver(side,locked,data['scene'],data['carrying'])
        if self.warmed_generation!=self.cache_generation:
            if planner.warmup(enable_graph=True,num_warmup_iterations=1) is not True:
                raise ValueError('official planner warmup did not confirm preparation')
            self.torch.cuda.synchronize();self.warmed_generation=self.cache_generation
        info['warmed']=True
        return {'prepared':True,'cache':info,'seconds':time.monotonic()-started,'metadata':self.metadata,
            'scope':'compile selected planner topology only; no path dispatch, grasp, object attachment in physics or task success',
            'hardware_access':False,'transfer_ready':False}

    @staticmethod
    def geometry_hash():
        return hashlib.sha256(b''.join((ROOT/'task/planners'/f).read_bytes()
                                      for f in ('openarm.json','openarm.urdf'))).hexdigest()

    def state(self, data):
        from curobo.types import JointState
        values = data['joints']
        if set(values) != set(self.names):
            raise ValueError('joint inventory differs from reviewed OpenArm geometry')
        q = np.array([values[n] for n in self.kin.joint_names],dtype=np.float32)
        if not np.isfinite(q).all() or np.max(abs(q))>7:
            raise ValueError('invalid joint state')
        return JointState.from_position(self.torch.tensor(q[None],device='cuda'),joint_names=self.kin.joint_names)

    def fk(self, data):
        state = self.kin.compute_kinematics(self.state(data)).tool_poses
        return {'position':state.position.detach().cpu().tolist(),
                'quaternion_wxyz':state.quaternion.detach().cpu().tolist(),
                'tool_frames':self.kin.tool_frames}

    def plan(self, data):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import JointState, GoalToolPose
        logging.info('plan: validate input')
        self.state(data)
        side = data['arm']
        if side not in ('left','right'):
            raise ValueError('invalid arm')
        target = np.asarray(data['position_m'],dtype=float)
        quat = np.asarray(data['quaternion_wxyz'],dtype=float)
        if target.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(np.r_[target,quat]).all() or abs(np.linalg.norm(quat)-1)>.02:
            raise ValueError('invalid goal pose')
        if np.any(abs(target)>2):
            raise ValueError('goal outside simulation bounds')
        scene = data['scene']
        validate_scene(scene)
        attached=data.get('attachment')
        if attached is not None:
            if set(attached)!={'object_id','spheres_tcp','removed_world_geoms','representation'} or attached['object_id']!='bottle':
                raise ValueError('only declared bottle attachment is supported')
            spheres=np.asarray(attached['spheres_tcp'],dtype=np.float32)
            if spheres.ndim!=2 or spheres.shape[1]!=4 or not 1<=len(spheres)<=32 or not np.isfinite(spheres).all() or np.max(abs(spheres[:,:3]))>.5 or np.any(spheres[:,3]<=0) or np.any(spheres[:,3]>.1):
                raise ValueError('invalid bounded attachment spheres')
            names=attached['removed_world_geoms']
            if names!=['bottle_body','bottle_neck','bottle_cap'] or any(n in scene['cuboid'] for n in names):
                raise ValueError('attached bottle must not remain a static world obstacle')
        active = [f'openarm_{side}_joint{i}' for i in range(1,8)]
        locked = {n:float(data['joints'][n]) for n in self.names if n not in active}
        start=time.monotonic()
        planner,cache=self._solver(side,locked,scene,attached is not None)
        js=JointState.from_position(self.torch.tensor([[data['joints'][n] for n in planner.joint_names]],device='cuda'),joint_names=planner.joint_names)
        cache['geometry_guard']=self._geometry_guard(planner,js,data)
        if attached is not None:
            logging.info('plan: update attachment')
            from curobo._src.collision.attachment_manager import AttachmentManager
            seen=set()
            for model in kinematic_models(planner):
                pointer=model.config.kinematics_config.link_spheres.data_ptr()
                if pointer in seen:continue
                seen.add(pointer)
                AttachmentManager(model).update(self.torch.as_tensor(spheres,device='cuda'),js,link_name='carried_bottle')
            cache['attachment_buffers_updated']=len(seen)
            # Development feedback, kept in receipts rather than the main UI.
            params=planner.attachment_manager.kinematics_params
            world=planner.kinematics.compute_kinematics(js).get_link_spheres().detach().cpu().numpy().reshape(-1,4)
            logging.info('plan: attachment world spheres available')
            ids=params.get_sphere_index_from_link_name('carried_bottle').detach().cpu().numpy()
            carried=world[ids];carried=carried[carried[:,3]>0]
            overlaps=[]
            for link in self.config['robot_cfg']['kinematics']['collision_link_names']:
                index=params.get_sphere_index_from_link_name(link).detach().cpu().numpy()
                robot=world[index];robot=robot[robot[:,3]>0]
                if not len(robot):continue
                gap=np.linalg.norm(carried[:,None,:3]-robot[None,:,:3],axis=-1)-carried[:,None,3]-robot[None,:,3]
                if gap.min()<0:overlaps.append({'link':link,'sphere_overlap_m':float(-gap.min())})
            attach_debug={'world_spheres':carried.tolist(),'overlapping_robot_spheres':overlaps}
            pairs=planner.kinematics.get_self_collision_config().collision_pairs.detach().cpu().numpy()
            gaps=np.linalg.norm(world[pairs[:,0],:3]-world[pairs[:,1],:3],axis=-1)-world[pairs[:,0],3]-world[pairs[:,1],3]
            attach_debug['checked_self_overlap_m']=float(max(0,-gaps.min()))
            labels={}
            for link in [*self.config['robot_cfg']['kinematics']['collision_link_names'],'carried_bottle']:
                for index in params.get_sphere_index_from_link_name(link).detach().cpu().tolist():labels[index]=link
            attach_debug['checked_overlapping_pairs']=[{'links':[labels.get(int(pairs[i,0])),labels.get(int(pairs[i,1]))],
                'depth_m':float(-gaps[i])} for i in np.argsort(gaps)[:10] if gaps[i]<0]
            logging.info('plan: attachment diagnostics complete')
        goal=GoalToolPose(tool_frames=planner.tool_frames,
            position=self.torch.tensor(target.reshape(1,1,1,1,3),device='cuda',dtype=self.torch.float32),
            quaternion=self.torch.tensor(quat.reshape(1,1,1,1,4),device='cuda',dtype=self.torch.float32))
        logging.info('plan: optimize trajectory')
        result=planner.plan_pose(goal,js,max_attempts=3)
        logging.info('plan: trajectory result available')
        success=bool(result is not None and result.success.any().item())
        response={'success':success,'seconds':time.monotonic()-start,'metadata':self.metadata,'cache':cache,
                  'status':'completed' if success else str(getattr(result,'status','IK found no feasible seed')),'joint_names':planner.joint_names}
        if attached is not None:
            response['attachment']={'object_id':'bottle','sphere_count':len(spheres),
                'removed_world_geoms':attached['removed_world_geoms'],'scope':'GPU collision prediction only; native path and contact verification required',
                'diagnostics':attach_debug}
        if result is None:
            logging.info('plan: diagnostic IK')
            diagnostic=planner.ik_solver.solve_pose(goal,return_seeds=4,current_state=js)
            response['ik_diagnostics']={k:getattr(diagnostic,k).detach().cpu().tolist()
                for k in ('success','position_error','rotation_error','feasible') if getattr(diagnostic,k,None) is not None}
            try:
                logging.info('plan: diagnostic constraint metrics')
                metrics=planner.ik_solver.metrics_rollout.compute_metrics_from_action(diagnostic.optimized_seeds.reshape(-1,1,len(active)))
                blocks=metrics.costs_and_constraints
                response['ik_diagnostics']['constraints']={kind:{name:{'minimum':float(value.detach().min().cpu()),'maximum':float(value.detach().max().cpu())}
                    for name,value in zip(getattr(blocks,kind).names,getattr(blocks,kind).values)}
                    for kind in ('constraints','hybrid_costs_constraints')}
            except Exception as exc:
                response['ik_diagnostics']['metrics_diagnostic_error']=str(exc)
            logging.info('plan: diagnostic IK complete')
        if success:
            interpolated=result.get_interpolated_plan()
            all_path=interpolated.position.detach().cpu().numpy()
            all_names=interpolated.joint_names
            if all_names is None or len(all_names)!=all_path.shape[-1] or not set(active)<=set(all_names):
                raise ValueError('interpolated trajectory joint inventory mismatch')
            path=all_path.reshape(-1,len(all_names))[:,[all_names.index(n) for n in planner.joint_names]]
            if not 2<=len(path)<=1200 or not np.isfinite(path).all():
                raise ValueError('planner path exceeds bounded execution contract')
            response.update(path=path.tolist(),dt=1/30)
        logging.info('plan: response ready')
        return response


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data):
        payload=json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(payload)));self.end_headers()
        try:self.wfile.write(payload)
        except (BrokenPipeError,ConnectionResetError):pass

    def allowed(self):
        return self.client_address[0]=='127.0.0.1' and not self.headers.get('Origin')

    def do_GET(self):
        if not self.allowed():return self.reply(403,{'error':'local worker only'})
        if self.path=='/metadata':return self.reply(200,self.server.planner.metadata)
        self.reply(404,{'error':'unknown route'})

    def do_POST(self):
        if not self.allowed():return self.reply(403,{'error':'local worker only'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=65536 or self.headers.get('Content-Type')!='application/json':
                raise ValueError('bounded JSON required')
            data=json.loads(self.rfile.read(size))
            if self.path=='/prepare':result=self.server.planner.prepare(data)
            elif self.path=='/fk':result=self.server.planner.fk(data)
            elif self.path=='/plan':result=self.server.planner.plan(data)
            else:return self.reply(404,{'error':'unknown route'})
            self.reply(200,result)
        except Exception as exc:
            self.reply(400,{'error':str(exc)})

    def log_message(self,*args):pass


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    server=HTTPServer(('127.0.0.1',8002),Handler)
    try:
        server.planner=Planner()
        print('OpenArm cuRobo worker ready: 127.0.0.1:8002',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
