"""cuRobo V2 global planning for the real harness. Plans only, on the GPU.

Used when the local IK search fails (e.g. the wrists start near their limits).
The caller retimes the geometric path and Planner.validate_path (FCL on mesh
hulls) must still accept it before any dispatch. Lock-refresh helpers follow
sim/curobo_cache.py from the simulation platform.
"""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .evidence import finite,transform
except ImportError:
    from evidence import finite,transform


SIDES={'left':0,'right':8}


def joints(side):return ['openarm_%s_joint%d'%(side,i) for i in range(1,8)]


def finger_axes(urdf):
    out={}
    for j in ET.parse(urdf).getroot().findall('joint'):
        child=j.find('child').get('link');limit=j.find('limit');axis=j.find('axis')
        if child.endswith('_finger') and limit is not None and axis is not None:
            a=np.array([float(v) for v in axis.get('xyz').split()]);out[child]=(a/np.linalg.norm(a),float(limit.get('upper')))
    return out


def box_spheres(T,size,radius):
    """Spheres whose union contains a box of `size` centred in frame T."""
    size=np.asarray(size,float);counts=np.maximum(1,np.ceil(size/(radius*1.2)).astype(int));out=[]
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                local=(np.array([i,j,k])+.5)/counts*size-size/2;cell=size/counts
                out.append({'center':(T[:3,:3]@local+T[:3,3]).tolist(),'radius':float(np.linalg.norm(cell)/2)})
    return out


def robot_dict(base,urdf,calibration):
    """Swept fingers (full URDF travel) and the measured wrist camera box."""
    k=copy.deepcopy(base['kinematics']);spheres=k['collision_spheres']
    for link,(axis,travel) in finger_axes(urdf).items():
        rows=spheres.get(link,[]);extra=[]
        for u in (.25,.5,.75,1.):
            extra+=[{'center':(np.asarray(r['center'])+axis*travel*u).tolist(),'radius':r['radius']} for r in rows]
        spheres[link]=rows+extra
    cam=calibration.get('camera',{})
    size=calibration.get('calibration_aids',{}).get('ruler',{}).get('camera_size_mm')
    if cam.get('T_link_camera') and size:
        w,h,d=(v/1000 for v in size);T=transform(cam['T_link_camera']).copy();T[:3,3]+=T[:3,:3]@[0,0,-d/2]
        # Hang the camera on the rigid parent of the hand (fixed joint), so the
        # handover hand-vs-hand ignore list never hides the camera itself.
        link=cam['mounted_link']
        for j in ET.parse(urdf).getroot().findall('joint'):
            if j.find('child').get('link')==link and j.get('type')=='fixed':
                from scipy.spatial.transform import Rotation
                o=j.find('origin');parent=np.eye(4)
                if o is not None:
                    parent[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in o.get('rpy','0 0 0').split()]).as_matrix()
                    parent[:3,3]=[float(v) for v in o.get('xyz','0 0 0').split()]
                T=parent@T;link=j.find('parent').get('link');break
        spheres.setdefault(link,[]).extend(box_spheres(T,[w+.01,h+.01,d+.01],.02))
    k['lock_joints']=None;k['extra_links']={};k['extra_collision_spheres']=None
    return {'kinematics':k}


def bottle_spheres(T_tool_object,obj):
    """Axial slabs of the upright cylinder, each enclosed by one sphere (tool frame).
    The cover protrudes about one radius past the ends: global moves need the
    bottle clear of supports; lifting off and setting down stay on local steps."""
    radius=obj['body_radius_m'];height=obj['body_height_m'];count=max(1,min(32,int(np.ceil(height/.012))))
    length=height/count;T=transform(T_tool_object);out=[]
    for i in range(count):
        p=T@np.array([0,0,obj['body_bottom_m']+(i+.5)*length,1])
        out.append([*p[:3].tolist(),float(np.hypot(radius,length/2))+.003])
    return np.array(out,np.float32)


def cuboids(scene,*,contact,carried,bottle=None,obj=None,basket=None):
    """Scene capsules -> enclosing oriented cuboids; table as a slab. A bottle
    held by the locked arm is an obstacle unless this move may touch it."""
    # The torso and shoulder mounts stand below the table plane, so the slab
    # starts in front of them; FCL still checks the full plane for every
    # non-mount link.
    out={'table':{'pose':[1.1,0,scene['table_z_m']-.05,1,0,0,0],'dims':[2.,4.,.1]}}
    rows=list(scene['obstacles'])
    if bottle is not None and not contact:
        a=bottle@np.array([0,0,obj['body_bottom_m'],1]);b=bottle@np.array([0,0,obj['body_bottom_m']+obj['body_height_m'],1])
        rows.append({'id':'held_bottle','capsules':[[a[:3].tolist(),b[:3].tolist(),obj['body_radius_m']+.005]]})
    if basket is not None:
        # Five boxes instead of the many thin bars FCL uses for the same walls.
        from scipy.spatial.transform import Rotation
        T=transform(scene['T_base_basket']);size=np.asarray(basket['inner_size_m'],float);w=2*basket['wall_radius_m'];f=basket['floor_z_m']
        w0,x0,y0,z0=np.roll(Rotation.from_matrix(T[:3,:3]).as_quat(),1)
        boxes={'floor':([0,0,f-w/2],[size[0]+2*w,size[1]+2*w,w])}
        for sign in (-1,1):
            boxes['wall_x%+d'%sign]=([sign*(size[0]+w)/2,0,f+size[2]/2],[w,size[1]+2*w,size[2]])
            boxes['wall_y%+d'%sign]=([0,sign*(size[1]+w)/2,f+size[2]/2],[size[0]+2*w,w,size[2]])
        for name,(c,d) in boxes.items():
            if name=='floor' and scene.get('allow_basket_support'):continue
            p=T@np.r_[c,1];out['basket_'+name]={'pose':[*p[:3].tolist(),w0,x0,y0,z0],'dims':list(map(float,d))}
        rows=[r for r in rows if r['id'] not in ('basket','basket_floor')]
    for row in rows:
        # Fingers may touch the bottle during contact moves; held bottles are attached.
        if row['id']=='bottle' and (carried or contact):continue
        if row['id']=='basket_floor' and scene.get('allow_basket_support'):continue
        for n,(a,b,r) in enumerate(row['capsules']):
            a=np.asarray(a,float);b=np.asarray(b,float);d=b-a;length=float(np.linalg.norm(d))
            z=d/length if length>1e-9 else np.array([0,0,1.])
            helper=np.array([1.,0,0]) if abs(z[0])<.9 else np.array([0.,1,0])
            x=np.cross(helper,z);x/=np.linalg.norm(x);y=np.cross(z,x)
            from scipy.spatial.transform import Rotation
            w,qx,qy,qz=np.roll(Rotation.from_matrix(np.column_stack([x,y,z])).as_quat(),1)
            c=(a+b)/2;out['%s_%d'%(row['id'],n)]={'pose':[*c.tolist(),w,qx,qy,qz],'dims':[2*r,2*r,length+2*r]}
    return {'cuboid':out}


class CuMotion:
    """One cached official MotionPlanner per (arm, carrying, scene)."""
    def __init__(self,root,hardware,calibration):
        import os
        cache=Path(root)/'.runtime/cache'   # keep JIT kernel caches inside the harness
        os.environ.setdefault('WARP_CACHE_PATH',str(cache/'warp'));os.environ.setdefault('CUDA_CACHE_PATH',str(cache/'nv'))
        import torch
        import yaml
        import curobo.runtime as runtime
        runtime.cache_dir=str(Path(root)/'.runtime/cache/curobo')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for cuRobo')
        self.torch=torch;self.cal=calibration;self.hw=hardware
        base=yaml.safe_load((Path(root)/'config/curobo.yml').read_text())
        self.base=robot_dict(base,hardware['urdf'],calibration);self.cache={};self.state={}
        self.links={j.get('name'):{'parent':j.find('parent').get('link'),'child':j.find('child').get('link'),
            'type':j.get('type'),'mimic':None} for j in ET.parse(hardware['urdf']).getroot().findall('joint')}

    def _config(self,side,locked,carried,handover=False):
        c=copy.deepcopy(self.base);k=c['kinematics'];tool=self.cal['motion']['tool_links'][side]
        k['tool_frames']=[tool];k['lock_joints']=dict(locked)
        if handover:
            # Hands meet at the handover; sphere covers are too coarse there,
            # so hand-vs-hand is left to the exact FCL mesh recheck.
            hands=self.cal['motion']['contact_links']
            for a in hands['left']:k['self_collision_ignore'].setdefault(a,[]).extend(hands['right'])
        if carried:
            link='carried_bottle'
            k['extra_links']={link:{'parent_link_name':tool,'link_name':link,'joint_name':'carried_bottle_fixed',
                'joint_type':'FIXED','fixed_transform':[0,0,0,1,0,0,0]}}
            k['extra_collision_spheres']={link:32};k['collision_link_names'].append(link)
            contacts=self.cal['motion']['contact_links'][side];k['self_collision_ignore'][link]=list(contacts)
            for name in contacts:k['self_collision_ignore'].setdefault(name,[]).append(link)
            k['self_collision_buffer'][link]=0.
        return c

    def _planner(self,side,locked,world,carried,handover=False):
        """One planner per (arm, carrying); scenes swap in, locks refresh in place."""
        from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
        from curobo._src.geom.types import SceneCfg
        key=(side,bool(carried),bool(handover))
        if key not in self.cache:
            self.cache[key]=MotionPlanner(MotionPlannerCfg.create(robot=self._config(side,locked,carried,handover),scene_model=world,
                collision_cache={'cuboid':64},num_ik_seeds=64,num_trajopt_seeds=4,use_cuda_graph=False,
                self_collision_check=True,interpolation_dt=1/30))
            self.state[key]=(dict(locked),repr(world))
        planner=self.cache[key];old_locked,old_world=self.state[key]
        if old_world!=repr(world):planner.update_world(SceneCfg.create(copy.deepcopy(world)))
        if old_locked!=locked:self._refresh(planner,side,locked,carried,handover)
        self.state[key]=(dict(locked),repr(world));return planner

    def warm(self,start,table_z):
        """Build all four planners once per session (JIT kernels are disk-cached)."""
        world={'cuboid':{'table':{'pose':[1.1,0,table_z-.05,1,0,0,0],'dims':[2.,4.,.1]}}}
        for side in ('left','right'):
            other='left' if side=='right' else 'right'
            locked={n:float(start[SIDES[other]+k]) for k,n in enumerate(joints(other))}
            for carried in (None,{'arm':side}):self._planner(side,locked,world,carried)
            self._planner(side,locked,world,None,True)

    def _refresh(self,planner,side,locked,carried,handover=False):
        """Update locked-arm fixed transforms in place (no rebuild)."""
        from curobo.kinematics import Kinematics,KinematicsCfg
        from curobo.types import JointState
        full=self._config(side,{},carried,handover);full['kinematics']['lock_joints']=None
        full['kinematics']['tool_frames']=list({r['child'] for r in self.links.values()}|{r['parent'] for r in self.links.values()})
        fk=Kinematics(KinematicsCfg.from_robot_yaml_file(full))
        state=JointState.from_position(self.torch.as_tensor([[locked.get(n,0.) for n in fk.joint_names]],device='cuda',
            dtype=self.torch.float32),joint_names=fk.joint_names)
        poses=fk.compute_kinematics(state).tool_poses;models=[]
        for solver in (planner.ik_solver,planner.trajopt_solver):
            for rollout in solver.get_all_rollout_instances():
                m=rollout.transition_model.robot_model
                if all(m is not x for x in models):models.append(m)
        for model in models:
            fresh=model.config.kinematics_config.clone()
            for name,value in locked.items():
                row=self.links[name];index=fresh.link_name_to_idx_map.get(row['child'])
                if index is None:continue
                t=poses.get_link_pose(row['parent']).inverse().multiply(poses.get_link_pose(row['child'])).get_affine_matrix()
                fresh.fixed_transforms[index].copy_(t.reshape_as(fresh.fixed_transforms[index]))
            fresh.lock_jointstate.position.copy_(self.torch.as_tensor([locked[n] for n in fresh.lock_jointstate.joint_names],
                device='cuda',dtype=fresh.lock_jointstate.position.dtype).reshape_as(fresh.lock_jointstate.position))
            model.update_kinematics_config(fresh)

    def plan_arm(self,start,side,target,scene,bottle_pose=None):
        """Geometric 16-D path for one arm; the other arm stays exactly locked."""
        from curobo.types import JointState,GoalToolPose
        from scipy.spatial.transform import Rotation
        start=finite(start,(16,));i=SIDES[side];active=joints(side)
        other='left' if side=='right' else 'right';j=SIDES[other]
        locked={n:float(start[j+k]) for k,n in enumerate(joints(other))}
        carried=scene.get('carried') if (scene.get('carried') or {}).get('arm')==side else None
        contact=bool(scene.get('contact_links'))
        world=cuboids(scene,contact=contact,carried=carried,bottle=bottle_pose,obj=self.cal['objects']['bottle'],basket=self.cal['objects']['basket'])
        links=set(scene.get('contact_links',[]));hands=self.cal['motion']['contact_links']
        handover=all(hands[s][0] in links for s in ('left','right'))
        planner=self._planner(side,locked,world,carried,handover)
        js=JointState.from_position(self.torch.tensor([[float(start[i+active.index(n)]) for n in planner.joint_names]],
            device='cuda',dtype=self.torch.float32),joint_names=planner.joint_names)
        if carried:
            from curobo._src.collision.attachment_manager import AttachmentManager
            spheres=self.torch.as_tensor(bottle_spheres(carried['T_tool_object'],self.cal['objects']['bottle']),device='cuda')
            seen=set()
            for solver in (planner.ik_solver,planner.trajopt_solver):
                for rollout in solver.get_all_rollout_instances():
                    m=rollout.transition_model.robot_model;ptr=m.config.kinematics_config.link_spheres.data_ptr()
                    if ptr not in seen:seen.add(ptr);AttachmentManager(m).update(spheres,js,link_name='carried_bottle')
        T=transform(target);w,x,y,z=np.roll(Rotation.from_matrix(T[:3,:3]).as_quat(),1)
        goal=GoalToolPose(tool_frames=planner.tool_frames,
            position=self.torch.tensor(T[:3,3].reshape(1,1,1,1,3),device='cuda',dtype=self.torch.float32),
            quaternion=self.torch.tensor(np.array([w,x,y,z]).reshape(1,1,1,1,4),device='cuda',dtype=self.torch.float32))
        result=planner.plan_pose(goal,js,max_attempts=4)
        if result is None or not bool(result.success.any().item()):
            raise RuntimeError('cuRobo found no collision-free trajectory (%s)'%getattr(result,'status','IK failed'))
        plan=result.get_interpolated_plan();names=plan.joint_names
        rows=plan.position.detach().cpu().numpy().reshape(-1,len(names))[:,[names.index(n) for n in active]]
        path=[];base=start.copy()
        for row in rows:
            q=base.copy();q[i:i+7]=row;path.append(q)
        if not 2<=len(path)<=2000 or not np.isfinite(path).all():raise RuntimeError('cuRobo path outside bounds')
        return path


def build(root):
    """config/curobo.yml from the bound URDF meshes. Sphere fits are coarse, so
    link pairs already overlapping at the ACT start pose (FCL-validated, used
    daily) are ignored by cuRobo; the FCL mesh recheck still covers them."""
    import os
    import yaml
    root=Path(root);os.environ.setdefault('WARP_CACHE_PATH',str(root/'.runtime/cache/warp'))
    from curobo.robot_builder import RobotBuilder
    from curobo.kinematics import Kinematics,KinematicsCfg
    from curobo.types import JointState
    import json
    import torch
    hw=json.loads((root/'config/hardware.json').read_text())
    b=RobotBuilder(hw['urdf'],tool_frames=['openarm_left_hand_tcp','openarm_right_hand_tcp'])
    b.fit_collision_spheres(sphere_density=1.);b.compute_collision_matrix(prune_collisions=False)
    out=root/'config/curobo.yml';b.save(b.build(),str(out))
    config=yaml.safe_load(out.read_text());k=config['kinematics']
    kin=Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(config)))
    q=dict(zip(joints('left')+joints('right'),np.r_[hw['startup_target'][0:7],hw['startup_target'][8:15]]))
    js=JointState.from_position(torch.tensor([[q[n] for n in kin.joint_names]],device='cuda',dtype=torch.float32),joint_names=kin.joint_names)
    world=kin.compute_kinematics(js).get_link_spheres().detach().cpu().numpy().reshape(-1,4)
    params=kin.config.kinematics_config;pairs=kin.get_self_collision_config().collision_pairs.detach().cpu().numpy();label={}
    for link in k['collision_link_names']:
        for i in params.get_sphere_index_from_link_name(link).detach().cpu().tolist():label[i]=link
    gap=np.linalg.norm(world[pairs[:,0],:3]-world[pairs[:,1],:3],axis=-1)-world[pairs[:,0],3]-world[pairs[:,1],3]
    added=sorted({tuple(sorted((label[int(a)],label[int(b)]))) for (a,b),g in zip(pairs,gap) if g<0})
    for a,b2 in added:k['self_collision_ignore'].setdefault(a,[]).append(b2)
    out.write_text(yaml.safe_dump(config,sort_keys=False))
    return {'config':str(out),'spheres':b.num_spheres,'ignored_at_start_pose':added}


def resample(path,*,rate=30,velocity=.15,acceleration=.3,jerk=2.):
    """Time-scale a geometric joint path with the same quintic profile as
    motion.retime: zero end velocity/acceleration and joint-rate limits."""
    p=np.asarray(path,float);seg=np.max(abs(np.diff(p,axis=0)),axis=1);s=np.r_[0,np.cumsum(seg)];total=float(s[-1])
    if total<1e-9:return [p[-1].copy()]
    duration=max(.3,1.875*total/velocity,np.sqrt(5.774*total/acceleration),np.cbrt(60*total/jerk))
    ticks=max(2,int(np.ceil(duration*rate)));u=np.arange(1,ticks+1)/ticks;arc=total*(10*u**3-15*u**4+6*u**5)
    out=[np.array([np.interp(a,s,p[:,k]) for k in range(p.shape[1])]) for a in arc[:-1]]
    return out+[p[-1].copy()]


if __name__=='__main__':
    import json
    import sys
    if sys.argv[1:]!=['build']:raise SystemExit('usage: cumotion.py build')
    print(json.dumps(build(Path(__file__).resolve().parents[1]),ensure_ascii=False,indent=2))
