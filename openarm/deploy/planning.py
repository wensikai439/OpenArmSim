"""Device-free FK/IK, conservative mesh envelopes and complete path checks.

No simulator is imported. Mesh envelopes are candidates until a separately
recorded physical geometry trial admits them; tiny URDF placeholder spheres
are never promoted into robot collision geometry.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .evidence import Kinematics,finite,transform,fresh
    from .motion import retime
except ImportError:
    from evidence import Kinematics,finite,transform,fresh
    from motion import retime


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def segment_distance(a,b,c,d):
    """Shortest distance between two closed 3D line segments, including points."""
    a,b,c,d=[finite(p,(3,)) for p in [a,b,c,d]]
    u=b-a;v=d-c;w=a-c;aa=u@u;bb=u@v;cc=v@v;dd=u@w;ee=v@w
    if aa<1e-14 and cc<1e-14:return float(np.linalg.norm(w))
    if aa<1e-14:return float(np.linalg.norm(a-(c+np.clip(ee/cc,0,1)*v)))
    if cc<1e-14:return float(np.linalg.norm(a+np.clip(-dd/aa,0,1)*u-c))
    den=aa*cc-bb*bb;s=np.clip((bb*ee-cc*dd)/den,0,1) if den>1e-14 else 0.
    t=(bb*s+ee)/cc
    if t<0:t=0.;s=np.clip(-dd/aa,0,1)
    elif t>1:t=1.;s=np.clip((bb-dd)/aa,0,1)
    return float(np.linalg.norm(w+s*u-t*v))


def mesh_envelopes(hardware):
    """Read existing visual meshes without changing their URDF/source files."""
    import trimesh
    path=Path(hardware['urdf']);root=ET.parse(path).getroot();kin=Kinematics(path)
    rows=[];sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            m=visual.find('geometry/mesh')
            if m is None:continue
            source=Path(m.get('filename'))
            if not source.is_absolute():source=path.parent/source
            if not source.is_file():raise ValueError('mesh missing: '+str(source))
            mesh=trimesh.load(source,force='mesh',process=False)
            vertices=np.array(mesh.vertices,dtype=float)*np.array([float(v) for v in m.get('scale','1 1 1').split()])
            # Preserve visual origin, including any negative mirror scale.
            origin=visual.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
            if origin is not None:
                xyz=np.array([float(v) for v in origin.get('xyz','0 0 0').split()])
                rpy=np.array([float(v) for v in origin.get('rpy','0 0 0').split()])
            from scipy.spatial.transform import Rotation
            vertices=vertices@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz
            center=vertices.mean(axis=0);_,_,axes=np.linalg.svd(vertices-center,full_matrices=False);axis=axes[0]
            along=(vertices-center)@axis
            radial=np.linalg.norm(vertices-center-along[:,None]*axis,axis=1)
            radius=float(radial.max());a=center+along.min()*axis;b=center+along.max()*axis
            if not .001<=radius<=.5:raise ValueError('implausible mesh envelope size')
            rows.append({'link':link.get('name'),'a':a.tolist(),'b':b.tolist(),'radius':radius,
                'mesh':str(source),'encloses_all_vertices':True})
            sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    if len(rows)<20:raise ValueError('incomplete bimanual mesh coverage')
    # Exclude only adjacent rigid bodies here. Extra exclusions require an
    # explicitly reviewed model, not all exclusions from the placeholder URDF.
    # Convex mesh hulls avoid the enormous false overlaps of a single capsule
    # around a curved link. They enclose visual vertices, not just their surface.
    hulls=[]
    for row in rows:
        # Reuse the same visual-origin and mirror transform by converting the
        # source vertices again; no original description files are rewritten.
        link=next(l for l in root.findall('link') if l.get('name')==row['link'])
        visual=next(v for v in link.findall('visual') if v.find('geometry/mesh') is not None)
        m=visual.find('geometry/mesh');mesh=trimesh.load(row['mesh'],force='mesh',process=False)
        mesh.vertices=np.array(mesh.vertices)*np.array([float(v) for v in m.get('scale','1 1 1').split()])
        origin=visual.find('origin');pose=np.eye(4)
        if origin is not None:
            pose[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in origin.get('rpy','0 0 0').split()]).as_matrix()
            pose[:3,3]=[float(v) for v in origin.get('xyz','0 0 0').split()]
        mesh.apply_transform(pose)
        # An outer k-DOP bounds ALL source vertices with a small fixed number
        # of planes. A raw convex hull of the DAE can contain 100k+ triangles.
        from scipy.spatial import HalfspaceIntersection
        from itertools import product
        directions=np.array([v for v in product([-1.,0.,1.],repeat=3) if any(v)])
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        bounds=np.max(np.array(mesh.vertices)@directions.T,axis=0)+.00025
        points=HalfspaceIntersection(np.c_[directions,-bounds],np.array(mesh.vertices).mean(axis=0)).intersections
        hull=trimesh.convex.convex_hull(points)
        hulls.append({'link':row['link'],'vertices':hull.vertices.tolist(),'faces':hull.faces.tolist(),
            'mesh_source':row['mesh'],'mesh_scale':[float(v) for v in m.get('scale','1 1 1').split()],
            'mesh_origin':pose.tolist()})
    parents={name:name for name in [kin.root,*kin.joints]}
    def group(name):
        while parents[name]!=name:name=parents[name]
        return name
    for child,j in kin.joints.items():
        if j['type']=='fixed':parents[group(child)]=group(j['parent'])
    adjacent={tuple(sorted([group(child),group(j['parent'])])) for child,j in kin.joints.items() if j['type']!='fixed'}
    ignored=set();names=[r['link'] for r in rows]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if group(a)==group(b) or tuple(sorted([group(a),group(b)])) in adjacent:ignored.add(tuple(sorted([a,b])))
    return {'schema':'openarm-collision-envelopes/2','qualification':'candidate convex visual hulls; finger sweep and physical geometry still need admission',
        'sources':sources,'capsules':rows,'hulls':hulls,'disabled_pairs':[list(p) for p in sorted(ignored)],'physical_trial_passed':False}


class Planner:
    identity='openarm-measured-fcl-dls/1'
    def __init__(self,hardware,calibration,collision):
        self.hardware=hardware;self.config=calibration;self.model=collision;self.kin=Kinematics(hardware['urdf'])
        self.motion=calibration['motion'];self.links=self.motion['tool_links'];self.margin=float(self.motion['collision_margin_m'])
        self.names={s:['openarm_'+s+'_joint'+str(i) for i in range(1,8)] for s in ['left','right']}
        self.limits={}
        for side in ['left','right']:
            limits=[next(j['limits'] for j in self.kin.joints.values() if j['name']==n) for n in self.names[side]]
            self.limits[side]=np.array(limits).T
            self.limits[side+'_gripper']=calibration['grippers'][side]['motor_limits']
        if not collision.get('capsules'):raise ValueError('real mesh collision envelopes required')
        for path,digest in collision['sources'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('collision mesh/source changed')
        self.disabled={tuple(sorted(p)) for p in collision['disabled_pairs']}
        import fcl
        import trimesh
        self.fcl=fcl;self.hulls=[];self._scene_ref=None;self._obstacles=[];self.global_planner=None;self._walking=False;self.global_attempts=3
        for row in collision['hulls']:
            vertices=np.array(row['vertices'],float);faces=np.array(row['faces'],dtype=np.int64)
            if row.get('jaw_sweep_hull_vertices'):
                sweep=trimesh.convex.convex_hull(np.array(row['jaw_sweep_hull_vertices'],float))
                vertices=np.array(sweep.vertices);faces=np.array(sweep.faces,dtype=np.int64)
            polygons=np.c_[np.full(len(faces),3,dtype=np.int64),faces].flatten()
            if row.get('mesh_source') and not row.get('jaw_sweep_hull_vertices'):
                mesh=trimesh.load(row['mesh_source'],force='mesh',process=False)
                actual=np.array(mesh.vertices,dtype=float)*np.array(row['mesh_scale'])
                origin=transform(row['mesh_origin']);actual=actual@origin[:3,:3].T+origin[:3,3]
                geometry=fcl.BVHModel();geometry.beginModel(len(actual),len(mesh.faces))
                geometry.addSubModel(actual,np.array(mesh.faces,dtype=np.int32));geometry.endModel()
                # The k-DOP hull encloses every mesh vertex: a clear hull proves a clear mesh.
                coarse=fcl.CollisionObject(fcl.Convex(vertices,len(faces),polygons))
            else:geometry=fcl.Convex(vertices,len(faces),polygons);coarse=None
            self.hulls.append((row['link'],vertices,fcl.CollisionObject(geometry),coarse))

    def joints(self,q):
        q=finite(q,(16,));return {n:float(q[i+k]) for side,i in [('left',0),('right',8)] for k,n in enumerate(self.names[side])}

    def tools(self,q):
        values=self.joints(q);cache={};return {s:self.kin.pose(link,values,cache) for s,link in self.links.items()}

    def capsules(self,q,scene):
        values=self.joints(q);out=[];cache={}
        for row in self.model['capsules']:
            pose=self.kin.pose(row['link'],values,cache)
            out.append((row['link'],*( (pose@np.r_[row[k],1])[:3] for k in ['a','b']),row['radius']))
        carried=scene.get('carried')
        if carried:
            pose=self.tools(q)[carried['arm']]@transform(carried['T_tool_object']);obj=self.config['objects']['bottle']
            bottom=obj['body_bottom_m'];height=obj['body_height_m'];radius=obj['body_radius_m']
            out.append(('carried_bottle',(pose@np.array([0,0,bottom,1]))[:3],
                (pose@np.array([0,0,bottom+height,1]))[:3],radius))
        return out

    def valid(self,q,scene):
        q=finite(q,(16,))
        for side,i in [('left',0),('right',8)]:
            lo,hi=self.limits[side]
            if np.any(q[i:i+7]<lo) or np.any(q[i:i+7]>hi):return False
            if not self.limits[side+'_gripper'][0]<=q[i+7]<=self.limits[side+'_gripper'][1]:return False
        tools=self.tools(q);lo,hi=self.motion['workspace_bounds_m']
        if any(np.any(p[:3,3]<lo) or np.any(p[:3,3]>hi) for p in tools.values()):return False
        contact=set(scene.get('contact_links',[]));values=self.joints(q);cache={};shapes=[]
        for name,vertices,shape,coarse in self.hulls:
            t=self.kin.pose(name,values,cache);pose=self.fcl.Transform(t[:3,:3],t[:3,3]);shape.setTransform(pose)
            if coarse is not None:coarse.setTransform(pose)
            points=vertices@t[:3,:3].T+t[:3,3];shapes.append((name,shape,points.min(axis=0),points.max(axis=0),coarse))
        carried_pose=None;carried_bottom=None;basket_support=False
        if scene.get('carried'):
            carried_pose=tools[scene['carried']['arm']]@transform(scene['carried']['T_tool_object']);obj=self.config['objects']['bottle']
            end=carried_pose@np.array([0,0,obj['body_bottom_m'],1]);axis=carried_pose[:3,2]
            carried_bottom=float(min(end[2],end[2]+axis[2]*obj['body_height_m'])-obj['body_radius_m']*np.sqrt(max(0.,1-axis[2]**2)))
            rel=np.linalg.inv(transform(scene['T_base_basket']))@carried_pose;basket=self.config['objects']['basket']
            basket_support=bool(scene.get('allow_basket_support') and rel[2,2]>.999 and
                np.all(abs(rel[:2,3])+obj['body_radius_m']+self.margin<np.array(basket['inner_size_m'][:2])/2) and
                rel[2,3]+obj['body_bottom_m']>=basket['floor_z_m']-self.config['support_tolerance_m'])
            cylinder=carried_pose.copy();cylinder[:3,3]+=(obj['body_bottom_m']+obj['body_height_m']/2)*cylinder[:3,2]
            shape=self.fcl.CollisionObject(self.fcl.Cylinder(obj['body_radius_m'],obj['body_height_m']),
                self.fcl.Transform(cylinder[:3,:3],cylinder[:3,3]))
            radius=obj['body_radius_m']+obj['body_height_m']/2
            shapes.append(('carried_bottle',shape,cylinder[:3,3]-radius,cylinder[:3,3]+radius,None))
        if self._scene_ref is not scene:
            self._obstacles=[]
            for row in scene['obstacles']:
                for c,d,r in row['capsules']:
                    c=np.array(c);d=np.array(d);delta=d-c;length=np.linalg.norm(delta)
                    axis=delta/max(length,1e-12)
                    helper=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0.,1,0])
                    x=np.cross(helper,axis);x/=max(np.linalg.norm(x),1e-12);rotation=np.column_stack([x,np.cross(axis,x),axis])
                    geometry=self.fcl.Sphere(r) if length<1e-10 else self.fcl.Capsule(r,length)
                    rotation=np.eye(3) if length<1e-10 else rotation
                    shape=self.fcl.CollisionObject(geometry,self.fcl.Transform(rotation,(c+d)/2))
                    self._obstacles.append((row['id'],shape,np.minimum(c,d)-r,np.maximum(c,d)+r))
            self._scene_ref=scene
        def exact(a,b,margin,ca=None,cb=None):
            if ca is not None or cb is not None:
                coarse=self.fcl.distance(ca or a,cb or b,self.fcl.DistanceRequest(),self.fcl.DistanceResult())
                if np.isfinite(coarse) and coarse>=margin:return True
            distance=self.fcl.distance(a,b,self.fcl.DistanceRequest(),self.fcl.DistanceResult())
            return bool(np.isfinite(distance) and distance>=margin)
        # Axis-aligned bounds separated by more than the margin are clear; all
        # remaining pairs get the exact FCL distance (same verdicts, vectorised).
        def near(lo1,hi1,lo2,hi2,margin):
            return ~(np.any(hi1[:,None]+margin<lo2[None],axis=2)|np.any(hi2[None]+margin<lo1[:,None],axis=2))
        lo=np.array([s[2] for s in shapes]);hi=np.array([s[3] for s in shapes]);selfm=self.motion.get('self_collision_margin_m',self.margin)
        pairs=near(lo,hi,lo,hi,selfm)
        if self._obstacles:
            olo=np.array([o[2] for o in self._obstacles]);ohi=np.array([o[3] for o in self._obstacles]);hits=near(lo,hi,olo,ohi,self.margin)
        for i,(name,shape,minimum,maximum,coarse) in enumerate(shapes):
            # Base mounts are stationary and may intersect the physical table.
            if name=='carried_bottle':
                if carried_bottom<scene['table_z_m']-self.config['support_tolerance_m']:return False
            elif name not in self.motion['table_mount_links'] and minimum[2]<scene['table_z_m']+self.margin:return False
            for j in np.flatnonzero(pairs[i,i+1:])+i+1:
                name2=shapes[j][0]
                if tuple(sorted([name,name2])) in self.disabled:continue
                if 'carried_bottle' in [name,name2] and (name in contact or name2 in contact):continue
                if not exact(shape,shapes[j][1],selfm,coarse,shapes[j][4]):return False
            if not self._obstacles:continue
            for j in np.flatnonzero(hits[i]):
                obstacle=self._obstacles[j]
                if obstacle[0]=='bottle' and (scene.get('carried') or name in contact):continue
                if obstacle[0]=='basket_floor' and name=='carried_bottle' and basket_support:continue
                if not exact(shape,obstacle[1],self.margin,coarse):return False
        return True

    def validate_path(self,path,scene,*,require_fresh=True):
        import time
        if require_fresh and not fresh(scene.get('timestamp_unix_s'),time.time(),self.motion['scene_max_age_s']):return False
        prior=None
        for q in path:
            q=finite(q,(16,))
            if prior is not None:
                # Densify independently of controller timing; never check just
                # IK endpoints. This is discrete checked, not continuous CCD.
                count=max(1,int(np.ceil(np.max(abs(q-prior))/.003)))
                if any(not self.valid(prior+(q-prior)*t/count,scene) for t in range(1,count+1)):return False
            elif not self.valid(q,scene):return False
            prior=q
        return prior is not None

    def plan(self,start,targets,scene):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        start=finite(start,(16,));active=[i+k for s,i in [('left',0),('right',8)] if s in targets for k in range(7)]
        if not active:raise ValueError('no tool targets')
        lo=np.concatenate([self.limits[s][0] for s in ['left','right'] if s in targets])
        hi=np.concatenate([self.limits[s][1] for s in ['left','right'] if s in targets])
        targets={s:transform(t) for s,t in targets.items()}
        def full(x):
            q=start.copy();q[active]=x;return q
        def residual(x):
            poses=self.tools(full(x));rows=[]
            for s,target in targets.items():
                rows.extend((poses[s][:3,3]-target[:3,3])*100)
                rows.extend(Rotation.from_matrix(target[:3,:3].T@poses[s][:3,:3]).as_rotvec()*5)
            rows.extend((x-start[active])*.01);return np.array(rows)
        # Deterministic nearby restarts, bounded time/pose range. Never execute
        # an approximate IK result and never clip a failed IK into a command.
        seeds=[start[active]]+[np.clip(start[active]+v,lo+1e-6,hi-1e-6) for v in [.1,-.1]]
        # Then joint configurations recorded at the operator-taught key poses:
        # the wrists start near their limits, so some targets need another
        # branch. Any result still passes the same full path validation.
        seeds+=[finite(row,(16,))[active] for row in self.motion.get('ik_seeds',[])]
        for seed in seeds:
            solved=least_squares(residual,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=80)
            goal=full(solved.x);poses=self.tools(goal)
            if any(np.linalg.norm(poses[s][:3,3]-t[:3,3])>.001 or
                np.linalg.norm(Rotation.from_matrix(t[:3,:3].T@poses[s][:3,:3]).as_rotvec())>.01 for s,t in targets.items()):continue
            path=retime(start,goal,velocity=self.motion['velocity_rad_s'],acceleration=self.motion['acceleration_rad_s2'],
                jerk=self.motion['jerk_rad_s3'])
            # Check geometry against the captured planning snapshot. Dispatch
            # re-observes and validates with require_fresh=True in Servo.follow.
            if self.validate_path([start,*path],scene,require_fresh=False):return path
        if not self._walking:
            try:return self._walk(start,targets,scene)
            except RuntimeError:pass
        if self.global_planner is not None:
            try:return self._global(start,targets,scene)
            except (RuntimeError,ValueError) as exc:
                raise RuntimeError('no fully checked path: local IK failed and '+str(exc)) from None
        raise RuntimeError('no fully checked path in bounded local IK search; no unchecked fallback')

    def _walk(self,start,targets,scene,step=.02):
        """Deterministic straight-line approach in 2 cm local IK steps per arm;
        every step passes the same dense FCL validation as a one-shot plan."""
        import time
        from scipy.spatial.transform import Rotation,Slerp
        q=finite(start,(16,)).copy();path=[];cached=self.global_planner;self.global_planner=None;self._walking=True
        try:
            for side,target in targets.items():
                a=self.tools(q)[side];target=transform(target)
                n=max(1,int(np.ceil(max(np.linalg.norm(target[:3,3]-a[:3,3])/step,
                    np.linalg.norm(Rotation.from_matrix(a[:3,:3].T@target[:3,:3]).as_rotvec())/.15))))
                if n>40:raise RuntimeError('straight-line walk too long')
                turn=Slerp([0,1],Rotation.from_matrix([a[:3,:3],target[:3,:3]]))
                for i in range(1,n+1):
                    w=np.eye(4);w[:3,:3]=turn(i/n).as_matrix();w[:3,3]=a[:3,3]+(target[:3,3]-a[:3,3])*i/n
                    segment=self.plan(q,{side:w},scene);path+=segment;q=segment[-1]
        finally:self.global_planner=cached;self._walking=False
        return path

    def _global(self,start,targets,scene,attempts=None):
        """cuRobo geometric path per arm, retimed to the same limits and then
        accepted only by the same dense FCL validation as local plans. cuRobo
        is stochastic, so an FCL rejection triggers a fresh attempt."""
        import time
        from scipy.spatial.transform import Rotation
        try:from .cumotion import resample
        except ImportError:from cumotion import resample
        reason='cuRobo path rejected by FCL recheck'
        for _ in range(attempts or self.global_attempts):
            q=finite(start,(16,)).copy();path=[]
            for side,target in targets.items():
                carried=scene.get('carried');bottle=None
                if carried and carried['arm']!=side:bottle=self.tools(q)[carried['arm']]@transform(carried['T_tool_object'])
                geometric=self.global_planner.plan_arm(q,side,target,scene,bottle_pose=bottle)
                segment=resample([q,*geometric],velocity=self.motion['velocity_rad_s'],
                    acceleration=self.motion['acceleration_rad_s2'],jerk=self.motion['jerk_rad_s3'])
                path+=segment;q=segment[-1]
            poses=self.tools(q)
            for s,t in targets.items():
                t=transform(t)
                if (np.linalg.norm(poses[s][:3,3]-t[:3,3])>.001 or
                    np.linalg.norm(Rotation.from_matrix(t[:3,:3].T@poses[s][:3,:3]).as_rotvec())>.01):
                    raise RuntimeError('cuRobo end pose outside 1 mm / 0.01 rad')
            # Same scene geometry the plan used; GPU planning can outlast the 0.2 s
            # snapshot age, and the caller re-observes before dispatch anyway.
            if self.validate_path([start,*path],scene,require_fresh=False):return path
        raise RuntimeError(reason)
