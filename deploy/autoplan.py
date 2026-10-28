"""Automatic task poses: grasp, handover, retreat and place, chosen by search.

Candidates come from the measured bottle/basket geometry; every segment of the
task is planned with the runtime planner (local IK, cuRobo fallback, FCL
recheck) from the ACT start pose. Nothing here moves the robot.
"""
import time
import types

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .evidence import transform
    from .runtime import Runtime,basket_clearance
except ImportError:
    from evidence import transform
    from runtime import Runtime,basket_clearance


def scene(planner,cal,q,objects,*,carried=None,contact=(),place=False):
    """The runtime's own obstacle construction for a hypothetical state."""
    tools=planner.tools(q);now=time.time()
    bottle=objects['bottle'] if carried is None else tools[carried['arm']]@transform(carried['T_tool_object'])
    state={'object_poses':{'bottle':np.asarray(bottle).tolist(),'basket':np.asarray(objects['basket']).tolist()},
        'held_by':[] if carried is None else [carried['arm']],'pose_revision':1,'timestamp_unix_s':now,
        'exposure_aligned_tools':{s:{'T_base_tool':t.tolist(),'timestamp_unix_s':now} for s,t in tools.items()}}
    return Runtime.scene(types.SimpleNamespace(cal=cal,operation={'kind':'place'} if place else None),state,contact=contact)


def side_tool(axis_point,start_rotation,yaw,depth):
    """Side grasp in the ACT start wrist orientation turned by `yaw` about the
    vertical: the wrists start near their limits, so this is the reachable
    family. TCP sits `depth` past the bottle axis along the approach."""
    r=Rotation.from_euler('z',yaw).as_matrix()@start_rotation
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=np.asarray(axis_point)+r[:,2]*depth;return t


class Search:
    def __init__(self,planner,cal,start,objects,log=print,max_seconds=60.):
        self.p=planner;self.cal=cal;self.start=np.asarray(start,float);self.obj=objects;self.log=log
        b=cal['objects']['bottle'];self.H=b['body_height_m'];self.r=b['body_radius_m'];self.bottom=b['body_bottom_m']
        self.m=cal['motion'];self.seeds=[];self.deadline=time.monotonic()+max_seconds

    def plan(self,q,side,target,**kw):
        """End joints of a move the runtime planner itself accepts (same code)."""
        if time.monotonic()>self.deadline:
            raise TimeoutError('在线规划超过时间预算；没有发送动作，请检查规划记录和物体摆放')
        end=self.p.plan(q,{side:target},scene(self.p,self.cal,q,self.obj,**kw))[-1]
        self.seeds.append(np.asarray(end).copy());return end

    def pregrasp(self,q,side,grasp,contact,end_contact,carried=None):
        """Shortest retreat along the tool axis that is reachable and clear."""
        for d in sorted({self.m['pregrasp_m'],.08,.10,.12}):
            pre=grasp.copy();pre[:3,3]-=pre[:3,2]*d
            try:qp=self.plan(q,side,pre,carried=carried,contact=end_contact)
            except RuntimeError:continue
            try:qg=self.plan(qp,side,grasp,carried=carried,contact=contact)
            except RuntimeError:continue
            return d,qp,qg
        return None

    def run(self):
        p=self.p;q0=self.start;tools=p.tools(q0);bottle=transform(self.obj['bottle']);basket=transform(self.obj['basket'])
        mid=.5*(tools['left'][:3,3]+tools['right'][:3,3]);axis=bottle[:3,3]
        # 1. right grasp on the lower half, approached from the right hand's side
        h=float(np.clip(.3*self.H,.035,.5*self.H-.04));out=None
        for yaw in (0.,.35,-.35,.7,-.7):
            g=side_tool(axis+[0,0,self.bottom+h],tools['right'][:3,:3],yaw,.02)
            found=self.pregrasp(q0,'right',g,('right',),())
            self.log('  右手抓取 偏转%+.0f°：%s'%(np.degrees(yaw),'可行' if found else '不可行'))
            if found:out=(g,found);break
        if out is None:raise RuntimeError('右手在初始姿态附近找不到可达的侧面抓取，请把瓶子挪近右手')
        g,(d_right,_,qg)=out;T_object_tool=np.linalg.inv(bottle)@g;carried={'arm':'right','T_tool_object':np.linalg.inv(T_object_tool).tolist()}
        self.seeds.append(qg)
        lift=p.tools(qg)['right'].copy();lift[2,3]+=self.m['lift_m'];ql=self.plan(qg,'right',lift,carried=carried,contact=('right',))
        # 2. handover point near the start mid-point; 3. left grasp on the upper half
        hl=float(np.clip(.75*self.H,h+.09,self.H-.025))
        # The wrists start near their limits: with the grasp orientation kept, the
        # right hand reaches only ~6 cm right of the mid-line, so search there.
        for dx in (.16,.20,.12):
            for dy in (-.08,-.11,-.05):
                for dz in (0.,.04,-.04):
                    target=transform(p.tools(ql)['right']@np.asarray(carried['T_tool_object'],float))   # bottle at the lift
                    target[:3,3]=mid+[dx,dy,dz]
                    try:qh=self.plan(ql,'right',target@T_object_tool,carried=carried,contact=('right',))
                    except RuntimeError as exc:self.log('  交接点 %s：右手到不了（%s）'%(np.round([dx,dy,dz],2),str(exc)[-60:]));continue
                    bh=p.tools(qh)['right']@transform(carried['T_tool_object'])
                    for yaw in (-.4,0.,-.8,.4):
                        L=side_tool(bh[:3,3]+bh[:3,2]*(self.bottom+hl),tools['left'][:3,:3],yaw,.02)
                        found=self.pregrasp(qh,'left',L,('left','right'),('right',),carried)
                        if not found:continue
                        d_left,_,qr=found;self.log('  交接点 %s，左手偏转%+.0f°：可行'%(np.round(mid+[dx,dy,dz]-mid,2),np.degrees(yaw)))
                        result=self.finish(qr,bh,L,carried,mid,d_right,d_left,T_object_tool,basket)
                        if result:self.seeds+= [qh,qr];return result
                    self.log('  交接点 %s：左手接不到'%np.round([dx,dy,dz],2))
        raise RuntimeError('找不到两只手都能到达的交接点，请把瓶子放得离身体中线近一些')

    def finish(self,qr,bh,L,carried,mid,d_right,d_left,T_object_tool,basket):
        p=self.p;T_bottle_left=np.linalg.inv(bh)@L;held={'arm':'left','T_tool_object':np.linalg.inv(T_bottle_left).tolist()}
        # 4. right retreat: first clear direction among back, outward and up
        right=p.tools(qr)['right'];outward=right[:3,3]-mid;outward[2]=0;outward/=np.linalg.norm(outward)
        for retreat in (-right[:3,2]*.08+[0,0,.02],outward*.08,np.array([0,0,.08]),-right[:3,2]*.06+outward*.05):
            target=right.copy();target[:3,3]+=retreat
            try:qd=self.plan(qr,'right',target,carried=held,contact=('left','right'));break
            except RuntimeError as exc:
                self.log('  右手撤离拒绝：'+str(exc));continue
        else:self.log('  右手撤离不可行');return None
        # 5. place upright at the basket centre; yaw is free for a cylinder
        b=self.cal['objects']['basket']
        for yaw in (0.,.6,-.6,1.2,-1.2,np.pi):
            place=basket.copy();place[:3,:3]=basket[:3,:3]@Rotation.from_euler('z',yaw).as_matrix()
            place[:3,3]=(basket@np.array([0,0,b['floor_z_m']+.002-self.bottom,1]))[:3]
            above=place.copy();above[2,3]+=basket_clearance(self.cal)
            phase='越过篮口'
            try:
                qa=self.plan(qd,'left',above@T_bottle_left,carried=held,contact=('left',))
                phase='篮内下降'
                qp=self.plan(qa,'left',place@T_bottle_left,carried=held,contact=('left',),place=True)
            except RuntimeError as exc:
                self.log('  %s偏转%+.0f°拒绝：%s'%(phase,np.degrees(yaw),str(exc)));continue
            self.seeds+=[qd,qa,qp];self.log('  放置偏转%+.0f°：可行'%np.degrees(yaw))
            pts=[t[:3,3] for q in [self.start,*self.seeds] for t in p.tools(q).values()]
            return {'T_object_tool':T_object_tool,'T_bottle_left_tool':T_bottle_left,'T_basket_bottle':np.linalg.inv(basket)@place,
                'handover_offset_m':bh[:3,3]-mid,'T_base_handover_bottle':bh,'donor_retreat_base_m':retreat,
                'pregrasp_m':max(d_right,d_left),'pregrasp_by_arm':{'right':d_right,'left':d_left},
                'ik_seeds':[np.asarray(q).tolist() for q in self.seeds[-16:]],'points':pts}
        self.log('  放进篮子不可行');return None


def session_poses(planner,cal,start,objects,log=print):
    """Task poses for this session's measured scene (one global attempt each)."""
    saved=planner.global_attempts;planner.global_attempts=1;t0=time.time()
    try:result=Search(planner,cal,start,objects,log).run()
    finally:planner.global_attempts=saved
    result['seconds']=time.time()-t0;return result
