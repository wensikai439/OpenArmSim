"""Real-only tool provider: measured relative goals, continuous hold and recovery.

Construction is inert. Only open(operator_confirmed=True) opens devices; the
service calls it only after the bound physical acceptance checks pass.
"""
from collections import deque
import copy
import json
from pathlib import Path
import threading
import time

import numpy as np

try:
    from .devices import Camera,LegacyArm,marker_poses
    from .evidence import fresh,transform
    from .motion import Servo,retime
    from .planning import Planner,canonical
    from .robot import TOOLS,right_region_error
    from .service import EvidencePipeline
except ImportError:
    from devices import Camera,LegacyArm,marker_poses
    from evidence import fresh,transform
    from motion import Servo,retime
    from planning import Planner,canonical
    from robot import TOOLS,right_region_error
    from service import EvidencePipeline


def attach_global(planner,root,hardware,calibration):
    """cuRobo fallback for moves the local IK cannot reach (GPU, plans only)."""
    if (Path(root)/'config/curobo.yml').is_file():
        try:from .cumotion import CuMotion
        except ImportError:from cumotion import CuMotion
        planner.global_planner=CuMotion(root,hardware,calibration)
        table=calibration.get('table_top_z_m');planner.global_planner.warm(hardware['startup_target'],table if table is not None else -10.)


def basket_clearance(calibration):
    """Clear the measured rim, including the carried-cylinder sphere cover."""
    m=calibration['motion'];b=calibration['objects']['basket'];o=calibration['objects']['bottle']
    return max(float(m['lift_m']),float(b['inner_size_m'][2])+float(o['body_radius_m'])+
               float(m['collision_margin_m'])+float(calibration['support_tolerance_m']))


def pregrasp_distance(calibration,side):
    m=calibration['motion']
    return float(m.get('pregrasp_by_arm',{}).get(side,m['pregrasp_m']))


class Runtime:
    tools=TOOLS
    def __init__(self,root,record):
        self.root=Path(root);self.record=record
        read=lambda n:json.loads((self.root/'config'/n).read_text())
        self.hardware=read('hardware.json');self.cal=read('calibration.json');self.collision=read('collision.json')
        self.driver=None;self.camera=None;self.servo=None;self.planner=None;self.capture_thread=None
        self.stop_event=threading.Event();self.failure=None;self.lock=threading.RLock();self.snapshot=None
        self.history=deque(maxlen=120);self.candidates={};self.checkpoints={};self.current_grasp=None
        self.last_failure=None;self.completed=[];self.operation=None;self.required_holds=set();self.started=None
        self.last_frame_record=0.

    def open(self,*,operator_confirmed=False):
        if not operator_confirmed:raise RuntimeError('on-site confirmation required')
        try:
            self.planner=Planner(self.hardware,self.cal,self.collision);self.pipeline=EvidencePipeline(self.cal)
            attach_global(self.planner,self.root,self.hardware,self.cal)
            camera=self.cal['camera'];cfg=copy.deepcopy(next(c for c in self.hardware['cameras'].values() if c.get('serial')==camera['serial']))
            cfg['clock_model']=camera['clock_model'];self.camera=Camera(cfg,self.record)
            self.driver=LegacyArm(self.hardware,self.record)
            self.camera.open(operator_confirmed=True);self.driver.open(operator_confirmed=True)
            # Fresh CAN feedback only; motors remain deferred until a complete
            # measured scene and collision-checked ACT startup path exist.
            for _ in range(12):
                self.history.append(self.driver.read());time.sleep(.025)
            self._capture()
            initial=self.observe()
            if initial['held_by'] or any(initial['touching'].values()):raise RuntimeError('initial contact/hold requires operator review')
            self.capture_thread=threading.Thread(target=self._capture_loop,name='openarm-perception',daemon=True)
            self.capture_thread.start()
            def startup_guard():
                self._ambient_guard();latest=self.observe()
                for name in ['bottle','basket']:
                    if np.linalg.norm(np.array(latest['object_poses'][name])[:3,3]-np.array(initial['object_poses'][name])[:3,3])>.003:
                        raise RuntimeError('scene moved during initialization')
                return self.stop_event.is_set()
            self.driver.initialize(self.planner.limits,startup_guard,planner=self.planner,scene=self.scene(initial))
            self.servo=Servo(self.driver,self.planner.limits,self.record)
            self.servo.guard=self._ambient_guard;self.servo.start()
            self.record('real_provider_started',{'tools':list(self.tools),'source':'real_devices'})
        except Exception:
            self.close();raise

    def _feedback(self,stamp):
        deadline=time.monotonic()+.05
        while self.servo and self.servo.thread and self.servo.feedback and self.servo.feedback[-1]['timestamp_unix_s']<stamp and time.monotonic()<deadline:
            time.sleep(.003)
        rows=list(self.servo.feedback) if self.servo and self.servo.feedback else list(self.history)
        rows=sorted(rows,key=lambda r:r['timestamp_unix_s'])
        before=[r for r in rows if r['timestamp_unix_s']<=stamp];after=[r for r in rows if r['timestamp_unix_s']>=stamp]
        if not before or not after:raise RuntimeError('exposure is outside recorded joint feedback history')
        a,b=before[-1],after[0];ta,tb=a['timestamp_unix_s'],b['timestamp_unix_s']
        if tb-ta>.05 or min(stamp-ta,tb-stamp)>.025:raise RuntimeError('no joint feedback within accepted exposure alignment')
        q=np.array(a['q']) if tb==ta else np.array(a['q'])+(np.array(b['q'])-a['q'])*(stamp-ta)/(tb-ta)
        measured=copy.deepcopy(a if stamp-ta<=tb-stamp else b)
        return q,measured

    def _capture(self):
        running=bool(self.servo and self.servo.thread)
        if not running:self.history.append(self.driver.read())
        frame=self.camera.read()
        if not running:self.history.append(self.driver.read())
        # A newer servo sample can arrive while RGB-D processing completes.
        q,feedback=self._feedback(frame['timestamp_unix_s'])
        cam=self.cal['camera'];base_camera=self.planner.kin.pose(cam['mounted_link'],self.planner.joints(q))@transform(cam['T_link_camera'])
        poses=marker_poses(frame,self.cal['objects'],base_camera)
        tools={s:{'T_base_tool':p,'timestamp_unix_s':frame['timestamp_unix_s']} for s,p in self.planner.tools(q).items()}
        state=self.pipeline.fuse(poses,tools,feedback)
        state['perception_frame']={'frame_number':frame['frame_number'],'exposure_unix_s':frame['timestamp_unix_s'],
            'arrival_unix_s':frame['arrival_unix_s'],'clock_domain':frame['clock_domain']}
        with self.lock:self.snapshot=state
        # Detailed feedback is stored in records, with no extra user interface.
        if time.monotonic()-self.last_frame_record>=1:
            folder=getattr(getattr(self.record,'__self__',None),'folder',None)
            if folder is not None:
                import cv2
                dest=Path(folder)/'frames';dest.mkdir(exist_ok=True)
                name=str(frame['frame_number'])
                if not cv2.imwrite(str(dest/(name+'.jpg')),frame['bgr']):raise RuntimeError('frame logging failed')
                if not cv2.imwrite(str(dest/(name+'-depth.png')),np.rint(frame['depth_m']*1000).astype(np.uint16)):
                    raise RuntimeError('depth logging failed')
                self.record('camera_frame',{'image':'frames/'+name+'.jpg','depth':'frames/'+name+'-depth.png',
                    'depth_unit_m':.001,**state['perception_frame']})
            self.last_frame_record=time.monotonic()

    def _capture_loop(self):
        try:
            while not self.stop_event.is_set():self._capture()
        except Exception as exc:self.stop('perception failed: '+str(exc))

    def _ambient_guard(self):
        if self.failure or self.stop_event.is_set():raise RuntimeError(self.failure or 'runtime stopped')
        with self.lock:s=copy.deepcopy(self.snapshot)
        if not s or not fresh(s['timestamp_unix_s'],time.time(),.15):raise RuntimeError('perception progress watchdog expired')
        if not self.required_holds<=set(s['held_by']):raise RuntimeError('required measured grip lost')
        if self.servo is not None and self.servo.state is not None:
            feedback=self.servo.state
            if not fresh(feedback['timestamp_unix_s'],time.time(),.1):raise RuntimeError('gripper effort feedback stale')
            for side in ['left','right']:
                motor=feedback['motors'][side][7]
                torque=float(motor['torque'])
                if not np.isfinite(torque) or abs(torque)>self.cal['grippers'][side]['contact_torque_range'][1]:
                    raise RuntimeError(side+' gripper exceeded accepted effort ceiling')

    def observe(self):
        self._ambient_guard()
        with self.lock:
            s=copy.deepcopy(self.snapshot)
            s.update(checkpoint_id=next(reversed(self.checkpoints),None),current_grasp_id=self.current_grasp,
                last_failure=copy.deepcopy(self.last_failure),completed_steps=list(self.completed),
                operation=copy.deepcopy(self.operation),remaining_seconds=120. if self.started is None else max(0.,120-(time.monotonic()-self.started)),
                terminal=False,invalid=False)
        if s['remaining_seconds']<=0:raise RuntimeError('real task time budget exceeded')
        return s

    def begin_session(self):
        state=self.observe()
        if state['held_by'] or any(state['touching'].values()) or self.operation is not None:
            raise RuntimeError('cannot begin a new task with a hold, contact or pending operation')
        feedback=self.servo.state
        if not fresh(feedback.get('timestamp_unix_s'),time.time(),.1):raise RuntimeError('fresh feedback required for ACT startup check')
        measured=np.array(feedback['q'],float);target=np.array(self.hardware['startup_target'],float)
        if measured.shape!=(16,) or target.shape!=(16,) or not np.isfinite(measured).all():
            raise RuntimeError('invalid startup feedback')
        error=abs(measured-target);arm=np.r_[0:7,8:15]
        if max(error[arm])>np.deg2rad(3) or max(error[[7,15]])>.1:
            raise RuntimeError('ACT startup pose required before EACH task; no automatic homing while state is unverified')
        with self.servo.lock:command=np.array(self.servo.target,float)
        if max(abs(command[arm]-target[arm]))>np.deg2rad(3) or max(abs(command[[7,15]]-target[[7,15]]))>.1:
            raise RuntimeError('pending servo target is not the ACT startup target')
        self.record('session_startup_verified',{'source':'fresh_measured_joint_feedback',
            'absolute_error':error.tolist(),'target':target.tolist()})
        self.started=time.monotonic();self.completed.clear();self.last_failure=None
        self.candidates.clear();self.checkpoints.clear();self.current_grasp=None

    def end_session(self):
        state=self.observe()
        if state['held_by'] or self.operation is not None:raise RuntimeError('task ended with unresolved hold/operation')
        self.started=None

    def scene(self,state=None,*,contact=()):
        s=state or self.observe();obj=self.cal['objects'];bottle=np.array(s['object_poses']['bottle']);basket=np.array(s['object_poses']['basket'])
        bottom=obj['bottle']['body_bottom_m'];height=obj['bottle']['body_height_m'];radius=obj['bottle']['body_radius_m']
        obstacles=[{'id':'bottle','capsules':[[(bottle@np.array([0,0,bottom,1]))[:3].tolist(),
            (bottle@np.array([0,0,bottom+height,1]))[:3].tolist(),radius+s.get('object_uncertainty_m',.003)]]
            }]
        # Basket four top edges and four vertical walls, conservatively sampled
        # as capsules. Interior geometry and table height are measured inputs.
        size=np.array(obj['basket']['inner_size_m']);floor=obj['basket']['floor_z_m'];wall=obj['basket']['wall_radius_m'];bars=[]
        for sign in [-1,1]:
            for axis in [0,1]:
                other=1-axis
                for z in np.linspace(floor,floor+size[2],max(2,int(np.ceil(size[2]/wall))+1)):
                    a=np.array([0.,0.,z]);b=a.copy();a[axis]=b[axis]=sign*(size[axis]/2+wall);a[other]=-size[other]/2;b[other]=size[other]/2
                    bars.append([(basket@np.r_[a,1])[:3].tolist(),(basket@np.r_[b,1])[:3].tolist(),wall])
                for end in [-1,1]:
                    a=np.array([0.,0.,floor]);a[axis]=sign*(size[axis]/2+wall);a[other]=end*size[other]/2;b=a.copy();b[2]+=size[2]
                    bars.append([(basket@np.r_[a,1])[:3].tolist(),(basket@np.r_[b,1])[:3].tolist(),wall])
        obstacles.append({'id':'basket','capsules':bars})
        floor_bars=[]
        for y in np.linspace(-size[1]/2,size[1]/2,max(2,int(np.ceil(size[1]/wall))+1)):
            floor_bars.append([(basket@np.array([-size[0]/2,y,floor-wall,1]))[:3].tolist(),
                (basket@np.array([size[0]/2,y,floor-wall,1]))[:3].tolist(),wall])
        obstacles.append({'id':'basket_floor','capsules':floor_bars})
        held=s['held_by'];carried=None
        if held:
            side='left' if held==['left'] else 'right'
            aligned=s['exposure_aligned_tools'][side]
            if abs(aligned['timestamp_unix_s']-s['timestamp_unix_s'])>.005:
                raise RuntimeError('carried object requires exposure-aligned tool pose')
            carried={'arm':side,'T_tool_object':(np.linalg.inv(transform(aligned['T_base_tool']))@bottle).tolist()}
        contact_links=[link for side in contact for link in self.cal['motion']['contact_links'][side]]
        return {'revision':s['pose_revision'],'timestamp_unix_s':s['timestamp_unix_s'],'table_z_m':self.cal['table_top_z_m'],
            'obstacles':obstacles,'carried':carried,'contact_links':contact_links,
            'T_base_basket':basket.tolist(),'allow_basket_support':bool(self.operation and self.operation['kind']=='place')}

    def _arrival(self):
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            self._ambient_guard();state=self.servo.state
            with self.servo.lock:target=self.servo.target.copy()
            idx=np.r_[0:7,8:15]
            if np.max(abs(target[idx]-np.array(state['q'])[idx]))<=.015:return
            time.sleep(.02)
        raise RuntimeError('measured arm arrival not reached')

    def _move(self,targets,*,contact=()):
        before=self.observe();scene=self.scene(before,contact=contact)
        with self.servo.lock:start=self.servo.target.copy()
        path=self.planner.plan(start,targets,scene)
        current=self.observe()
        for name in ['basket']+([] if scene['carried'] else ['bottle']):
            if np.linalg.norm(np.array(current['object_poses'][name])[:3,3]-np.array(before['object_poses'][name])[:3,3])>.003:
                raise RuntimeError('object moved during planning; fresh planning required')
        scene=self.scene(current,contact=contact)
        fixed=['basket']+([] if scene['carried'] else ['bottle'])
        def guard():
            self._ambient_guard();latest=self.observe()
            for name in fixed:
                a=np.array(latest['object_poses'][name]);b=np.array(current['object_poses'][name])
                if np.linalg.norm(a[:3,3]-b[:3,3])>.003 or np.linalg.norm(a[:3,:3]-b[:3,:3])>.02:
                    raise RuntimeError('scene changed during dispatch; stop and replan')
        self.servo.follow(path,self.planner,scene,guard=guard);self._arrival()

    def _jaw(self,side,opened):
        if side not in ['left','right']:raise ValueError('unknown arm')
        state=self.observe();support=state['support_evidence']
        if opened and side in state['held_by'] and not (set(state['held_by'])-set([side])):
            if not any(v.get('verified') and v.get('stationary') for v in support.values()):
                raise RuntimeError('sole holder cannot release unsupported bottle')
        with self.servo.lock:start=self.servo.target.copy()
        goal=start.copy();goal[7 if side=='left' else 15]=self.cal['grippers'][side]['open_motor' if opened else 'close_motor']
        path=retime(start,goal,velocity=.1,acceleration=.2,jerk=self.cal['motion']['jerk_rad_s3'])
        stop_on_hold=None if opened else lambda:side in self.observe()['held_by']
        self.servo.follow(path,self.planner,self.scene(state,contact=('left','right')),
            guard=self._ambient_guard,until=stop_on_hold)

    def _wait_hold(self,side,wanted=True):
        deadline=time.monotonic()+1.2
        while time.monotonic()<deadline:
            state=self.observe()
            if (side in state['held_by']) is wanted:return state
            time.sleep(.025)
        return self.observe()

    def _result(self,ok=True,done=True,code=None):
        if not ok:self.last_failure={'code':code,'timestamp_unix_s':time.time()}
        self.record('skill_feedback',{'ok':ok,'done':done,'error_code':code,'observation':self.observe()})
        return {'ok':ok,'done':done,**({'error_code':code} if code else {})}

    def call(self,name,*args,**kwargs):
        if name not in self.tools:raise ValueError('unknown tool')
        self._ambient_guard()
        return getattr(self,name)(*args,**kwargs)

    def propose_grasps(self,exclude_ids=()):
        state=self.observe();pose=transform(state['object_poses']['bottle']);candidates=[]
        for row in self.cal['motion']['right_grasps']:
            ident=row['id']
            if ident in exclude_ids:continue
            target=pose@transform(row['T_object_tool']);pre=target.copy();pre[:3,3]-=target[:3,2]*pregrasp_distance(self.cal,'right')
            self.candidates[ident]={'target':target,'pre':pre,'bottle':pose.copy()}
            candidates.append({'id':ident,'source':'measured_object_relative_transform','pose_revision':state['pose_revision']})
        return {'candidates':candidates}

    def _candidate(self,ident):
        if ident not in self.candidates:raise ValueError('unproposed grasp')
        c=self.candidates[ident];pose=np.array(self.observe()['object_poses']['bottle'])
        if np.linalg.norm(pose[:3,3]-c['bottle'][:3,3])>.003:raise RuntimeError('grasp object moved; repropose')
        return c

    def pregrasp(self,ident):
        c=self._candidate(ident)
        if self.observe()['held_by']:raise RuntimeError('pregrasp requires empty hands')
        self._jaw('right',True);self._move({'right':c['pre']});self.current_grasp=ident
        with self.servo.lock:self.checkpoints['pregrasp-'+ident]=self.servo.target.copy()
        return self._result()

    def approach(self,ident):
        self._move({'right':self._candidate(ident)['target']},contact=('right',));return self._result()

    def close_gripper(self,arm='right'):
        self._jaw(arm,False);s=self._wait_hold(arm)
        if arm not in s['held_by']:return self._result(False,code='grasp_not_confirmed' if arm=='right' else 'receiver_hold_not_confirmed')
        if arm=='right' and right_region_error(s):return self._result(False,code=right_region_error(s))
        if not (arm=='left' and 'right' in s['held_by']):self.required_holds.add(arm)
        return self._result()

    def open_gripper(self,arm='right'):
        state=self.observe()
        if arm in state['held_by'] and not set(state['held_by'])-set([arm]):raise RuntimeError('use verified supported release for sole holder')
        self.required_holds.discard(arm);self._jaw(arm,True);return self._result()

    def verify_hold(self,arm='right'):
        s=self._wait_hold(arm);return self._result(arm in s['held_by'],code=None if arm in s['held_by'] else 'hold_not_confirmed')

    def lift(self):
        state=self.observe()
        if state['held_by']!=['right'] or right_region_error(state):raise RuntimeError('lower-body right hold required')
        target=self.planner.tools(self.servo.state['q'])['right'];target[2,3]+=self.cal['motion']['lift_m']
        self._move({'right':target},contact=('right',));return self._result()

    def retreat(self,checkpoint):
        if self.observe()['held_by']:raise RuntimeError('retreat cannot rewind a carried object or contacts')
        if checkpoint not in self.checkpoints:raise ValueError('unknown checkpoint')
        goal=self.checkpoints[checkpoint];targets=self.planner.tools(goal)
        self._move(targets);return self._result()

    def release_on_support(self):
        state=self.observe();support=state['support_evidence']['table']
        if 'left' in state['held_by'] or not support['verified'] or not support['stationary']:raise RuntimeError('verified stationary table support required')
        self.required_holds.discard('right');self._jaw('right',True);self._wait_hold('right',False);return self._result()

    def start_transfer(self,placement_aware=True):
        state=self.observe()
        if placement_aware is not True or state['held_by']!=['right'] or right_region_error(state):raise RuntimeError('lower-body right hold and placement-aware plan required')
        self.operation={'kind':'transfer','phase':'position','index':0};return self._result(done=False)

    def finish_transfer(self):
        if set(self.observe()['held_by'])!={'left','right'}:return self._result(False,code='dual_hold_not_confirmed')
        self.required_holds.add('left');self.required_holds.discard('right');self._jaw('right',True)
        state=self._wait_hold('right',False)
        if state['held_by']!=['left']:raise RuntimeError('donor release not measured')
        self.operation=None;self.completed.append('handover');return self._result()

    def start_place(self):
        if self.observe()['held_by']!=['left']:raise RuntimeError('left-only hold required')
        self.operation={'kind':'place','phase':'above_basket','index':0};return self._result(done=False)

    def advance(self):
        if not self.operation:raise RuntimeError('no active operation')
        op=self.operation;phase=op['phase'];s=self.observe();tools=self.planner.tools(self.servo.state['q']);bottle=transform(s['object_poses']['bottle'])
        if op['kind']=='transfer':
            if phase=='position':
                desired=transform(self.cal['motion']['T_base_handover_bottle'])
                right=desired@np.linalg.inv(bottle)@tools['right']
                self._move({'right':right},contact=('right',));op['phase']='receiver_pregrasp'
            elif phase=='receiver_pregrasp':
                left=bottle@transform(self.cal['motion']['T_bottle_left_tool'])
                left[:3,3]-=left[:3,2]*pregrasp_distance(self.cal,'left')
                self._move({'left':left},contact=('right',));op['phase']='receiver_approach'
            elif phase=='receiver_approach':
                self._move({'left':bottle@transform(self.cal['motion']['T_bottle_left_tool'])},contact=('left','right'));op['phase']='receiver_close'
            elif phase=='receiver_close':
                r=self.close_gripper('left')
                if not r['ok']:return self._result(False,code='receiver_hold_not_confirmed')
                op['phase']='release_right'
            elif phase=='release_right':return self.finish_transfer()
            else:raise ValueError('unknown transfer phase')
        else:
            target_bottle=transform(s['object_poses']['basket'])@transform(self.cal['motion']['T_basket_bottle'])
            target=target_bottle@np.linalg.inv(bottle)@tools['left']
            if phase=='above_basket':
                target[2,3]+=basket_clearance(self.cal);self._move({'left':target},contact=('left',));op['phase']='lower'
            elif phase=='lower':
                self._move({'left':target},contact=('left',));op['phase']='release_left'
            elif phase=='release_left':
                support=s['support_evidence']['basket']
                if not support['verified'] or not support['stationary']:return self._result(False,code='basket_support_not_confirmed')
                self.required_holds.discard('left');self._jaw('left',True);self._wait_hold('left',False);op['phase']='verify_placement'
            elif phase=='verify_placement':
                time.sleep(.25);s=self.observe();support=s['support_evidence']['basket']
                if s['held_by'] or not support['verified'] or not support['stationary']:return self._result(False,code='placement_not_confirmed')
                self.operation=None;self.completed.append('placement');return self._result()
            else:raise ValueError('unknown placement phase')
        op['index']+=1;return self._result(done=False)

    def stop(self,reason='runtime stop requested'):
        self.failure=str(reason);self.stop_event.set()
        if self.driver:self.driver.stopped=True
        if self.servo:self.servo.halt(reason)

    def close(self):
        self.stop('runtime closed');errors=[]
        for device in [self.servo,self.camera,self.driver]:
            if device:
                try:device.close()
                except Exception as exc:errors.append(str(exc))
        if self.capture_thread and self.capture_thread is not threading.current_thread():self.capture_thread.join(2)
        if errors:
            try:self.record('error',{'reason':'device cleanup failures','errors':errors})
            except Exception:pass
