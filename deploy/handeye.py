"""First wrist-camera calibration, separate from admission of robot tasks.

Default: camera preview only. --execute additionally requires an on-site
clearance confirmation. It holds the measured pose; it does NOT run ACT home.
Only wrist joints 6/7 may change, <=0.10 rad, with <=25 mm mesh displacement.
No object, grip, table collision or task acceptance is fabricated here.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

try:
    from . import calib_math as cm
    from .calib import Store,session_calibration
    from .devices import Camera,LegacyArm
    from .evidence import finite
    from .motion import Servo,retime
    from .planning import Planner
    from .service import Journal,preflight
except ImportError:
    import calib_math as cm
    from calib import Store,session_calibration
    from devices import Camera,LegacyArm
    from evidence import finite
    from motion import Servo,retime
    from planning import Planner
    from service import Journal,preflight


BOARD={'kind':'checkerboard','inner_corners':[8,6],'square_m':.020,
       'source':'user supplied board photograph; independent RGB-D scale cross-check'}
ROOT=Path(__file__).resolve().parents[1]


class WristEnvelope:
    """Self-collision + measured-pose envelope, NOT absolute table clearance.

    The operator must separately confirm 50 mm of free space around BOTH
    wrists. This narrowly scoped commissioning check is never a task planner.
    """
    identity='onsite-wrist-envelope/1'
    def __init__(self,planner,anchor,side):
        self.planner=planner;self.anchor=finite(anchor,(16,)).copy();self.side=side
        i=0 if side=='left' else 8;self.active=[i+5,i+6];self.fixed=[k for k in range(16) if k not in self.active]
        self.reference=self.vertices(self.anchor)

    def vertices(self,q):
        p=self.planner;values=p.joints(q);cache={};out=[]
        for row in p.model['hulls']:
            T=p.kin.pose(row['link'],values,cache)
            v=np.asarray(row.get('jaw_sweep_hull_vertices') or row['vertices'],float)
            out.append(v@T[:3,:3].T+T[:3,3])
        return np.concatenate(out)

    def displacement(self,q):return float(np.max(np.linalg.norm(self.vertices(q)-self.reference,axis=1)))

    def valid(self,q):
        q=finite(q,(16,))
        if np.max(abs(q[self.fixed]-self.anchor[self.fixed]))>1e-8:return False
        if np.max(abs(q[self.active]-self.anchor[self.active]))>.100001:return False
        if self.displacement(q)>.025:return False
        # This scene explicitly supplies self-collision only. It is not a
        # measured table plane, and must not be reused by a grasping service.
        s={'timestamp_unix_s':time.time(),'table_z_m':-float('inf'),'obstacles':[],
           'carried':None,'contact_links':[],'T_base_basket':np.eye(4).tolist(),
           'allow_basket_support':False,'qualification':'self_collision_only; operator-reviewed free wrist volume'}
        return bool(self.planner.valid(q,s))

    def validate_path(self,path,scene):
        prior=None
        for q in path:
            q=finite(q,(16,));n=1 if prior is None else max(1,int(np.ceil(np.max(abs(q-prior))/.003)))
            if any(not self.valid(q if prior is None else prior+(q-prior)*t/n) for t in range(1,n+1)):return False
            prior=q
        return prior is not None

    def targets(self):
        out=[self.anchor.copy()]
        for amount in (.08,-.08,.10,-.10):
            for index in self.active:
                q=self.anchor.copy();q[index]+=amount
                if self.validate_path([self.anchor,q],{}):out.append(q)
        if len(out)<7:raise RuntimeError('not enough bounded wrist poses are self-collision free at this posture')
        return out


def clock_check(frame):
    """A static global-clock sanity check, not moving exposure calibration."""
    age=frame['arrival_unix_s']-frame['device_timestamp_ms']/1000
    return {'domain':frame['clock_domain'],'host_minus_device_s':float(age),
            'global_clock':('global' in frame['clock_domain'].lower()),'moving_timing_verified':False}


def preview(camera,board,folder):
    import cv2
    last=None
    for _ in range(20):
        try:
            f=camera.read(diagnostic=True);p=cm.board_pose(f,board)
            if p is not None:
                last=(p,f)
                if p['reprojection_rms_px']<=1.5*f['bgr'].shape[1]/640:break
        except RuntimeError as exc:
            if 'arrive' not in str(exc):raise
    if last is None:raise RuntimeError('complete 8 x 6 checkerboard not visible; no CAN opened')
    p,f=last;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if p['reprojection_rms_px']>1.5*f['bgr'].shape[1]/640:
        raise RuntimeError('board reprojection error too high; no CAN opened')
    cv2.imwrite(str(folder/'board.jpg'),f['bgr'])
    np.savez_compressed(folder/'depth.npz',depth_m=f['depth_m'],K=f['K'],distortion=f['distortion'])
    return p,f


def validate_solution(base_link,camera_board):
    """Refit independent leave-one-out subsets; no self-consistency-only pass."""
    full=cm.solve_hand_eye(base_link,camera_board);fits=[]
    for omit in range(len(base_link)):
        fit=cm.solve_hand_eye([p for i,p in enumerate(base_link) if i!=omit],
                              [p for i,p in enumerate(camera_board) if i!=omit])
        fits.append(fit['T_link_camera'])
    translation,rotation=cm.spread(fits)
    full['leave_one_out_translation_spread_m']=translation;full['leave_one_out_rotation_spread_rad']=rotation
    full['stable']=bool(full['views']>=7 and full['translation_rms_m']<=.003 and
        full['rotation_rms_rad']<=np.deg2rad(.6) and translation<=.005 and rotation<=np.deg2rad(.6))
    return full


def run(root=ROOT,*,side='right',execute=False,confirmed=False,ask=input):
    root=Path(root);store=Store(root);hw=store.read('hardware.json');cal=store.read('calibration.json')
    problems=[m for m in preflight(root)['missing'] if 'reference' in m or 'ACT' in m or 'startup target' in m]
    if problems:raise RuntimeError('software reference check failed: '+'; '.join(problems))
    folder=root/'records'/(time.strftime('%Y%m%d-%H%M%S-handeye-')+uuid.uuid4().hex[:6]);journal=Journal(folder,{'mode':'wrist_calibration','execute_requested':execute})
    board_spec=copy.deepcopy(BOARD);board=cm.calibration_board(board_spec)
    cam_config=dict(hw['cameras'][side]);cam_config.update(fps=15,resolution=[1280,720])
    camera=Camera(cam_config,journal.record);driver=None;servo=None;stop=threading.Event();reader=None
    sensor={'last_seen':0.,'pose':None,'frame':None,'error':None};lock=threading.Lock();views=[]
    clock_rows=[];moving_views=[];feedback={}
    report={'board':board_spec,'camera':side,'serial':cam_config['serial'],'fps':15,
            'task_accepted':False,'physical_acceptance':False,'motion_commands':0,'can_opened':False}
    try:
        camera.open(operator_confirmed=True);p,f=preview(camera,board,folder)
        report.update(preview_reprojection_rms_px=p['reprojection_rms_px'],clock=clock_check(f))
        if not execute:
            report['status']='preview_only';print('棋盘已识别；预览结束，没有打开 CAN 或电机。',flush=True);return report
        if not confirmed:raise RuntimeError('--execute requires --confirm-wrist-clearance; no CAN opened')
        print('首次手眼采样：不做 ACT 初始化，从实测姿态保持双臂；只转动%s腕第6/7关节，最多±0.10rad（5.7°），模型点位移最多25mm。'%side,
              '必须：两手空夹、两个手腕和夹爪周围都至少50mm无障碍；板固定；现场有人掌握急停。',
              '绝对桌面碰撞模型尚未标定，这一阶段依赖你确认上述空隙；不等于抓瓶任务准入。',sep='\n',flush=True)
        if ask('确认这些条件并允许标定采样，请输入 CALIBRATE：').strip()!='CALIBRATE':
            raise RuntimeError('operator did not confirm; no CAN opened')
        report['onsite_clearance_confirmed']=True
        # All hardware use below is explicit, commissioning-only and journaled.
        sys.setswitchinterval(.001);driver=LegacyArm(hw,journal.record);driver.open(operator_confirmed=True);report['can_opened']=True
        state=driver.read();anchor=finite(state['q'],(16,))
        sample_cal=session_calibration(cal);collision=store.read('collision.json');cm.jaw_sweeps(collision,hw['urdf'])
        planner=Planner(hw,sample_cal,collision)
        envelope=WristEnvelope(planner,anchor,side);targets=envelope.targets()
        journal.record('commissioning_plan',{'anchor':anchor.tolist(),'targets':[q.tolist() for q in targets],
            'active_joints':envelope.active,'max_mesh_displacement_m':max(envelope.displacement(q) for q in targets),
            'absolute_table_collision_checked':False,'operator_clearance_mm':50})
        if np.max(abs(finite(driver.read()['q'],(16,))-anchor))>.01:raise RuntimeError('robot moved during preview; activation refused')
        sensor.update(last_seen=0.,pose=p,frame=f)
        initial_plane_distance=abs(float(p['T_camera_board'][:3,2]@p['T_camera_board'][:3,3]))
        def feed():
            prior=None
            while not stop.is_set():
                try:
                    frame=camera.read(diagnostic=True);pose=cm.board_pose(frame,board)
                    clock=clock_check(frame)
                    if not clock['global_clock'] or not -.02<=clock['host_minus_device_s']<=.20:
                        raise RuntimeError('global exposure timestamp not fresh enough for commissioning')
                    if prior is not None and frame['frame_number']<=prior:raise RuntimeError('duplicate camera frame')
                    prior=frame['frame_number']
                    clock_rows.append((frame['device_timestamp_ms'],frame['arrival_unix_s']))
                    if servo is not None:
                        for value in list(servo.feedback):feedback[value['timestamp_unix_s']]=list(value['q'])
                    with lock:
                        if pose is not None and pose['reprojection_rms_px']<=1.5*frame['bgr'].shape[1]/640:
                            distance=abs(float(pose['T_camera_board'][:3,2]@pose['T_camera_board'][:3,3]))
                            if distance<max(.08,initial_plane_distance-.025):raise RuntimeError('camera-to-board clearance shrank beyond commissioning envelope')
                            sensor.update(last_seen=time.monotonic(),pose=pose,frame=frame)
                            moving_views.append({'device_ms':frame['device_timestamp_ms'],'R_camera_board':pose['T_camera_board'][:3,:3].copy()})
                except Exception as exc:
                    with lock:sensor['error']=str(exc)
                    return
        reader=threading.Thread(target=feed,daemon=True);reader.start()
        def guard():
            with lock:last=sensor['last_seen'];error=sensor['error']
            if error or time.monotonic()-last>.6:raise RuntimeError('calibration camera watchdog: '+str(error or 'board lost'))
            if servo is not None:
                q=finite(servo.state['q'],(16,))
                if np.max(abs(q[envelope.fixed]-anchor[envelope.fixed]))>.02:raise RuntimeError('non-wrist joint drifted')
                if np.max(abs(q[envelope.active]-anchor[envelope.active]))>.125:raise RuntimeError('measured wrist envelope exceeded')
        deadline=time.monotonic()+2.
        while time.monotonic()<deadline:
            with lock:seen=sensor['last_seen'];error=sensor['error']
            if seen or error:break
            time.sleep(.01)
        guard()
        if np.max(abs(finite(driver.read()['q'],(16,))-anchor))>.01:raise RuntimeError('robot moved before activation')
        driver.controller.activate_at_current_position(anchor,limits=planner.limits,motion_checks=True)
        servo=Servo(driver,planner.limits,journal.record,rate=30,tracking_rad=.02);servo.guard=guard;servo.start()
        # Enable may settle by a small amount. Keep the newly measured fixed
        # joints at their hold command; never command them back to an old read.
        with servo.lock:held_anchor=servo.target.copy()
        if np.max(abs(held_anchor-anchor))>.01:raise RuntimeError('enable settling exceeded commissioning limit')
        anchor=held_anchor;envelope=WristEnvelope(planner,anchor,side);targets=envelope.targets()
        paths=[];prior=anchor
        for target in [*targets,anchor]:
            path=retime(prior,target,velocity=.04,acceleration=.08,jerk=.5)
            if not envelope.validate_path([prior,*path],{}):raise RuntimeError('full commissioning path rejected before first wrist move')
            paths.append(path);prior=target
        link='openarm_'+side+'_hand'
        for index,target in enumerate(targets):
            guard()
            servo.follow(paths[index],envelope,{'revision':index},guard=guard);report['motion_commands']+=1
            time.sleep(.5);poses=[];qs=[];frames=[]
            for _ in range(5):
                guard()
                with lock:pose=copy.deepcopy(sensor['pose']);frame=dict(sensor['frame'])
                q=finite(driver.read()['q'],(16,))
                if np.max(abs(q-target))>.02:raise RuntimeError('static calibration pose not reached')
                poses.append(pose['T_camera_board']);qs.append(q);frames.append(frame);time.sleep(.09)
            if np.max(np.ptp(np.asarray(qs)[:,np.r_[0:7,8:15]],axis=0))>.003:
                raise RuntimeError('arm did not settle enough for a static hand-eye measurement')
            q=np.median(qs,axis=0);b=cm.mean_pose(poses);a=planner.kin.pose(link,planner.joints(q))
            view={'q':q.tolist(),'T_base_link':a.tolist(),'T_camera_board':b.tolist(),
                  'reprojection_rms_px':pose['reprojection_rms_px'],'frame_number':frame['frame_number']}
            journal.record('handeye_view',view);views.append(view)
            import cv2
            cv2.imwrite(str(folder/('view-%02d.jpg'%index)),frame['bgr'])
            print('手眼采样 %d/%d，重投影 %.2fpx'%(index+1,len(targets),view['reprojection_rms_px']),flush=True)
        servo.follow(paths[-1],envelope,{'revision':100},guard=guard)
        result=validate_solution([np.array(v['T_base_link']) for v in views],[np.array(v['T_camera_board']) for v in views])
        report.update(status='sampled',handeye=result,views=views)
        if len(clock_rows)>=20 and len(feedback)>=30:
            try:
                ds,arrivals=zip(*list(clock_rows));clock=cm.clock_envelope(ds,arrivals)
                def base_camera(q):return planner.kin.pose(link,planner.joints(q))@result['T_link_camera']
                rows=sorted(list(feedback.items()));motion=list(moving_views);chosen=[]
                for value in motion[::max(1,len(motion)//200)]:
                    stamp=value['device_ms']*clock['seconds_per_ms']+clock['offset_unix_s']
                    a=cm.interpolate(rows,stamp-.02);b=cm.interpolate(rows,stamp+.02)
                    if a is not None and b is not None and cm.angle(base_camera(a)[:3,:3].T@base_camera(b)[:3,:3])/.04>.03:
                        chosen.append(value)
                fit=cm.time_offset(chosen,rows,base_camera,result['T_base_board'],clock=clock)
                report['timing']={'domain':f['clock_domain'],'seconds_per_ms':clock['seconds_per_ms'],
                    'offset_unix_s':fit['offset_unix_s'],'maximum_error_s':fit['maximum_error_s'],'fit':fit,'envelope':clock}
            except Exception as exc:report['timing_error']=str(exc)
        # Save an accurately labelled candidate, never fill acceptance flags or
        # claim a timing calibration from stationary frames.
        if result['stable']:
            cal=store.read('calibration.json');cal['camera'].update(serial=cam_config['serial'],mounted_link=link,
                T_link_camera=result['T_link_camera'].tolist(),stream_profile={'resolution':cam_config['resolution'],'fps':cam_config['fps']},
                time_sync_verified=False,clock_model=None)
            aids=cal.setdefault('calibration_aids',{});aids.update(board=board_spec,handeye_candidate=str(folder.relative_to(root)))
            aids['board_table_candidate_z_m']=float(result['T_base_board'][2,3])
            if abs(float(result['T_base_board'][2,2]))>=.995:
                cal['table_top_z_m']=float(result['T_base_board'][2,3])
                if cal.get('support_tolerance_m') is None:cal['support_tolerance_m']=.008
            timing=report.get('timing',{})
            if timing.get('maximum_error_s',9)<=.02:
                cal['camera'].update(time_sync_verified=True,clock_model={'serial':cam_config['serial'],
                    'domain':timing['domain'],'seconds_per_ms':timing['seconds_per_ms'],
                    'offset_unix_s':timing['offset_unix_s'],'maximum_error_s':timing['maximum_error_s'],
                    'valid_until_unix_s':time.time()+2*3600,'source':str(folder.relative_to(root))})
            store.write('calibration.json',cal)
        return report
    except BaseException as exc:
        report.update(status='failed',error=str(exc),error_type=type(exc).__name__)
        raise
    finally:
        stop.set()
        if servo is not None:
            try:servo.close()
            except Exception as exc:report['hold_cleanup_error']=str(exc)
            try:
                before=finite(driver.read()['q'],(16,));drift=0.;end=time.monotonic()+3
                while time.monotonic()<end:
                    current=finite(driver.read()['q'],(16,));drift=max(drift,float(np.max(abs(current[np.r_[0:7,8:15]]-before[np.r_[0:7,8:15]]))))
                    time.sleep(.05)
                report['post_stop_hold']={'seconds':3,'maximum_joint_drift_rad':drift,'within_0_01_rad':drift<=.01}
                if drift>.01:print('停止后检测到关节漂移，不能准入；请现场接管。',flush=True)
            except Exception as exc:report['post_stop_hold']={'error':str(exc)}
        if reader is not None:reader.join(2)
        try:camera.close()
        except Exception as exc:report['camera_cleanup_error']=str(exc)
        if driver is not None:
            try:driver.close()
            except Exception as exc:report['driver_cleanup_error']=str(exc)
        report['task_accepted']=False;report['physical_acceptance']=False
        (folder/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda v:np.asarray(v).tolist())+'\n')
        journal.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',choices=['left','right'],default='right')
    p.add_argument('--execute',action='store_true');p.add_argument('--confirm-wrist-clearance',action='store_true');a=p.parse_args(argv)
    print(json.dumps(run(side=a.camera,execute=a.execute,confirmed=a.confirm_wrist_clearance),ensure_ascii=False,indent=2,default=lambda v:np.asarray(v).tolist()))


if __name__=='__main__':main()
