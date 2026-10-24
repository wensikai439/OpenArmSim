"""Explicit opt-in hardware adapters; importing this module opens no devices.

Do not import the old camera wrapper: its module body enumerates cameras.
Do not import the old policy launcher: its dependencies include policy runtimes.
"""
import ast
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

try:
    from .evidence import finite,transform
except ImportError:
    from evidence import finite,transform


def pinned(path,digest):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('reference changed: '+str(path))
    return path


class Camera:
    def __init__(self,config,record):
        self.config=config;self.record=record;self.pipeline=None

    def open(self,*,operator_confirmed=False):
        if not operator_confirmed:raise RuntimeError('camera requires an explicit on-site device session')
        import pyrealsense2 as rs
        self.rs=rs;self.pipeline=rs.pipeline();c=rs.config()
        c.enable_device(self.config['serial'])
        w,h=self.config['resolution'];fps=self.config['fps']
        c.enable_stream(rs.stream.color,w,h,rs.format.bgr8,fps)
        c.enable_stream(rs.stream.depth,w,h,rs.format.z16,fps)
        try:
            profile=self.pipeline.start(c)
            self.scale=profile.get_device().first_depth_sensor().get_depth_scale()
            self.align=rs.align(rs.stream.color)
        except Exception:
            self.close();raise

    def read(self):
        started=time.time();frames=self.pipeline.wait_for_frames(500)
        frames=self.align.process(frames);rgb=frames.get_color_frame();depth=frames.get_depth_frame()
        if not rgb or not depth:raise RuntimeError('incomplete RGB-D frame')
        # Host arrival is not exposure time. Preserve SDK clock/domain and
        # reject slow fetches; acceptance also needs a measured latency bound.
        arrived=time.time()
        if arrived-started>.15:raise RuntimeError('camera fetch latency exceeded')
        intr=rgb.profile.as_video_stream_profile().intrinsics
        return {'bgr':np.asarray(rgb.get_data()).copy(),
            'depth_m':np.asarray(depth.get_data()).copy()*self.scale,
            'K':np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]]),
            'distortion':np.array(intr.coeffs),'distortion_model':str(intr.model),
            'timestamp_unix_s':arrived,'acquisition_started_unix_s':started,
            'device_timestamp_ms':rgb.get_timestamp(),'clock_domain':str(rgb.get_frame_timestamp_domain()),
            'frame_number':rgb.get_frame_number(),'serial':self.config['serial']}

    def close(self):
        if self.pipeline is not None:
            try:self.pipeline.stop()
            except RuntimeError:pass
            self.pipeline=None


def marker_poses(frame,config,base_camera):
    """Registered marker baseline; requires physical IDs/sizes/extrinsics.

    No marker is assumed to exist on the user's bottle. Missing marker => no
    pose. Wrist base_camera must be recomputed from synchronized feedback.
    """
    import cv2
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector=cv2.aruco.ArucoDetector(dictionary,cv2.aruco.DetectorParameters())
    corners,ids,_=detector.detectMarkers(frame['bgr']);result={}
    if ids is None:return result
    for pixels,ident in zip(corners,ids.flatten()):
        matches=[(name,c) for name,c in config.items() if c.get('marker_id')==int(ident)]
        if len(matches)!=1:continue
        name,c=matches[0];half=c['marker_size_m']/2
        points=np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]])
        pixels=np.asarray(pixels,dtype=float).reshape(4,2)
        ok,rvec,tvec=cv2.solvePnP(points,pixels,frame['K'],frame['distortion'],flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or tvec[2,0]<=0:continue
        projected,_=cv2.projectPoints(points,rvec,tvec,frame['K'],frame['distortion'])
        rms=float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-pixels)**2,axis=1))))
        if rms>c['maximum_reprojection_px']:continue
        # Depth is checked on the marker itself, not on the object centre or
        # on an unrelated bounding-box centre. Missing/reflective depth fails.
        mask=np.zeros(frame['depth_m'].shape,dtype=np.uint8)
        cv2.fillConvexPoly(mask,np.rint(pixels).astype(np.int32),1)
        depths=frame['depth_m'][mask.astype(bool)];depths=depths[(depths>.05)&(depths<3.)]
        if len(depths)<20 or abs(float(np.median(depths))-float(tvec[2,0]))>c['maximum_depth_residual_m']:continue
        camera_marker=np.eye(4);camera_marker[:3,:3]=cv2.Rodrigues(rvec)[0];camera_marker[:3,3]=tvec[:,0]
        pose=transform(base_camera)@camera_marker@transform(c['marker_to_object'])
        result[name]={'T_base_object':pose.tolist(),'timestamp_unix_s':frame['timestamp_unix_s'],
            'source':'measured_marker_rgbd','frame_number':frame['frame_number'],
            'marker_id':int(ident),'reprojection_rms_px':rms,'uncertainty_m':c['accepted_pose_uncertainty_m']}
    return result


class LegacyArm:
    """Read-only reuse of original driver and the exact ACT startup function.

    Construction is inert. open(), initialize() and send() are hardware calls
    and are never used by preflight, package checks or offline tests.
    """
    def __init__(self,config,record):
        self.config=config;self.record=record;self.controller=None
        self.target=finite(config['startup_target'],(16,));self.stopped=False

    def open(self,*,operator_confirmed=False):
        if not operator_confirmed:raise RuntimeError('CAN requires an explicit on-site device session')
        c=self.config;root=Path(c['deploy_root'])
        for path,digest in c['source_hashes'].items():pinned(root/path,digest)
        # The source path must be explicit. No policy/camera launcher import.
        for path in [str(root),str(root/'openarm')]:
            if path not in sys.path:sys.path.insert(0,path)
        spec=importlib.util.spec_from_file_location('openarm_harness_legacy',root/'openarm/openarm_controller.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.controller=module.OpenArmController(enable_left=True,enable_right=True,startup_mode='deferred')
        original=self.controller.get_position
        def audited_position(arm):
            value=original(arm)
            side='left' if arm is self.controller.left_arm else 'right'
            reader=self.controller._feedback_readers[side]
            self.record('can_feedback',{'side':side,'motors':[
                {'position':s.position,'velocity':s.velocity,'torque':s.torque,'rx_ns':s.rx_ns,
                 'can_id':s.can_id,'raw_hex':s.raw.hex()} for s in reader.last_snapshot]})
            return value
        self.controller.get_position=audited_position
        self.record('hardware_open',{'enabled':False,'startup_mode':'deferred'})

    def read(self):
        c=self.controller;rows={};q=[]
        # get_position creates and uses the original fresh readers. Preserve
        # position/velocity/torque/RX timestamp instead of SDK cached position.
        for side in ['left','right']:
            c.get_position(getattr(c,side+'_arm'))
            reader=c._feedback_readers[side]
            reader.require_fresh_snapshot();samples=reader.last_snapshot
            q.extend(s.position for s in samples)
            rows[side]=[{'position':s.position,'velocity':s.velocity,'torque':s.torque,
                'timestamp_unix_s':s.rx_ns/1e9,'can_id':s.can_id,'raw_hex':s.raw.hex()} for s in samples]
        result={'q':q,'motors':rows,'timestamp_unix_s':min(r['timestamp_unix_s'] for a in rows.values() for r in a)}
        self.record('feedback',result);return result

    def send(self,target,state):
        q=finite(target,(16,));measured=finite(state['q'],(16,));c=self.controller
        if self.stopped:raise RuntimeError('driver stop is latched')
        age=time.time()-state['timestamp_unix_s']
        if not -.01<=age<=.1:raise RuntimeError('stale or invalid command feedback')
        self.record('command_requested',{'target':q.tolist(),'measured':measured.tolist()})
        c.remember_validated_state(measured)
        for side,i in [('left',0),('right',8)]:
            getattr(c,'set_'+side+'_position')(q[i:i+7].tolist(),float(q[i+7]),measured[i:i+7].tolist(),float(measured[i+7]))
        self.record('command_returned',{'target':q.tolist(),'qualification':'driver returned, not proof of motion or arrival'})

    def initialize(self,limits,should_stop):
        """Explicit caller must obtain on-site confirmation before open()."""
        c=self.controller;state=np.array(self.read()['q']);c.remember_validated_state(state)
        c.activate_at_current_position(state,limits=limits,motion_checks=True)
        # Reuse the exact interpolation/5s hold from the pinned source via an
        # AST-selected function set. No top-level launcher/model imports run.
        source=Path(self.config['deploy_root'])/'model_launchers/models/dp/deploy_dp_dual_joint.py'
        names={'move_arms_to_initial_position','read_joint_state','get_current_state',
            'joints_to_state_vector','parse_dp_action','initial_joint_state','log_initialization_error'}
        tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=names:raise RuntimeError('startup functions changed')
        def validate(values,bounds):
            q=finite(values,(16,))
            for side,i in [('left',0),('right',8)]:
                low,high=bounds[side]
                if np.any(q[i:i+7]<low) or np.any(q[i:i+7]>high):raise ValueError('startup outside URDF limits')
            return q
        namespace={'np':np,'time':time,'LOGGER':logging.getLogger('startup'),
            '_STARTUP_TARGET':self.target.copy(),'ARM':np.r_[0:7,8:15],'GRIP':np.array([7,15]),'validate_state':validate}
        # Use postponed annotations without evaluating the original types.
        for node in nodes:
            node.returns=None
            for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:arg.annotation=None
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
        # Audit every old-controller command and measured state. Original source
        # is not edited, patched on disk, or imported as a policy launcher.
        old_left,old_right=c.set_left_position,c.set_right_position
        def logged(side,original):
            def invoke(q,g,current,cg):
                if should_stop():raise InterruptedError('initialization cancelled')
                self.record('initialization_command',{'side':side,'target':list(q)+[float(g)],'feedback':list(current)+[float(cg)]})
                return original(q,g,current,cg)
            return invoke
        c.set_left_position=logged('left',old_left);c.set_right_position=logged('right',old_right)
        try:namespace['move_arms_to_initial_position'](c,config=None,limits=limits,waypoints=[],should_stop=should_stop)
        finally:c.set_left_position=old_left;c.set_right_position=old_right
        measured=np.array(self.read()['q']);error=abs(measured-self.target)
        arrived=bool(max(error[np.r_[0:7,8:15]])<=np.deg2rad(3) and max(error[[7,15]])<=.1)
        self.record('initialization_result',{'target':self.target.tolist(),'measured':measured.tolist(),
            'absolute_error':error.tolist(),'arrived':arrived,'source':'same ACT target and interpolation; added arrival check'})
        if not arrived:raise RuntimeError('actual startup pose not reached')

    def close(self):
        self.stopped=True
        if self.controller is not None:
            c=self.controller
            try:
                if c._activated_sides:c.hold_last_validated_position()
            finally:c.close_feedback_readers()
            self.record('hardware_close',{'qualification':'last-valid hold attempted; not an emergency stop or motor disable'})
