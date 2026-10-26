"""Strict calibration schema and evidence binding; never invent measurements."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

try:
    from .evidence import finite,transform
    from .planning import canonical
except ImportError:
    from evidence import finite,transform
    from planning import canonical


CATEGORIES=('camera_extrinsics','object_geometry','gripper_contact','timing','motion_planner','watchdog')


def implementation(root):
    result={name:hashlib.sha256((Path(root)/'app'/(name+'.py')).read_bytes()).hexdigest()
        for name in ['robot','service','devices','evidence','motion','planning','runtime','relative','calibration']
        if (Path(root)/'app'/(name+'.py')).is_file()}
    for name in ['app/skill/program.py','config/robot.json']:
        p=Path(root)/name
        if p.is_file():result[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def subject(calibration,hardware,collision,code=None):
    values={k:v for k,v in calibration.items() if k not in ('acceptance','status','note')}
    return canonical({'calibration':values,'hardware':hardware,'collision':collision,'implementation':code or {}})


def check(cal,hardware,collision,root):
    missing=[]
    config_path=Path(root)/'config/robot.json'
    relative_required=(config_path.is_file() and
        json.loads(config_path.read_text()).get('robot_contract')=='openarm-relative-v1/1')
    def need(name,fn):
        try:
            if fn() is not True:raise ValueError('unverified')
        except (KeyError,TypeError,ValueError,IndexError):missing.append(name)
    def positive(value):return type(value) in (float,int) and math.isfinite(value) and value>0
    need('table geometry',lambda:math.isfinite(cal['table_top_z_m']) and positive(cal['support_tolerance_m']) and cal['support_tolerance_m']<=.01)
    need('camera identity/mount',lambda:bool(cal['camera']['serial']) and bool(cal['camera']['mounted_link']))
    need('camera extrinsic transform',lambda:bool(transform(cal['camera']['T_link_camera']).shape==(4,4)))
    def clock():
        c=cal['camera'];m=c['clock_model']
        return (c['time_sync_verified'] is True and m['serial']==c['serial'] and bool(m['domain']) and
            .00095<=m['seconds_per_ms']<=.00105 and math.isfinite(m['offset_unix_s']) and
            0<=m['maximum_error_s']<=.02 and m['valid_until_unix_s']>time.time())
    need('camera time synchronization',clock)
    for name in ['bottle','basket']:
        def marker(name=name):
            c=cal['objects'][name]
            return (type(c['marker_id']) is int and 0<=c['marker_id']<50 and positive(c['marker_size_m']) and
                positive(c['maximum_reprojection_px']) and positive(c['maximum_depth_residual_m']) and
                positive(c['accepted_pose_uncertainty_m']) and transform(c['marker_to_object']).shape==(4,4))
        need(name+' registered RGB-D geometry',marker)
    need('bottle dimensions',lambda:positive(cal['objects']['bottle']['body_height_m']) and
        positive(cal['objects']['bottle']['body_radius_m']) and math.isfinite(cal['objects']['bottle']['body_bottom_m']))
    need('basket interior',lambda:bool(np.all(finite(cal['objects']['basket']['inner_size_m'],(3,))>0)) and
        math.isfinite(cal['objects']['basket']['floor_z_m']) and positive(cal['objects']['basket']['wall_radius_m']))
    for side in ['left','right']:
        def grip(side=side):
            c=cal['grippers'][side]
            for name in ['motor_limits','contact_motor_range','contact_torque_range']:
                a=finite(c[name],(2,))
                if not a[0]<a[1]:return False
            for name in ['open_motor','close_motor']:
                if not c['motor_limits'][0]<=c[name]<=c['motor_limits'][1]:return False
            bounds=finite(c['object_center_bounds_tool_m'],(2,3))
            if not np.all(bounds[0]<bounds[1]):return False
            pads=np.asarray(c['pad_extent_points_tool_m'],float)
            return (pads.ndim==2 and pads.shape[1]==3 and len(pads)>=8 and np.isfinite(pads).all() and
                all(positive(c[n]) for n in ['maximum_slip_m','maximum_rotation_rad','accepted_tool_uncertainty_m']))
        need(side+' complete grip calibration',grip)
    def motion():
        c=cal['motion'];bounds=finite(c['workspace_bounds_m'],(2,3))
        if not np.all(bounds[0]<bounds[1]):return False
        if set(c['tool_links'])!={'left','right'} or not all(c['tool_links'].values()):return False
        if not all(positive(c[k]) for k in ['collision_margin_m','velocity_rad_s','acceleration_rad_s2','scene_max_age_s','lift_m','pregrasp_m']):return False
        if c['velocity_rad_s']>.15 or c['acceleration_rad_s2']>.3 or c['scene_max_age_s']>.2:return False
        if not c['right_grasps'] or len(c['right_grasps'])>32:return False
        for row in c['right_grasps']:transform(row['T_object_tool'])
        transform(c['T_bottle_left_tool']);transform(c['T_basket_bottle'])
        if 'relative' in c or relative_required:
            r=c['relative']
            if r['schema']!='openarm-relative-goals/1' or r['handover_frame']!='session_start_tool_midpoint':return False
            for name in ['handover_offset_m','donor_retreat_base_m']:
                vector=finite(r[name],(3,))
                if np.linalg.norm(vector)>.3:return False
            if np.linalg.norm(r['donor_retreat_base_m'])<.01:return False
            if not 0<r['correction_step_m']<=.006 or not 0<r['correction_tolerance_m']<=.003:return False
            if not 0<r['rotation_tolerance_rad']<=.03 or not 0<r['maximum_target_shift_m']<=.02:return False
        else:transform(c['T_base_handover_bottle'])
        return bool(c['contact_links']['left']) and bool(c['contact_links']['right']) and isinstance(c['table_mount_links'],list)
    need('measured relative grasp/transfer/place and motion limits',motion)
    need('mesh collision model and full jaw sweep',lambda:isinstance(collision,dict) and collision.get('schema')=='openarm-collision-envelopes/2'
        and len(collision.get('hulls',[]))>=20 and collision.get('physical_trial_passed') is True
        and collision.get('full_jaw_sweep_verified') is True)
    def jaw_sweeps():
        fingers=[r for r in collision['hulls'] if r['link'].endswith('_finger')]
        if len(fingers)!=4:return False
        for row in fingers:
            points=np.asarray(row['jaw_sweep_hull_vertices'],float)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<8 or not np.isfinite(points).all():return False
        return True
    need('measured full-range finger collision volumes',jaw_sweeps)
    digest=subject(cal,hardware,collision,implementation(root))
    for key in CATEGORIES:
        def evidence(key=key):
            e=cal['acceptance'][key];p=(Path(root)/e['record']).resolve()
            if not p.is_relative_to(Path(root).resolve()) or not p.is_file():return False
            if hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256'] or e['physical_trial_passed'] is not True:return False
            row=json.loads(p.read_text())
            return (row.get('schema')=='openarm-physical-check/1' and row.get('category')==key and
                row.get('passed') is True and row.get('subject_sha256')==digest and
                row.get('valid_until_unix_s',0)>time.time() and bool(row.get('operator')) and bool(row.get('measurements')))
        need(key+' bound physical acceptance',evidence)
    return missing
