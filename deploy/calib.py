"""现场标定向导：./run calib 查看进度和下一步。

Steps: markers -> measure -> handeye -> grip -> accept, all task-independent;
task poses are searched online per session. Only handeye and grip open devices, and only after the operator types an on-site confirmation. Every
derived value is shown before it is written; old configs are backed up.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import select
import sys
import threading
import time
import types

import numpy as np

try:
    from . import calib_math as cm
    from .calibration import check,subject,implementation
    from .devices import Camera,LegacyArm,marker_poses
    from .evidence import Kinematics,transform
    from .motion import Servo,retime
    from .planning import Planner,mesh_envelopes
    from .runtime import Runtime,attach_global
    from .service import Journal,preflight
except ImportError:
    import calib_math as cm
    from calibration import check,subject,implementation
    from devices import Camera,LegacyArm,marker_poses
    from evidence import Kinematics,transform
    from motion import Servo,retime
    from planning import Planner,mesh_envelopes
    from runtime import Runtime,attach_global
    from service import Journal,preflight

ROOT=Path(__file__).resolve().parents[1]
WIDE=[[-1.5,-1.5,-1.],[1.5,1.5,2.]]
CATEGORY_NAMES={'camera_extrinsics':'相机外参','object_geometry':'瓶篮几何','gripper_contact':'夹持判定',
    'timing':'曝光对时','motion_planner':'规划与避碰','watchdog':'停止与看门狗'}


class Quit(Exception):
    """Operator ended the session; the arms hold their last position."""


# ---------------------------------------------------------------- files

class Store:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.state=self.root/'records/calib'

    def read(self,name):return json.loads((self.root/'config'/name).read_text())

    def write(self,name,value):
        path=self.root/'config'/name;backup=self.state/'backups';backup.mkdir(parents=True,exist_ok=True)
        if path.is_file():(backup/(time.strftime('%Y%m%d-%H%M%S-')+name)).write_bytes(path.read_bytes())
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

    def save(self,kind,data):
        self.state.mkdir(parents=True,exist_ok=True)
        path=self.state/(kind+'-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v:np.asarray(v).tolist())+'\n')
        return path

    def latest(self,kind):
        rows=sorted(self.state.glob(kind+'-*.json')) if self.state.is_dir() else []
        return (rows[-1],json.loads(rows[-1].read_text())) if rows else (None,None)


# ---------------------------------------------------------------- console

class Console:
    """Line prompts and single-key input. Tests pass scripted answers/keys."""
    def __init__(self,lines=None,keys=None):
        self.lines=None if lines is None else iter(lines);self.keys=None if keys is None else iter(keys)
        self.fd=None;self.saved=None

    def say(self,*rows):print('\n'.join(str(r) for r in rows),flush=True)

    def line(self,prompt):
        if self.lines is not None:
            value=next(self.lines);print(prompt+value);return value
        if self.saved is None:return input(prompt)
        # Typed answers need echo and line editing even inside a key session.
        import termios,tty
        termios.tcsetattr(self.fd,termios.TCSADRAIN,self.saved)
        try:return input(prompt)
        finally:tty.setcbreak(self.fd)

    def yes(self,prompt,default=False):
        while True:
            v=self.line(prompt+(' [Y/n] ' if default else ' [y/N] ')).strip().lower()
            if not v:return default
            if v in ('y','yes','是','好','对'):return True
            if v in ('n','no','否','不'):return False

    def number(self,prompt,default=None,lo=None,hi=None,unit='mm'):
        while True:
            hint='' if default is None else ' [%g]'%default
            v=self.line('%s（%s）%s：'%(prompt,unit,hint)).strip()
            if not v and default is not None:return float(default)
            try:value=float(v)
            except ValueError:self.say('  请输入数字');continue
            if not math.isfinite(value) or (lo is not None and value<lo) or (hi is not None and value>hi):
                self.say('  超出合理范围 %s~%s %s，请检查'%(lo,hi,unit));continue
            return value

    def choice(self,prompt,options,default):
        while True:
            v=self.line(prompt+' '+'/'.join(options)+' [%s]：'%default).strip() or default
            if v in options:return v

    def __enter__(self):
        if self.keys is None and sys.stdin.isatty():
            import termios,tty
            self.fd=sys.stdin.fileno();self.saved=termios.tcgetattr(self.fd);tty.setcbreak(self.fd)
        return self

    def __exit__(self,*_):
        if self.saved is not None:
            import termios
            termios.tcsetattr(self.fd,termios.TCSADRAIN,self.saved);self.saved=None

    def key(self):
        if self.keys is not None:
            try:return next(self.keys)
            except StopIteration:return 'q'
        if self.fd is None:return (sys.stdin.read(1) or 'q').lower()
        ch=os.read(self.fd,1).decode(errors='ignore')
        if ch=='\x1b':
            while select.select([self.fd],[],[],.02)[0]:os.read(self.fd,8)
            return ''
        return ch.lower()

    def flush(self):
        if self.fd is not None:
            import termios
            termios.tcflush(self.fd,termios.TCIFLUSH)


def on_site(console,extra=()):
    console.say('','⚠ 这一步会打开相机、使能电机并移动机械臂。开始前请确认：',
        '  · 有人站在机器人旁边，急停/断电开关在手边',
        '  · 机械臂运动范围内没有人和多余物品',*['  · '+v for v in extra],
        '  · 程序先用 ACT 同款初始化把双臂移到初始姿态（第71条），全程慢速',
        '  · 任何时候按 Ctrl+C 或 Q 都会停止并原地保持；结束后手臂保持在最后位置',
        '    （和 ACT 程序退出时一样），收臂请按你们平时的做法')
    if console.line('确认现场条件都满足，请输入 y 开始：').strip().lower()!='y':
        raise Quit('未确认现场条件，没有打开设备')


# ---------------------------------------------------------------- devices

def session_calibration(cal,table_z=None):
    """Planner inputs for an on-site session: ACT jaw range until measured,
    a wide workspace (taught bounds come from this session)."""
    c=copy.deepcopy(cal)
    for side in ['left','right']:
        if not c['grippers'][side].get('motor_limits'):c['grippers'][side]['motor_limits']=list(cm.ACT_JAW_RANGE)
    c['motion']['workspace_bounds_m']=copy.deepcopy(WIDE)
    measured=c.get('table_top_z_m') if table_z is None else table_z
    c['table_top_z_m']=-10. if measured is None else float(measured)
    if c.get('support_tolerance_m') is None:c['support_tolerance_m']=.008
    return c


class Rig:
    """Camera + both arms for one calibration session, servoed at 30 Hz."""
    def __init__(self,root,record,*,camera,hardware,calibration,collision,driver=None,device=None):
        self.root=Path(root);self.record=record;self.hw=hardware;self.collision=collision
        self.cal=session_calibration(calibration);self.camera_side=camera
        self.planner=Planner(self.hw,self.cal,self.collision)
        self.driver=driver or LegacyArm(self.hw,record)
        self.camera=device or Camera(dict(self.hw['cameras'][camera]),record)
        self.servo=None;self.stop=threading.Event();self.torque_ceiling=2.5;self.rest=None;self.servo_rate=30
        self.start=np.array(self.hw['startup_target'],float);self.camera_name=None;self.frames_seen=[]

    @property
    def table_z(self):return self.cal['table_top_z_m']

    def set_table(self,z):self.cal['table_top_z_m']=float(z)

    def open(self):
        table=self.cal.get('table_top_z_m')
        if table is None or not math.isfinite(table) or table<=-5:
            raise RuntimeError('现场桌面高度尚未实测；拒绝禁用桌面碰撞检查来初始化。先运行 ./run calib measure')
        # Perception, planning and the 30 Hz servo share one interpreter; a short
        # switch interval keeps the servo thread from queueing behind the others.
        sys.setswitchinterval(.001)
        self.camera.open(operator_confirmed=True)
        try:
            import pyrealsense2 as rs
            self.camera_name=self.camera.pipeline.get_active_profile().get_device().get_info(rs.camera_info.name)
        except Exception:pass
        self.driver.open(operator_confirmed=True)
        attach_global(self.planner,self.root,self.hw,self.cal)
        self.rest=np.array(self.driver.read()['q'],float)
        self.record('calibration_rest_state',{'q':self.rest.tolist(),'camera':self.camera_name})
        # Calibration initialization retains the measured table collision plane.
        self.driver.initialize(self.planner.limits,self.stop.is_set,planner=self.planner,scene=self.scene())
        self.servo=Servo(self.driver,self.planner.limits,self.record,rate=self.servo_rate)
        self.servo.guard=self._guard;self.servo.start()

    def _guard(self):
        if self.stop.is_set():raise RuntimeError('operator stop')
        s=self.servo.state if self.servo else None
        if s and 'motors' in s:
            for side in ['left','right']:
                if abs(float(s['motors'][side][7]['torque']))>self.torque_ceiling:
                    raise RuntimeError(side+' jaw torque above session ceiling')

    def alive(self):return bool(self.servo and not self.servo.failure)

    def scene(self,objects=None,*,contact=(),carried=None,place=False):
        """Runtime scene geometry (same obstacle construction), fresh timestamp."""
        now=time.time()
        if objects is None:
            return {'revision':0,'timestamp_unix_s':now,'table_z_m':self.table_z,'obstacles':[],'carried':None,
                'contact_links':[l for s in contact for l in self.cal['motion']['contact_links'][s]],
                'T_base_basket':np.eye(4).tolist(),'allow_basket_support':False}
        q=self.servo.target if self.servo else self.start;tools=self.planner.tools(q)
        bottle=objects['bottle'] if carried is None else tools[carried['arm']]@transform(carried['T_tool_object'])
        state={'object_poses':{'bottle':np.asarray(bottle).tolist(),'basket':np.asarray(objects['basket']).tolist()},
            'held_by':[] if carried is None else [carried['arm']],'pose_revision':1,'timestamp_unix_s':now,
            'exposure_aligned_tools':{s:{'T_base_tool':t.tolist(),'timestamp_unix_s':now} for s,t in tools.items()}}
        return Runtime.scene(types.SimpleNamespace(cal=self.cal,operation={'kind':'place'} if place else None),state,contact=contact)

    def tools(self,q=None):return self.planner.tools(self.servo.target if q is None else q)

    def arrive(self,timeout=3.):
        deadline=time.monotonic()+timeout;idx=np.r_[0:7,8:15]
        while time.monotonic()<deadline:
            if not self.alive():raise RuntimeError(self.servo.failure or 'servo stopped')
            with self.servo.lock:target=self.servo.target.copy()
            if np.max(abs(target[idx]-np.array(self.servo.state['q'])[idx]))<=.015:return
            time.sleep(.02)
        raise RuntimeError('measured arm arrival not reached')

    def move_joints(self,goal,scene_fn):
        with self.servo.lock:start=self.servo.target.copy()
        m=self.cal['motion'];path=retime(start,goal,velocity=m['velocity_rad_s'],acceleration=m['acceleration_rad_s2'],jerk=m['jerk_rad_s3'])
        self.servo.follow(path,self.planner,scene_fn(),guard=self._guard);self.arrive()

    def move_tool(self,targets,scene_fn):
        with self.servo.lock:start=self.servo.target.copy()
        path=self.planner.plan(start,targets,scene_fn())
        self.servo.follow(path,self.planner,scene_fn(),guard=self._guard);self.arrive()

    def jaw(self,side,command,scene_fn,*,stall=False):
        """Move one jaw at 0.1 unit/s; optionally stop at the first stable stall.
        Returns measured samples (command, position, velocity, torque)."""
        i=7 if side=='left' else 15
        with self.servo.lock:start=self.servo.target.copy()
        goal=start.copy();goal[i]=command;samples=[];direction=np.sign(command-start[i]) or 1.
        def until():
            s=self.servo.state;m=s['motors'][side][7]
            with self.servo.lock:cmd=float(self.servo.target[i])
            samples.append({'t':s['timestamp_unix_s'],'command':cmd,'position':float(m['position']),
                'velocity':float(m.get('velocity',0.)),'torque':float(m['torque'])})
            if not stall or len(samples)<6:return False
            last=samples[-6:]
            return all(abs(r['velocity'])<.05 and (r['command']-r['position'])*direction>.06 for r in last)
        path=retime(start,goal,velocity=.1,acceleration=.2,jerk=self.cal['motion']['jerk_rad_s3'])
        self.servo.follow(path,self.planner,scene_fn(),guard=self._guard,until=until)
        stalled=bool(stall and samples and until())
        time.sleep(.5);settle=[]
        for _ in range(10):
            s=self.servo.state;m=s['motors'][side][7]
            settle.append({'position':float(m['position']),'torque':float(m['torque'])});time.sleep(.033)
        return {'stalled':stalled,'samples':samples,'settled':settle}

    def frame(self):
        self.camera.read(diagnostic=True);f=self.camera.read(diagnostic=True)
        f['q']=list(self.servo.state['q']) if self.servo else None;self.frames_seen.append(f['device_timestamp_ms'])
        return f

    def still(self,n=3,settle=.4):
        time.sleep(settle);return [self.frame() for _ in range(n)]

    def close(self):
        self.stop.set();errors=[]
        for device in [self.servo,self.camera,self.driver]:
            if device:
                try:device.close()
                except Exception as exc:errors.append(str(exc))
        if errors:self.record('error',{'reason':'calibration cleanup','errors':errors})
        return errors


def camera_pose(rig,X,q):
    cam=rig.cal['camera']['mounted_link'];return rig.planner.kin.pose(cam,rig.planner.joints(q))@transform(X)


# ---------------------------------------------------------------- steps

def status(store=None,console=None):
    store=store or Store();console=console or Console();cal=store.read('calibration.json')
    collision=store.read('collision.json') if (store.root/'config/collision.json').is_file() else {}
    aids=cal.get('calibration_aids',{});missing=check(cal,store.read('hardware.json'),collision,store.root)
    software=[m for m in preflight(store.root)['missing'] if 'reference' in m or 'ACT' in m or 'startup' in m]
    steps=[('markers','打印标记','打印标定板和瓶子/篮子标记',bool(aids.get('markers_printed'))),
        ('measure','量尺寸','尺子量瓶子、篮子、标记，填进去（不开设备）',bool(aids.get('measured'))),
        ('handeye','手眼标定','现场：转动右手腕拍标定板，算相机位置和对时',bool(cal['camera'].get('time_sync_verified'))),
        ('grip','夹爪检查','现场：只动夹爪，测夹持力矩、桌面高度、相机能否看到夹住的瓶子',bool(aids.get('grip'))),
        ('accept','现场验收','逐项确认测量结果，生成验收记录',not missing)]
    console.say('','OpenArm 真机标定进度')
    for i,(cmd,name,desc,done) in enumerate(steps,1):
        console.say('  [%s] %d %-6s ./run calib %-8s %s'%('✓' if done else ' ',i,name,cmd,desc))
    if software:console.say('','  ⚠ ACT 参考源码不一致：'+'；'.join(software))
    nxt=next((cmd for cmd,_,_,done in steps if not done),None)
    console.say('','下一步：'+('./run calib '+nxt if nxt else '全部完成，可以运行 ./run grab'))
    return {'next':nxt,'missing':missing,'software':software}


def markers(store=None,console=None,dpi=300):
    store=store or Store();console=console or Console();cal=store.read('calibration.json')
    folder=store.state/'markers';folder.mkdir(parents=True,exist_ok=True)
    names=['1-handeye-board.png','2-bottle-basket-markers.png']
    for name,page in zip(names,cm.marker_sheets(dpi=dpi)):(folder/name).write_bytes(cm.png_dpi(page,dpi))
    cal.setdefault('calibration_aids',{})['markers_printed']=True;store.write('calibration.json',cal)
    console.say('','已生成两张 A4 图片：',*['  '+str(folder/n) for n in names],'',
        '打印要求：A4、按"实际大小/100%"打印，不要"适合页面"。打印后：',
        '  1. 标定板（第1张）贴在硬纸板上保持平整',
        '  2. 瓶子标记（ID 1）贴在一小块硬卡片上，再固定到瓶身侧面，箭头朝上；也可以平贴在瓶盖上',
        '  3. 篮子标记（ID 2）贴在篮子一个外侧面，箭头朝上',
        '  4. 用尺子量每张的黑色方块实际边长，下一步要填',
        '','取回本机打印：scp data-cs:'+str(folder)+'/*.png .','','下一步：./run calib measure')


def measure(store=None,console=None):
    """Ruler measurements; no device. Object frames: origin at the bottle bottom
    centre / basket inner-floor centre, +z up, +x along the marked face normal."""
    store=store or Store();console=console or Console()
    cal=store.read('calibration.json');hw=store.read('hardware.json');aids=cal.setdefault('calibration_aids',{})
    old=aids.get('ruler',{});n=console.number;o=cal['objects']
    console.say('','── 量尺寸（只填数，不开设备；回车用方括号里的值）','','[标定板]')
    r={'board_marker_mm':n('标定板单个黑方块边长',old.get('board_marker_mm',40),20,80),
       'board_gap_mm':n('标定板方块之间的间距',old.get('board_gap_mm',10),3,30)}
    console.say('','[桌面] 可量固定肩关节转轴中心到桌面的竖直距离，无需寻找基座原点。',
        '固定肩关节是最靠机架的第1关节，不是移动的肘部；认不准就先停下。')
    previous=cal.get('table_top_z_m')
    method=console.choice('量肩关节到桌面(shoulder)，或已知基座坐标 z(base)？',['shoulder','base'],old.get('table_reference','shoulder'))
    r['table_reference']=method
    if method=='shoulder':
        kin=Kinematics(hw['urdf']);zero={j['name']:0. for j in kin.joints.values()}
        zs=[kin.pose('openarm_'+side+'_link1',zero)[2,3]*1000 for side in ['left','right']]
        if abs(zs[0]-zs[1])>1.:raise ValueError('双肩高度不一致，不能使用肩关节量法')
        shoulder=float(np.mean(zs));console.say('  按当前 URDF：肩关节中心 z=%.1fmm；桌面 z=此值减去实测竖直距离。'%shoulder)
        r['shoulder_table_drop_mm']=n('肩关节转轴中心高于桌面的竖直距离',old.get('shoulder_table_drop_mm'),-500,1500)
        r['table_top_z_mm']=shoulder-r['shoulder_table_drop_mm']
        r['shoulder_reference_z_mm']=shoulder
    else:
        r['table_top_z_mm']=n('桌面 z（桌面高于基座原点为正）',old.get('table_top_z_mm',None if previous is None else previous*1000),-500,1500)
    cal['table_top_z_m']=r['table_top_z_mm']/1000
    console.say('','[瓶子]  以瓶底中心为原点')
    r['bottle_height_mm']=n('瓶子总高度（含瓶盖）',old.get('bottle_height_mm'),50,400)
    r['bottle_diameter_mm']=n('瓶身最粗处直径',old.get('bottle_diameter_mm'),20,120)
    r['bottle_marker_mm']=n('瓶子标记黑方块边长',old.get('bottle_marker_mm',40),15,80)
    r['bottle_mount']=console.choice('瓶子标记贴在 侧面(side) 还是 瓶盖顶上(top)？',['side','top'],old.get('bottle_mount','side'))
    if r['bottle_mount']=='side':
        r['bottle_marker_height_mm']=n('瓶子标记中心离桌面的高度',old.get('bottle_marker_height_mm'),10,r['bottle_height_mm'])
        r['bottle_marker_distance_mm']=n('标记表面到瓶子中心轴的水平距离（≈半径+卡片厚）',
            old.get('bottle_marker_distance_mm',round(r['bottle_diameter_mm']/2+2,1)),5,100)
        to_object=cm.side_marker(r['bottle_marker_height_mm']/1000,r['bottle_marker_distance_mm']/1000)
    else:
        r['bottle_marker_height_mm']=n('瓶盖上标记表面离桌面的高度',old.get('bottle_marker_height_mm',r['bottle_height_mm']),30,400)
        to_object=cm.top_marker(r['bottle_marker_height_mm']/1000)
    o['bottle'].update(marker_id=cm.BOTTLE_ID,marker_size_m=r['bottle_marker_mm']/1000,marker_to_object=to_object.tolist(),
        body_height_m=r['bottle_height_mm']/1000,body_bottom_m=0.,body_radius_m=r['bottle_diameter_mm']/2000)
    console.say('','[篮子]  以内底中心为原点，x 轴=贴标记那面的朝外方向')
    r['basket_floor_mm']=n('篮子内底离桌面的高度（底板厚度）',old.get('basket_floor_mm',3),0,100)
    r['basket_depth_mm']=n('内腔尺寸：垂直于贴标记那面的方向',old.get('basket_depth_mm'),60,600)
    r['basket_width_mm']=n('内腔尺寸：平行于贴标记那面的方向',old.get('basket_width_mm'),60,600)
    r['basket_inner_height_mm']=n('内腔高度（内底到篮口）',old.get('basket_inner_height_mm'),20,400)
    r['basket_wall_mm']=n('篮壁厚度',old.get('basket_wall_mm',5),1,40)
    r['basket_marker_mm']=n('篮子标记黑方块边长',old.get('basket_marker_mm',60),15,120)
    r['basket_marker_height_mm']=n('篮子标记中心离桌面的高度',old.get('basket_marker_height_mm'),5,400)
    distance=r['basket_depth_mm']/2+r['basket_wall_mm']+.5
    r['basket_marker_distance_mm']=n('篮子标记表面到篮子中心的水平距离',old.get('basket_marker_distance_mm',distance),30,400)
    o['basket'].update(marker_id=cm.BASKET_ID,marker_size_m=r['basket_marker_mm']/1000,
        marker_to_object=cm.side_marker((r['basket_marker_height_mm']-r['basket_floor_mm'])/1000,r['basket_marker_distance_mm']/1000).tolist(),
        inner_size_m=[r['basket_depth_mm']/1000,r['basket_width_mm']/1000,r['basket_inner_height_mm']/1000],
        floor_z_m=0.,wall_radius_m=max(r['basket_wall_mm'],6)/2000)
    console.say('','[腕部相机]  含支架的外形尺寸，用于避碰（保守一点填）')
    r['camera_size_mm']=[n('相机宽',(old.get('camera_size_mm') or [100,45,45])[0],20,200),
        n('相机高',(old.get('camera_size_mm') or [100,45,45])[1],15,120),
        n('相机厚（镜头到背面，含支架）',(old.get('camera_size_mm') or [100,45,45])[2],15,150)]
    for name in ['bottle','basket']:
        # Initial acceptance limits; the grip session replaces them with
        # bounds derived from the measured marker statistics.
        for key,value in [('maximum_reprojection_px',3.),('maximum_depth_residual_m',.02),('accepted_pose_uncertainty_m',.004 if name=='bottle' else .006)]:
            if o[name].get(key) is None:o[name][key]=value
    if cal.get('support_tolerance_m') is None:cal['support_tolerance_m']=.008
    # Task poses are searched online; the workspace is a generic box in front.
    cal['motion']['task_poses']='online'
    if cal['motion'].get('workspace_bounds_m') is None:cal['motion']['workspace_bounds_m']=[[-.3,-.9,-.5],[1.,.9,1.5]]
    collision=store.read('collision.json') if (store.root/'config/collision.json').is_file() else mesh_envelopes(hw)
    cm.jaw_sweeps(collision,hw['urdf'])
    aids['ruler']=r;aids['measured']=True;aids['board']={'marker_m':r['board_marker_mm']/1000,'gap_m':r['board_gap_mm']/1000}
    console.say('','将写入：',
        '  瓶子 高%.0fmm 半径%.1fmm，标记ID1 %.0fmm（%s）'%(r['bottle_height_mm'],r['bottle_diameter_mm']/2,r['bottle_marker_mm'],r['bottle_mount']),
        '  篮子 内腔%.0f×%.0f×%.0fmm，壁厚%.0fmm，标记ID2 %.0fmm'%(r['basket_depth_mm'],r['basket_width_mm'],r['basket_inner_height_mm'],r['basket_wall_mm'],r['basket_marker_mm']),
        '  夹爪手指按 URDF 全行程（每指 44mm）生成扫掠碰撞体积；现场 grip 时会让你量一次实际张开距离核对')
    if not console.yes('确认写入？',True):console.say('未写入');return None
    store.write('collision.json',collision);store.write('calibration.json',cal)
    console.say('已保存。下一步（需要到机器旁边）：./run calib handeye');return r


# Single-joint wrist moves (index, rad) around the ACT start pose, then pairs.
OFFSETS=[(14,.25),(14,-.25),(13,.25),(13,-.25),(12,.25),(12,-.25),(11,.15),(11,-.15),
    ((12,13),(.18,.18)),((12,13),(-.18,-.18)),((13,14),(.18,-.2)),((12,14),(-.18,.2))]


def handeye(store=None,console=None,*,camera='right',rig_factory=None):
    store=store or Store();console=console or Console()
    cal=store.read('calibration.json');hw=store.read('hardware.json');aids=cal.get('calibration_aids',{})
    if not aids.get('measured'):console.say('请先完成：./run calib measure');return None
    collision=store.read('collision.json');board=cm.grid_board(aids['board']['marker_m'],aids['board']['gap_m'])
    console.say('','── 手眼标定（%s腕相机）'%('右' if camera=='right' else '左'),
        '把标定板放在机器人正前方约 40~60cm（平放在桌上，或靠着东西立起来略向后仰），',
        '让%s手腕相机在初始姿态能看到整块板。标定过程中不要碰标定板。'%('右' if camera=='right' else '左'),
        '程序会把双臂移到 ACT 初始姿态，然后只转动%s手腕附近的关节（每次≤0.25rad，约14°），拍 12 个角度，'%('右' if camera=='right' else '左'),
        '再来回转动手腕测相机对时，最后测试停止保持。首次现场耗时尚未实测。')
    on_site(console,['标定板已放好、固定'])
    flat=console.yes('标定板是平放在桌面上的吗？（用于核对桌面高度）',True)
    journal=Journal(store.root/'records'/(time.strftime('%Y%m%d-%H%M%S')+'-calib-handeye'),{'mode':'calibration_handeye','camera':camera})
    rig=(rig_factory or Rig)(store.root,journal.record,camera=camera,hardware=hw,calibration=cal,collision=collision)
    link='openarm_%s_hand'%camera;rig.cal['camera']['mounted_link']=link
    base=7 if camera=='left' else 15;offset=-8 if camera=='left' else 0
    views=[];all_frames=[];result=None;sweep=None;watchdog=None
    try:
        console.say('正在打开设备并初始化到 ACT 初始姿态…')
        rig.open();start=rig.servo.target.copy();scene=lambda:rig.scene()
        def look(tag):
            poses=[]
            for f in rig.still(5):
                all_frames.append((f['device_timestamp_ms'],f['arrival_unix_s']))
                p=cm.board_pose(f,board)
                if p:poses.append((p,f))
            if len(poses)<2:return None
            p=poses[-1];return {'tag':tag,'q':p[1]['q'],'T_camera_board':cm.mean_pose([x[0]['T_camera_board'] for x in poses]),
                'reprojection_rms_px':max(x[0]['reprojection_rms_px'] for x in poses),'markers':min(x[0]['markers'] for x in poses),
                'K':p[1]['K'].tolist(),'domain':p[1]['clock_domain']}
        first=look('start')
        if first is None:
            console.say('✗ 初始姿态下看不到标定板（至少要看清 3 个方块）。请调整标定板位置后重新运行。');return None
        views.append(first);console.say('  初始姿态：看到 %d 个方块，重投影 %.2fpx'%(first['markers'],first['reprojection_rms_px']))
        for k,(index,delta) in enumerate(OFFSETS,1):
            goal=start.copy()
            for i,d in zip(np.atleast_1d(index),np.atleast_1d(delta)):goal[int(i)+offset]+=d
            try:rig.move_joints(goal,scene)
            except (ValueError,RuntimeError) as exc:
                if not rig.alive():raise
                console.say('  %2d/%d 跳过（%s）'%(k,len(OFFSETS),exc));continue
            v=look(k)
            console.say('  %2d/%d %s'%(k,len(OFFSETS),'看不到标定板，跳过' if v is None else
                '%d 个方块，重投影 %.2fpx'%(v['markers'],v['reprojection_rms_px'])))
            if v:views.append(v)
        rig.move_joints(start,scene)
        a=[rig.planner.kin.pose(link,rig.planner.joints(v['q'])) for v in views]
        result=cm.solve_hand_eye(a,[v['T_camera_board'] for v in views])
        console.say('','手眼结果：%d 个视角，标定板位置一致性 RMS %.1fmm / %.2f°'%(result['views'],
            result['translation_rms_m']*1000,math.degrees(result['rotation_rms_rad'])))
        sweep=timing(rig,board,result,start,base,scene,console,all_frames)
        if console.yes('现在做停止保持测试吗？程序会故意卡住 0.5 秒，软件看门狗应立即让手臂原地保持',True):
            watchdog=stall_test(rig)
            console.say('  看门狗 %.2f 秒后触发，之后 3 秒内关节最大漂移 %.2f°'%(watchdog['halt_latency_s'],math.degrees(watchdog['drift_rad'])))
    except Quit as exc:console.say('已停止：'+str(exc))
    except KeyboardInterrupt:console.say('已按 Ctrl+C 停止，手臂原地保持')
    finally:
        errors=rig.close()
        if watchdog is not None:watchdog['after_close']=after_close(rig)
        journal.close()
    if errors:console.say('⚠ 关闭设备时有错误：'+'；'.join(errors))
    if result is None or sweep is None:return None
    board_base=result['T_base_board'];table=float(board_base[2,3]) if flat else None
    report={'camera':camera,'camera_name':rig.camera_name,'serial':hw['cameras'][camera]['serial'],'mounted_link':link,
        'hand_eye':{k:v for k,v in result.items()},'views':[{k:v for k,v in x.items() if k!='q'}|{'q':list(x['q'])} for x in views],
        'board_flat_on_table':flat,'board_table_z_m':table,'board_normal_z':float(board_base[2,2]),
        'timing':sweep,'watchdog':watchdog,'record':str(journal.folder.relative_to(store.root)),
        'rest_q':None if rig.rest is None else rig.rest.tolist()}
    good=hand_eye_ok(result)
    console.say('','── 结果',
        '  相机：%s（序列号 %s），装在 %s'%(rig.camera_name or '未知型号',report['serial'],link),
        '  手眼一致性 RMS：%.1fmm / %.2f°  %s'%(result['translation_rms_m']*1000,math.degrees(result['rotation_rms_rad']),'✓' if good else '✗ 偏大（要求 RMS≤3mm/0.6°，≥7个视角）'),
        '  曝光对时：时钟域 %s，误差上界 %.1fms  %s'%(sweep['domain'],sweep['maximum_error_s']*1000,'✓' if sweep['maximum_error_s']<=.02 else '✗'),
        *(['  桌面高度（标定板）：z=%.3fm，板面倾斜 %.1f°'%(table,math.degrees(math.acos(min(1,abs(board_base[2,2])))))] if flat else []))
    path=store.save('handeye',report)
    if not good or sweep['maximum_error_s']>.02:
        console.say('','结果不合格，没有写入配置。建议：标定板放近一点、更平整，或换个位置让各角度都能看到，然后重跑。','记录：'+str(path));return report
    if not console.yes('写入相机外参和对时？',True):return report
    cal=store.read('calibration.json');collision=store.read('collision.json')
    clock={'serial':report['serial'],'domain':sweep['domain'],'seconds_per_ms':sweep['seconds_per_ms'],
        'offset_unix_s':sweep['offset_unix_s'],'maximum_error_s':sweep['maximum_error_s'],
        'valid_until_unix_s':time.time()+(30*86400 if 'global' in sweep['domain'].lower() else 2*3600),
        'source':'moving-board timing fit '+path.name}
    cal['camera'].update(serial=report['serial'],mounted_link=link,T_link_camera=result['T_link_camera'].tolist(),
        time_sync_verified=True,clock_model=clock)
    cal.setdefault('calibration_aids',{})['handeye']=path.name
    if table is not None:cal['calibration_aids']['board_table_z_m']=table
    size=[v/1000 for v in cal['calibration_aids']['ruler']['camera_size_mm']]
    cm.camera_hull(collision,link,result['T_link_camera'],size)
    store.write('collision.json',collision);store.write('calibration.json',cal)
    if 'global' not in sweep['domain'].lower():
        console.say('⚠ 相机时钟不是 global time，对时只在 2 小时内有效；建议在 RealSense 设置里打开 global time 后重做。')
    console.say('已保存。下一步：./run calib grip');return report


def hand_eye_ok(h):
    return h['views']>=7 and h['translation_rms_m']<=.003 and h['rotation_rms_rad']<=math.radians(.6)


def timing(rig,board,result,start,base,scene,console,all_frames):
    """Sweep the last wrist joint and fit the exposure clock against CAN feedback."""
    console.say('','测曝光对时：手腕来回转动约 10 秒…')
    # Sweep the wrist joint with the most room on both sides of the start pose.
    lo,hi=rig.planner.limits[rig.camera_side];first=base-7
    room={i:min(start[i]-lo[i-first],hi[i-first]-start[i]) for i in range(base-3,base)}
    joint=max(room,key=room.get);amp=min(.2,room[joint]-.03)
    if amp<.08:raise RuntimeError('wrist joints too close to their limits for a timing sweep')
    X=result['T_link_camera'];goal=lambda d:start+np.eye(16)[joint]*d
    rows={};views=[];error=[]
    def feed():
        for s in list(rig.servo.feedback):rows[s['timestamp_unix_s']]=list(s['q'])
    def run():
        try:
            for g in [goal(amp),goal(-amp),start]:rig.move_joints(g,scene)
        except Exception as exc:error.append(exc)
    worker=threading.Thread(target=run,daemon=True);worker.start();domain=None
    while worker.is_alive():
        f=rig.camera.read(diagnostic=True);feed();domain=f['clock_domain']
        all_frames.append((f['device_timestamp_ms'],f['arrival_unix_s']))
        p=cm.board_pose(f,board)
        if p:views.append({'device_ms':f['device_timestamp_ms'],'R_camera_board':p['T_camera_board'][:3,:3]})
    worker.join();feed()
    if error:raise error[0]
    d,a=zip(*all_frames);clock=cm.clock_envelope(d,a)
    fit=cm.time_offset(views,sorted(rows.items()),lambda q:camera_pose(rig,X,q),result['T_base_board'],clock=clock)
    return {'domain':domain,'seconds_per_ms':clock['seconds_per_ms'],'offset_unix_s':fit['offset_unix_s'],
        'maximum_error_s':fit['maximum_error_s'],'envelope':clock,'fit':{k:v for k,v in fit.items() if k!='offset_unix_s'},
        'frames':len(all_frames),'moving_views':len(views)}


def stall_test(rig):
    """Block the servo loop once; the independent watchdog must latch a hold."""
    blocked={'at':None}
    def guard():
        if blocked['at'] is None:blocked['at']=time.monotonic();time.sleep(.5)
    q0=np.array(rig.servo.state['q'],float);rig.servo.guard=guard
    deadline=time.monotonic()+2
    while not rig.servo.failure and time.monotonic()<deadline:time.sleep(.005)
    latency=time.monotonic()-(blocked['at'] or time.monotonic())
    drift=0.;end=time.monotonic()+3;idx=np.r_[0:7,8:15]
    while time.monotonic()<end:
        q=np.array(rig.driver.read()['q'],float);drift=max(drift,float(np.max(abs(q[idx]-q0[idx]))));time.sleep(.05)
    return {'halted':bool(rig.servo.failure),'reason':str(rig.servo.failure),'halt_latency_s':float(latency),'drift_rad':drift}


def after_close(rig):
    """Feedback for 3 s after the session sent its final hold and stopped."""
    try:
        q0=np.array(rig.driver.read()['q'],float);drift=0.;end=time.monotonic()+3;idx=np.r_[0:7,8:15]
        while time.monotonic()<end:
            q=np.array(rig.driver.read()['q'],float);drift=max(drift,float(np.max(abs(q[idx]-q0[idx]))));time.sleep(.05)
        rig.driver.controller.close_feedback_readers()
        return {'drift_rad':drift,'seconds':3}
    except Exception as exc:return {'error':str(exc)}


# ---------------------------------------------------------------- grip

def depth_residual(frame,camera_marker,config):
    import cv2
    half=config['marker_size_m']/2;points=np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0],[0,0,0]])
    pixels,_=cv2.projectPoints(points[:4],cv2.Rodrigues(camera_marker[:3,:3])[0],camera_marker[:3,3],frame['K'],frame['distortion'])
    mask=np.zeros(frame['depth_m'].shape,np.uint8);cv2.fillConvexPoly(mask,np.rint(pixels.reshape(4,2)).astype(np.int32),1)
    depths=frame['depth_m'][mask.astype(bool)];depths=depths[(depths>.05)&(depths<3.)]
    return None if len(depths)<20 else float(abs(np.median(depths)-camera_marker[2,3]))


def short(exc):
    text=str(exc)
    if 'local IK failed and ' in text:return '规划不出无碰撞路径（全局规划：%s）'%text.split('local IK failed and ',1)[1][:80]
    for key,value in [('no fully checked path','规划不出无碰撞路径'),('planner did not explicitly accept','路径碰撞检查不通过'),
                      ('measured arm arrival','没有到位'),('exceeds URDF','超出关节限位')]:
        if key in text:return value
    return text[:80]


def detect(rig,cal,stats=None):
    """Bottle/basket poses in one frame with relaxed limits; stats set thresholds."""
    f=rig.frame();cam=camera_pose(rig,cal['camera']['T_link_camera'],f['q']);relaxed=copy.deepcopy(cal['objects'])
    for o in relaxed.values():o.update(maximum_reprojection_px=8.,maximum_depth_residual_m=.06)
    found=marker_poses(f,relaxed,cam)
    if stats is not None:
        for name,p in found.items():
            marker=np.linalg.inv(cam)@np.array(p['T_base_object'])@np.linalg.inv(transform(cal['objects'][name]['marker_to_object']))
            stats.append({'object':name,'reprojection_rms_px':p['reprojection_rms_px'],'depth_residual_m':depth_residual(f,marker,cal['objects'][name])})
    return {k:np.array(v['T_base_object']) for k,v in found.items()}


def observe(rig,cal,n=6,stats=None):
    rows={};time.sleep(.4)
    for _ in range(n):
        for k,v in detect(rig,cal,stats).items():rows.setdefault(k,[]).append(v)
    return {k:cm.mean_pose(v) for k,v in rows.items() if len(v)>=max(2,n//2)}


def grip(store=None,console=None,*,rig_factory=None):
    """Task-independent: jaw contact model, table height, marker limits and
    whether the camera sees a held bottle. Only the jaws move after startup."""
    store=store or Store();console=console or Console()
    cal=store.read('calibration.json');hw=store.read('hardware.json')
    if not cal['camera'].get('time_sync_verified'):console.say('请先完成：./run calib handeye');return None
    console.say('','── 夹爪与场景检查（双臂到初始姿态后只有夹爪会动）',
        '准备：瓶子竖直放在桌上平时的位置，篮子放好，标记朝向机器人。')
    on_site(console,['瓶子和篮子已放好'])
    journal=Journal(store.root/'records'/(time.strftime('%Y%m%d-%H%M%S')+'-calib-grip'),{'mode':'calibration_grip'})
    side_cam='left' if 'left' in cal['camera']['mounted_link'] else 'right'
    rig=(rig_factory or Rig)(store.root,journal.record,camera=side_cam,hardware=hw,calibration=cal,collision=store.read('collision.json'))
    d={'marker_stats':[]};done=False
    try:
        console.say('正在初始化到 ACT 初始姿态…');rig.open();scene=lambda:rig.scene()
        found=observe(rig,rig.cal,stats=d['marker_stats'])
        if set(found)!={'bottle','basket'}:raise Quit('初始姿态下看不到瓶子或篮子的标记，请调整摆放后重来')
        bottom=float((found['bottle']@np.array([0,0,rig.cal['objects']['bottle']['body_bottom_m'],1]))[2])
        d.update(table_z_m=bottom,bottle=found['bottle'],basket=found['basket'],upright_z=float(found['bottle'][2,2]));rig.set_table(bottom)
        console.say('  桌面 z=%.3f，瓶子竖直度 %.3f'%(bottom,found['bottle'][2,2]))
        for side in ['right','left']:
            name='右' if side=='right' else '左';i=7 if side=='left' else 15
            hi=rig.cal['grippers'][side]['motor_limits'][1];opened=float(rig.start[i])
            r=rig.jaw(side,hi,scene,stall=True);empty=r['settled'][-1];free=[abs(s['torque']) for s in r['samples']]
            rig.jaw(side,opened,scene)
            console.line('把瓶子放进%s手两指之间（瓶身下半部分），扶住后按回车：'%name)
            r=rig.jaw(side,hi,scene,stall=True)
            if not r['stalled']:raise Quit(name+'手没有检测到夹住瓶子')
            squeeze=rig.jaw(side,hi,scene)['settled'];held=[]
            for _ in range(6):
                b=detect(rig,rig.cal).get('bottle')
                if b is not None:held.append(np.linalg.inv(rig.tools(rig.servo.state['q'])[side])@b)
            rig.jaw(side,opened,scene);console.line('已松开，把瓶子放回桌上原位后按回车：')
            d[side]={'empty':empty,'free':free+[abs(s['torque']) for s in r['samples'][:-6]],'stall':r['samples'][-1],
                'squeeze':squeeze,'seen':len(held)>=3,'center':cm.mean_pose(held)[:3,3].tolist() if len(held)>=3 else None}
            console.say('  %s手：夹住位置 %.3f，力矩 %.3f；相机%s看到手里的瓶子'%(name,squeeze[-1]['position'],squeeze[-1]['torque'],'' if len(held)>=3 else '没有'))
        d['jaw_gap_mm']=console.number('现在夹爪是张开的，请量一只夹爪两指内侧的距离',None,10,200);done=True
    except Quit as exc:console.say('已停止：'+str(exc))
    except KeyboardInterrupt:console.say('已按 Ctrl+C 停止，手臂原地保持')
    finally:
        errors=rig.close();journal.close()
    if errors:console.say('⚠ 关闭设备时有错误：'+'；'.join(errors))
    d.update(record=str(journal.folder.relative_to(store.root)),rest_q=None if rig.rest is None else rig.rest.tolist(),completed=done)
    if not done:store.save('grip',d);return None
    cal=store.read('calibration.json');collision=store.read('collision.json');b=cal['objects']['bottle']
    for side in ['right','left']:
        g=d[side];cal['grippers'][side].update(cm.grip(side,open_motor=float(rig.start[7 if side=='left' else 15]),
            limits=rig.cal['grippers'][side]['motor_limits'],free=g['free'],empty=g['empty'],stall=g['stall'],
            squeeze=g['squeeze'],center_tool=[0,0,0]))
        # Online grasps vary the height, so bound the bottle origin generically:
        # below the pads (tool x is up), on the jaw centre line, near the TCP.
        cal['grippers'][side]['object_center_bounds_tool_m']=[[-b['body_height_m']/2-.03,-.02,-.07],[.0,.02,.01]]
        cal['grippers'][side]['pad_extent_points_tool_m']=cm.pad_points(collision,rig.planner.kin,cal['motion']['tool_links'][side],b['body_radius_m'])
    for name in ['bottle','basket']:
        rp=[s['reprojection_rms_px'] for s in d['marker_stats'] if s['object']==name]
        dr=[s['depth_residual_m'] for s in d['marker_stats'] if s['object']==name and s['depth_residual_m'] is not None]
        if rp:cal['objects'][name]['maximum_reprojection_px']=float(np.clip(3*np.percentile(rp,95),1.,4.))
        if dr:cal['objects'][name]['maximum_depth_residual_m']=float(np.clip(3*np.percentile(dr,95),.006,.03))
    cal['table_top_z_m']=d['table_z_m'];travel=min(t['travel_m'] for t in cm.finger_travel(hw['urdf']).values())
    collision['full_jaw_sweep_verified']=bool(d['jaw_gap_mm']/2000<=travel+.002);d['jaw_sweep_verified']=collision['full_jaw_sweep_verified']
    d['startup_path_check']=startup_check(rig,cal,collision,hw,d)
    path=store.save('grip',d);cal.setdefault('calibration_aids',{})['grip']=path.name
    store.write('collision.json',collision);store.write('calibration.json',cal)
    console.say('','已保存。初始化路径检查：'+('✓' if d['startup_path_check'] is None else '✗ '+d['startup_path_check']),
        *(['⚠ 夹住后相机看不到瓶子标记：运行时无法确认持瓶，需要调整标记位置或相机'] if not (d['right']['seen'] and d['left']['seen']) else []),
        '下一步：./run calib accept')
    return d


def startup_check(rig,cal,collision,hw,d):
    """Replay the runtime's own startup check with the final calibration."""
    if rig.rest is None:return '没有记录到静止姿态'
    try:
        planner=Planner(hw,cal,collision);path=[rig.rest+(rig.start-rig.rest)*i/400 for i in range(401)]
        now=time.time()
        state={'object_poses':{'bottle':np.asarray(d['bottle']).tolist(),'basket':np.asarray(d['basket']).tolist()},
            'held_by':[],'pose_revision':1,'timestamp_unix_s':now,'exposure_aligned_tools':{}}
        scene=Runtime.scene(types.SimpleNamespace(cal=cal,operation=None),state,contact=())
        if planner.validate_path(path,scene):return None
        low=min(float(t[2,3]) for q in path[::20] for t in planner.tools(q).values())
        return '被拒绝（可能是静止时手臂低于桌面平面 z=%.3f，最低点 %.3f）；运行前先让手臂处在桌面以上'%(cal['table_top_z_m'],low)
    except Exception as exc:return str(exc)[:120]



# ---------------------------------------------------------------- accept

def accept(store=None,console=None):
    store=store or Store();console=console or Console()
    cal=store.read('calibration.json');hw=store.read('hardware.json');collision=store.read('collision.json')
    _,he=store.latest('handeye');_,gr=store.latest('grip')
    if he is None or gr is None or not gr.get('completed'):console.say('请先完成 handeye 和 grip');return None
    pending=[m for m in check(cal,hw,collision,store.root) if 'bound physical acceptance' not in m and 'mesh collision model' not in m]
    if pending:console.say('还缺这些测量，先补齐：',*['  · '+p for p in pending]);return None
    h=he['hand_eye'];t=he['timing'];w=he.get('watchdog') or {};g=cal['grippers']
    board=cal.get('calibration_aids',{}).get('board_table_z_m');table_gap=None if board is None else gr['table_z_m']-board
    hull=any(r.get('kind')=='wrist_camera' for r in collision['hulls']);seen=gr['right']['seen'] and gr['left']['seen']
    checks={
        'camera_extrinsics':({'hand_eye':h,'table_vs_board_m':table_gap},hand_eye_ok(h) and abs(table_gap or 0)<=.008,
            '手眼一致性 RMS %.1fmm/%.2f°，%d 视角；瓶底与标定板桌面差 %s'%(h['translation_rms_m']*1000,math.degrees(h['rotation_rms_rad']),h['views'],
            '—' if table_gap is None else '%.1fmm'%(table_gap*1000))),
        'object_geometry':({'upright_z':gr['upright_z'],'ruler':cal.get('calibration_aids',{}).get('ruler'),'objects':cal['objects']},
            gr['upright_z']>.99,'瓶子竖直度 %.3f，瓶篮标记阈值已按实测设定'%gr['upright_z']),
        'gripper_contact':({'left':g['left'],'right':g['right'],'held_bottle_seen':{s:gr[s]['seen'] for s in ['left','right']}},seen,
            '夹住后相机能看到瓶子：右%s 左%s'%tuple('✓' if gr[s]['seen'] else '✗' for s in ['right','left'])),
        'timing':({'clock_model':cal['camera']['clock_model'],'fit':t.get('fit')},cal['camera']['clock_model']['maximum_error_s']<=.02,
            '曝光对时误差上界 %.1fms（%s）'%(cal['camera']['clock_model']['maximum_error_s']*1000,t['domain'])),
        'motion_planner':({'startup_path_check':gr.get('startup_path_check'),'jaw_sweep_verified':collision.get('full_jaw_sweep_verified'),'camera_hull':hull},
            gr.get('startup_path_check') is None and collision.get('full_jaw_sweep_verified') is True and hull,
            '初始化路径%s；夹爪扫掠%s；相机%s加入碰撞模型'%('✓' if gr.get('startup_path_check') is None else '✗ '+str(gr.get('startup_path_check')),
            '✓' if collision.get('full_jaw_sweep_verified') else '✗','已' if hull else '未')),
        'watchdog':({'stall_test':w},bool(w.get('halted')) and w.get('halt_latency_s',9)<=.3 and w.get('drift_rad',9)<=.01
            and (w.get('after_close') or {}).get('drift_rad',9)<=.01,
            '卡住后 %.2fs 触发保持，漂移 %.2f°；停止发指令后 3 秒漂移 %s'%(w.get('halt_latency_s',float('nan')),math.degrees(w.get('drift_rad',float('nan'))),
            '%.2f°'%math.degrees(w['after_close']['drift_rad']) if 'drift_rad' in (w.get('after_close') or {}) else '未测'))}
    console.say('','── 现场验收：逐项核对自动测量结果，再由你确认现场观察')
    for key,(_,auto,text) in checks.items():console.say('  [%s] %s：%s'%('✓' if auto else '✗',CATEGORY_NAMES[key],text))
    failed=[CATEGORY_NAMES[k] for k,(_,auto,_) in checks.items() if not auto]
    if failed:console.say('','这些项自动检查没通过，不能验收：'+'、'.join(failed),'按上面的提示重做对应步骤。');return None
    operator=console.line('操作员姓名：').strip()
    if not operator:return None
    for key in checks:
        if not console.yes('「%s」现场观察也正常吗？（没有意外碰撞、夹持稳定等）'%CATEGORY_NAMES[key],True):
            console.say('未验收，配置未改动');return None
    collision['physical_trial_passed']=True;store.write('collision.json',collision)
    digest=subject(cal,hw,collision,implementation(store.root));folder=store.root/'records/acceptance';folder.mkdir(parents=True,exist_ok=True)
    import hashlib
    for key,(measurements,_,_) in checks.items():
        record=cm.acceptance(key,subject=digest,operator=operator,measurements=measurements,passed=True)
        path=folder/(time.strftime('%Y%m%d-%H%M%S-')+key+'.json')
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=lambda v:np.asarray(v).tolist())+'\n')
        cal['acceptance'][key]={'physical_trial_passed':True,'record':str(path.relative_to(store.root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    cal['status']='physically_accepted';store.write('calibration.json',cal)
    missing=preflight(store.root)['missing']
    console.say('','验收记录已写入 records/acceptance/（30 天有效，改代码/配置后需重做）。',
        '预检：'+('✓ 全部通过，可以运行 ./run grab' if not missing else '还缺：'+'；'.join(missing)))
    return missing


def main(argv=None):
    p=argparse.ArgumentParser(prog='./run calib',description=__doc__)
    p.add_argument('step',nargs='?',default='status',choices=['status','markers','measure','handeye','grip','accept'])
    p.add_argument('--camera',choices=['left','right'],default='right')
    a=p.parse_args(argv)
    try:
        if a.step=='handeye':handeye(camera=a.camera)
        elif a.step in ('status','markers','measure','grip','accept'):globals()[a.step]()
    except Quit as exc:print(str(exc));return 1
    except (KeyboardInterrupt,EOFError):print('\n已取消');return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
