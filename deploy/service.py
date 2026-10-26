"""Local real-robot service host and audit trail; default is NO DEVICES.

An HTTP listener is not evidence that cameras, grasp detection or a collision-
checked motion provider have passed physical acceptance. Never advertise that.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import secrets
import threading
import time
import sys
import importlib.metadata
import signal

try:
    from .robot import TOOLS,tools_for,clean,validate_observation,verify
    from .evidence import finite,fresh,relative_target,transform,HoldEvidence,grasp_region
    from .calibration import check as check_calibration
except ImportError:
    from robot import TOOLS,tools_for,clean,validate_observation,verify
    from evidence import finite,fresh,relative_target,transform,HoldEvidence,grasp_region
    from calibration import check as check_calibration

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(Path(path).read_text())


class Journal:
    """One operation directory; write-ahead requests + monotonic time + hashes."""
    def __init__(self,folder,metadata):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.lock=threading.Lock();self.previous='0'*64;self.sequence=0
        self.file=(self.folder/'events.jsonl').open('x',encoding='utf8')
        (self.folder/'session.json').write_text(json.dumps(clean(metadata),ensure_ascii=False,indent=2)+'\n')
        self.record('session_started',metadata)

    def record(self,kind,value):
        with self.lock:
            row={'sequence':self.sequence,'kind':kind,'unix_ns':time.time_ns(),
                'monotonic_ns':time.monotonic_ns(),'previous_sha256':self.previous,'data':clean(value)}
            data=json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
            self.previous=hashlib.sha256(data.encode()).hexdigest();row['sha256']=self.previous
            self.file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.file.flush()
            # Persist dispatch intention before a driver call, not just when a
            # successful reply returns. No retry if a command outcome is unknown.
            if kind in ('tool_requested','command_requested','initialization_command','stop','error'):
                os.fsync(self.file.fileno())
            self.sequence+=1

    def close(self):
        self.record('session_closed',{'events_before_close':self.sequence})
        self.file.close()


def audit_log(folder):
    previous='0'*64;count=0
    for line in (Path(folder)/'events.jsonl').read_text().splitlines():
        row=json.loads(line);digest=row.pop('sha256')
        actual=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if row['previous_sha256']!=previous or digest!=actual or row['sequence']!=count:
            raise ValueError('audit chain broken at '+str(count))
        previous=digest;count+=1
    return {'events':count,'sha256_chain_valid':True,'scope':'log integrity, not hardware-success proof'}


def preflight(root=ROOT):
    root=Path(root);hardware=load(root/'config/hardware.json');cal=load(root/'config/calibration.json')
    missing=[];references=[]
    runtime=hardware.get('runtime_contract',{})
    if runtime:
        if '.'.join(map(str,sys.version_info[:2]))!=runtime['python']:missing.append('hardware Python ABI mismatch')
        for name,version in runtime['packages'].items():
            try:match=importlib.metadata.version(name)==version
            except importlib.metadata.PackageNotFoundError:match=False
            if not match:missing.append('runtime package missing/changed: '+name)
        native=runtime['native_can'];p=Path(native['path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=native['sha256']:missing.append('native CAN extension missing/changed')
    for name,digest in hardware['source_hashes'].items():
        p=Path(hardware['deploy_root'])/name
        match=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
        references.append({'path':str(p),'sha256_match':match})
        if not match:missing.append('reference missing or changed: '+name)
    init_path=Path(hardware['deploy_root'])/hardware['startup_target_file']
    init=load(init_path) if init_path.is_file() else {}
    if init.get('target')!=hardware['startup_target']:missing.append('startup target differs from ACT or unavailable')
    if init.get('feature_names')!=hardware['feature_names']:missing.append('ACT state ordering changed or unavailable')
    collision=load(root/'config/collision.json') if (root/'config/collision.json').is_file() else {}
    missing.extend(check_calibration(cal,hardware,collision,root))
    urdf=Path(hardware['urdf'])
    if not urdf.is_file() or hashlib.sha256(urdf.read_bytes()).hexdigest()!=hardware['urdf_sha256']:
        missing.append('original URDF missing or changed')
    return {'software_reference_match':all(r['sha256_match'] for r in references),
        'same_act_startup_target':init.get('target')==hardware['startup_target'],
        'calibration_verified':not missing,'missing':missing,'references':references,
        'setup_groups':setup_groups(missing),
        'hardware_access':False,'devices_started':False,'motion_commands':0,
        'qualification':'read-only files and evidence hashes; no physical acceptance performed here'}


def setup_groups(missing):
    """Actionable diagnostics only. Never populate measurements or accept trials."""
    rules=[('physical_acceptance','现场验收',('bound physical acceptance',),
            '真实分段试验及硬件停止验收；不能将标志直接改成 true。'),
        ('perception','相机与定位',('camera','registered RGB-D'),
            '确定实际相机/标记，测量外参与曝光对时；已有 VR 标定不替代相机标定。'),
        ('grip','夹持证据',('grip calibration',),
            '采集位置/力矩/实际持瓶证据，验证夹持区间、滑移及闭爪力矩上限。'),
        ('geometry_motion','场景与动作',('table geometry','bottle dimensions','basket interior','relative grasp','collision','finger'),
            '实测桌面、瓶篮和夹爪几何，确认工作空间、相对抓姿、交接偏移和撤离向量。')]
    groups={};unassigned=[]
    for item in missing:
        for key,title,words,action in rules:
            if any(w in item for w in words):
                group=groups.setdefault(key,{'name':title,'missing':[],'next_action':action})
                group['missing'].append(item);break
        else:unassigned.append(item)
    if unassigned:groups['software']={'name':'软件引用','missing':unassigned,
        'next_action':'核对环境、原源码和 ACT 初始化目标；保持原部署只读。'}
    return groups


class EvidencePipeline:
    """One synchronized, measured snapshot; unknown is never coerced to empty."""
    def __init__(self,calibration):
        self.config=calibration;self.holds=HoldEvidence(calibration['grippers']);self.revision=0
        self.last_stamp=None;self.support_rows=[]

    def fuse(self,poses,tool_poses,feedback,*,now=None):
        now=time.time() if now is None else now;c=self.config
        if set(['bottle','basket'])-poses.keys():raise ValueError('bottle or basket not visible')
        stamps=[poses[n]['timestamp_unix_s'] for n in ['bottle','basket']]+[feedback['timestamp_unix_s']]
        if not all(fresh(s,now) for s in stamps) or max(stamps)-min(stamps)>.05:raise ValueError('unsynchronized pose/feedback')
        stamp=poses['bottle']['timestamp_unix_s']
        if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('cached/out-of-order evidence')
        self.last_stamp=stamp;self.revision+=1
        bottle=transform(poses['bottle']['T_base_object']);basket=transform(poses['basket']['T_base_object'])
        held=[];evidence={};region={};touching={}
        for side,i in [('left',0),('right',8)]:
            tool=tool_poses[side]
            if not isinstance(tool,dict) or not fresh(tool.get('timestamp_unix_s'),now) or abs(tool['timestamp_unix_s']-stamp)>.005:
                raise ValueError('exposure-aligned measured TCP required')
            grip=feedback['motors'][side][7]
            evidence[side]=self.holds.read(side,object_pose=bottle,tool_pose=tool['T_base_tool'],
                feedback=grip,stamp=stamp,now=now)
            if not evidence[side]['verified']:raise ValueError('grip evidence unknown')
            touching[side]=evidence[side]['touching']
            if evidence[side]['held']:held.append(side)
            if evidence[side]['held']:
                g=c['grippers'][side];obj=c['objects']['bottle']
                region[side]=grasp_region(bottle,tool_poses[side]['T_base_tool'],height_m=obj['body_height_m'],
                    body_bottom_m=obj['body_bottom_m'],pad_points_tool=g['pad_extent_points_tool_m'],
                    uncertainty_m=poses['bottle']['uncertainty_m']+g['accepted_tool_uncertainty_m'])
        center=bottle[:3,3];obj=c['objects']['bottle']
        bottom=float(center[2]+bottle[2,2]*obj['body_bottom_m'])
        support=abs(bottom-c['table_top_z_m'])<=c['support_tolerance_m']
        self.support_rows.append((stamp,center.copy()));self.support_rows=[r for r in self.support_rows if stamp-r[0]<=.35]
        stationary=bool(len(self.support_rows)>=4 and stamp-self.support_rows[0][0]>=.2 and
            max(np.linalg.norm(p-center) for _,p in self.support_rows)<c['support_tolerance_m'])
        relative=np.linalg.inv(basket)@bottle;container=c['objects']['basket'];size=np.array(container['inner_size_m'])
        in_basket=bool(np.all(abs(relative[:2,3])+obj['body_radius_m']+poses['bottle']['uncertainty_m']<size[:2]/2)
            and abs(relative[2,3]+obj['body_bottom_m']-container['floor_z_m'])<=c['support_tolerance_m'])
        return {'source':'measured_perception','frame':'robot_base','timestamp_unix_s':stamp,
            'pose_revision':self.revision,'held_by':held,'touching':touching,
            'upright':bool(bottle[2,2]>.98),'bottle_center_m':center.tolist(),
            'object_poses':{'bottle':bottle.tolist(),'basket':basket.tolist()},'hold_evidence':evidence,
            'exposure_aligned_tools':{side:{'T_base_tool':transform(tool_poses[side]['T_base_tool']).tolist(),
                'timestamp_unix_s':tool_poses[side]['timestamp_unix_s']} for side in ['left','right']},
            'hold_evidence_verified':True,'object_pose_verified':True,'grasp_region_evidence':region,
            'support_evidence':{name:{'verified':bool(candidate and stationary),'stationary':stationary,
                'source':'accepted_geometry_and_measured_stationarity','timestamp_unix_s':stamp,
                'qualification':'geometric support inference, not a tactile contact sensor'}
                for name,candidate in [('table',support),('basket',in_basket)]},
            'gripper_motor_feedback_rad':{s:feedback['motors'][s][7]['position'] for s in ['left','right']},
            'gripper_open_rad':{s:c['grippers'][s]['open_motor'] for s in ['left','right']},
            'checkpoint_id':None,'current_grasp_id':None,'last_failure':None,'completed_steps':[],
            'operation':None,'remaining_seconds':120.,'terminal':False,'invalid':False}


class Service:
    def __init__(self,config,readiness,journal,provider=None):
        self.config=config;self.readiness=readiness;self.journal=journal;self.provider=provider
        self.tools=tools_for(config)
        self.lock=threading.Lock();self.failed=False;self.seen=set();self.calls=0
        self.token=secrets.token_urlsafe(32)
        self.session=None;self.session_deadline=0.;self.session_lock=threading.Lock()

    def begin(self):
        with self.session_lock:
            if not self.status()['hardware_execution_available']:raise ValueError('real service not accepted')
            if self.session is not None:raise ValueError('another skill owns the robot')
            if hasattr(self.provider,'begin_session'):self.provider.begin_session()
            self.session=secrets.token_urlsafe(24);self.session_deadline=time.monotonic()+120
            self.seen.clear();self.calls=0
            return {'session_id':self.session,'lease_seconds':120}

    def end(self,ident):
        with self.session_lock:
            if ident!=self.session or ident is None:raise ValueError('not session owner')
            if self.lock.locked():raise ValueError('tool still active; abort or inspect outcome')
            if not self.failed and hasattr(self.provider,'end_session'):self.provider.end_session()
            self.session=None;return {'released':True}

    def abort(self,reason):
        self.failed=True
        try:
            if self.provider:self.provider.stop(reason)
        finally:
            try:self.journal.record('stop',{'reason':reason,'execution_outcome':'unknown until measured review'})
            except Exception:pass

    def expire(self):
        if self.session is not None and time.monotonic()>self.session_deadline:
            self.abort('skill lease expired');return True
        return False

    def status(self):
        # No provider is installed automatically from a module string. A new
        # motion backend needs independent path/collision/hold/watchdog trials.
        tools=list(self.provider.tools) if self.provider else []
        provider_failure=getattr(self.provider,'failure',None)
        ready=bool(self.provider and self.readiness['calibration_verified'] and set(self.tools)<=set(tools) and not self.failed and not provider_failure)
        return {'contract':self.config['robot_contract'],'source':'real_robot' if self.provider else 'disabled_service',
            'calibration_verified':self.readiness['calibration_verified'],
            'hardware_execution_available':ready,'tools':tools,
            'tool_implementations_included':(Path(__file__).parent/'runtime.py').is_file(),
            'device_session_active':bool(self.provider and not provider_failure),
            'missing':self.readiness['missing']+([] if self.provider else ['accepted real motion provider not loaded'])+
                (['real provider stopped: '+provider_failure] if provider_failure else []),
            'devices_started':bool(self.provider),'tool_requests':self.calls,
            'motion_commands':None if self.calls else 0}

    def observation(self):
        if not self.provider:raise ValueError('devices not started; measured observation unavailable')
        state=self.provider.observe();validate_observation(state)
        self.journal.record('observation',state);return state

    def tool(self,name,body):
        if not self.lock.acquire(blocking=False):raise ValueError('one skill owns the robot; busy')
        try:
            if not self.status()['hardware_execution_available']:raise ValueError('real action service not physically accepted')
            if body.get('session_id')!=self.session or self.session is None:raise ValueError('exclusive skill session required')
            if self.expire():raise ValueError('session expired')
            self.session_deadline=time.monotonic()+120
            if name not in self.tools:raise ValueError('unknown skill tool')
            ident=body.get('request_id')
            if not isinstance(ident,str) or not ident or ident in self.seen:raise ValueError('missing or duplicate request id; never resend uncertain motion')
            if len(self.seen)>=(192 if len(self.tools)>len(TOOLS) else 64):raise ValueError('tool budget exhausted')
            self.seen.add(ident)
            try:state=self.observation()
            except Exception:
                self.abort('pre-tool perception failed');raise
            if state['terminal'] or state['invalid']:raise ValueError('terminal state')
            self.journal.record('tool_requested',{'name':name,'request':body,'observation_timestamp':state['timestamp_unix_s']})
            self.calls+=1
            try:
                result=self.provider.call(name,*body.get('args',[]),**body.get('kwargs',{}))
                self.journal.record('tool_returned',{'name':name,'request_id':ident,'result':result})
                return result
            except Exception:
                self.abort('tool failed: '+name);raise
        finally:self.lock.release()


def serve(service,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def response(self,status,value):
            data=json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path=='/status':return self.response(200,service.status())
                if self.path=='/observation':return self.response(200,service.observation())
                self.response(404,{'error':'unknown route'})
            except (ValueError,RuntimeError) as exc:self.response(409,{'error':str(exc)})
        def do_POST(self):
            if not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+service.token):
                return self.response(403,{'error':'explicit service session required'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('invalid request size')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('request object required')
                if self.path=='/session':return self.response(200,service.begin())
                if self.path=='/session/end':return self.response(200,service.end(body.get('session_id')))
                if self.path=='/abort':
                    if body.get('session_id')!=service.session or service.session is None:raise ValueError('not session owner')
                    service.abort('client requested abort');return self.response(200,{'stop_latched':True})
                if not self.path.startswith('/tools/'):return self.response(404,{'error':'unknown route'})
                result=service.tool(self.path[len('/tools/'):],body)
                self.response(200,result)
            except (ValueError,RuntimeError,KeyError,TypeError) as exc:self.response(409,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','service','audit-log'])
    p.add_argument('folder',nargs='?');p.add_argument('--port',type=int,default=8772)
    p.add_argument('--devices',action='store_true',help='explicit on-site device and initialization session')
    p.add_argument('--confirm-on-site',action='store_true');a=p.parse_args()
    if a.devices and (a.command!='service' or not a.confirm_on_site):raise ValueError('--devices requires service and --confirm-on-site')
    verify()
    if a.command=='audit-log':result=audit_log(a.folder)
    else:
        readiness=preflight()
        if a.command=='preflight':result=readiness
        else:
            if a.devices and not readiness['calibration_verified']:
                print(json.dumps({'status':'not_ready','reason':'实测配置或现场验收未完成，设备未打开',
                    'setup_groups':readiness['setup_groups'],'next_command':'./run preflight',
                    'devices_started':False,'motion_commands':0},ensure_ascii=False,indent=2))
                return 2
            folder=ROOT/'records'/(time.strftime('%Y%m%d-%H%M%S')+'-service-'+secrets.token_hex(2))
            journal=Journal(folder,{'mode':'devices' if a.devices else 'disabled','preflight':readiness,'hardware_access':a.devices})
            service=Service(load(ROOT/'config/robot.json'),readiness,journal)
            server=serve(service,a.port)
            token_path=ROOT/'.runtime/session-token';token_path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600);os.fchmod(fd,0o600)
            with os.fdopen(fd,'w') as f:f.write(service.token)
            stop_monitor=threading.Event()
            def monitor():
                while not stop_monitor.wait(.1):service.expire()
            threading.Thread(target=monitor,name='skill-lease',daemon=True).start()
            def interrupted(*_):raise KeyboardInterrupt()
            previous_term=signal.signal(signal.SIGTERM,interrupted)
            try:
                if a.devices:
                    if service.config.get('robot_contract')=='openarm-relative-v1/1':
                        try:from .relative import RelativeRuntime as Runtime
                        except ImportError:from relative import RelativeRuntime as Runtime
                    else:
                        try:from .runtime import Runtime
                        except ImportError:from runtime import Runtime
                    service.provider=Runtime(ROOT,journal.record)
                    service.provider.open(operator_confirmed=True)
                print(json.dumps({'service':'http://127.0.0.1:'+str(server.server_port),'devices_started':a.devices,'motion_enabled':a.devices}),flush=True)
                server.serve_forever()
            except KeyboardInterrupt:pass
            finally:
                stop_monitor.set()
                signal.signal(signal.SIGTERM,previous_term)
                try:
                    if service.provider:service.provider.close()
                finally:
                    server.server_close()
                    if token_path.is_file() and token_path.read_text()==service.token:token_path.unlink()
                    journal.close()
            return 0
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'not_ready','reason':str(exc),'devices_started':False},ensure_ascii=False));raise SystemExit(2)
