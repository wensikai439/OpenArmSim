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

try:
    from .robot import TOOLS,clean,validate_observation,verify
    from .evidence import finite,fresh,relative_target,transform,HoldEvidence,grasp_region
except ImportError:
    from robot import TOOLS,clean,validate_observation,verify
    from evidence import finite,fresh,relative_target,transform,HoldEvidence,grasp_region

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
    for name,digest in hardware['source_hashes'].items():
        p=Path(hardware['deploy_root'])/name
        match=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
        references.append({'path':str(p),'sha256_match':match})
        if not match:missing.append('reference missing or changed: '+name)
    init_path=Path(hardware['deploy_root'])/hardware['startup_target_file']
    init=load(init_path) if init_path.is_file() else {}
    if init.get('target')!=hardware['startup_target']:missing.append('startup target differs from ACT or unavailable')
    if init.get('feature_names')!=hardware['feature_names']:missing.append('ACT state ordering changed or unavailable')
    for key in ['camera_extrinsics','object_geometry','gripper_contact','timing','motion_planner','watchdog']:
        evidence=cal.get('acceptance',{}).get(key,{})
        relative=evidence.get('record');digest=evidence.get('sha256')
        p=(root/relative).resolve() if relative else None
        if (not p or not p.is_relative_to(root.resolve()) or not p.is_file() or
                not digest or hashlib.sha256(p.read_bytes()).hexdigest()!=digest or
                evidence.get('physical_trial_passed') is not True):missing.append(key+' physical acceptance missing')
    urdf=Path(hardware['urdf'])
    if not urdf.is_file() or hashlib.sha256(urdf.read_bytes()).hexdigest()!=hardware['urdf_sha256']:
        missing.append('original URDF missing or changed')
    if cal.get('camera',{}).get('T_link_camera') is None:missing.append('measured camera extrinsics not configured')
    for side in ['left','right']:
        if any(v is None for v in cal.get('grippers',{}).get(side,{}).values()):missing.append(side+' grip calibration not configured')
    return {'software_reference_match':all(r['sha256_match'] for r in references),
        'same_act_startup_target':init.get('target')==hardware['startup_target'],
        'calibration_verified':not missing,'missing':missing,'references':references,
        'hardware_access':False,'devices_started':False,'motion_commands':0,
        'qualification':'read-only files and evidence hashes; no physical acceptance performed here'}


class EvidencePipeline:
    """One synchronized, measured snapshot; unknown is never coerced to empty."""
    def __init__(self,calibration):
        self.config=calibration;self.holds=HoldEvidence(calibration['grippers']);self.revision=0
        self.last_stamp=None

    def fuse(self,poses,tool_poses,feedback,*,now=None):
        now=time.time() if now is None else now;c=self.config
        if set(['bottle','basket'])-poses.keys():raise ValueError('bottle or basket not visible')
        stamps=[poses[n]['timestamp_unix_s'] for n in ['bottle','basket']]+[feedback['timestamp_unix_s']]
        if not all(fresh(s,now) for s in stamps) or max(stamps)-min(stamps)>.05:raise ValueError('unsynchronized pose/feedback')
        stamp=min(stamps)
        if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('cached/out-of-order evidence')
        self.last_stamp=stamp;self.revision+=1
        bottle=transform(poses['bottle']['T_base_object']);basket=transform(poses['basket']['T_base_object'])
        held=[];evidence={};region={};touching={}
        for side,i in [('left',0),('right',8)]:
            grip=feedback['motors'][side][7]
            evidence[side]=self.holds.read(side,object_pose=bottle,tool_pose=tool_poses[side],
                feedback=grip,stamp=stamp,now=now)
            if not evidence[side]['verified']:raise ValueError('grip evidence unknown')
            touching[side]=evidence[side]['touching']
            if evidence[side]['held']:held.append(side)
            if evidence[side]['held']:
                g=c['grippers'][side];obj=c['objects']['bottle']
                region[side]=grasp_region(bottle,tool_poses[side],height_m=obj['body_height_m'],
                    body_bottom_m=obj['body_bottom_m'],pad_points_tool=g['pad_extent_points_tool_m'],
                    uncertainty_m=poses['bottle']['uncertainty_m']+g['accepted_tool_uncertainty_m'])
        center=bottle[:3,3];obj=c['objects']['bottle']
        bottom=float(center[2]+bottle[2,2]*obj['body_bottom_m'])
        support=abs(bottom-c['table_top_z_m'])<=c['support_tolerance_m']
        return {'source':'measured_perception','frame':'robot_base','timestamp_unix_s':stamp,
            'pose_revision':self.revision,'held_by':held,'touching':touching,
            'upright':bool(bottle[2,2]>.98),'bottle_center_m':center.tolist(),
            'object_poses':{'bottle':bottle.tolist(),'basket':basket.tolist()},'hold_evidence':evidence,
            'hold_evidence_verified':True,'object_pose_verified':True,'grasp_region_evidence':region,
            # Support position alone cannot prove contact or stationarity.
            'support_evidence':{'table':{'verified':False,'stationary':False,'geometric_candidate':support}},
            'gripper_motor_feedback_rad':{s:feedback['motors'][s][7]['position'] for s in ['left','right']},
            'gripper_open_rad':{s:c['grippers'][s]['open_motor'] for s in ['left','right']},
            'checkpoint_id':None,'current_grasp_id':None,'last_failure':None,'completed_steps':[],
            'operation':None,'remaining_seconds':120.,'terminal':False,'invalid':False}


class Service:
    def __init__(self,config,readiness,journal,provider=None):
        self.config=config;self.readiness=readiness;self.journal=journal;self.provider=provider
        self.lock=threading.Lock();self.failed=False;self.seen=set();self.calls=0
        self.token=secrets.token_urlsafe(32)

    def status(self):
        # No provider is installed automatically from a module string. A new
        # motion backend needs independent path/collision/hold/watchdog trials.
        tools=list(self.provider.tools) if self.provider else []
        ready=bool(self.provider and self.readiness['calibration_verified'] and set(TOOLS)<=set(tools) and not self.failed)
        return {'contract':self.config['robot_contract'],'source':'real_robot' if self.provider else 'disabled_service',
            'calibration_verified':self.readiness['calibration_verified'],
            'hardware_execution_available':ready,'tools':tools,
            'missing':self.readiness['missing']+([] if self.provider else ['accepted real motion provider not loaded']),
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
            if name not in TOOLS:raise ValueError('unknown skill tool')
            ident=body.get('request_id')
            if not isinstance(ident,str) or not ident or ident in self.seen:raise ValueError('missing or duplicate request id; never resend uncertain motion')
            if len(self.seen)>=64:raise ValueError('tool budget exhausted')
            self.seen.add(ident);state=self.observation()
            if state['terminal'] or state['invalid']:raise ValueError('terminal state')
            self.journal.record('tool_requested',{'name':name,'request':body,'observation_timestamp':state['timestamp_unix_s']})
            self.calls+=1
            try:
                result=self.provider.call(name,*body.get('args',[]),**body.get('kwargs',{}))
                self.journal.record('tool_returned',{'name':name,'request_id':ident,'result':result})
                return result
            except Exception:
                self.failed=True;self.journal.record('error',{'tool':name,'request_id':ident,'outcome':'unknown; stop latched'})
                self.provider.stop();raise
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
                if not self.path.startswith('/tools/'):return self.response(404,{'error':'unknown route'})
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('invalid request size')
                body=json.loads(self.rfile.read(length));result=service.tool(self.path[len('/tools/'):],body)
                self.response(200,result)
            except (ValueError,RuntimeError) as exc:self.response(409,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','service','audit-log'])
    p.add_argument('folder',nargs='?');p.add_argument('--port',type=int,default=8772);a=p.parse_args()
    verify()
    if a.command=='audit-log':result=audit_log(a.folder)
    else:
        readiness=preflight()
        if a.command=='preflight':result=readiness
        else:
            # Intentionally no --execute/--devices switch until a complete
            # real provider is admitted. Starting this listener opens nothing.
            folder=ROOT/'records'/(time.strftime('%Y%m%d-%H%M%S')+'-service-'+secrets.token_hex(2))
            journal=Journal(folder,{'mode':'disabled','preflight':readiness,'hardware_access':False})
            service=Service(load(ROOT/'config/robot.json'),readiness,journal)
            server=serve(service,a.port)
            print(json.dumps({'service':'http://127.0.0.1:'+str(server.server_port),'devices_started':False,'motion_enabled':False}),flush=True)
            try:server.serve_forever()
            except KeyboardInterrupt:pass
            finally:server.server_close();journal.close()
            return 0
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
