"""Task-only bridge. No ROS, SDK, CAN, model loading or hardware lifecycle."""
from __future__ import annotations
import os
import argparse, hashlib, importlib.util, json, math, re, socket, struct, subprocess, sys, time
from pathlib import Path

ROOT = Path(str(Path(os.environ["PAOS_ROOT"])))
CONTROL = Path(str(Path(os.environ.get("PIPER_CONTROL_ROOT", str(Path(os.environ["PAOS_ROOT"]) / "external/robot-control")))))
PROTOCOL = 'runtime-task/v1'
TERMINAL = {'completed', 'failed', 'cancelled', 'unknown'}
CAMERAS = ['cam_left', 'cam_middle', 'cam_right']
ORDER = [*(f'left_joint{i}' for i in range(1,7)), 'left_gripper',
         *(f'right_joint{i}' for i in range(1,7)), 'right_gripper']

class BridgeError(RuntimeError):
    def __init__(self, message, response=None):
        super().__init__(message); self.response=response

def digest(data): return hashlib.sha256(data).hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)); tmp.replace(path)
def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise BridgeError('Expected a prepared task identifier, not a path')
    return value

def contract(name):
    """Load the native hardware-free contract source in this Skill's environment."""
    path=CONTROL/'piper_baseline'/name
    spec=importlib.util.spec_from_file_location('_paos_native_'+path.stem,path)
    if spec is None or spec.loader is None: raise BridgeError('Native offline contract unavailable: '+str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

PROGRAMS={'single_arm_program':('program_core/independent_pick_place.py','independent-pick-place'),
          'bimanual_program':('program_core/bimanual_program.py','bimanual-demo')}

def check_program(program,task_type='bimanual_program'):
    if task_type not in PROGRAMS: raise BridgeError('Unsupported program task type')
    native=contract(PROGRAMS[task_type][0]);checked=native.validate_program(program)
    if checked.get('ok') is not True: raise BridgeError('Native program rejection: '+json.dumps(checked.get('errors')))
    # The native estimate assigns 0.1 s to a human checkpoint. Allow its full
    # bounded program budget instead of cancelling while the operator answers.
    checked['execution_budget_s']=native.MAX_BUDGET_S if task_type=='single_arm_program' else checked['total_budget_s']
    return checked

def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise BridgeError(name+' must be a finite number')
    return float(value)

def check_step(step):
    if not isinstance(step,dict): raise BridgeError('step must be an object')
    skill=step.get('skill');clean=dict(step)
    if skill=='move_right_tcp_relative':
        allowed={'skill','dx_m','dy_m','dz_m','frame','step_level_m','timeout_s','poll_s'}
        if set(step)-allowed: raise BridgeError('Unknown/unsafe relative step fields')
        native=contract('relative_tcp.py')
        try:
            move=native.validate_relative_translation(*(number(step.get(k),k) for k in ['dx_m','dy_m','dz_m']),
                frame=step.get('frame',native.RIGHT_BASE_FRAME),step_level_m=step.get('step_level_m',native.DEFAULT_STEP_M))
        except ValueError as exc: raise BridgeError('Native relative step rejection: '+str(exc)) from exc
        timeout=number(step.get('timeout_s',native.DEFAULT_TIMEOUT_S),'timeout_s')
        poll=number(step.get('poll_s',native.DEFAULT_POLL_S),'poll_s')
        if not 0<timeout<=30 or not 0<poll<=1: raise BridgeError('Native timeout_s/poll_s limits exceeded')
        clean={'skill':skill,'dx_m':move.dx_m,'dy_m':move.dy_m,'dz_m':move.dz_m,
               'frame':move.frame,'step_level_m':move.step_level_m,'timeout_s':timeout,'poll_s':poll}
        budget=timeout
    elif skill=='set_gripper':
        if set(step)-{'skill','side','position_m','unit'} or step.get('side','right')!='right' or step.get('unit','m')!='m':
            raise BridgeError('Basic gripper route requires right side and metres')
        width=number(step.get('position_m'),'position_m')
        if width<0: raise BridgeError('position_m must be non-negative; native hardware bounds still apply')
        clean={'skill':skill,'side':'right','unit':'m','position_m':width};budget=5.0
    elif skill=='bounded_joint_step':
        if set(step)-{'skill','candidate_id'} or step.get('candidate_id') not in {f'right_joint{i}_{d}' for i in range(1,7) for d in ['plus','minus']}:
            raise BridgeError('Use one bounded right joint candidate; live feedback is supplied by the bridge')
        budget=5.0
    else: raise BridgeError('Unsupported basic action')
    return {'normalized':clean,'total_budget_s':budget}

def check_policy(model_config, duration_s, include_gripper):
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s) or not 0 < duration_s <= 60):
        raise BridgeError('Finite policy duration_s must be within (0, 60]')
    if not isinstance(include_gripper, bool): raise BridgeError('include_gripper must be boolean')
    path=Path(model_config)
    if not path.is_absolute(): raise BridgeError('Model config needs an absolute server path')
    result=subprocess.run([os.environ.get('PAOS_PYTHON',str(ROOT/'.venv/bin/python')),'-B','-m','console.real_preparation',
                           '--model-config',str(path)],cwd=ROOT,text=True,capture_output=True,timeout=30)
    try: report=json.loads(result.stdout)
    except ValueError as exc: raise BridgeError('Offline towel configuration checker failed: '+result.stderr[-1000:]) from exc
    if result.returncode or report.get('ok') is not True:
        raise BridgeError('Towel configuration rejected: '+json.dumps(report.get('errors'),ensure_ascii=False))
    return {'model_config_path':str(path.resolve()),'model_config_sha256':report['source_sha256'],
            'expected_model_config_sha256':report['native_checker']['resolved_config_sha256'],
            'model_files':report['model_files'],
            'dataset_metadata_sha256':report['dataset_metadata_sha256'],
            'duration_s':float(duration_s),'include_gripper':include_gripper,
            'total_budget_s':float(duration_s)}

class Client:
    def __init__(self,path,session,runtime_id=None,timeout=2.):
        self.path,self.session,self.runtime_id,self.timeout=Path(path),session,runtime_id,timeout
    def request(self,operation,**fields):
        import uuid
        request_id=uuid.uuid4().hex
        payload={'protocol_version':PROTOCOL,'request_id':request_id,'operation':operation,
                 'client_session_id':self.session,**fields}
        if self.runtime_id is not None: payload['runtime_instance_id']=self.runtime_id
        raw=(json.dumps(payload,allow_nan=False)+'\n').encode()
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as peer:
                peer.settimeout(max(self.timeout,12.) if operation=='cancel_task' else self.timeout); peer.connect(str(self.path)); peer.sendall(raw)
                buf=b''
                while b'\n' not in buf and len(buf)<=1024*1024:
                    part=peer.recv(65536)
                    if not part: break
                    buf+=part
        except OSError as exc: raise BridgeError('Native transport unresolved: '+str(exc)) from exc
        if b'\n' not in buf: raise BridgeError('Incomplete native response; do not resubmit')
        try: reply=json.loads(buf.split(b'\n',1)[0])
        except (ValueError,UnicodeDecodeError) as exc: raise BridgeError('Invalid native JSON; do not resubmit') from exc
        if not isinstance(reply,dict): raise BridgeError('Invalid native reply object; do not resubmit')
        if (reply.get('protocol_version'),reply.get('request_id'))!=(PROTOCOL,request_id):
            raise BridgeError('Native response identity mismatch',reply)
        if self.runtime_id is not None and reply.get('runtime_instance_id')!=self.runtime_id:
            raise BridgeError('Native runtime changed; do not resubmit',reply)
        if reply.get('ok') is not True: raise BridgeError(str(reply.get('error')),reply)
        result=reply.get('result')
        if not isinstance(result,dict): raise BridgeError('Native result is not an object',reply)
        return {**result,'runtime_instance_id':reply.get('runtime_instance_id')}

def fresh(stamp,limit):
    if isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp):
        raise BridgeError('Missing source timestamp')
    if not -.02 <= time.time()-stamp <= limit: raise BridgeError('Stale/future source timestamp')

def snapshot(path):
    def exact(peer,n):
        data=bytearray()
        while len(data)<n:
            part=peer.recv(n-len(data))
            if not part: raise BridgeError('Incomplete observation payload')
            data.extend(part)
        return bytes(data)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as peer:
        peer.settimeout(2);peer.connect(str(path))
        raw=json.dumps({'op':'observe'}).encode();peer.sendall(struct.pack('!II',len(raw),0)+raw)
        header,size=struct.unpack('!II',exact(peer,8))
        if not 0<header<=65536 or size!=3*480*640*3: raise BridgeError('Missing/invalid native RGB payload')
        meta=json.loads(exact(peer,header));pixels=exact(peer,size)
    if meta.get('ok') is not True or meta.get('schema')!='piper-shadow-v1': raise BridgeError('Native observation rejected')
    if meta.get('order')!=ORDER or len(meta.get('state',[]))!=14 or meta.get('camera_names')!=CAMERAS: raise BridgeError('Observation contract mismatch')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in meta['state']): raise BridgeError('Nonfinite native state')
    if meta.get('image_shape')!=[480,640,3] or meta.get('encoding')!='rgb8': raise BridgeError('RGB contract mismatch')
    if set(meta.get('state_stamps',{}))!={s+'_'+k for s in ['left','right'] for k in ['joints','gripper']}:
        raise BridgeError('Incomplete state sources')
    if set(meta.get('image_stamps',{}))!=set(CAMERAS): raise BridgeError('Incomplete image sources')
    for stamp in meta['state_stamps'].values(): fresh(stamp,.15)
    for stamp in meta['image_stamps'].values(): fresh(stamp,.2)
    fresh(meta.get('snapshot_wall'),.2)
    if max(meta['image_stamps'].values())-min(meta['image_stamps'].values())>.1: raise BridgeError('Camera skew >100ms')
    return meta,pixels

class Bridge:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.config_path=self.root/'config/real_runtime.json'
        self.run=self.root/'runs/real-runtime';self.run.mkdir(parents=True,exist_ok=True)
    def config(self): return read_json(self.config_path)
    def client(self,session):
        cfg=self.config()
        if cfg.get('transport')=='native-four-can-replay/v1': raise BridgeError('Current four-CAN chain does not use the legacy UDS client')
        if cfg.get('mode')!='live': raise BridgeError('OFFLINE_ONLY: collection is independent; native I/O is disabled')
        rid=cfg.get('runtime_instance_id')
        if not isinstance(rid,str) or not rid: raise BridgeError('A fresh operator-started Runtime identity is required')
        return Client(cfg['socket_path'],session,rid)
    def capabilities(self):
        cfg=self.config()
        if cfg.get('transport')=='native-four-can-replay/v1':
            import native_replay_bridge as replay
            status=replay.worker_snapshot(self) if cfg.get('mode')=='live' else {}
            return {'mode':cfg.get('mode'),'source':'existing_native_four_can_entry',
                'transport':replay.TRANSPORT,'connected':status.get('available',False),'motion_ready':False,
                'prepare_task_types':[replay.TASK_TYPE],'supported_task_types':[replay.TASK_TYPE],
                'raw_target_authoring_by_agent':False,'program_source':'operator_registered_native_trajectory',
                'basic_actions':'registered_joint_trajectory_only',
                'unavailable':['legacy_UDS_station_program','TCP_step','policy_execution','camera_observation'],
                'field_release_present':bool(cfg.get('release')),'business_success':None,
                'blockers':['offline_only'] if cfg.get('mode')!='live' else []}
        if cfg.get('mode')!='live':
            return {'mode':'offline','connected':False,'motion_ready':False,
                    'source':'configuration_only','native_protocol':PROTOCOL,
                    'blockers':['native_io_disabled','no_field_task_release'],
                    'supported_task_types':['single_arm_program','bimanual_program','visual_servo_step'],
                    'basic_actions':['move_right_tcp_relative','set_gripper','bounded_joint_step'],
                    'basic_action_scope':{'side':'right','frame':'right_base','joint_step_requires_native_capability':True},
                    'policy_status':'legacy_interface_preserved_not_adapted_to_current_control_chain',
                    'policy_contract':{'duration_s_maximum':60.0,'loaded_model_config_binding_required':True},
                    'business_success':None}
        return {'mode':'live','source':'native_runtime',**self.client('paos-real-observer').request('get_capabilities')}
    def state(self,session='paos-real-observer'):
        if self.config().get('transport')=='native-four-can-replay/v1':
            import native_replay_bridge as replay
            replay.ensure_live(self)
            return replay.worker_snapshot(self)
        return {'source':'native_runtime','state':self.client(session).request('get_state'),'business_success':None}
    def observe(self,session='paos-real-observer'):
        if self.config().get('transport')=='native-four-can-replay/v1':
            raise BridgeError('Use the existing control console cameras; no camera route is exposed by native four-CAN replay')
        client=self.client(session);caps=client.request('get_capabilities')
        state=client.request('get_state')
        for side in ['left','right']:
            arm=state['arms'][side]
            fresh(arm.get('feedback_timestamp'),.15)
            fresh(arm.get('gripper',{}).get('feedback_timestamp'),.15)
            if len(arm.get('q_rad',[]))!=6: raise BridgeError('Native joint state incomplete')
        obs=caps.get('observation',{})
        if obs.get('available') is not True or obs.get('protocol')!='piper-shadow-v1': raise BridgeError('Native three-camera observation unavailable')
        meta,pixels=snapshot(obs['socket'])
        frames=[]
        from PIL import Image
        destination=self.run/'observations'/str(time.time_ns());destination.mkdir(parents=True)
        for index,camera in enumerate(CAMERAS):
            path=destination/(camera+'.png')
            Image.frombytes('RGB',(640,480),pixels[index*480*640*3:(index+1)*480*640*3]).save(path)
            frames.append({'camera':camera,'path':str(path),'sha256':digest(path.read_bytes()),'source_stamp':meta['image_stamps'][camera]})
        result={'source':'native_runtime','state':state,'snapshot':meta,'frames':frames,'business_success':None}
        save(destination/'observation.json',result)
        return result
    def task_path(self,task_id): return self.root/'prepared/real_tasks'/(identifier(task_id)+'.json')
    def store_task(self,task_id,artifact):
        path=self.task_path(task_id);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f: json.dump(artifact,f,ensure_ascii=False,indent=2,allow_nan=False)
        return {'prepared_id':task_id,'path':str(path),'sha256':digest(path.read_bytes()),
                'task_type':artifact['task_type'],'field_approved':False,'native_io_performed':False}
    def prepare(self,task_id,program,task_type=None):
        if task_type is None:
            task_type='single_arm_program' if isinstance(program,dict) and program.get('schema_version')=='independent-pick-place/v1' else 'bimanual_program'
        checked=check_program(program,task_type)
        artifact={'schema':'paos-real-task/v1','task_type':task_type,'program_config_id':PROGRAMS[task_type][1],
                  'program_version':identifier(task_id),'program':checked['normalized'],
                  'program_digest':checked['program_digest'],'total_budget_s':checked['total_budget_s'],
                  'field_approved':False,'physical_success':None}
        return self.store_task(task_id,artifact)
    def prepare_step(self,task_id,step):
        checked=check_step(step)
        return self.store_task(task_id,{'schema':'paos-real-task/v1','task_type':'visual_servo_step',
            'visual_servo_config_id':'right-visual-servo-v1','visual_servo_step':checked['normalized'],
            'total_budget_s':checked['total_budget_s'],'field_approved':False,'physical_success':None})
    def prepare_task(self,prepared_id,task_type,program=None,step=None):
        if task_type=='four_can_replay' and step is None:
            import native_replay_bridge as replay
            return replay.prepare(self,prepared_id,program)
        if task_type in PROGRAMS and step is None: return self.prepare(prepared_id,program,task_type)
        if task_type=='visual_servo_step' and program is None: return self.prepare_step(prepared_id,step)
        raise BridgeError('Expected a restricted native program or one basic step')
    def prepare_policy(self,task_id,model_config,duration_s,include_gripper,act_config_id='act-approved-demo'):
        checked=check_policy(model_config,duration_s,include_gripper)
        artifact={'schema':'paos-real-task/v1','task_type':'act_approved','task_label':'毛巾任务',
                  'act_config_id':identifier(act_config_id),'program_version':identifier(task_id),
                  **checked,'field_approved':False,'physical_success':None}
        path=self.task_path(task_id)
        with path.open('x') as f: json.dump(artifact,f,ensure_ascii=False,indent=2)
        return {'prepared_id':task_id,'path':str(path),'sha256':digest(path.read_bytes()),'field_approved':False}
    def receipt(self,key): return self.run/'attempts'/(digest(key.encode())+'.json')
    def preflight(self,task_id,key):
        if self.config().get('transport')=='native-four-can-replay/v1':
            import native_replay_bridge as replay
            return replay.preflight(self,task_id,key)
        cfg=self.config();self.client('preflight-no-request')  # Check offline gate before any native I/O.
        approval=cfg.get('release') or {};path=self.task_path(task_id)
        raw=path.read_bytes();task=json.loads(raw)
        if task.get('task_type') in PROGRAMS:
            checked=check_program(task['program'],task['task_type'])
            if task.get('program_config_id')!=PROGRAMS[task['task_type']][1]: raise BridgeError('Program config id changed')
            if task.get('program_digest')!=checked['program_digest']: raise BridgeError('Prepared program digest changed')
        elif task.get('task_type')=='visual_servo_step':
            checked=check_step(task['visual_servo_step'])
            if task.get('visual_servo_config_id')!='right-visual-servo-v1' or task['visual_servo_step']!=checked['normalized']:
                raise BridgeError('Prepared basic step changed')
        elif task.get('task_type')=='act_approved':
            checked=check_policy(task['model_config_path'],task['duration_s'],task['include_gripper'])
            if any(task.get(k)!=v for k,v in checked.items()): raise BridgeError('Prepared policy configuration or model files changed')
            identifier(task.get('act_config_id'))
        else: raise BridgeError('Unsupported prepared task type')
        if approval.get('prepared_id')!=task_id or approval.get('sha256')!=digest(raw): raise BridgeError('No release for this exact prepared artifact')
        if not 0<float(approval.get('valid_until_unix',0))-time.time()<=300: raise BridgeError('Fresh five-minute field release required')
        approval_id=identifier(approval.get('release_id'))
        if not approval.get('operator_confirmation_ref'): raise BridgeError('Missing field confirmation reference')
        if task['task_type']=='single_arm_program':
            if approval.get('field_verified') is not True or not isinstance(approval.get('field_review_id'),str) or not approval['field_review_id'].strip():
                raise BridgeError('Existing operator field_review_id and field_verified=true are required; Agent cannot create them')
        if self.receipt(key).exists(): raise BridgeError('Existing execution receipt: recover status, never resubmit')
        for old in (self.run/'attempts').glob('*.json'):
            record=read_json(old)
            if record.get('phase') in {'dispatch_pending','running','unknown'}: raise BridgeError('An unresolved owned task blocks a new dispatch')
        return cfg,task,checked,approval_id
    def execute(self,task_id,key,cancel_event):
        if self.config().get('transport')=='native-four-can-replay/v1':
            import native_replay_bridge as replay
            return replay.execute(self,task_id,key,cancel_event)
        cfg,task,checked,approval_id=self.preflight(task_id,key)
        session='paos-real-'+digest(key.encode())[:24]
        client=self.client(session);caps=client.request('get_capabilities')
        task_type=task['task_type'];kind=caps.get('task_types',{}).get(task_type,{})
        if kind.get('enabled') is not True: raise BridgeError('Native '+task_type+' executor is disabled/unconfigured')
        if task_type=='act_approved':
            binding=kind.get('loaded_model',{})
            if (kind.get('config_id')!=task['act_config_id'] or
                    binding.get('sha256')!=checked['expected_model_config_sha256']):
                raise BridgeError('Native loaded model configuration does not match this prepared policy')
            maximum=kind.get('finite_duration_s',{}).get('maximum')
            if not isinstance(maximum,(int,float)) or not checked['duration_s']<=maximum<=60:
                raise BridgeError('Native finite policy interface is unavailable; existing Runtime may require a later operator restart')
            dispatch={'task_type':task_type,'act_config_id':task['act_config_id'],
                      'duration_s':checked['duration_s'],'include_gripper':checked['include_gripper'],
                      'expected_model_config_sha256':checked['expected_model_config_sha256']}
        elif task_type in PROGRAMS:
            if kind.get('config_id')!=task['program_config_id']: raise BridgeError('Native program configuration id mismatch')
            dispatch={'task_type':task_type,'program_config_id':task['program_config_id'],
                      'program':checked['normalized'],'program_version':task['program_version'],
                      'program_digest':checked['program_digest']}
        else:
            if kind.get('config_id')!=task['visual_servo_config_id']: raise BridgeError('Native basic action configuration id mismatch')
            dispatch={'task_type':task_type,'visual_servo_config_id':task['visual_servo_config_id'],
                      'visual_servo_step':dict(checked['normalized'])}
        if caps.get('execution_rights')!='available': raise BridgeError('Native Runtime is owned by another task')
        before=self.observe(session) if task_type=='act_approved' else self.state(session)
        if task_type=='visual_servo_step' and checked['normalized']['skill']=='bounded_joint_step':
            joint=kind.get('skills',{}).get('bounded_joint_step',{})
            step=dispatch['visual_servo_step']
            if joint.get('enabled') is not True or step['candidate_id'] not in joint.get('candidate_ids',[]):
                raise BridgeError('Native bounded_joint_step capability/candidate is disabled')
            arm=before['state'].get('arms',{}).get('right',{});q=arm.get('q_rad')
            if not isinstance(q,list) or len(q)!=6: raise BridgeError('Native right joint state incomplete')
            step['expected_q_rad']=[number(v,'right joint') for v in q]
            age=number(joint.get('max_observation_age_s'),'native max_observation_age_s')
            if age<=0: raise BridgeError('Native bounded joint freshness limit unavailable')
            fresh(arm.get('feedback_timestamp'),age)
            step['expected_feedback_timestamp']=arm['feedback_timestamp']
        if self.config()!=cfg or time.time()>=cfg['release']['valid_until_unix']: raise BridgeError('Field release/configuration changed before dispatch')
        if cancel_event.is_set(): return {'phase':'cancelled','business_success':None,'native_dispatch_attempted':False,'physical_stop':'not_requested'}
        used=self.run/'releases'/(approval_id+'.used');used.parent.mkdir(parents=True,exist_ok=True)
        with used.open('x') as f: f.write(key)
        record={'phase':'dispatch_pending','key':key,'receipt_reference':digest(key.encode()),'prepared_id':task_id,'runtime_instance_id':client.runtime_id,
                'client_session_id':session,'idempotency_key':'paos-real-'+digest(key.encode()),'before':before,
                'business_success':None,'physical_stop':'unknown','cancel_sent':False,'dispatch_intent':dispatch}
        save(self.receipt(key),record)
        try:
            provenance={'source':'PhyAgentOS Forge','execution_key':key,'prepared_id':task_id,
                        'operator_confirmation_ref':cfg['release']['operator_confirmation_ref']}
            if task_type=='single_arm_program':
                provenance.update(field_review_id=cfg['release']['field_review_id'],field_verified=cfg['release']['field_verified'])
            native=client.request('submit_task',idempotency_key=record['idempotency_key'],
                                  **dispatch,decision_provenance=provenance)
        except BridgeError as exc:
            if exc.response and isinstance(exc.response.get('result'),dict) and exc.response['result'].get('task_id'):
                native=exc.response['result']
            elif exc.response:
                record.update(phase='failed',native_rejection=exc.response,error=str(exc),physical_stop='not_requested')
                save(self.receipt(key),record);return record
            else:
                # Read-only reconciliation once; no submit or action replay.
                try: native=client.request('get_task',idempotency_key=record['idempotency_key'])
                except BridgeError as lookup:
                    record.update(phase='unknown',error=str(exc),reconciliation_error=str(lookup))
                    save(self.receipt(key),record);return record
        record.update(phase='running',task_id=native['task_id'],native=native);save(self.receipt(key),record)
        deadline=time.monotonic()+checked.get('execution_budget_s',checked['total_budget_s'])+5
        cancel_deadline=None
        while True:
            if native.get('task_state') in TERMINAL:
                if native['task_state']=='unknown' or native.get('execution_rights')=='available': break
            if cancel_event.is_set() or time.monotonic()>=deadline:
                if not record['cancel_sent']:
                    record['cancel_sent']=True;save(self.receipt(key),record)
                    try: native=client.request('cancel_task',task_id=record['task_id'])
                    except BridgeError as exc:
                        record.update(phase='unknown',error=str(exc));save(self.receipt(key),record);return record
                    cancel_deadline=time.monotonic()+3
                if time.monotonic()>cancel_deadline:
                    record.update(phase='unknown',error='Native cancellation/release not confirmed',native=native)
                    save(self.receipt(key),record);return record
            time.sleep(.1)
            try: native=client.request('get_task',task_id=record['task_id'])
            except BridgeError as exc:
                record.update(phase='unknown',error=str(exc));save(self.receipt(key),record);return record
            changed=native.get('pending_confirmation')!=record['native'].get('pending_confirmation')
            record['native']=native
            if changed: save(self.receipt(key),record)
        record.update(phase=native['task_state'],native=native)
        # Original native evidence remains authoritative; never infer object success or physical stop.
        try: record['after']=self.observe(session) if task_type=='act_approved' else self.state(session)
        except Exception as exc: record['after_error']=str(exc)
        save(self.receipt(key),record);return record
    def recover(self,reference):
        if self.config().get('transport')=='native-four-can-replay/v1':
            import native_replay_bridge as replay
            return replay.recover(self,reference)
        if not isinstance(reference,str) or not re.fullmatch('[0-9a-f]{64}',reference): raise BridgeError('Expected receipt SHA reference')
        path=self.run/'attempts'/(reference+'.json');record=read_json(path)
        client=self.client(record['client_session_id'])
        if client.runtime_id!=record['runtime_instance_id']: raise BridgeError('Recovery belongs to a previous Runtime')
        fields={'task_id':record['task_id']} if record.get('task_id') else {'idempotency_key':record['idempotency_key']}
        native=client.request('get_task',**fields);record['native']=native
        if native.get('task_state') in TERMINAL and native.get('execution_rights')=='available': record['phase']=native['task_state']
        elif native.get('task_state') not in TERMINAL: record['phase']='running'
        save(path,record);return record


    def confirm_owned(self,reference,step_index,arm,checkpoint,answer,operator_ref):
        """Operator CLI only; deliberately absent from the LLM Tool descriptor."""
        if answer not in {'yes','no','unknown'} or not isinstance(operator_ref,str) or not operator_ref.strip():
            raise BridgeError('An explicit human answer and operator reference are required')
        record=self.recover(reference);native=record.get('native',{})
        pending=native.get('pending_confirmation') or []
        if isinstance(pending,dict): pending=[pending]
        if not any(p.get('run_id')==record.get('task_id') and p.get('step_index')==step_index and p.get('arm')==arm and p.get('checkpoint')==checkpoint for p in pending):
            raise BridgeError('No matching pending task/step/arm/checkpoint; confirmation was not sent')
        request={'task_id':record['task_id'],'step_index':step_index,'arm':arm,'checkpoint':checkpoint,
                 'answer':answer,'source':operator_ref}
        result=self.client(record['client_session_id']).request('confirm_task',**request)
        evidence=self.run/'confirmations'/(str(time.time_ns())+'.json')
        save(evidence,{'receipt_reference':reference,'request':request,'native_response':result})
        return {'source':'native_runtime','confirmation':result,'evidence_path':str(evidence),'business_success':None}

    def cancel_owned(self,reference):
        if self.config().get('transport')=='native-four-can-replay/v1':
            import native_replay_bridge as replay
            return replay.cancel_owned(self,reference)
        record=self.recover(reference)
        if record.get('phase') in TERMINAL and record.get('native',{}).get('execution_rights')=='available': return record
        if record.get('cancel_sent'): return record
        if not record.get('task_id'): raise BridgeError('Native task identity unresolved; cancellation cannot be guessed')
        record['cancel_sent']=True;path=self.run/'attempts'/(reference+'.json');save(path,record)
        try:
            native=self.client(record['client_session_id']).request('cancel_task',task_id=record['task_id'])
            record['native']=native
            record['phase']=native['task_state'] if native.get('task_state') in TERMINAL and native.get('execution_rights')=='available' else 'unknown'
        except BridgeError as exc: record.update(phase='unknown',error=str(exc))
        save(path,record);return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('status',help='Offline configuration only; never connects')
    subs.add_parser('capabilities');subs.add_parser('state')
    p=subs.add_parser('prepare');p.add_argument('prepared_id');p.add_argument('program_file')
    p.add_argument('--task-type',choices=list(PROGRAMS))
    p=subs.add_parser('prepare-step');p.add_argument('prepared_id');p.add_argument('step_file')
    p=subs.add_parser('prepare-replay',help='Register an operator-provided native trajectory; never starts hardware')
    p.add_argument('prepared_id');p.add_argument('trajectory_file')
    p=subs.add_parser('cancel');p.add_argument('receipt_reference')
    p=subs.add_parser('confirm',help='Human operator only; not exposed as an LLM tool')
    p.add_argument('receipt_reference');p.add_argument('--step-index',type=int,required=True)
    p.add_argument('--arm',choices=['left','right'],required=True)
    p.add_argument('--checkpoint',choices=['after_lift','after_release','after_retreat'],required=True)
    p.add_argument('--answer',choices=['yes','no','unknown'],required=True)
    p.add_argument('--operator-ref',required=True)
    p=subs.add_parser('prepare-policy',help='Freeze an existing towel ACT config; no Runtime connection or release')
    p.add_argument('prepared_id');p.add_argument('model_config');p.add_argument('--duration-s',type=float,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--include-gripper',dest='include_gripper',action='store_true')
    group.add_argument('--no-include-gripper',dest='include_gripper',action='store_false')
    p.add_argument('--act-config-id',default='act-approved-demo')
    p=subs.add_parser('recover');p.add_argument('receipt_reference')
    args=parser.parse_args();bridge=Bridge()
    if args.command=='status':
        config=bridge.config()
        print(json.dumps({'configured_mode':config.get('mode'),'configured_transport':config.get('transport',PROTOCOL),'native_io_performed':False,
                         'field_release_present':bool(config.get('release')),'business_success':None},indent=2));return
    if args.command=='capabilities': result=bridge.capabilities()
    elif args.command=='state': result=bridge.state()
    elif args.command=='cancel': result=bridge.cancel_owned(args.receipt_reference)
    elif args.command=='confirm': result=bridge.confirm_owned(args.receipt_reference,args.step_index,args.arm,args.checkpoint,args.answer,args.operator_ref)
    elif args.command=='prepare': result=bridge.prepare(args.prepared_id,read_json(args.program_file),args.task_type)
    elif args.command=='prepare-step': result=bridge.prepare_step(args.prepared_id,read_json(args.step_file))
    elif args.command=='prepare-replay':
        import native_replay_bridge as replay
        registered=replay.register(bridge,args.prepared_id,args.trajectory_file)
        result=bridge.prepare_task(args.prepared_id,'four_can_replay',{'registered_replay_id':registered['registered_replay_id']})
        result['native_preview']=registered['native_preview']
    elif args.command=='prepare-policy':
        result=bridge.prepare_policy(args.prepared_id,args.model_config,args.duration_s,args.include_gripper,args.act_config_id)
    else: result=bridge.recover(args.receipt_reference)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
