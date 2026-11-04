"""One adapter to the existing four-CAN replay Worker. No CAN/ROS sender here."""
from __future__ import annotations
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

TRANSPORT = 'native-four-can-replay/v1'
TASK_TYPE = 'four_can_replay'
_ACTIVE = {}
_ACTIVE_LOCK = threading.Lock()

def api():
    import runtime_bridge
    return runtime_bridge

def native_module():
    """Import the existing stdlib-only supervisor, never a control environment."""
    key = '_paos_existing_four_can_replay_web'
    if key not in sys.modules:
        path = api().CONTROL/'tools/replay_web.py'
        tools = str(path.parent)
        if tools not in sys.path:
            sys.path.insert(0, tools)
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]

def registry(bridge, name):
    return bridge.root/'prepared/native_replays'/(api().identifier(name)+'.json')

def register(bridge, name, source):
    """Operator CLI only: validate with the real native preview, freeze exact bytes."""
    source = Path(source)
    if not source.is_absolute():
        raise api().BridgeError('Native trajectory requires an absolute server path')
    raw = source.read_bytes()
    # Validate the bytes that will be registered, including concurrent edits.
    target = registry(bridge, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(raw)
    command = [os.environ.get('PIPER_CONTROL_PYTHON','/usr/bin/python3'), '-B', str(api().CONTROL/'tools/replay_handoff.py'),
               '--prepared-can', '--no-video', '--trajectory', str(target), '--preview']
    checked = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if checked.returncode:
        # Preserve the rejected input, but never allow it to become a prepared task.
        api().save(target.with_suffix('.rejected.json'), {'returncode':checked.returncode,
            'stderr':checked.stderr, 'native_preview':checked.stdout})
        raise api().BridgeError('Native trajectory preview rejected; evidence: '+str(target))
    preview = json.loads(checked.stdout)
    api().save(target.with_suffix('.preview.json'), preview)
    return {'registered_replay_id':name, 'path':str(target), 'sha256':api().digest(raw),
            'native_preview':preview, 'field_approved':False, 'physical_execution_attempted':False}

def prepare(bridge, name, program):
    if not isinstance(program, dict) or set(program) != {'registered_replay_id'}:
        raise api().BridgeError('Use an operator-registered replay ID; raw targets are not accepted')
    replay_id = api().identifier(program['registered_replay_id'])
    path = registry(bridge, replay_id)
    if path.with_suffix('.rejected.json').exists() or not path.with_suffix('.preview.json').is_file():
        raise api().BridgeError('Native replay has no successful validation record')
    preview = api().read_json(path.with_suffix('.preview.json'))['trajectory']
    raw = path.read_bytes()
    if preview['trajectory_sha256'] != api().digest(raw):
        raise api().BridgeError('Registered native trajectory changed')
    return bridge.store_task(name, {'schema':'paos-real-task/v1', 'task_type':TASK_TYPE,
        'transport':TRANSPORT, 'registered_replay_id':replay_id,
        'trajectory_sha256':api().digest(raw), 'duration_s':preview['replay_duration_s'],
        'field_approved':False, 'physical_success':None})

def worker_snapshot(bridge):
    with _ACTIVE_LOCK:
        active = _ACTIVE.get(str(bridge.root))
    if active is None:
        return {'source':'existing_native_replay_worker', 'available':False,
                'feedback_available':False, 'reason':'no_owned_native_worker', 'business_success':None}
    worker, event, receipt_path = active
    return {'source':'existing_native_replay_worker', 'available':True,
            'native':worker.status(), 'receipt_reference':receipt_path.stem,
            'feedback_available':False, 'business_success':None}

def ensure_live(bridge):
    cfg = bridge.config()
    if cfg.get('mode') != 'live':
        raise api().BridgeError('OFFLINE_ONLY: native four-CAN worker is not started')
    if cfg.get('transport') != TRANSPORT:
        raise api().BridgeError('Current chain requires the registered native four-CAN replay transport')
    return cfg

def lock_available():
    with (api().CONTROL/'run/unified_runtime.lock').open('a') as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

def process_identity(pid):
    try:
        path = Path('/proc')/str(int(pid))
        return {'pid':int(pid), 'start_time':(path/'stat').read_text().rsplit(')',1)[1].split()[19],
                'cmdline_sha256':hashlib.sha256((path/'cmdline').read_bytes()).hexdigest()}
    except (OSError, ValueError, IndexError, TypeError):
        return None

def native_evidence(worker):
    status = worker.status()
    directory = (status.get('recording') or {}).get('directory')
    summary = None
    if directory:
        candidate = Path(directory).resolve().parent/'summary.json'
        allowed = (api().CONTROL/'runs').resolve()
        if candidate.is_relative_to(allowed) and candidate.is_file():
            summary = {'path':str(candidate), 'data':api().read_json(candidate)}
    return status, summary

def preflight(bridge, name, key):
    cfg = ensure_live(bridge)
    artifact_path = bridge.task_path(name)
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)
    if artifact.get('task_type') != TASK_TYPE or artifact.get('transport') != TRANSPORT:
        raise api().BridgeError('Legacy station/TCP/policy tasks are not adapted to the current four-CAN chain')
    path = registry(bridge, artifact['registered_replay_id'])
    if api().digest(path.read_bytes()) != artifact['trajectory_sha256']:
        raise api().BridgeError('Native replay bytes changed after preparation')
    approval = cfg.get('release') or {}
    if approval.get('prepared_id') != name or approval.get('sha256') != api().digest(raw):
        raise api().BridgeError('No field release for this exact prepared artifact')
    if not 0 < float(approval.get('valid_until_unix',0))-time.time() <= 300:
        raise api().BridgeError('Fresh five-minute field release required')
    release_id = api().identifier(approval.get('release_id'))
    if not approval.get('operator_confirmation_ref'):
        raise api().BridgeError('Missing onsite operator confirmation; never generated by the Agent')
    if bridge.receipt(key).exists():
        raise api().BridgeError('Existing execution receipt: recover, never replay')
    for old in (bridge.run/'attempts').glob('*.json'):
        if api().read_json(old).get('phase') in {'dispatch_pending','running','unknown'}:
            raise api().BridgeError('An unresolved owned task blocks a new native worker')
    if not lock_available():
        raise api().BridgeError('Native control lock is owned by another session')
    return cfg, artifact, path, release_id

def execute(bridge, name, key, cancel):
    cfg, artifact, trajectory_path, release_id = preflight(bridge, name, key)
    native = native_module()
    # Identity/link resolution is the unmodified native implementation, read-only.
    native.require_prepared_can()
    trajectory = native.Trajectory(trajectory_path)
    reference = api().digest(key.encode())
    destination = bridge.run/'four-can'/reference
    worker = native.Worker(trajectory, destination, preview=False, managed=True, prepared_can=True)
    if worker.preview or '--prepared-can' not in worker.build_command(False):
        raise api().BridgeError('Native hardware transport mismatch')
    if bridge.config() != cfg or time.time() >= cfg['release']['valid_until_unix']:
        raise api().BridgeError('Field release changed before dispatch')
    if cancel.is_set():
        return {'phase':'cancelled','native_dispatch_attempted':False,'business_success':None,'physical_stop':'not_requested'}
    used = bridge.run/'releases'/(release_id+'.used')
    used.parent.mkdir(parents=True,exist_ok=True)
    with used.open('x') as f:
        f.write(key)
    record = {'transport':TRANSPORT,'phase':'dispatch_pending','key':key,
        'receipt_reference':reference,'prepared_id':name,'trajectory_sha256':artifact['trajectory_sha256'],
        'business_success':None,'physical_success':None,'physical_stop':'unknown',
        'native_output':str(destination),'native':{'execution_rights':'owned'},'worker_identity':None}
    receipt_path = bridge.receipt(key)
    api().save(receipt_path,record)
    with _ACTIVE_LOCK:
        _ACTIVE[str(bridge.root)] = (worker,cancel,receipt_path)
    outcome = 'unknown'
    try:
        # Existing Worker starts the existing managed native owner; no CAN code here.
        worker.start(False)
        record.update(phase='running',worker_identity=process_identity(worker.process.pid))
        api().save(receipt_path,record)
        deadline = time.monotonic()+120
        while True:
            state = worker.status()
            if cancel.is_set():
                outcome = 'cancelled'; break
            if state.get('error') or state.get('state') == 'FAULT':
                outcome = 'failed';record['error'] = state.get('error') or 'Native worker fault';break
            if state.get('state') == 'HOLD' and state.get('cursor_s',0) >= trajectory.duration-.001:
                outcome = 'completed';break
            if not state.get('running'):
                outcome = 'failed';record['error'] = 'Native worker exited before trajectory completion';break
            if time.monotonic() >= deadline:
                outcome = 'failed';record['error'] = 'Native preparation/execution exceeded the 120-second task budget';break
            cancel.wait(.1)
    except Exception as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        # A start exception may occur after spawn. Never retry it.
        record['worker_identity'] = process_identity(worker.process.pid) if worker.process else None
    finally:
        try:
            worker.stop()
        except Exception as exc:
            record['cleanup_error'] = type(exc).__name__+': '+str(exc)
        status, summary = native_evidence(worker)
        exited = worker.process is None or worker.process.poll() is not None
        released = exited and not status.get('forced_stop') and lock_available()
        if not released:
            outcome = 'unknown'
        if outcome == 'completed' and (status.get('exit_code') != 0 or not summary or summary['data'].get('error')):
            outcome = 'failed' if summary else 'unknown'
        record.update(phase=outcome,native_status=status,native_summary=summary,
            native={'execution_rights':'available' if released else 'unknown',
                'release_evidence':{'worker_exited':exited,'native_lock_available':released,
                    'physical_stop':'unknown','role_restoration_observed':False}},
            result_refs=[str(destination)]+([summary['path']] if summary else []))
        api().save(receipt_path,record)
        with _ACTIVE_LOCK:
            _ACTIVE.pop(str(bridge.root),None)
    return record

def recover(bridge, reference):
    if not re.fullmatch(r'[0-9a-f]{64}', reference):
        raise api().BridgeError('Expected an owned receipt SHA256')
    path = bridge.run/'attempts'/(reference+'.json')
    record = api().read_json(path)
    if record.get('transport') != TRANSPORT:
        raise api().BridgeError('Receipt is not a native four-CAN task')
    with _ACTIVE_LOCK:
        active = _ACTIVE.get(str(bridge.root))
    if active and active[2] == path:
        return {**record,'native_status':active[0].status()}
    if record.get('phase') in {'dispatch_pending','running'}:
        # Parent death is handled by the original native supervisor. Without its
        # final evidence do not promote completion or redispatch an old task.
        record.update(phase='unknown',worker_still_present=(record.get('worker_identity') is not None and
            process_identity(record['worker_identity']['pid']) == record['worker_identity']))
        api().save(path,record)
    return record

def cancel_owned(bridge, reference):
    record = recover(bridge, reference)
    with _ACTIVE_LOCK:
        active = _ACTIVE.get(str(bridge.root))
    if active and active[2].stem == reference:
        active[1].set()
        return {'phase':'completed','cancellation_requested':True,'cancel_request_only':True,
                'receipt_reference':reference,'physical_stop':'unknown','business_success':None}
    if record.get('phase') == 'unknown' and record.get('worker_still_present'):
        # No signal to a process after losing its native Worker owner object.
        raise api().BridgeError('Original native owner must be reconciled; no blind signal or replay')
    return {**record,'cancellation_requested':False}
