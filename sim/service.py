"""A single owner for physics and EGL. HTTP threads never touch MuJoCo."""
from __future__ import annotations

import copy
import io
import json
import os
import queue
import threading
import time
import uuid
import hashlib
import urllib.request
import mujoco
import numpy as np
from concurrent.futures import Future

from PIL import Image

from .environment import OpenArmEnv, ROOT, STAGES, TASK
from .policy import make_client, load_config
from .recording import Recorder, dump, serial
from .timing import PolicyTiming
from .decision_timing import DecisionWait
from .planning_timing import PlanningTiming
from .replay import ReferenceReplay
from .tamp import TampPlanner, PlanningFailure
from .curobo import CuRoboPlanner, metadata as curobo_metadata
from .relocation import RelocationPlanner
from .view import FRONT,validate_view
from .execution import ExecutionProviders, run_execution
from harness.execution_contracts import execution_lifecycle, planner_provenance


def physical_memory_contract(manifest):
    """Fingerprint actual physics/data; planners, workers and tests are separate."""
    physical = {k:manifest.get(k) for k in ('schema','engine','profile','physics_dt','contract')}
    modules={'sim/environment.py','sim/controller.py','sim/contacts.py','sim/gripper.py','sim/reference_clock.py'}
    physical['hashes'] = {p:value for p,value in manifest.get('hashes', {}).items()
        if p in modules or p in {'sim/profile.json','calibrate.json','source.json'}
        or p.startswith('openarm_mujoco/') or p.startswith('reference/')
        or p.startswith('task/') and p.endswith('.xml')}
    evidence=manifest.get('model_evidence',{})
    if evidence:
        physical['model_identity']={
            'compiled':evidence['compiled']['sha256'],
            'inputs':evidence['inputs']['content_sha256'],
            'runtime_parameters':evidence.get('runtime_parameters',{}).get('sha256'),
            'scope':'compiled initial candidate and frozen assets; no physical measurement'}
    return physical


def memory_context(manifest, policy_config, robot):
    """Compatibility concerns robot/control/sensor contracts, not UI/tests/RAG."""
    physical=physical_memory_contract(manifest)
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {'schema':'openarm-memory/4','environment':'simulation','robot':robot,
            'contract':digest(physical),'policy_contract':digest(policy_config or {})}


def service_env_kwargs(selection):
    values={'images':False}
    if selection is not None:values['selection']=list(selection)
    return values


class SimulationService:
    def __init__(self, *, environment_factory=None, initial_scene_selection=None,
                 policy_client_factory=None, policy_timing_class=None,
                 tamp_planner_factory=None, relocation_planner_factory=None,
                 curobo_planner_factory=None, curobo_metadata_provider=None,
                 recorder_class=None):
        self.environment_factory = OpenArmEnv if environment_factory is None else environment_factory
        self.initial_scene_selection = (None if initial_scene_selection is None
            else tuple(initial_scene_selection))
        self.policy_client_factory = policy_client_factory or make_client
        self.policy_timing_class = policy_timing_class or PolicyTiming
        self.tamp_planner_factory = tamp_planner_factory
        self.relocation_planner_factory = relocation_planner_factory
        self.curobo_planner_factory = curobo_planner_factory
        self.curobo_metadata_provider = curobo_metadata_provider
        self.recorder_class = recorder_class
        self.commands = queue.Queue()
        self.completed = queue.Queue()
        self.lock = threading.Lock()
        self.stop_job = threading.Event()
        self.shutdown = threading.Event()
        self.ready = threading.Event()
        self.cache = {"state": "starting"}
        self.frames = {}
        self.frame_history = {}
        self.view_request=validate_view(FRONT)
        self.view_revision=0
        self.applied_view_revision=-1
        self.job = None
        self.decision_wait = None
        self.decision_receipt = None
        self.decision_publish = 0.
        self.backend_refresh=0.
        self.curobo_status={'ready':False,'reason':'checking local worker'}
        self.policy_status={'ready':False,'reason':'policy not configured'}
        self.thread = threading.Thread(target=self._worker, name="openarm-physics", daemon=True)
        self.thread.start()
        if not self.ready.wait(30):
            raise RuntimeError("simulation initialization timed out")
        if self.cache.get("state") == "error":
            raise RuntimeError(self.cache["error"])

    def _tamp_planner_factory(self):
        """Resolve an injected router, or the current module default for tests."""
        return getattr(self, 'tamp_planner_factory', None) or TampPlanner

    def _relocation_planner_factory(self):
        """Resolve an injected relocation router, or the current module default."""
        return getattr(self, 'relocation_planner_factory', None) or RelocationPlanner

    def _curobo_planner_factory(self):
        """Resolve an injected GPU route, or the current module default."""
        return getattr(self, 'curobo_planner_factory', None) or CuRoboPlanner

    def _curobo_metadata(self):
        """Probe the worker paired with this service's selected CuRobo route."""
        provider = getattr(self, 'curobo_metadata_provider', None)
        return provider() if provider is not None else curobo_metadata()

    def _recorder_class(self):
        """Resolve an instance recorder, or the current module default for tests."""
        return getattr(self, 'recorder_class', None) or Recorder

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.cache)

    def frame(self, camera, observation_id=None):
        with self.lock:
            frames = self.frames if not observation_id else self.frame_history.get(observation_id)
            if frames is None:
                raise ValueError("camera observation expired; request a fresh status snapshot")
            if camera not in frames:
                raise ValueError("unknown camera")
            return frames[camera]

    def set_view(self, values):
        config=validate_view(values)
        with self.lock:
            if config!=self.view_request:
                self.view_request=config
                self.view_revision+=1
            return {'accepted':True,'revision':self.view_revision}

    def _render_operator(self):
        with self.lock:
            view=copy.deepcopy(self.view_request)
            revision=self.view_revision
        buf=io.BytesIO()
        Image.fromarray(self.env.render('operator',view=view)).save(buf,format='JPEG',quality=90)
        return buf.getvalue(),view,revision

    def _refresh_operator(self):
        # The physics owner applies coalesced UI requests even when paused.
        # No sensor rerender, new observation ID, or physics step is needed.
        if self.view_revision==self.applied_view_revision:
            return
        frame,view,revision=self._render_operator()
        with self.lock:
            self.frames['operator']=frame
            self.cache['viewer']={**view,'revision':revision}
            self.applied_view_revision=revision
            self.frame_history[self.cache['observation_id']]['operator']=frame

    def call(self, command, **arguments):
        if command == "stop":
            self.stop_job.set()
            return {"state": "stop_requested"}
        if command != 'end_decision' and self.snapshot().get("state") == "running":
            raise ValueError("当前实验正在运行；请先停止，再重置或开始另一轮")
        future = Future()
        self.commands.put((command, arguments, future))
        try:
            return future.result(timeout=35)
        except TimeoutError:
            future.cancel()
            raise RuntimeError("simulation busy; command expired") from None

    def _publish(self):
        if not self.job and time.monotonic()-self.backend_refresh>10:
            self.curobo_status=self._curobo_metadata()
            self.policy_status={'ready':False,'reason':'compatible checkpoint/endpoint required'}
            if self.policy_config:
                try:
                    if self.policy_config['backend']=='lerobot_http':
                        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        with opener.open(self.policy_config['uri'].rstrip('/')+'/metadata',timeout=.5) as response:
                            meta=json.loads(response.read(65536))
                        if meta.get('checkpoint')!=self.policy_config['checkpoint'] or not meta.get('strict_weights_verified'):
                            raise ValueError('checkpoint identity or strict weight check mismatch')
                        self.policy_status={'ready':True,'checkpoint':meta['checkpoint'],'scope':'inference connection verified; task performance is separate'}
                    else:
                        self.policy_status={'ready':True,'scope':'configured; connection checked when invoked'}
                except (OSError,ValueError) as exc:
                    self.policy_status={'ready':False,'reason':str(exc)}
            self.backend_refresh=time.monotonic()
        frames = {}
        for camera in ("overview", "head", "left", "right"):
            rgb = self.env.render(camera)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
            frames[camera] = buf.getvalue()
        operator,view,revision=self._render_operator()
        frames['operator']=operator
        tcp = {}
        for side, body in self.env.ee_ids.items():
            rotation = self.env.data.xmat[body].reshape(3, 3)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
            tcp[side] = {'frame': 'world', 'position_m': (self.env.data.xpos[body] + rotation @ np.array([0., .006, .075])).tolist(),
                         'quaternion_xyzw': np.r_[quat[1:], quat[0]].tolist(), 'source': 'robot_forward_kinematics'}
        cache = {"state": "running" if self.job or self.decision_wait else "ready", "backend": "native-mujoco", "robot_model": self.env.robot_version,
                 'memory_context': memory_context(self.env.manifest(), self.policy_config, self.env.robot_version),
                 "physics_owner": "server", "task": TASK, "stages": STAGES,
                 "observation": {"state": self.env.state().tolist(), "sim_time": float(self.env.data.time),
                                 "tick": self.env.tick, "cameras": list(self.env.profile["camera_map"]), 'tcp': tcp, 'episode_finished':self.env._done, 'clock_mode':self.env._clock_mode,'task_window':self.env.task_window()},
                 "diagnostics": self.env.diagnostics(), "job": self.job,
                 "last_run": getattr(self, "last_run", None), "keyframes": self.keyframes,
                 'decision_timing':self.decision_wait.summary() if self.decision_wait else self.decision_receipt,
                 "viewer":{**view,'revision':revision},
                 "transfer_ready": False, "calibration_gaps": self.env.profile["uncertainties"],
                 "policy_configured": bool(self.policy_config) and self.policy_status['ready'],
                 'policy_status':self.policy_status,
                 'curobo':self.curobo_status,
                 'policy_type': (self.policy_config or {}).get('policy_type', 'pi05'),
                 'policy_config_error': self.policy_config_error,
                 "observation_id": f"{self.epoch}:{self.env.tick}",
                 "recording": "server-owned; not tied to browser frame rate"}
        with self.lock:
            self.cache = copy.deepcopy(cache)
            self.frames = frames
            self.applied_view_revision=revision
            self.frame_history[cache["observation_id"]] = frames
            while len(self.frame_history) > 32:
                self.frame_history.pop(next(iter(self.frame_history)))

    def _capture(self, label, recorder=None, *, sensor_evidence=False,camera=None):
        selected=camera or ('operator' if recorder is None else 'overview')
        if selected not in {'operator','overview','head','left','right'}:
            raise ValueError('unknown capture camera')
        if recorder is None:
            recorder = self._recorder_class()(self.env, "snapshot", record_images=False, seed=0)
            own = True
        else:
            own = False
        folder = recorder.path / "keyframes"
        folder.mkdir(exist_ok=True)
        path = folder / f"{self.env.tick:06d}-{len(self.keyframes):02d}.jpg"
        view=None
        if selected=='operator':
            pixels,view,_=self._render_operator();path.write_bytes(pixels)
        else:Image.fromarray(self.env.render(selected)).save(path, quality=95)
        item = {"label": label, "sim_time": float(self.env.data.time), "tick": self.env.tick,
                "url": "/runs/" + str(path.relative_to(ROOT / "runs")), "camera": '操作者视角' if selected=='operator' else selected,
                'manual':own,'observation_id':f'{self.epoch}:{self.env.tick}'}
        if view:item['operator_view']=view
        if sensor_evidence:
            item['sensor_urls'] = {}
            item['observation_id'] = f'{self.epoch}:{self.env.tick}'
            for camera in ('head','left','right'):
                sensor_path = path.with_name(path.stem+'-'+camera+'.jpg')
                Image.fromarray(self.env.render(camera)).save(sensor_path, quality=95)
                item['sensor_urls'][camera] = '/runs/' + str(sensor_path.relative_to(ROOT/'runs'))
        self.keyframes.append(item)
        self.keyframes = self.keyframes[-100:]
        recorder.event({"type": "keyframe", **item})
        if own:
            recorder.finish("snapshot", self.env.info())
        return item

    def _worker(self):
        self.env = None
        try:
            self.env = self.environment_factory(**service_env_kwargs(self.initial_scene_selection))
            self.env.reset(seed=0)
            self.policy_config = None
            self.policy_config_error = None
            if os.environ.get('OPENARM_POLICY_CONFIG'):
                try:
                    self.policy_config = load_config(os.environ['OPENARM_POLICY_CONFIG'])
                except (ValueError, OSError) as exc:
                    self.policy_config_error = str(exc)
            self.epoch = uuid.uuid4().hex[:12]
            self.keyframes = []
            self._publish()
            self.ready.set()
            while not self.shutdown.is_set():
                try:
                    command, args, future = self.commands.get(timeout=.01 if self.decision_wait else .1)
                except queue.Empty:
                    if self.decision_wait:
                        self.decision_wait.advance(cancelled=self.stop_job.is_set() or self.shutdown.is_set())
                        if time.monotonic()-self.decision_publish>.15:
                            self._publish();self.decision_publish=time.monotonic()
                    self._refresh_operator()
                    if time.monotonic()-self.backend_refresh>10:self._publish()
                    continue
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if command == 'prepare':
                        if args.get('observation_id') != f'{self.epoch}:{self.env.tick}':
                            raise ValueError('准备任务需要当前观测，先 observe')
                        mode=args.get('mode','preserve')
                        if mode=='reset':self.env.reset(seed=0,options={'episode':self.env.episode})
                        elif mode=='preserve':self.env.prepare_task()
                        else:raise ValueError('unknown prepare mode')
                        self.epoch=uuid.uuid4().hex[:12];self.keyframes=[];self.last_run=None;self.decision_receipt=None
                        self._publish();result={'state':'prepared','mode':mode,'observation_id':self.cache['observation_id'],
                            'physical_state_preserved':mode=='preserve','fresh_observation_required':True}
                    elif command == "reset":
                        self.env.reset(seed=int(args.get("seed", 0)), options={"episode": int(args.get("episode", 0))})
                        self.epoch = uuid.uuid4().hex[:12]
                        self.keyframes = []
                        self.last_run = None
                        self.decision_receipt = None
                        self._publish()
                        result = self.snapshot()
                    elif command == "capture":
                        result = self._capture(str(args.get("label", "手动采集"))[:80],camera=args.get('camera'))
                        self._publish()
                    elif command == 'task_budget':
                        self.env.require_current_sources()
                        result=self.env.set_task_budget(args.get('seconds'))
                        self._publish()
                    elif command == 'begin_decision':
                        if args.get('observation_id')!=f'{self.epoch}:{self.env.tick}':
                            raise ValueError('decision capture is stale')
                        self.env.require_current_sources();self.stop_job.clear()
                        self.decision_wait=DecisionWait(self.env,args['root'],args['request_id'],args['observation_id'])
                        self._capture('模型决策前',self.decision_wait.recorder,sensor_evidence=True)
                        self._publish();result=self.decision_wait.summary()
                    elif command == 'end_decision':
                        wait=self.decision_wait
                        if not wait or args.get('root')!=wait.details['root'] or args.get('request_id')!=wait.details['request_id']:
                            raise ValueError('decision wait identity mismatch')
                        self.decision_receipt=wait.finish(cancelled=self.stop_job.is_set() or self.shutdown.is_set())
                        self._capture('模型决策后',wait.recorder,sensor_evidence=True)
                        self.decision_wait=None;self._publish();result=self.decision_receipt
                    elif command in {"replay", "policy", "tamp"}:
                        self.env.require_current_sources()
                        self.stop_job.clear()
                        self.job = {"kind": command, "progress": 0, "episode": args.get("episode", 0),
                                    "execution_id":uuid.uuid4().hex, "status":"accepted", "phase":"accepted",
                                    "initiator":args.get('initiator',{'type':'direct-command'})}
                        self._publish()
                        future.set_result({"state": "started", "kind": command, "execution_id":self.job['execution_id']})
                        self._run(command, args)
                        continue
                    elif command == "manifest":
                        result = self.env.manifest()
                    else:
                        raise ValueError("unknown simulation command")
                    future.set_result(result)
                except Exception as exc:
                    if command in {'begin_decision','end_decision'} and self.decision_wait:
                        self.decision_receipt=self.decision_wait.finish(cancelled=True)
                        self.decision_wait=None
                        self._publish()
                    if not future.done():
                        future.set_exception(exc)
        except Exception as exc:
            with self.lock:
                self.cache = {"state": "error", "error": str(exc)}
            self.ready.set()
        finally:
            if self.decision_wait:
                self.decision_wait.finish(cancelled=True)
            if self.env:
                self.env.close()

    def _run(self, kind, args):
        # Keep this entrypoint and resolve current defaults when a job starts.
        providers = ExecutionProviders(
            load_policy_config=load_config,
            policy_client=getattr(self, 'policy_client_factory', make_client),
            policy_timing=getattr(self, 'policy_timing_class', PolicyTiming),
            replay=ReferenceReplay,
            planning_timing=PlanningTiming,
        )
        return run_execution(self, kind, args, providers)

    def close(self):
        self.stop_job.set()
        self.shutdown.set()
        self.thread.join(timeout=65)
