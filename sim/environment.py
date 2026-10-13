"""MuJoCo environment with an explicit robot contract and independent evaluator.

Policy commands use 30 Hz; audited reference replay uses recorded intervals.
Physics uses 1 ms steps, never the browser's animation clock.
Ground-truth object poses are restricted to diagnostics and the evaluator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from .contacts import Contacts
from .gripper import GripperMap
from .model_evidence import ModelInputs, ParameterAudit, compiled_evidence

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
STAGES = ("初始场景", "右手夹持", "右手抬起", "双手交接", "左手持瓶", "松开放置", "稳定入篮")
TASK = "Pass the black bottle from the right hand to the left, then place it into the white box."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Capture code when this process imports the environment, not when a later
# experiment happens to write its manifest. A live server must never label old
# imported code with the hash of newly edited files on disk.
_CODE_HASHES = {str(p.relative_to(ROOT)): digest(p)
                for folder in ('sim', 'harness') for p in sorted((ROOT/folder).glob('*.py'))}
_CODE_HASHES['server.py'] = digest(ROOT/'server.py')


class OpenArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, *, images=True, profile=None):
        self.profile = json.loads((ROOT / "sim/profile.json").read_text())
        if profile:
            self.profile.update(profile)
        self.controller_mode=self.profile.get('controller_mode','position_pd_bias')
        if self.controller_mode not in ('position_pd_bias','source_mit_candidate'):
            raise ValueError('unknown controller mode')
        if self.controller_mode=='source_mit_candidate' and self.profile['gravity_compensation']!='none':
            raise ValueError('source-MIT candidate must explicitly disable extra bias compensation')
        self.source_controller=None
        self.controller_feedback={}
        self.source_initial_integral=self.profile.get('source_initial_integral','zero')
        if self.controller_mode=='source_mit_candidate':
            from .controller import SourceMIT
            self.source_controller=SourceMIT()
            self.source_controller.contract(self.source_initial_integral)  # source identity before experiment
        self.images = images
        self.robot_version = self.profile.get('robot_version', 'v2')
        if self.robot_version not in ('v1', 'v2'):
            raise ValueError('unknown official OpenArm robot model')
        if self.source_controller and self.robot_version!='v1':raise ValueError('audited source controller only matched to candidate v1')
        self.calibration = json.loads((ROOT / "calibrate.json").read_text())
        self.reference = json.loads((ROOT / "reference/real_trajectory.json").read_text())["episodes"]
        cell_path = ROOT/'task/cell-v1.xml' if self.robot_version=='v1' else ROOT/'openarm_mujoco/v2/cell/cell.xml'
        self._model_inputs = ModelInputs(ROOT, ROOT/'task/bottle_to_box.xml',
            overrides={('task/bottle_to_box.xml','cell.xml'):cell_path})
        xml = self._model_inputs.xml(ROOT/'task/bottle_to_box.xml')
        if self.robot_version == 'v1':
            # Resolve the two attachments in memory. The upstream home key is
            # unused (reset comes from recorded encoders) and cannot be safely
            # namespaced through nested attachments. Keep upstream files intact.
            robot = ET.fromstring(self._model_inputs.xml(ROOT/'openarm_mujoco/v1/openarm_bimanual.xml'))
            robot.remove(robot.find('keyframe'))
            robot.find('compiler').set('meshdir', str(ROOT/'openarm_mujoco/v1/meshes'))
            cell = ET.fromstring(self._model_inputs.xml(cell_path))
            cell.find('compiler').set('meshdir', str(ROOT/'openarm_mujoco/v2/assets'))
            cell.find('asset/model').set('file', 'openarm-v1-runtime.xml')
            spec = mujoco.MjSpec.from_string(xml.replace('file="cell.xml"', 'file="cell-v1-runtime.xml"'),
                include={**self._model_inputs.includes, 'cell-v1-runtime.xml': ET.tostring(cell),
                         'openarm-v1-runtime.xml': ET.tostring(robot)}, assets=self._model_inputs.assets)
        else:
            spec = mujoco.MjSpec.from_string(xml.replace('file="cell.xml"', f'file="{cell_path}"'),
                include=self._model_inputs.includes, assets=self._model_inputs.assets)
        # The laboratory robot has a fixed mounting transform. A locked slide
        # should be a model constraint, not qpos overwritten during each step.
        spec.delete(spec.actuator("lifter_ctrl"))
        spec.delete(spec.joint("openarm_lifter_joint"))
        spec.body("openarm_lifter_link").pos[2] += self.profile["mount_height"]
        if self.robot_version == 'v1':
            # v1 defines a full 698 mm torso below the shoulder mount. The
            # existing cell attaches its robot at the shoulder, not at the foot.
            spec.body('openarm_body_link0').pos[2] -= .698
            for side in SIDES:
                hand = spec.body(f'openarm_{side}_hand')
                hand.add_camera(name=f'camera_wrist_{side}', pos=[-.025,0,-.035],
                    xyaxes=[0,1,0, 1,0,0], fovy=90, resolution=[640,480])
        for name in self.calibration["scene"]["disable_cell_collisions"] + ["cell_vis"]:
            spec.delete(spec.geom(name))
        for side in SIDES:
            # Drop the historical peak-as-limit auxiliary actuator. Keep exactly
            # one actuator per command channel and record the chosen limit.
            spec.delete(spec.actuator(f"{side}_finger_limit"))
        self.model = spec.compile()
        if self.model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("physics contact timing currently requires a single-step integrator")
        self.data = mujoco.MjData(self.model)
        self.render_mode = "rgb_array"
        self._renderers = {}
        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self._option.sitegroup[:] = 0
        self.model.vis.quality.shadowsize = 2048
        self.model.vis.global_.offwidth = max(size[0] for size in self.profile["camera_sizes"].values())
        self.model.vis.global_.offheight = max(size[1] for size in self.profile["camera_sizes"].values())
        if self.profile.get("background_style") == "neutral":
            floor = self.model.geom("floor").id
            self.model.geom_matid[floor] = -1
            self.model.geom_rgba[floor] = [.32, .34, .32, 1.]
        if self.profile.get("gripper_mapping") != "shared_motor_angle_v1":
            raise ValueError("a shared motor-angle gripper mapping is required")
        if self.profile.get("gripper_force_limit_frame") != "actuated_finger_joint":
            raise ValueError("the gripper force limit must declare its actuated-finger-joint frame")
        self.qids, self.dids, self.aids, self.fingers, self.grippers, self.ee_ids = {}, {}, {}, {}, {}, {}
        lo, hi = [], []
        for side in SIDES:
            joints = [self.model.joint(f"openarm_{side}_joint{i}") for i in range(1, 8)]
            self.qids[side] = np.array([j.qposadr[0] for j in joints])
            self.dids[side] = np.array([j.dofadr[0] for j in joints])
            self.aids[side] = np.array([self.model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, 8)])
            self.fingers[side] = [self.model.joint(f"openarm_{side}_finger_joint{i}") for i in (1, 2)]
            finger = self.fingers[side][0]
            opening = float(max(finger.range, key=abs))
            unit = 'm' if self.model.jnt_type[finger.id] == mujoco.mjtJoint.mjJNT_SLIDE else 'rad'
            expected_force_unit = 'N' if unit == 'm' else 'N·m'
            if self.profile['gripper_force_limit_unit'] != expected_force_unit:
                raise ValueError('gripper force limit unit does not match the active joint type')
            self.grippers[side] = GripperMap(self.profile["state_gripper_open"],
                                             self.profile["state_gripper_closed"], opening, unit)
            ee_name = f'openarm_{side}_hand' if self.robot_version == 'v1' else f'openarm_{side}_ee_base_link'
            self.ee_ids[side] = self.model.body(ee_name).id
            for i, aid in enumerate(self.aids[side], 1):
                kp = self.calibration["actuators"]["kp"][f"j{i}"]
                kd = self.calibration["actuators"]["kd"][f"j{i}"]
                self.model.actuator_gainprm[aid, 0] = kp
                self.model.actuator_biasprm[aid, 1:3] = [-kp, -kd]
            aid = self.model.actuator(f"{side}_finger1_ctrl").id
            # q_joint = a * (q_motor - closed). Virtual work reflects motor
            # stiffness/damping by 1/a². This is a coordinate conversion, not
            # additional fitted stiffness. v1 joint effort is N; v2 is N.m.
            kp = self.grippers[side].joint_gain(self.calibration["actuators"]["kp"]["gripper"])
            kd = self.grippers[side].joint_gain(self.calibration["actuators"]["kd"]["gripper"])
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1:3] = [-kp, -kd]
            self.model.actuator_forcerange[aid] = [-self.profile["gripper_force_limit_value"], self.profile["gripper_force_limit_value"]]
            lo.extend([j.range[0] for j in joints] + [self.profile["action_gripper_open"]])
            hi.extend([j.range[1] for j in joints] + [self.profile["action_gripper_closed"]])
        self.action_space = gym.spaces.Box(np.array(lo, np.float32), np.array(hi, np.float32))
        obs_spaces = {"state": gym.spaces.Box(-np.inf, np.inf, (16,), np.float32)}
        if images:
            obs_spaces["images"] = gym.spaces.Dict({name: gym.spaces.Box(0, 255,
                (self.profile["camera_sizes"][name][1], self.profile["camera_sizes"][name][0], 3), np.uint8)
                for name in self.profile["camera_map"]})
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.bottle_id = self.model.body("task_bottle").id
        self.bottle_geoms = np.flatnonzero(self.model.geom_bodyid == self.bottle_id)
        self.box_id = self.model.body("target_box").id
        self.box_floor_id = self.model.geom("box_floor").id
        self.table_id = self.model.geom('cell_table_col').id
        self.placement_target = None
        self.contacts = Contacts(self.model, self.bottle_id, self.box_floor_id, self.table_id)
        self.base_to_world = np.array([.185, 0., .642 + self.profile["mount_height"]])
        self.tick = 0
        self._clock_mode = None
        self.placement_target = None
        self.last_action = None
        self.events = []
        self._done = False
        self.task_termination='legacy_benchmark'
        self._source = json.loads((ROOT/'source.json').read_text())
        resources = ['sim/profile.json', 'source.json', 'calibrate.json', 'task/bottle_to_box.xml',
                     'reference/real_trajectory.json', 'reference/acquisition.json', 'task/cell-v1.xml',
                     f'openarm_mujoco/{self.robot_version}/openarm_bimanual.xml']
        self._runtime_hashes = {**_CODE_HASHES, **{p:digest(ROOT/p) for p in resources}}
        self._runtime_hashes.update(self._model_inputs.hashes())
        self._model_evidence = {'inputs':self._model_inputs.describe(self.model),
                               'compiled':compiled_evidence(self.model)}
        self._parameter_audit=ParameterAudit(self.model)
        # Keep immutable hashes and disk metadata, release the VFS byte copies.
        self._model_inputs.contents.clear()
        self._model_inputs.includes.clear()
        self._model_inputs.assets.clear()
        self.require_current_sources()

    def source_status(self):
        changed = [p for p, value in self._runtime_hashes.items() if p not in self._model_inputs.rows
                   if not (ROOT/p).is_file() or digest(ROOT/p) != value]
        changed = sorted(set(changed+self._model_inputs.changed()))
        return {'matches_disk': not changed, 'restart_required': bool(changed), 'changed_files': changed}

    def require_current_sources(self):
        if self.source_status()['restart_required']:
            raise ValueError('运行中的代码与磁盘版本不同；请重启 ./run 后再开始实验，避免结果版本混淆')
        if self._parameter_audit.summary()['changed_fields']:
            raise ValueError('运行模型参数已改变；请重启 ./run 后再开始实验，避免候选模型身份混淆')

    def manifest(self):
        return {"schema": "openarm-sim/1", "engine": f"mujoco-{mujoco.__version__}",
                'task_window':self.task_window(),
                "source": {**self._source, 'robot_model': f'OpenArm {self.robot_version}'},
                "profile": self.profile, "physics_dt": float(self.model.opt.timestep),
                "contract": {"state": "left q1..q7(rad), measured gripper motor angle(rad), right q1..q7(rad), measured gripper motor angle(rad)",
                             'controller_semantics':self.source_controller.contract(self.source_initial_integral) if self.source_controller else
                                {'mode':'position_pd_bias','scope':'candidate position PD plus selected compensation; not source-MIT equivalent'},
                             "action": "left q1..q7(rad), target gripper motor angle(rad), right q1..q7(rad), target gripper motor angle(rad)",
                             "policy_controller": "per-arm 7-D L2 step cap, then joint-limit clipping; requested and applied actions logged separately",
                             "controller_source": "/home/sj/Desktop/wuji/lerobot_20260116/openarm_deploy/model_launchers/models/dp/deploy_dp_dual_joint.py:614-651 (read-only audit 2026-09-30)",
                             "state_gripper_mapping": "shared motor-angle affine transmission for target and feedback; not physically calibrated",
                             "gripper_transmissions": {side: mapping.describe() for side, mapping in self.grippers.items()},
                             "gripper_force_limit_frame": "actuated finger joint; hardware motor limit remains unverified",
                             "gripper_force_limit_unit": self.profile['gripper_force_limit_unit'],
                             "images": "RGB HWC uint8, head/left/right; no robot masking",
                             "step": "obs(t), action(t), obs(t+dt); policy defaults to 30 Hz; reference replay/calibration use audited logger intervals; actual per-run clock is declared in details",
                             "contact_sampling": "1 ms solved physics substeps; continuous dwell at this resolution",
                             "inference_timing": "per-run details.timing declares paused or measured-delay discrete-event integration; not a validated wall-clock real-time controller",
                             "capabilities": ["atomic_vla_requires_matching_checkpoint", "tamp_oracle_pick_handover_place"],
                             "hardware_access": False},
                "model_evidence":{**self._model_evidence,'runtime_parameters':self._parameter_audit.summary()},
                "source_status": self.source_status(), "hashes": dict(self._runtime_hashes)}

    def reference_action(self, episode, frame):
        e = self.reference[str(episode)]
        return np.array([v for s in SIDES for v in
                         [*e[f"{s}_action_joints"][frame], e[f"{s}_action_gripper"][frame]]], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.episode = int(options.get("episode", 0))
        e = self.reference[str(self.episode)]
        mujoco.mj_resetData(self.model, self.data)
        self._clock_mode = None
        self._periodic_offset = 0.
        self.placement_target = None
        if self.source_controller:self.source_controller.reset()
        self.controller_feedback={}
        for side in SIDES:self.model.actuator_biasprm[self.aids[side],0]=0.
        for side in SIDES:
            self.data.qpos[self.qids[side]] = e[f"{side}_joints"][0]
            value = e[f"{side}_gripper"][0]
            for j in self.fingers[side]:
                self.data.qpos[j.qposadr[0]] = np.clip(self.grippers[side].to_joint(value), *j.range)
        self.tick = 0
        self._done = False
        self.task_termination='legacy_benchmark'
        self._task_start = 0.
        self._task_budget=float(self.profile['episode_seconds'])
        self.last_action = self.reference_action(options.get("episode", 0), 0)
        self._set_command(self.last_action,update_controller=False)
        mujoco.mj_forward(self.model, self.data)
        if self.source_controller and self.source_initial_integral=='static_gravity_proxy':
            for side in SIDES:
                self.source_controller.accumulated[side]=np.clip(self.data.qfrc_bias[self.dids[side]],-10.,10.).copy()
        self.events = [{"stage": 0, "name": STAGES[0], "time": 0., "evidence": "environment reset"}]
        self._dwell = 0.
        self._bad_dwell = 0.
        self._right_lifted = self._dual_held = self._left_held = False
        self._max_penetration = 0.
        self._interval_penetration = 0.
        self.contacts.reset()
        self.evaluation = self._evaluate(0.)
        from .grasp_feedback import GraspFeedback
        self.grasp_feedback=GraspFeedback()
        self.grasp_feedback.read(self)
        return self.observe(), self.info()

    def task_window(self):
        requested=getattr(self,'_task_budget',float(self.profile['episode_seconds']))
        budget=float(self.profile['episode_seconds']) if self.task_termination=='legacy_benchmark' else requested
        start=getattr(self,'_task_start',0.)
        return {'start_sim_seconds':float(start),'requested_seconds':requested,'active_seconds':budget,
                'remaining_seconds':max(0.,float(start+budget-self.data.time)),
                'legacy_benchmark_seconds':float(self.profile['episode_seconds'])}

    def set_task_budget(self,seconds):
        if type(seconds) is not int or seconds not in {40,60,90,120}:
            raise ValueError('仿真任务预算只能为 40、60、90 或 120 秒')
        if self._done or self.evaluation['invalid'] or abs(self.data.time-self._task_start)>self.model.opt.timestep/2:
            raise ValueError('时间预算只能在新任务开始、任何物理动作之前锁定；不能在失败后加时')
        self._task_budget=float(seconds);self.task_termination='skill'
        return self.task_window()

    def prepare_task(self):
        """New simulation window; preserve physical state and actuator targets."""
        if self.evaluation['invalid'] or self._clock_mode not in (None,'periodic'):
            raise ValueError('当前状态无效或为记录时钟；必须明确 reset 后才能开始新任务')
        self._done=False
        self.task_termination='legacy_benchmark'
        self.placement_target = None
        self.contacts.placement_seconds = 0.
        self._task_start=float(self.data.time)
        self._task_budget=float(self.profile['episode_seconds'])
        self.events=[{'stage':0,'name':'当前场景','time':float(self.data.time),'evidence':'new task; physical state preserved'}]
        self._right_lifted=self._dual_held=self._left_held=False
        self._dwell=self._bad_dwell=self._max_penetration=self._interval_penetration=0.
        # Contact dwell is physical history and remains valid in this unchanged state.
        self.evaluation=self._evaluate(0.)
        return self.info()

    def state(self):
        values = []
        for side in SIDES:
            encoder = float(self.grippers[side].to_motor(self.data.qpos[self.fingers[side][0].qposadr[0]]))
            values.extend([*self.data.qpos[self.qids[side]], encoder])
        return np.array(values, np.float32)

    def observe(self):
        obs = {"state": self.state()}
        if self.images:
            obs["images"] = {name: self.render(name) for name in self.profile["camera_map"]}
        return obs

    def validate_action(self, action, *, check_delta=True, native_fingers=False):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (16,) or not np.isfinite(a).all():
            raise ValueError("action must be 16 finite numbers")
        lower,upper=self.action_space.low.copy(),self.action_space.high.copy()
        if native_fingers:
            for index,side in zip((7,15),SIDES):
                motor=self.grippers[side].to_motor(self.fingers[side][0].range)
                lower[index],upper[index]=min(motor),max(motor)
        if np.any(a < lower - 1e-5) or np.any(a > upper + 1e-5):
            raise ValueError("action exceeds joint/gripper command limits; command rejected, not clipped")
        if check_delta:
            state = self.state()
            for start in (0, 8):
                if np.linalg.norm(a[start:start+7] - state[start:start+7]) > self.profile["max_action_delta_rad"] + 1e-5:
                    raise ValueError("applied command exceeds per-arm 7-D L2 measured-state jump limit")
        return a

    def prepare_policy_action(self, action):
        """Match the reviewed deployment's order: L2 cap, then joint bounds.

        The gripper's device-side saturation remains unverified. Retain the
        explicit command-range guard there instead of inventing a motor limit.
        """
        requested = np.asarray(action, np.float32)
        if requested.shape != (16,) or not np.isfinite(requested).all():
            raise ValueError("policy action must contain 16 finite numbers")
        if np.any(requested[[7,15]] < self.action_space.low[[7,15]] - 1e-5) or np.any(requested[[7,15]] > self.action_space.high[[7,15]] + 1e-5):
            raise ValueError("gripper output outside the verified command convention")
        applied = requested.copy()
        state = self.state()
        report = {}
        for side, start in zip(SIDES, (0, 8)):
            sl = slice(start, start + 7)
            delta = requested[sl] - state[sl]
            distance = float(np.linalg.norm(delta))
            cap = self.profile["max_action_delta_rad"]
            limited = state[sl] + delta * (cap / distance) if distance > cap else requested[sl]
            applied[sl] = np.clip(limited, self.action_space.low[sl], self.action_space.high[sl])
            report[side] = {"requested_distance_l2_rad": distance, "step_limited": distance > cap,
                            "joint_limit_clipped": bool(np.any(applied[sl] != limited))}
        return applied, report

    def _set_command(self, action, *, update_controller=True):
        for i, side in enumerate(SIDES):
            self.data.ctrl[self.aids[side]] = action[i * 8:i * 8 + 7]
            self.data.ctrl[self.model.actuator(f"{side}_finger1_ctrl").id] = self.grippers[side].to_joint(action[i * 8 + 7])
            if self.source_controller and update_controller:
                feedback=self.source_controller.update(side,action[i*8:i*8+7],self.data.qpos[self.qids[side]])
                kd=-self.model.actuator_biasprm[self.aids[side],2]
                # Native affine actuator b0: the total PD+dq+tau force is
                # limited by forcerange. Adding qfrc_applied would bypass it.
                self.model.actuator_biasprm[self.aids[side],0]=kd*feedback['velocity_input']+feedback['feedforward_input']
                self.controller_feedback[side]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in feedback.items()}

    def step(self, action):
        return self._advance(action, (self.tick + 1) / self.profile["control_hz"]+getattr(self,'_periodic_offset',0.), "periodic")

    def step_tamp(self, action):
        """Oracle planning may use the official full finger slide range.

        Arm guards and physics are identical. This explicit simulation-only
        extension does not enlarge the recorded/VLA command contract.
        """
        return self._advance(action,(self.tick+1)/self.profile['control_hz']+getattr(self,'_periodic_offset',0.),'periodic',native_fingers=True)

    def hold_tamp(self, action, seconds):
        """Planner-only wait on the existing 1ms solver grid, same targets.

        Offset subsequent 30Hz command times by this extra wait. The recorded
        clock, ordinary command intervals, controller and evaluator are intact.
        """
        dt=float(self.model.opt.timestep)
        if not np.array_equal(np.asarray(action),self.last_action):
            raise ValueError('planner wait must preserve the exact previous target')
        if not np.isfinite(seconds) or seconds<dt or seconds>1/self.profile['control_hz']+dt:
            raise ValueError('planner hold must be 1ms to one control period')
        if abs(seconds/dt-round(seconds/dt))>1e-6:
            raise ValueError('planner wait must use the existing physics grid')
        result=self._advance(action,float(self.data.time)+seconds,'periodic',native_fingers=True)
        self._periodic_offset=getattr(self,'_periodic_offset',0.)+seconds-1/self.profile['control_hz']
        return result

    def step_until(self, action, target_time):
        """Replay/calibration: hold a command until an audited logger timestamp.

        The native 1 ms solver and contact sampling stay unchanged. Times are
        rounded to the nearest physics tick, with at most 0.5 ms phase error.
        A reset is required to switch between periodic and recorded clocks.
        """
        return self._advance(action, float(target_time), "recorded")

    def _advance(self, action, target, mode, *, native_fingers=False):
        if self._done:
            raise RuntimeError("episode finished; reset before stepping")
        if self._clock_mode not in (None, mode):
            raise ValueError("reset before switching the control clock")
        if not np.isfinite(target) or not self.model.opt.timestep / 2 < target - self.data.time <= .25:
            raise ValueError("target time must advance by at most 250 ms")
        a = self.validate_action(action,native_fingers=native_fingers)
        self._clock_mode = mode
        self._set_command(a)
        before = float(self.data.time)
        self.contacts.begin(before)
        self._interval_penetration = 0.
        self.tick += 1
        while self.data.time + self.model.opt.timestep / 2 < target:
            sample_time = float(self.data.time)
            bid = self.model.joint("bottle_free").dofadr[0]
            bottle_velocity = self.data.qvel[bid:bid+6].copy()
            if self.profile["gravity_compensation"] == "bias":
                for side in SIDES:
                    self.data.qfrc_applied[self.dids[side]] = self.data.qfrc_bias[self.dids[side]]
            mujoco.mj_step(self.model, self.data)
            # mj_step solves contacts at the beginning of this substep, then
            # integrates qpos/qvel. Derived positions still match sample_time.
            contact = self.contacts.read(self.data)
            self.contacts.sample(contact, float(self.model.opt.timestep), sample_time,
                                 stable=self._stable_placement(contact, bottle_velocity),
                                 placement_stable=self._placement_condition(contact, bottle_velocity))
        # mj_step's position-derived fields otherwise lag the integrated qpos.
        mujoco.mj_forward(self.model, self.data)
        contact = self.contacts.read(self.data)
        bid = self.model.joint("bottle_free").dofadr[0]
        self.contacts.sample(contact, 0., float(self.data.time),
                             stable=self._stable_placement(contact, self.data.qvel[bid:bid+6]),
                             placement_stable=self._placement_condition(contact, self.data.qvel[bid:bid+6]))
        self._interval_penetration = self.contacts.penetration
        self.last_action = a.copy()
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("non-finite physics state")
        self.evaluation = self._evaluate(float(self.data.time) - before)
        self.grasp_feedback.read(self)
        terminated = self.evaluation["invalid"] or (self.task_termination=='legacy_benchmark' and self.evaluation["success"])
        budget=self.profile['episode_seconds'] if self.task_termination=='legacy_benchmark' else self._task_budget
        truncated = self.data.time + self.model.opt.timestep / 2 >= self._task_start+budget
        self._done = terminated or truncated
        return self.observe(), float(self.evaluation["success"]), terminated, truncated, self.info()

    def _contacts(self):
        c = self.contacts.read(self.data)
        return c["held"], c["touching"], c["supported"], c["penetration_m"], c["peak_normal_force_N"], c["robot_contact_count"]

    def _bottle_bounds(self):
        # Exact world AABB of each cylindrical part, including rotation.
        lows, highs = [], []
        for gid in self.bottle_geoms:
            axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
            r, half = self.model.geom_size[gid, :2]
            ext = r * np.sqrt(np.maximum(0, 1 - axis ** 2)) + half * abs(axis)
            lows.append(self.data.geom_xpos[gid] - ext)
            highs.append(self.data.geom_xpos[gid] + ext)
        return np.min(lows, axis=0), np.max(highs, axis=0)

    def _stable_placement(self, contact, velocity):
        if not contact["supported"] or any(contact["touching"].values()):
            return False
        if np.linalg.norm(velocity[:3]) >= .02 or np.linalg.norm(velocity[3:]) >= .1:
            return False
        lo, hi = self._bottle_bounds()
        box = self.data.xpos[self.box_id]
        return bool(np.all(lo[:2] > box[:2] - [.100, .084]) and np.all(hi[:2] < box[:2] + [.100, .084]))

    def placement_surface(self, surface):
        """Declared horizontal support geometry; this is explicitly oracle data."""
        if surface == 'table':
            center=self.data.geom_xpos[self.table_id]
            half=self.model.geom_size[self.table_id]
            return {'xy_min':center[:2]-half[:2], 'xy_max':center[:2]+half[:2],
                    'top_z':float(center[2]+half[2])}
        if surface == 'basket':
            center=self.data.xpos[self.box_id]
            return {'xy_min':center[:2]-[.100,.084], 'xy_max':center[:2]+[.100,.084],
                    'top_z':float(self.data.geom_xpos[self.box_floor_id,2]+self.model.geom_size[self.box_floor_id,2])}
        raise ValueError('unknown support surface')

    def set_placement_target(self, surface, xy, tolerance_m=None):
        xy=np.asarray(xy,dtype=float)
        area=self.placement_surface(surface)
        # Entire upright bottle footprint plus 10 mm clearance must fit.
        margin=max(self.model.geom_size[self.bottle_geoms,0])+.010
        if xy.shape!=(2,) or not np.isfinite(xy).all() or np.any(xy<area['xy_min']+margin) or np.any(xy>area['xy_max']-margin):
            raise ValueError('placement target exceeds the support surface footprint')
        from .placement import placement_tolerance
        tolerance=placement_tolerance(surface,xy) if tolerance_m is None else tolerance_m
        if type(tolerance) not in (int,float) or not np.isfinite(tolerance) or not .0005<=tolerance<=placement_tolerance(surface,xy):
            raise ValueError('invalid object placement tolerance')
        self.placement_target={'surface':surface,'xy_m':xy.tolist(),'tolerance_m':float(tolerance)}
        self.contacts.placement_seconds=0.

    def _placement_condition(self, contact, velocity):
        target=self.placement_target
        if not target or any(contact['touching'].values()):return False
        supported=contact['table_supported'] if target['surface']=='table' else contact['supported']
        if not supported or np.linalg.norm(velocity[:3])>=.02 or np.linalg.norm(velocity[3:])>=.1:return False
        if target['surface']=='table' and self.data.xmat[self.bottle_id].reshape(3,3)[2,2]<.98:return False
        lo,hi=self._bottle_bounds();area=self.placement_surface(target['surface'])
        return bool(np.all(lo[:2]>area['xy_min']) and np.all(hi[:2]<area['xy_max'])
                    and np.linalg.norm(self.data.xpos[self.bottle_id,:2]-target['xy_m'])<=target['tolerance_m'])

    def _event(self, stage, evidence):
        if not any(e["stage"] == stage for e in self.events):
            self.events.append({"stage": stage, "name": STAGES[stage], "time": float(self.data.time), "evidence": evidence})

    def _evaluate(self, dt):
        contact_held, touching, supported, penetration, force, robot_contacts = self._contacts()
        held = {s: self.contacts.hold_seconds[s] >= .1 and contact_held[s] for s in SIDES}
        penetration = max(penetration, self._interval_penetration)
        lo, hi = self._bottle_bounds()
        box = self.data.xpos[self.box_id]
        inside = bool(np.all(lo[:2] > box[:2] - [.100, .084]) and np.all(hi[:2] < box[:2] + [.100, .084]))
        bid = self.model.joint("bottle_free").dofadr[0]
        linear_speed = float(np.linalg.norm(self.data.qvel[bid:bid + 3]))
        angular_speed = float(np.linalg.norm(self.data.qvel[bid + 3:bid + 6]))
        no_hands = not any(touching.values())
        if held["right"]:
            self._event(1, "two right finger bodies contact bottle, normal force > 0.05 N")
        if held["right"] and lo[2] > 1.035:
            self._right_lifted = True
            self._event(2, "two-finger right contact and bottle bottom > table + 30 mm")
        if self._right_lifted and held["right"] and held["left"]:
            self._dual_held = True
            self._event(3, "both grippers have two-finger contact after right lift")
        if self._dual_held and held["left"] and not touching["right"]:
            self._left_held = True
            self._event(4, "left contact persists after right releases")
        if self._left_held and inside and no_hands:
            self._event(5, "whole bottle XY footprint inside basket; both grippers released")
        stable = inside and supported and no_hands and linear_speed < .02 and angular_speed < .1
        self._dwell = self.contacts.stable_seconds if stable else 0.
        self._max_penetration = max(self._max_penetration, penetration)
        self._bad_dwell = self.contacts.penetration_seconds
        invalid = bool(lo[2] < .80 or self.contacts.deep_penetration)
        reasons = (["bottle fell below the work surface"] if lo[2] < .80 else []) + (["penetration exceeded 10 mm continuously for 100 ms of physics samples"] if self.contacts.deep_penetration else [])
        success = bool(self._left_held and self._dwell >= 1. and not invalid)
        if success:
            self._event(6, "handover order + whole bottle in basket + floor support + released + stable 1 s")
        return {"success": success, "invalid": invalid, "held": held,
                "touching": touching, "failure_reasons": reasons,
                "inside_basket": inside, "supported_by_basket": supported,
                "stable_seconds": round(self._dwell, 4), "linear_speed_m_s": linear_speed,
                "penetration_m": penetration, "max_penetration_m": self._max_penetration,
                "peak_normal_force_N": max(force, self.contacts.peak_force), "robot_contact_count": robot_contacts,
                "contact_window": self.contacts.summary(),
                "placement": {**(self.placement_target or {}), 'stable_seconds':round(self.contacts.placement_seconds,4),
                    'confirmed':bool(self.placement_target and self.contacts.placement_seconds>=1. and not invalid)},
                "evidence_source": "privileged simulator evaluator; not a camera measurement",
                "transfer_ready": False}

    def info(self):
        return {"tick": self.tick, "sim_time": float(self.data.time),
                "evaluation": dict(self.evaluation), "events": list(self.events),
                'controller_feedback':self.controller_feedback,'task_termination':self.task_termination,
                'grasp_feedback':self.grasp_feedback.summary()}

    def diagnostics(self):
        return {"privileged": True, "tcp_base": {s: (self.data.site(f"{s}_ee_control_point").xpos - self.base_to_world).tolist() for s in SIDES},
                "bottle_world": self.data.xpos[self.bottle_id].tolist(),
                'manipulation_scene':{'object_id':'bottle', 'position_m':self.data.xpos[self.bottle_id].tolist(),
                    'radius_m':float(max(self.model.geom_size[self.bottle_geoms,0])),
                    'upright':bool(self.data.xmat[self.bottle_id].reshape(3,3)[2,2]>.98),
                    'held_by':[side for side,value in self.evaluation['held'].items() if value],
                    'grasp_quality':self.grasp_feedback.summary(),
                    'surfaces':{surface:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.placement_surface(surface).items()}
                                for surface in ('table','basket')}}, **self.info()}

    def render(self, camera="overview", *, view=None):
        size = tuple(self.profile["camera_sizes"].get('overview' if camera=='operator' else camera, ()))
        if len(size) != 2:
            raise ValueError("unknown camera")
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, height=size[1], width=size[0])
        renderer = self._renderers[size]
        if camera in ("overview",'operator'):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [.39, 0, 1.18]
            cam.distance, cam.azimuth, cam.elevation = 1.48, 0, -33
            if camera=='operator':
                from .view import FRONT,validate_view
                config=validate_view(view or FRONT)
                cam.lookat[:]=config['lookat']
                cam.distance,cam.azimuth,cam.elevation=config['distance'],config['azimuth'],config['elevation']
        else:
            if camera not in self.profile["camera_map"]:
                raise ValueError("unknown camera")
            cam = self.profile["camera_map"][camera]
        renderer.update_scene(self.data, camera=cam, scene_option=self._option)
        return renderer.render().copy()

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
