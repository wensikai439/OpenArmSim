"""PhyAgentOS-only wrapper: measured telemetry and cooperative cancellation.

The task/controller and model come directly from the independent shared asset.
The RPC adapter is contained in this PhyAgentOS bundle. Cancellation is an episode-ending interruption at a native control tick.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import mujoco
import numpy as np

# Optional project collection scene; default profiles remain unchanged.
_selection_path = os.environ.get("PAOS_SCENE_SELECTION_FILE")
if _selection_path and Path(_selection_path).is_file():
    _selection = json.loads(Path(_selection_path).read_text())
    _scene_bytes = Path(_selection["config_path"]).read_bytes()
    if hashlib.sha256(_scene_bytes).hexdigest() != _selection["sha256"]:
        raise RuntimeError("Selected scene hash mismatch before native setup")
    _scene = json.loads(_scene_bytes)
    os.environ["PAOS_CONTROL_HZ"] = str(_scene["world"]["control_hz"])
    _timing = _scene.get("paos_motion_timing_profile", "legacy")
    if _timing != "legacy":
        if _timing != "feedback_joint_v1":
            raise RuntimeError("Unknown project collection timing")
        os.environ["PAOS_MOTION_TIMING_PROFILE"] = _timing
        os.environ["PAOS_FROZEN_SIM_SOURCE_LOCK"] = _scene["paos_asset_source_lock"]
_source_lock_path = os.environ.get("PAOS_FROZEN_SIM_SOURCE_LOCK")
if _source_lock_path:
    _lock = json.loads(Path(_source_lock_path).read_text())
    if _lock["asset_root"] != str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(os.environ["PAOS_ROOT"]) / "external/piper_robocasa")))):
        raise RuntimeError("Unexpected collection asset root")
    for _entry in _lock["modules"].values():
        if hashlib.sha256(Path(_entry["path"]).read_bytes()).hexdigest() != _entry["sha256"]:
            raise RuntimeError("Collection asset source hash mismatch")
    import piper_robocasa
    piper_robocasa.__path__.insert(0,str(Path(_source_lock_path).parent/"piper_robocasa"))

import sim_rpc as env_server
from piper_robocasa import air_handoff_task
from piper_robocasa.pose_preflight import check_pose_path

_cancel_signal = False
_active_execution = None
_cancel_request = Path(os.environ["PAOS_CANCEL_REQUEST"])


def _request_cancel(_signum, _frame):
    global _cancel_signal
    _cancel_signal = True


class CooperativeCancellation(RuntimeError):
    """Handled by the native task's failure path, preserving its real evidence."""


class CooperativeTask(air_handoff_task.PiperAirHandoffTask):
    BINARY_OPEN_COMMAND = -1.0
    BINARY_CLOSE_COMMAND = 0.998
    BINARY_CLOSED_CLEARANCE_M = 0.0001

    def __init__(self, *args, **kwargs):
        self._paos_rate_scale = int(os.environ.get("PAOS_CONTROL_HZ", "20")) / 20
        self._paos_gripper_mode = os.environ.get("PAOS_GRIPPER_MODE", "legacy")
        if self._paos_gripper_mode not in {"legacy", "binary_endpoints"}:
            raise ValueError("Unknown PhyAgentOS gripper mode")
        selected = None
        selection_file = os.environ.get("PAOS_SCENE_SELECTION_FILE")
        if not kwargs.get("config_path") and selection_file and Path(selection_file).exists():
            selected = json.loads(Path(selection_file).read_text())
            kwargs["config_path"] = selected["config_path"]
        scene_path = Path(kwargs.get("config_path") or
            str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(os.environ["PAOS_ROOT"]) / "external/piper_robocasa"))) / 'scenes/piper_air_handoff_v1/scene_spec.json')).resolve()
        scene_bytes = scene_path.read_bytes()
        scene_sha = hashlib.sha256(scene_bytes).hexdigest()
        if selected is not None and selected["sha256"] != scene_sha:
            raise RuntimeError("Selected initial scene SHA mismatch")
        scene_data = json.loads(scene_bytes)
        self._paos_binary_contact_wait = scene_data.get("paos_binary_contact_wait", "legacy")
        if self._paos_binary_contact_wait not in {"legacy", "feedback_v1"}:
            raise ValueError("Unknown project binary contact wait")
        scenario = scene_data.get("paos_scenario", {})
        self._scene_initialization = {
            "config_path": str(scene_path), "source_sha256": scene_sha,
            "scenario_id": scenario.get("scenario_id", "original-baseline"),
            "split": scenario.get("split"), "configuration_seed": scenario.get("configuration_seed"),
            "semantics": "scene constructed once before physics; no live state relocation"}
        super().__init__(*args, **kwargs)
        if hashlib.sha256(scene_path.read_bytes()).hexdigest() != scene_sha:
            raise RuntimeError("Initial scene changed while being constructed")
        (self.output / "scene_initialization.json").write_text(
            json.dumps(self._scene_initialization, indent=2) + "\n")
        if self._paos_gripper_mode == "binary_endpoints":
            # Factory commands are OPEN; no physical state or qpos is assigned.
            self.ws.binary_task_modes = {"left": 1, "right": 1}
            for side in ("left", "right"):
                index = self.ws.robot._action_split_indexes[side + "_gripper"][0]
                self.ws.action[index] = self.BINARY_OPEN_COMMAND
            path = self.output / "binary_gripper_contract.json"
            path.write_text(json.dumps({
                "mode": "binary_endpoints",
                "task_mode_values": {"CLOSE": 0, "OPEN": 1},
                "open_width_target_m": 0.1,
                "closed_width_target_m": self.BINARY_CLOSED_CLEARANCE_M,
                "open_native_command": self.BINARY_OPEN_COMMAND,
                "close_native_command": self.BINARY_CLOSE_COMMAND,
                "actuator_force_limits": "Shared model unchanged: +/-10 N per finger",
                "contact_semantics": "Binary CLOSE intent: advance toward 0.1mm by <=2mm per control tick; hold first measured bilateral bottle-contact target",
                "close_width_step_m": 0.002 / self._paos_rate_scale,
                "contact_hold_grid_m": 0.002,
                "assistance": "Shared bilateral-contact-triggered assistance remains explicit",
                "safety": "Shared collision and 10mm penetration guards unchanged",
            }, indent=2) + "\n")
        if os.environ.get("PAOS_MOTION_TIMING_PROFILE") == "feedback_joint_v1":
            from joint_collection_timing import metadata
            actual_gains = {side:self.ws.robot.part_controllers[side].kp.tolist() for side in ("left","right")}
            self._paos_motion_timing = {**metadata(),"controller_kp":actual_gains,
                                       "asset_source_lock":_source_lock_path,
                                       "binary_contact_wait":self._paos_binary_contact_wait}
            (self.output/"project_motion_timing.json").write_text(json.dumps(self._paos_motion_timing,indent=2)+"\n")
            contract_path=self.output/"joint_command_contract.json"
            contract=json.loads(contract_path.read_text())
            contract["kp"]=actual_gains
            contract["motion_timing"]=self._paos_motion_timing
            contract_path.write_text(json.dumps(contract,indent=2)+"\n")

    def _wait_binary_contact_feedback(self, side, attached=False):
        if self._paos_binary_contact_wait != "feedback_v1":
            self.tick(int(np.ceil(8 * self._paos_rate_scale)))
            return 8
        # Consume actual MuJoCo ticks; preserve the original maximum wait when
        # physical two-finger contact and measured velocity are not yet stable.
        ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                          for name in self.ws.robot.gripper[side].joints])
        if len(ids) != 2 or np.any(ids < 0):
            raise RuntimeError("Missing measured two-finger joints")
        previous = None
        stable = 0
        for count in range(1, int(np.ceil(8 * self._paos_rate_scale)) + 1):
            self.tick()
            positions = self.data.qpos[self.model.jnt_qposadr[ids]].copy()
            speeds = self.data.qvel[self.model.jnt_dofadr[ids]]
            contacts = self._finger_contacts_now(side)
            ready = (len(contacts) >= 2 and np.max(np.abs(speeds)) <= .01
                     and previous is not None and np.max(np.abs(positions-previous)) <= .0001
                     and (not attached or self._held_sides()[side]))
            stable = stable + 1 if ready else 0
            previous = positions
            if stable >= int(np.ceil(3 * self._paos_rate_scale)):
                break
        self.events.append({"kind": "binary_contact_settle_feedback",
                            "side": side, "attached": attached, "actual_ticks": count,
                            "stable_contact_samples": stable, "contact_geoms": contacts,
                            "control_step": int(self.ws.step_count), "time": float(self.data.time)})
        return count

    def _operate_gripper(self, side, operation):
        if self._paos_gripper_mode != "binary_endpoints":
            return super()._operate_gripper(side, operation)
        if operation == "open":
            self.ws.binary_task_modes[side] = 1
            result = super()._operate_gripper(side, operation)
        elif operation == "close":
            self.ws.binary_task_modes[side] = 0
            other = "right" if side == "left" else "left"
            index = self.ws.robot._action_split_indexes[side + "_gripper"][0]
            width = 0.05 * (1.0 - float(self.ws.action[index]))
            contacts = self._finger_contacts_now(side)
            # A binary CLOSE is a task command. Servo targets approach the
            # endpoint and stop at actual bilateral contact with a held object.
            # The target/intent/feedback are logged separately, never relabelled.
            for _ in range(int(np.ceil(50 * self._paos_rate_scale)) + 1):
                if contacts or width <= self.BINARY_CLOSED_CLEARANCE_M + 1e-12:
                    break
                width = max(self.BINARY_CLOSED_CLEARANCE_M, width - 0.002 / self._paos_rate_scale)
                self.gripper(side, 1.0 - width / 0.05, 1)
                contacts = self._finger_contacts_now(side)
            if contacts and self._paos_rate_scale != 1:
                # Preserve the old 2mm CLOSE target grid at the faster recording
                # rate. A sub-grid first contact otherwise leaves 1.33mm less
                # preload than the previously validated 20Hz grasp.
                grid_width = max(self.BINARY_CLOSED_CLEARANCE_M,
                                 np.floor((width + 1e-10) / 0.002) * 0.002)
                if width - grid_width > 1e-10:
                    self.gripper(side, 1.0 - grid_width / 0.05, 1)
                    width = grid_width
            self._wait_binary_contact_feedback(side)
            contacts = self._finger_contacts_now(side)
            hold_command = float(self.ws.action[index])
            self.events.append({
                "kind": "binary_close_contact_stop" if contacts else "binary_close_empty_endpoint",
                "side": side, "task_mode": 0,
                "requested_closed_width_m": self.BINARY_CLOSED_CLEARANCE_M,
                "actual_servo_width_target_m": 0.05 * (1.0 - hold_command),
                "measured_opening_m": self.observe()["gripper_feedback"][side]["joint_opening_m"],
                "native_command": hold_command, "contact_geoms": contacts,
                "control_step": int(self.ws.step_count), "time": float(self.data.time),
            })
            evidence = None
            held = self._held_sides()
            if contacts and not held[side]:
                if held[other]:
                    self.receiver_ready = side
                else:
                    evidence = self._attach_measured(side)
                    self._wait_binary_contact_feedback(side, attached=True)
            index = self.ws.robot._action_split_indexes[side + "_gripper"][0]
            consumed = float(self.ws.action[index])
            result = {
                "kind": "gripper_contact_control",
                "open_or_close": operation,
                "commanded_value": consumed,
                "contact_geoms": self._finger_contacts_now(side),
                "assist_active": self._held_sides(),
                "receiver_ready": self.receiver_ready,
                "evidence": evidence,
                "verified": abs(consumed - hold_command) <= 1e-9 and (bool(contacts) or abs(width - self.BINARY_CLOSED_CLEARANCE_M) <= 1e-9),
            }
        else:
            raise ValueError("Binary gripper expects open or close")
        result.update(gripper_command_mode="binary_endpoints",
                      task_mode=1 if operation == "open" else 0,
                      requested_endpoint_width_m=0.1 if operation == "open" else self.BINARY_CLOSED_CLEARANCE_M,
                      servo_width_target_m=0.05 * (1.0 - result["commanded_value"]))
        return result

    def tick(self, count=1):
        global _cancel_signal
        for _ in range(count):
            if _cancel_signal and _active_execution is not None:
                _cancel_signal = False
                request = json.loads(_cancel_request.read_text())
                if request.get("execution_id") == _active_execution:
                    self._paos_cancelled_at = int(self.ws.step_count)
                    raise CooperativeCancellation(
                        "PAOS_CANCELLED_AT_CONTROL_STEP:" + str(self._paos_cancelled_at)
                    )
            super().tick()

    def observe(self):
        state = super().observe()
        state["scene_initialization"] = dict(self._scene_initialization)
        state["control_hz"] = int(round(20 * self._paos_rate_scale))
        state["physics_dt_s"] = float(self.model.opt.timestep)
        if hasattr(self,"_paos_motion_timing"):
            state["project_motion_timing"] = dict(self._paos_motion_timing)
        state["joint_positions_rad"] = {
            side: self.data.qpos[self.ws.qindexes[side]].astype(float).tolist()
            for side in ("left", "right")
        }
        state["joint_names"] = {
            side: [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(i))
                   for i in self.ws.joint_ids[side]]
            for side in ("left", "right")
        }
        fingers = {}
        for side in ("left", "right"):
            values = {}
            for finger in ("l", "r"):
                name = f"gripper0_{side}_{finger}_finger_joint"
                index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if index < 0 or self.model.jnt_type[index] != mujoco.mjtJoint.mjJNT_SLIDE:
                    raise RuntimeError("Expected native slide joint: " + name)
                values[name] = float(self.data.qpos[self.model.jnt_qposadr[index]])
            fingers[side] = {"finger_joint_qpos_m": values,
                             "joint_opening_m": sum(abs(v) for v in values.values())}
        state["gripper_feedback"] = fingers
        state["gripper_command_mode"] = getattr(self, "_paos_gripper_mode", "legacy")
        if state["gripper_command_mode"] == "binary_endpoints":
            state["binary_gripper_commands"] = {
                side: {
                    "task_mode": int(self.ws.binary_task_modes[side]),
                    "native_command": float(self.ws.action[self.ws.robot._action_split_indexes[side + "_gripper"][0]]),
                    "servo_width_target_m": 0.05 * (1.0 - float(self.ws.action[self.ws.robot._action_split_indexes[side + "_gripper"][0]])),
                    "measured_opening_m": fingers[side]["joint_opening_m"],
                } for side in ("left", "right")
            }
            state["binary_gripper_contract_path"] = str(self.output / "binary_gripper_contract.json")
        state["telemetry_source"] = "native MuJoCo data.qpos; arm ws.qindexes; named slide finger joints"
        state["cancelled_at_control_step"] = getattr(self, "_paos_cancelled_at", None)
        state["reset_required"] = bool(self.failed)
        joint_mode = hasattr(self.ws, "joint_command")
        state["controller_mode"] = "JOINT_POSITION:absolute" if joint_mode else "OSC_POSE:absolute"
        if joint_mode:
            state["joint_command_targets_rad"] = {
                side: self.ws.joint_command[slice(*self.ws.robot._action_split_indexes[side])].tolist()
                for side in ("left", "right")
            }
            state["controller_consumed_joint_targets_rad"] = {
                side: self.ws.robot.part_controllers[side].goal_qpos.tolist()
                for side in ("left", "right")
            }
            state["wrist_pitch_kinematics"] = {
                side: {
                    "joint_name": state["joint_names"][side][4],
                    "axis_world": self.data.xaxis[self.ws.joint_ids[side][4]].tolist(),
                    "anchor_world_m": self.data.xanchor[self.ws.joint_ids[side][4]].tolist(),
                    "source": "actual MuJoCo xaxis/xanchor; read-only pose planning geometry",
                } for side in ("left", "right")
            }
            state["joint_command_trace_path"] = str(self.output / "joint_commands.jsonl")
            state["joint_command_contract_path"] = str(self.output / "joint_command_contract.json")
        return state


class TelemetryBackend(env_server.PiperSimReadOnlyBackend):
    def check_pose(self, side, target_world_m, target_rotation_rad=None):
        task = self._task
        payload = {"side": side, "target_world_m": target_world_m}
        if target_rotation_rad is not None:
            payload["target_rotation_rad"] = target_rotation_rad
        task._validate_composable_action({"id": "move_arm", "payload": payload})
        before_qpos = task.data.qpos.copy()
        before_qvel = task.data.qvel.copy()
        before_time = float(task.data.time)
        before_steps = int(task.ws.step_count)
        state = task.observe()
        verification = check_pose_path(task.ws, side, target_world_m, target_rotation_rad)
        unchanged = (np.array_equal(before_qpos, task.data.qpos)
                     and np.array_equal(before_qvel, task.data.qvel)
                     and before_time == float(task.data.time)
                     and before_steps == int(task.ws.step_count))
        if not unchanged:
            raise RuntimeError("Native pose Query changed live physical state")
        return {"verification": verification, "read_only_confirmed": unchanged,
                "episode_id": state["episode_id"], "physics_step": state["physics_step"],
                "sim_time_s": state["sim_time_s"], "side": side,
                "start_tcp_world_m": state["tcp_world_m"][side],
                "start_rotation_rad": state["tcp_rotation_rad"][side],
                "live_qpos_sha256": hashlib.sha256(before_qpos.tobytes()).hexdigest(),
                "live_qvel_sha256": hashlib.sha256(before_qvel.tobytes()).hexdigest(),
                "scope": "Native joint-limited IK/FK path check only; actual Action completion, contact and dynamics still require execution evidence"}

    def step_action(self, action, execution_id=None):
        global _active_execution
        if execution_id is not None and (not isinstance(execution_id, str) or not execution_id):
            raise ValueError("execution_id must be a nonempty string")
        _active_execution = execution_id
        try:
            result = super().step_action(action)
            cancelled = getattr(self._task, "_paos_cancelled_at", None)
            if cancelled is not None:
                result = dict(result)
                result.update(adapter_cancelled=True, reset_required=True,
                              adapter_execution_id=execution_id,
                              cancelled_at_control_step=cancelled)
            return result
        finally:
            _active_execution = None


class ExecutionFacade(env_server.PiperSimEnvFacade):
    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["env.check_pose"] = self._backend.check_pose
        self._readonly_methods.add("env.check_pose")

    def step_action(self, action, execution_id=None):
        return self._backend.step_action(action, execution_id=execution_id)


def main():
    if "--fake" in sys.argv:
        raise ValueError("The PhyAgentOS PiPER profile requires the real simulator")
    mode = os.environ.get("PAOS_EXPERT_CONTROLLER", "osc_pose")
    if mode not in {"osc_pose", "joint_position"}:
        raise ValueError("Unknown PhyAgentOS expert controller: " + mode)
    if mode == "joint_position":
        from joint_expert_adapter import install
        install()
    if os.environ.get("PAOS_MOTION_TIMING_PROFILE") == "feedback_joint_v1":
        if mode != "joint_position":raise RuntimeError("Collection timing requires joint controller")
        from joint_collection_timing import install as install_timing
        install_timing()
    signal.signal(signal.SIGUSR1, _request_cancel)
    # Changes are scoped to this owned child process; original source files are untouched.
    air_handoff_task.PiperAirHandoffTask = CooperativeTask
    env_server.PiperSimReadOnlyBackend = TelemetryBackend
    env_server.PiperSimEnvFacade = ExecutionFacade
    return env_server.main()


if __name__ == "__main__":
    raise SystemExit(main())
