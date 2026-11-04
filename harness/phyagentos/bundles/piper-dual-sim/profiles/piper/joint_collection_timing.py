"""Opt-in timing for project-owned joint expert processes; shared files stay unchanged."""
from pathlib import Path
import hashlib
import os
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from piper_robocasa import air_handoff_task
from piper_robocasa.pose_preflight import interpolate_rotation, motion_orientations

PROFILE = "feedback_joint_v1"
CONTROL_HZ = int(os.environ.get("PAOS_CONTROL_HZ", "20"))
RATE_SCALE = CONTROL_HZ / 20
def ticks_at_rate(original_20hz_ticks):
    return int(math.ceil(original_20hz_ticks * RATE_SCALE))
SPEC = {
    "profile": PROFILE, "control_hz": CONTROL_HZ,
    "translation_step_m": .02 / RATE_SCALE, "rotation_step_rad": .10 / RATE_SCALE,
    "max_interpolation_steps": ticks_at_rate(240), "max_move_settle_steps": ticks_at_rate(40),
    "finish_position_m": .004, "finish_rotation_rad": .03,
    "native_accept_position_m": .02, "native_accept_rotation_rad": .10,
    "stable_joint_speed_rad_s": .02, "stable_feedback_scope": "measured joint speed and unchanged native pose acceptance",
    "stable_feedback_ticks": ticks_at_rate(3), "controller_kp": "from existing selected scene; logged per arm after initialization",
    "physics_dt_s": 1 / 3000 if CONTROL_HZ == 30 else .0005,
    "early_finish": "actual 4mm/0.03rad tracking OR stable servo plateau; unchanged native pose acceptance and contact checks remain",
    "source_contact_approach": "original 4mm/0.02rad steps within measured near-bottle zone",
    "gripper": "original binary close; feedback open; settling only after actual release",
}

def _move_basic(self, side, target, rot=None):
    target = np.asarray(target, dtype=float)
    a, b = self.ws.robot._action_split_indexes[side]
    start = self.tcp(side)
    initial, final = motion_orientations(self.ws, side, rot)
    angle = (Rotation.from_rotvec(final) * Rotation.from_rotvec(initial).inv()).magnitude()
    near_source = (not any(self._held_sides().values())
                   and target[2] < self.bottle()[2]+float(self.config["objects"]["bottle"]["height_m"])+.02
                   and np.linalg.norm(target[:2]-self.bottle()[:2]) < .15)
    translation_step, rotation_step = (.004,.02) if near_source else (.02,.10)
    # Opt-in for the measured donor wrist peak: keep the successful full pose
    # path and original gains/guards; only use more ticks for its world-Z turn.
    yaw_step = self.config.get("paos_donor_yaw_rotation_step_rad")
    if yaw_step is not None:
        if not isinstance(yaw_step, (int, float)) or not .02 <= yaw_step <= .10:
            raise ValueError("donor yaw step must be between 0.02 and 0.10rad")
        delta_rot = (Rotation.from_rotvec(final) * Rotation.from_rotvec(initial).inv()).as_rotvec()
        held = self._held_sides()
        if (side == "right" and held["right"] and not held["left"] and angle > .5
                and np.linalg.norm(target-start) < .04 and abs(delta_rot[2]) > .95 * angle):
            rotation_step = float(yaw_step)
    count = max(1, int(np.ceil(max(np.linalg.norm(target-start)/translation_step, angle/rotation_step) * RATE_SCALE)))
    if count > ticks_at_rate(240):
        raise ValueError("basic movement exceeds bounded interpolation steps")
    for k in range(1, count+1):
        fraction = k/count
        self.ws.action[a:a+3] = start+(target-start)*fraction
        self.ws.action[a+3:b] = interpolate_rotation(initial, final, fraction)
        self.tick()
    desired = Rotation.from_rotvec(final).as_matrix()
    previous = None
    stable = 0
    settle = 0
    finish = "max_settle"
    for _ in range(ticks_at_rate(40)):
        error = float(np.linalg.norm(target-self.tcp(side)))
        rotation_error = self._site_rotation_error(side, desired)
        if error <= .004 and rotation_error <= .03:
            finish = "measured_4mm_0.03rad_tracking"
            break
        speeds = self.data.qvel[self.model.jnt_dofadr[self.ws.joint_ids[side]]]
        ready = (previous is not None and error <= .02 and rotation_error <= .10
                 and abs(error-previous[0]) <= .0001 / RATE_SCALE
                 and abs(rotation_error-previous[1]) <= .001 / RATE_SCALE
                 and np.max(np.abs(speeds)) <= .02
                )
        stable = stable+1 if ready else 0
        if stable >= ticks_at_rate(3):
            finish = "measured_servo_plateau_original_acceptance"
            break
        previous = (error, rotation_error)
        self.tick()
        settle += 1
    error = float(np.linalg.norm(target-self.tcp(side)))
    if error > .02:
        raise RuntimeError(f"{side} target not reached: {error:.4f} m")
    self.events.append({
        "kind": "basic_move", "side": side, "timing_profile": PROFILE,
        "interpolation_steps": count, "settle_steps": settle,
        "finish_feedback": finish, "target": target.tolist(), "error_m": error,
        "rotation_error_rad": self._site_rotation_error(side, desired),
        "time": float(self.data.time),
    })

def _open_feedback(self, side):
    joints = self.ws.robot.gripper[side].joints
    ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,n) for n in joints])
    if len(ids) != 2 or np.any(ids < 0):
        raise RuntimeError("missing two-finger feedback")
    target = self.model.jnt_range[ids][[0,1],[1,0]]
    positions = self.data.qpos[self.model.jnt_qposadr[ids]]
    speeds = self.data.qvel[self.model.jnt_dofadr[ids]]
    return bool(np.all(np.abs(positions-target) <= .001) and np.all(np.abs(speeds) <= .01))

def install():
    task = air_handoff_task.PiperAirHandoffTask
    if getattr(task, "_paos_timing_installed", False):
        return
    original_gripper = task._operate_gripper
    def _operate_gripper(self, side, operation):
        if operation != "open":
            return original_gripper(self, side, operation)
        other = "right" if side == "left" else "left"
        held = self._held_sides()
        evidence = None
        if held[side] and self.receiver_ready == other:
            evidence = self._attach_measured(other, donor=side)
            self.receiver_ready = None
        elif held[side]:
            before = self.data.qpos.copy()
            self.data.eq_active[self.eq[side]] = 0
            mujoco.mj_forward(self.model,self.data)
            if not np.array_equal(before,self.data.qpos):
                raise RuntimeError("release changed qpos")
            evidence = {"kind":"composable_release","side":side,"qpos_change":0}
            self.events.append(evidence)
        if self.receiver_ready == side:
            self.receiver_ready = None
        index = self.ws.robot._action_split_indexes[side+"_gripper"][0]
        self.ws.action[index] = -1.
        consecutive = 0
        for feedback_ticks in range(1,ticks_at_rate(20)+1):
            self.tick()
            consecutive = consecutive+1 if _open_feedback(self,side) else 0
            if consecutive >= ticks_at_rate(3):
                break
        if not _open_feedback(self,side):
            raise RuntimeError("gripper did not physically reach original open endpoint")
        release_ticks = 0
        if evidence and evidence["kind"] == "composable_release":
            consecutive = 0
            for release_ticks in range(1,ticks_at_rate(50)+1):
                self.tick()
                state = self.observe()
                ready = (not any(state["assist_active"].values())
                         and state["bottle_speed_norm"] < .05
                         and (state["bucket_bottom_contact"] or self._bottle_worktable_contact()))
                consecutive = consecutive+1 if ready else 0
                if release_ticks >= ticks_at_rate(6) and consecutive >= ticks_at_rate(3):
                    break
        return {"kind":"gripper_contact_control","timing_profile":PROFILE,
                "feedback_steps":feedback_ticks,"release_settle_steps":release_ticks,
                "open_or_close":operation,"commanded_value":float(self.ws.action[index]),
                "contact_geoms":self._finger_contacts_now(side),"assist_active":self._held_sides(),
                "receiver_ready":self.receiver_ready,"evidence":evidence,"verified":True}
    task._move_basic = _move_basic
    task._operate_gripper = _operate_gripper
    task._paos_timing_installed = True

def metadata():
    return {**SPEC, "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
