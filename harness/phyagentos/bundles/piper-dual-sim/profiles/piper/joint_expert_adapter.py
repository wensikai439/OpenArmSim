"""Independent joint-command expert adapter for the shared PiPER asset.

Only this owned process changes controller instances. Shared source, motor
actuators, contacts and task thresholds are reused unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robosuite.controllers.parts.generic.joint_pos import JointPositionController

from piper_robocasa import photo_relay_task
from piper_robocasa.pose_preflight import cross_arm_penetration
from run_photo_pick_place_demo import DemoWorkstation

SIDES = ("left", "right")


class JointTargetWorkstation(DemoWorkstation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_hz = int(os.environ.get("PAOS_CONTROL_HZ", "20"))
        if self.control_hz not in (20, 30):
            raise ValueError("Only existing 20Hz and derived 30Hz are supported")
        if abs(round(1 / self.control_hz / self.model.opt.timestep) * self.model.opt.timestep - 1 / self.control_hz) > 1e-12:
            raise ValueError("Control period must contain an integer number of physics steps")
        self.joint_command = np.zeros(self.robot.action_dim, dtype=float)
        self.last_pose = {}
        self.ik_scratch = mujoco.MjData(self.model)
        self.joint_trace = (self.output.parent / "joint_commands.jsonl").open("x")
        for side in SIDES:
            params = dict(self.robot.composite_controller.part_controller_config[side])
            bounds = self.model.jnt_range[self.joint_ids[side]].T.copy()
            params.update(input_type="absolute", impedance_mode="fixed", kp=150,
                          damping_ratio=1, policy_freq=self.control_hz, lite_physics=False,
                          qpos_limits=bounds, interpolator=None)
            controller = JointPositionController(**params)
            controller.update(force=True)
            controller.reset_goal()
            self.robot.part_controllers[side] = controller
            a, b = self.robot._action_split_indexes[side]
            self.joint_command[a:b] = self.data.qpos[self.qindexes[side]]
        self.robot.composite_controller.setup_action_split_idx()
        spec = {
            "schema": "phyagentos.executed_joint_targets.v1",
            "controller": "robosuite.JointPositionController",
            "input_type": "absolute", "joint_unit": "rad", "kp": 150,
            "damping_ratio": 1, "control_hz": self.control_hz,
            "physics_dt_s": float(self.model.opt.timestep),
            "actuation": "unchanged motor torque output, mass-matrix PD + compensation",
            "native_action_splits": self.robot._action_split_indexes,
            "joint_names": {s: [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(i))
                                 for i in self.joint_ids[s]] for s in SIDES},
            "joint_limits_rad": {s: self.model.jnt_range[self.joint_ids[s]].tolist() for s in SIDES},
            "model_sha256": self.model_hash,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
            "record_alignment": "pre-state -> actual consumed absolute joint command -> post-state",
            "pose_action": "expert intent only; NOT the dataset joint command",
            "assistance": "existing asset contact-triggered assisted grasp; no new constraint",
            "training_admission": "separate physical, official Task and cross-domain contract checks required"
        }
        (self.output.parent / "joint_command_contract.json").write_text(
            json.dumps(spec, indent=2) + "\n")

    def _solve_command(self, side):
        a, b = self.robot._action_split_indexes[side]
        pose = self.action[a:b].copy()
        previous = self.joint_command[a:b].copy()
        if side in self.last_pose and np.array_equal(pose, self.last_pose[side]):
            return previous, {"reused_unchanged_pose": True}
        scratch = self.ik_scratch
        scratch.qpos[:] = self.data.qpos
        scratch.mocap_pos[:] = self.data.mocap_pos
        scratch.mocap_quat[:] = self.data.mocap_quat
        indexes = self.qindexes[side]
        site = self.eef_ids[side]
        bounds = self.model.jnt_range[self.joint_ids[side]].T.copy()
        desired = Rotation.from_rotvec(pose[3:])
        def residual(q):
            scratch.qpos[indexes] = q
            mujoco.mj_kinematics(self.model, scratch)
            actual = Rotation.from_matrix(scratch.site_xmat[site].reshape(3, 3))
            return np.r_[scratch.site_xpos[site] - pose[:3],
                         (desired * actual.inv()).as_rotvec() * 0.12]
        def objective(q):
            return np.r_[residual(q), (q - previous) * 0.005]
        local_bounds = bounds
        if os.environ.get("PAOS_MOTION_TIMING_PROFILE") == "feedback_joint_v1":
            # The existing 0.5 rad command limit belongs inside the solve too.
            local_bounds = np.stack((np.maximum(bounds[0]+1e-8,previous-.5+1e-8),
                                     np.minimum(bounds[1]-1e-8,previous+.5-1e-8)))
        fit = least_squares(objective, np.clip(previous, local_bounds[0]+1e-8, local_bounds[1]-1e-8),
                            bounds=local_bounds, max_nfev=80, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = residual(fit.x)
        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]) / 0.12)
        delta = float(np.max(np.abs(fit.x - previous)))
        if position_error > .005 or rotation_error > .05 or delta > .5:
            raise RuntimeError(f"Joint expert IK rejected {side}: position={position_error:.6f}m "
                               f"rotation={rotation_error:.6f}rad joint_delta={delta:.6f}rad")
        mujoco.mj_fwdPosition(self.model, scratch)
        collision = cross_arm_penetration(self.model, scratch)
        if collision is not None:
            raise RuntimeError("Joint expert IK collision: " + str(collision))
        self.last_pose[side] = pose
        return fit.x.copy(), {"position_error_m": position_error,
                             "rotation_error_rad": rotation_error,
                             "joint_delta_rad": delta, "nfev": fit.nfev}

    def _sample(self):
        return {"time_s": float(self.data.time), "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(), "ctrl": self.data.ctrl.tolist(),
                "full_state": self.full_state().tolist(),
                "arm_qpos_rad": {s: self.data.qpos[self.qindexes[s]].tolist() for s in SIDES},
                "tcp_world_m": {s: self.data.site_xpos[self.eef_ids[s]].tolist() for s in SIDES},
                "eq_active": self.data.eq_active.astype(int).tolist(),
                "warnings": [int(w.number) for w in self.data.warning]}

    def step(self, action=None):
        if action is not None:
            self.action[:] = action
        row = {"transition_index": int(self.step_count),
               "source": "reviewed_expert_pose_to_executed_joint_target",
               "pre": self._sample(), "expert_pose_intent": self.action.tolist()}
        if hasattr(self, "binary_task_modes"):
            row["gripper_task_modes"] = dict(self.binary_task_modes)
        count = int(round((1.0 / self.control_hz) / self.model.opt.timestep))
        done = 0
        try:
            ik = {}
            for side in SIDES:
                goal, ik[side] = self._solve_command(side)
                a, b = self.robot._action_split_indexes[side]
                self.joint_command[a:b] = goal
                g = slice(*self.robot._action_split_indexes[side + "_gripper"])
                self.joint_command[g] = self.action[g]
            command = self.joint_command.copy()
            row.update(command_native=command.tolist(), ik=ik,
                       command_created_before_step=True)
            ctrl_peak = np.zeros(self.model.nu)
            for i in range(count):
                self.robot.control(command, policy_step=(i == 0))
                if i == 0:
                    consumed = {s: self.robot.part_controllers[s].goal_qpos.copy() for s in SIDES}
                    if not all(np.array_equal(consumed[s], command[slice(*self.robot._action_split_indexes[s])])
                               for s in SIDES):
                        raise RuntimeError("Actual controller goal differs from logged command")
                    row["controller_consumed_joint_targets_rad"] = {s: q.tolist() for s, q in consumed.items()}
                    row["ctrl_first"] = self.data.ctrl.tolist()
                ctrl_peak = np.maximum(ctrl_peak, np.abs(self.data.ctrl))
                self.sim.step()
                done += 1
                self.safety()
            self.step_count += 1
            row.update(completed=True, ctrl_abs_peak=ctrl_peak.tolist(),
                       tracking_error_rad={s: (command[slice(*self.robot._action_split_indexes[s])]
                                              - self.data.qpos[self.qindexes[s]]).tolist() for s in SIDES})
        except Exception as exc:
            self.paused = True
            self.last_error = str(exc)
            row.update(completed=False, error=str(exc))
            row["failure_contacts"] = [
                {"dist_m": float(c.dist),
                 "geoms": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                           for g in (c.geom1, c.geom2)]}
                for c in self.data.contact[:self.data.ncon] if c.dist < -0.001
            ]
            raise
        finally:
            row.update(physics_substeps_done=done, physics_substeps_expected=count,
                       post=self._sample())
            self.joint_trace.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.joint_trace.flush()

    def close(self):
        if hasattr(self, "joint_trace") and not self.joint_trace.closed:
            self.joint_trace.close()
        super().close()


def install():
    """Select this workstation only inside an owned simulator process."""
    photo_relay_task.DemoWorkstation = JointTargetWorkstation
