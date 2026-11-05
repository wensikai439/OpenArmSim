"""Regression checks for the physical/data contract; no external service calls."""
import json
import tempfile
import unittest
from unittest.mock import patch
import threading
from pathlib import Path

import msgpack
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from .environment import ROOT, OpenArmEnv
from .policy import pack_array, unpack_array, load_config
from .recording import Recorder
from .policy import OpenPiClient
from harness.agent import Harness
from .replay import ReferenceReplay
from .reference_clock import load as load_reference_clock
from .dataset import validate_training_clock


class ContractChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = OpenArmEnv(images=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=42)

    def test_clock_has_no_drift(self):
        for frame in range(90):
            self.env.step(self.env.reference_action(0, frame))
            self.assertLessEqual(abs(self.env.data.time - (frame + 1) / 30), .000501)
        self.assertAlmostEqual(self.env.data.time, 3., places=10)

    def test_reject_invalid_and_no_mutation(self):
        before = self.env.data.qpos.copy()
        for action in ([0.] * 7, [float("nan")] * 16, [100.] * 16):
            with self.assertRaises(ValueError):
                self.env.step(action)
            np.testing.assert_array_equal(before, self.env.data.qpos)
            self.assertEqual(self.env.tick, 0)

    def test_nonuniform_clock_is_bounded_and_cannot_mix_with_policy_clock(self):
        env = self.env
        action = env.reference_action(0, 0)
        before = env.data.qpos.copy()
        for target in (0, -1, float("nan"), 1):
            with self.assertRaises(ValueError):
                env.step_until(action, target)
            np.testing.assert_array_equal(before, env.data.qpos)
        for target in (.033804, .073, .113795):
            env.step_until(action, target)
            self.assertLessEqual(abs(float(env.data.time)-target), .000501)
        before = env.data.qpos.copy()
        with self.assertRaises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(before, env.data.qpos)
        env.reset()
        env.step(action)
        with self.assertRaises(ValueError):
            env.step_until(action, .073)

    def test_replay_delimits_commands_by_observations_and_scores_last_sample(self):
        audit = load_reference_clock()
        # A short recorded excerpt has three observations and only two known intervals.
        audit["episodes"]["0"]["recorded_seconds"] = [0., .03523, .17214]
        with patch("sim.replay.load", return_value=audit):
            source = ReferenceReplay(self.env, 0)
        self.assertEqual(source.intervals, 2)
        first, _, d0 = source.advance()
        second, _, d1 = source.advance()
        np.testing.assert_array_equal(first, self.env.reference_action(0, 0))
        np.testing.assert_array_equal(second, self.env.reference_action(0, 1))
        self.assertEqual(d0["next_reference_frame"], d1["reference_frame"])
        self.assertEqual(d1["next_reference_frame"], 2)
        self.assertAlmostEqual(self.env.data.time, .172, places=9)
        metrics = source.metrics()
        self.assertEqual(metrics["scored_observations"], 3)
        self.assertLessEqual(metrics["max_clock_phase_error_seconds"], .000501)
        expected = self.env.state()[[*range(7), *range(8,15)]] - np.array([
            v for side in ("left", "right") for v in self.env.reference["0"][f"{side}_joints"][2]])
        np.testing.assert_array_equal(d1["next_joint_error_rad"], expected)
        before = self.env.data.qpos.copy()
        with self.assertRaises(StopIteration):
            source.advance()
        np.testing.assert_array_equal(before, self.env.data.qpos)

    def test_recorded_replay_cannot_be_silently_relabelled_as_uniform_training_data(self):
        manifest = {"profile": {"control_hz": 30}, "details": {}}
        def rows(times):
            return [{"observation": {"sim_time": a}, "next_observation": {"sim_time": b}}
                    for a, b in zip(times[:-1], times[1:])]
        validate_training_clock(manifest, rows([0, .033, .067, .100]), "imitation")
        # Same mean frequency is insufficient: individual intervals matter.
        nonuniform = rows([0, .020, .090, .100])
        with self.assertRaises(ValueError):
            validate_training_clock(manifest, nonuniform, "imitation")
        result = validate_training_clock(manifest, nonuniform, "diagnostic")
        self.assertFalse(result["eligible_clock_for_imitation"])
        manifest["details"] = {"replay_clock": {"mode": "recorded"}}
        with self.assertRaises(ValueError):
            validate_training_clock(manifest, rows([0, .033]), "imitation")

    def test_gripper_reports_measured_angle_and_matches_target_only_at_equilibrium(self):
        obs = self.env.state()
        action = self.env.reference_action(0, 0)
        self.assertNotAlmostEqual(obs[7], action[7], places=2)
        self.assertNotAlmostEqual(obs[15], action[15], places=2)
        for target in (-.9, -.4, .05):
            action[[7, 15]] = target
            before = self.env.state().copy()
            self.env._set_command(action)
            np.testing.assert_array_equal(before, self.env.state())
            # A motor that reaches its target must report that same angle,
            # including the mirrored right hand, rather than a rescaled value.
            for side in ("left", "right"):
                q = self.env.data.ctrl[self.env.model.actuator(f"{side}_finger1_ctrl").id]
                for joint in self.env.fingers[side]:
                    self.env.data.qpos[joint.qposadr[0]] = q
            np.testing.assert_allclose(self.env.state()[[7, 15]], target, atol=1e-7)
        self.assertEqual(self.env.model.nu, 16)
        self.assertEqual(mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_lifter_joint"), -1)

    def test_gripper_actuator_preserves_motor_torque_under_coordinate_change(self):
        env = self.env
        action = env.reference_action(0, 0)
        action[[7, 15]] = -.2
        for side in ("left", "right"):
            for joint in env.fingers[side]:
                env.data.qpos[joint.qposadr[0]] = env.grippers[side].to_joint(-.4)
        env.data.qvel[:] = 0
        env._set_command(action)
        mujoco.mj_forward(env.model, env.data)
        for side in ("left", "right"):
            aid = env.model.actuator(f"{side}_finger1_ctrl").id
            torque = env.grippers[side].motor_torque(env.data.actuator_force[aid])
            self.assertAlmostEqual(float(torque), env.calibration["actuators"]["kp"]["gripper"] * .2, places=6)

    def test_active_mount_preserves_official_v1_chain_and_slide_units(self):
        env = self.env
        self.assertEqual(env.robot_version, 'v1')
        official = mujoco.MjModel.from_xml_path(str(ROOT/'openarm_mujoco/v1/openarm_bimanual.xml'))
        source = mujoco.MjData(official)
        for episode in env.reference.values():
            for frame in (0, episode['frames']//2, episode['frames']-1):
                for side in ('left', 'right'):
                    for i, q in enumerate(episode[f'{side}_joints'][frame], 1):
                        for model, data in ((official, source), (env.model, env.data)):
                            data.qpos[model.joint(f'openarm_{side}_joint{i}').qposadr[0]] = q
                mujoco.mj_forward(official, source)
                mujoco.mj_forward(env.model, env.data)
                for side in ('left', 'right'):
                    for suffix in ('link1', 'link7', 'hand_tcp'):
                        name = f'openarm_{side}_{suffix}'
                        a, b = env.model.body(name).id, official.body(name).id
                        np.testing.assert_allclose(env.data.xpos[a], source.xpos[b]+env.base_to_world, atol=1e-10)
                        np.testing.assert_allclose(env.data.xmat[a], source.xmat[b], atol=1e-10)
        for side in ('left', 'right'):
            self.assertEqual(env.grippers[side].joint_unit, 'm')
            for joint in env.fingers[side]:
                self.assertEqual(env.model.jnt_type[joint.id], mujoco.mjtJoint.mjJNT_SLIDE)
                np.testing.assert_allclose(joint.range, [0, .044])
        self.assertEqual(env.profile['gripper_force_limit_unit'], 'N')
        with self.assertRaisesRegex(ValueError, 'unit does not match'):
            OpenArmEnv(images=False, profile={'gripper_force_limit_unit':'N·m'})

    def test_deterministic_replay(self):
        rows = []
        for _ in range(2):
            self.env.reset(seed=19)
            for frame in range(30):
                self.env.step(self.env.reference_action(0, frame))
            rows.append(self.env.data.qpos.copy())
        np.testing.assert_array_equal(*rows)

    def test_basket_pose_alone_cannot_fake_task_success(self):
        env = self.env
        addr = env.model.joint("bottle_free").qposadr[0]
        env.data.qpos[addr:addr+3] = env.data.xpos[env.box_id] + [0, 0, .016]
        mujoco.mj_forward(env.model, env.data)
        for _ in range(80):
            env.step(env.reference_action(0, 0))
        self.assertTrue(env.evaluation["inside_basket"])
        self.assertTrue(env.evaluation["supported_by_basket"])
        self.assertFalse(env.evaluation["success"])
        self.assertEqual([e["stage"] for e in env.events], [0])

    def test_full_footprint_required(self):
        env = self.env
        addr = env.model.joint("bottle_free").qposadr[0]
        env.data.qpos[addr:addr+3] = env.data.xpos[env.box_id] + [.09, 0, .14]
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._evaluate(0)["inside_basket"])

    def test_msgpack_matches_openpi_arrays(self):
        values = {"state": self.env.state(), "image": np.zeros((4, 6, 3), np.uint8)}
        decoded = msgpack.unpackb(msgpack.packb(values, default=pack_array), raw=False, object_hook=unpack_array, strict_map_key=False)
        for name in values:
            np.testing.assert_array_equal(values[name], decoded[name])

    def test_recorded_transition_alignment_and_images(self):
        with tempfile.TemporaryDirectory(prefix="openarm-check-") as tmp:
            rec = Recorder(self.env, "check", record_images=True, seed=42, root=tmp)
            before = rec.observation(self.env)
            action = self.env.reference_action(0, 0)
            _, _, term, trunc, info = self.env.step(action)
            after = rec.observation(self.env)
            rec.append(before, action, after, info, terminated=term, truncated=trunc)
            rec.finish("partial_diagnostic", info)
            row = json.loads((rec.path / "transitions.jsonl").read_text())
            self.assertEqual(row["observation"]["tick"], 0)
            self.assertEqual(row["next_observation"]["tick"], 1)
            self.assertGreater(row["next_observation"]["sim_time"], row["observation"]["sim_time"])
            self.assertEqual(len(list(rec.path.glob("images/*/*.png"))), 6)
            for path in row["observation"]["images"].values():
                self.assertTrue((rec.path / path).is_file())

    def test_gym_reset_contract(self):
        obs, info = self.env.reset(seed=5)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertNotIn("bottle_world", obs)
        self.assertFalse(info["evaluation"]["transfer_ready"])

    def test_policy_example_cannot_accidentally_run(self):
        with self.assertRaises(ValueError):
            load_config(Path(__file__).resolve().parents[1] / "harness/policy.example.json")

    def test_harness_requires_fresh_observation_and_rejects_false_success(self):
        class LocalState:
            def __init__(self):
                self.lock = threading.Lock()
                self.frames = {cam: b"local-test-image" for cam in ("head", "left", "right")}
                self.cache = {"observation_id": "a:0", "observation": {"tick": 0, "state": [0.] * 16},
                              "job": None, "last_run": None, "diagnostics": {"evaluation": {"success": False}}}
            def snapshot(self):
                return self.cache
        local = LocalState()
        harness = Harness(local)
        with patch.object(harness, "_log"):
            with self.assertRaises(ValueError):
                harness.invoke("run_pi05_vla", {"observation_id": "a:0"})
            observed = harness.invoke("observe", {})
            self.assertNotIn("diagnostics", observed)
            self.assertEqual(set(observed["images"]), {"head", "left", "right"})
            with self.assertRaises(ValueError):
                harness.invoke("finish", {"observation_id": "a:0", "verdict": "success", "reason": "script exited"})
            local.cache["observation_id"] = "new-reset:0"
            with self.assertRaises(ValueError):
                harness.invoke("run_pi05_vla", {"observation_id": "a:0"})
            with self.assertRaises(ValueError):
                harness.invoke("run_shell", {"command": "ignored"})

    def test_policy_rejects_bad_chunk_before_execution(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "harness/policy.example.json").read_text())
        config["checkpoint"] = "protocol-test-only"
        client = OpenPiClient(config)
        sent = []
        class Transport:
            def send(self, data):
                sent.append(msgpack.unpackb(data, raw=False, object_hook=unpack_array, strict_map_key=False))
        client.socket = Transport()
        qpos = self.env.data.qpos.copy()
        with patch.object(client, "_receive", return_value={"actions": np.zeros((50, 7), np.float32)}):
            with self.assertRaises(ValueError):
                client.infer(self.env, "protocol test")
        np.testing.assert_array_equal(qpos, self.env.data.qpos)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][config["state_key"]].shape, (16,))
        for key in config["image_keys"].values():
            self.assertEqual(sent[0][key].shape, (960, 1280, 3) if key.endswith("head") else (480, 640, 3))

    def test_deployment_l2_limit_is_per_arm_not_per_joint(self):
        requested = self.env.reference_action(0, 0)
        requested[:7] = self.env.state()[:7] + .5
        applied, report = self.env.prepare_policy_action(requested)
        self.assertTrue(report["left"]["step_limited"])
        self.assertFalse(report["right"]["step_limited"])
        self.assertLessEqual(float(np.linalg.norm(applied[:7]-self.env.state()[:7])), 1.00001)
        self.env.step(applied)
