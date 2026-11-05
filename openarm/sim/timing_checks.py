"""Physical timing regressions; recorded commands are not a learned policy."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .environment import OpenArmEnv
from .recording import Recorder
from .service import SimulationService
from .timing import PolicyTiming, hold_ticks
from .decision_checks import DecisionChecks  # Same clock audit suite; avoid changing physics compatibility through the CLI registry.
from .planning_checks import PlanningChecks


class ChunkSource:
    metadata = {"kind": "test double; not a trained VLA"}

    def __init__(self, action, fail=False):
        self.action, self.fail = action, fail
        self.inputs = []

    def infer(self, env, task):
        self.inputs.append((env.tick, env.state().copy()))
        if self.fail:
            raise TimeoutError("controlled inference timeout")
        return np.repeat(self.action[None, :], 2, axis=0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class TimingChecks(unittest.TestCase):
    def setUp(self):
        self.env = OpenArmEnv(images=False)
        self.env.reset(seed=0)

    def tearDown(self):
        self.env.close()

    def drive(self, timing, source, commands=3, cancel=lambda: False):
        rows, events = [], []
        for requested, diagnostic, held in timing.actions(self.env, source, {"max_steps": commands}, events.append, cancel):
            before = float(self.env.data.time)
            action = requested if held else self.env.prepare_policy_action(requested)[0]
            self.env.step(action)
            timing.committed(held, before, float(self.env.data.time))
            rows.append((action.copy(), diagnostic, self.env.state().copy()))
            if self.env._done:
                break
        return rows, events

    def test_zero_delay_is_exactly_the_unchanged_physics(self):
        source = ChunkSource(self.env.reference_action(0, 1))
        self.drive(PolicyTiming("measured_delay", fixed_delay=0), source)
        actual = self.env.data.qpos.copy()
        self.env.reset(seed=0)
        for _ in range(3):
            self.env.step(self.env.prepare_policy_action(source.action)[0])
        np.testing.assert_array_equal(actual, self.env.data.qpos)

    def test_delay_holds_exact_previous_target_and_ages_observation(self):
        held = self.env.last_action.copy()
        source = ChunkSource(self.env.reference_action(0, 100))
        timing = PolicyTiming("measured_delay", fixed_delay=.1)
        rows, events = self.drive(timing, source)
        self.assertEqual((timing.commands, timing.wait_ticks, self.env.tick), (3, 6, 9))
        self.assertAlmostEqual(self.env.data.time, .3)
        for row in rows[:3]:
            np.testing.assert_array_equal(row[0], held)
        self.assertEqual(rows[3][1]["timing"]["observation_tick"], 0)
        self.assertAlmostEqual(rows[3][1]["timing"]["observation_age_sim_seconds"], .1)
        self.assertEqual(source.inputs[1][0], 5)
        self.assertEqual(len(events), 2)

    def test_timeout_integrates_wait_but_never_applies_candidate(self):
        timing = PolicyTiming("measured_delay", fixed_delay=.1)
        with self.assertRaises(TimeoutError):
            self.drive(timing, ChunkSource(self.env.last_action.copy(), fail=True))
        self.assertEqual((timing.commands, timing.wait_ticks), (0, 3))
        self.assertAlmostEqual(self.env.data.time, .1)

    def test_cancel_discards_received_actions_and_stops_wait_steps(self):
        timing = PolicyTiming("measured_delay", fixed_delay=.3)
        self.drive(timing, ChunkSource(self.env.last_action.copy()), cancel=lambda: self.env.tick >= 2)
        self.assertEqual((timing.commands, timing.wait_ticks), (0, 2))
        self.assertFalse(self.env.evaluation["success"])

    def test_duration_quantization_and_terminal_during_wait(self):
        self.assertEqual(hold_ticks(.1, 30), 3)
        self.assertEqual(hold_ticks(.101, 30), 4)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                hold_ticks(value, 30)
        self.env.profile["episode_seconds"] = .1
        timing = PolicyTiming("measured_delay", fixed_delay=.3)
        self.drive(timing, ChunkSource(self.env.last_action.copy()))
        self.assertEqual((timing.commands, timing.wait_ticks), (0, 3))
        self.assertTrue(self.env._done)

    def test_service_records_waits_and_keeps_command_budget_separate(self):
        import queue
        service = SimulationService.__new__(SimulationService)
        service.env, service.epoch = self.env, "timing-test"
        service.stop_job, service.shutdown = threading.Event(), threading.Event()
        service.completed, service.job = queue.Queue(), {"kind": "policy"}
        service.keyframes = []
        service._publish = lambda: None
        service._capture = lambda *args, **kwargs: None
        service.snapshot = lambda: {'observation_id':f'{service.epoch}:{self.env.tick}',
            'observation':{'tick':self.env.tick,'state':self.env.state().tolist(),'sim_time':float(self.env.data.time)}}
        config = {"max_steps": 3, "timing_mode": "measured_delay"}
        service.policy_config = config  # Frozen configuration established by normal service startup.
        source = ChunkSource(self.env.reference_action(0, 5))
        timing = PolicyTiming("measured_delay", fixed_delay=.1)
        with tempfile.TemporaryDirectory(prefix="openarm-timing-") as tmp:
            with patch.dict("os.environ", {"OPENARM_POLICY_CONFIG": "test-only"}), \
                 patch("sim.service.load_config", return_value=config), \
                 patch("sim.service.make_client", return_value=source), \
                 patch("sim.service.PolicyTiming", return_value=timing), \
                 patch("sim.service.Recorder", side_effect=lambda *a, **k: Recorder(*a, root=tmp, **k)):
                service._run("policy", {"observation_id": "timing-test:0", "record_images": False, "realtime": False})
            folder = next(Path(tmp).iterdir())
            summary = json.loads((folder/"summary.json").read_text())
            rows = [json.loads(s) for s in (folder/"transitions.jsonl").read_text().splitlines()]
            self.assertEqual(summary["status"], "budget_exhausted")
            self.assertEqual(summary["transitions"], 9)
            self.assertEqual(summary["metrics"]["timing"]["policy_commands"], 3)
            self.assertEqual(sum(r["diagnostics"]["timing"]["action_source"] == "inference_hold" for r in rows), 6)
            self.assertAlmostEqual(summary["sim_seconds"], .3)
            receipt = service.completed.get_nowait()
            self.assertEqual(receipt['result']['metrics']['timing']['policy_commands'],3)
            self.assertEqual(receipt['observation']['tick'],9)
            self.env.reset(seed=0)
            self.assertEqual(receipt['observation']['tick'],9)
