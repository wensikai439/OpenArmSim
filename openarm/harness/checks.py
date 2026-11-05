"""Cross-role data isolation, end-of-run review and owned job cancellation."""
import copy
import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .agent import Harness
from sim.learning import LearningJobs
from sim.evaluate import validate_split


class LocalState:
    def __init__(self, with_queue=False):
        self.lock = threading.Lock()
        self.frames = {c: b"test-image" for c in ("head", "left", "right")}
        self.cache = {"state": "ready", "observation_id": "episode:0", "observation": {"tick": 0, "state": [0.] * 16},
                      "job": None, "last_run": {"run_id": "run-a", "status": "invalid", "error": "SECRET_CONTACT_FAILURE"},
                      "diagnostics": {"evaluation": {"success": False, "held": {"left": True}, "secret": "OBJECT_TRUTH"}}}
        self.calls = []
        if with_queue:
            self.completed = queue.Queue()

    def snapshot(self):
        return copy.deepcopy(self.cache)

    def call(self, name, **arguments):
        self.calls.append(name)
        return {"state": "started", "execution_id":'call-'+str(len(self.calls))}


class HarnessChecks(unittest.TestCase):
    def test_replayed_source_cannot_cross_train_test_split(self):
        source = {"kind": "replay", "details": {"episode": 0}, "hashes": {"reference/real_trajectory.json": "same-recording"}}
        first = {"source_run": "run-a", "source_transitions_sha256": "a"}
        second = {"source_run": "run-b", "source_transitions_sha256": "b"}
        with self.assertRaises(ValueError):
            validate_split(source, source, first, second)
        other = {**source, "details": {"episode": 1}}
        validate_split(source, other, first, second)
        with self.assertRaises(ValueError):
            validate_split(source, other, first, first)

    def test_execution_status_and_assessment_do_not_reveal_or_change_truth(self):
        state = LocalState()
        agent = Harness(state)
        with patch.object(agent, "_log") as log:
            status = agent.invoke("get_run_status", {})
            self.assertNotIn("evaluation", status)
            for word in ("OBJECT_TRUTH", "SECRET_CONTACT_FAILURE", "invalid", "held"):
                self.assertNotIn(word, json.dumps(status))
            obs = agent.invoke("observe", {})
            result = agent.invoke("finish", {"observation_id": obs["observation_id"], "verdict": "success", "reason": "visual claim"})
            self.assertTrue(result["assessment_recorded"])
            self.assertFalse(state.cache["diagnostics"]["evaluation"]["success"])
            self.assertNotIn("simulator_success", result)
            self.assertTrue(any(call.args[0] == "independent_assessment" for call in log.call_args_list))

    def test_role_and_numeric_tool_validation(self):
        agent = Harness(LocalState(), LearningJobs())
        for name, role in (("train_model", "execute"), ("evaluate_timing", "execute"), ("fit_controller", "execute"), ("get_calibration", "execute"), ("run_pi05_vla", "research"), ("observe", "invalid")):
            with self.assertRaises(ValueError):
                agent.invoke(name, {}, role=role)
        for steps in (True, "5", 0, 10001):
            with self.assertRaises(ValueError):
                agent.invoke("train_model", {"run_id": "a", "recipe": "act_smoke", "steps": steps, "seed": 42}, role="research")

    def test_end_of_run_review_uses_fresh_views_once(self):
        state = LocalState(with_queue=True)
        state.cache['policy_configured'] = True
        agent = Harness(state)
        received = []
        def response(request, **kwargs):
            payload = json.loads(request.data)
            received.append(payload)
            self.assertNotIn("OBJECT_TRUTH", json.dumps(payload))
            self.assertNotIn("SECRET_CONTACT_FAILURE", json.dumps(payload))
            self.assertNotIn("RESEARCH_ONLY", json.dumps(payload))
            if len(received) == 1:
                msg = {"role":"assistant", "tool_calls":[{"id":"plan","type":"function","function":{"name":"set_plan","arguments":json.dumps({"observation_id":"episode:0","steps":[{"id":"run","title":"执行交接策略","checkpoint":"tool_completed"}]})}}]}
            elif len(received) == 2:
                msg = {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "run_policy", "arguments": '{"observation_id":"episode:0","backend":"pi05","skill_id":"atomic_right_to_left_handover"}'}}]}
            else:
                encoded = json.dumps(payload)
                self.assertEqual([t["function"]["name"] for t in payload["tools"]], ["observe", "finish"])
                self.assertIn("episode:3", encoded)
                self.assertNotIn("OLD_ORACLE_HISTORY", encoded)
                self.assertNotIn("ORACLE_REVIEW_SECRET", encoded)
                self.assertNotIn("Historical reference data", encoded)
                self.assertNotIn('"decision_timing"', encoded)
                msg = {"role": "assistant", "content": "当前画面不足以确认成功。"}
                if len(received) == 3:
                    msg = {"role":"assistant","tool_calls":[{"id":"fresh","type":"function","function":{"name":"observe","arguments":"{}"}}]}
                elif len(received) == 4:
                    msg = {"role":"assistant","tool_calls":[{"id":"finish","type":"function","function":{"name":"finish","arguments":json.dumps({"observation_id":"episode:3","verdict":"uncertain","reason":"图像不足以确认"})}}]}
            class Reply:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self, _): return json.dumps({"choices": [{"message": msg}]}).encode()
            return Reply()
        try:
            agent.history["research"].append({"role": "assistant", "content": "RESEARCH_ONLY"})
            agent.history["execute"].append({"role": "assistant", "content": "OLD_ORACLE_HISTORY"})
            with patch.dict(os.environ, {"OPENARM_LLM_KEY": "local-test-placeholder", "OPENARM_LLM_VISION": "1"}), patch("urllib.request.urlopen", side_effect=response), patch.object(agent, "_log"):
                agent.chat("运行交接仿真")
                state.cache["observation_id"] = "episode:3"
                state.cache["observation"]["tick"] = 3
                state.cache["observation"]["oracle_truth"] = "ORACLE_REVIEW_SECRET"
                state.cache["observation"]["decision_timing"] = {"task_failed": True, "score": "ORACLE_REVIEW_SECRET"}
                state.completed.put({"epoch": "episode", "execution_id":'call-1', "run_id": "run-a"})
                deadline = time.monotonic() + 3
                while not agent.notifications() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(len(agent.notifications()), 1)
                # Plan + policy start + blind view + fresh view assessment.
                # The blind-review turn gets neither old history nor scoring fields.
                self.assertEqual(len(received), 4)
                self.assertEqual(state.calls, ["policy"])
        finally:
            agent.close()
            agent.review_thread.join(1)

    def test_learning_paths_and_stale_process_handles(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sim.learning.ROOT", Path(tmp)):
            source = Path(tmp) / "runs/source"
            source.mkdir(parents=True)
            jobs = LearningJobs()
            for name in ("../source", str(source), "source/other"):
                with self.assertRaises(ValueError): jobs.run_path(name)
            (source / "job.json").write_text(json.dumps({"job_id": "source", "status": "running", "kind": "train"}))
            self.assertEqual(jobs.status("source")["status"], "unknown")
            with self.assertRaises(ValueError): jobs.stop("source")

    def test_owned_learning_process_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sim.learning.ROOT", Path(tmp)):
            (Path(tmp) / "runs/source").mkdir(parents=True)
            jobs = LearningJobs()
            with patch("sim.learning.runtime", return_value=([sys.executable, "-B", "-c", "import time; time.sleep(120)"], dict(os.environ))):
                job = jobs.start("export", "source")
                with self.assertRaises(ValueError): jobs.start("export", "source")
                jobs.stop(job["job_id"])
                jobs.thread.join(12)
                self.assertFalse(jobs.thread.is_alive())
                self.assertEqual(jobs.status()["status"], "cancelled")
                self.assertIsNone(jobs.process)
                saved = json.loads((Path(tmp) / "runs" / job["job_id"] / "summary.json").read_text())
                self.assertEqual(saved["status"], "cancelled")
