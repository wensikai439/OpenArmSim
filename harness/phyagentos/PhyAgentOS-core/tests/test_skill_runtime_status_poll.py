"""Regression checks for runtime ownership updates during healthy status polling."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PhyAgentOS.skill_runtime.integration import PersistentInvocationSet, PersistentTaskBindingSet
from PhyAgentOS.skill_runtime.manager import RuntimeManager, RuntimeManagerError
from PhyAgentOS.skill_runtime.state import RuntimeState, RuntimeStateStore


class RuntimeStatusPollTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = RuntimeStateStore(root / "state")
        self.skill = "status-poll-regression"
        self.manifest = SimpleNamespace(name=self.skill, tools_for_profile=lambda profile: ("observe",))
        self.manager = RuntimeManager(
            catalog=SimpleNamespace(get=lambda name: self.manifest),
            state_store=self.store, runtime_root=root / "runtime", logs_root=root / "logs",
        )
        self.manager._flow_running = lambda flow: True
        self.manager._gateway_snapshot = lambda manifest: {"ok": True}
        self.manager._tool_context_readiness = lambda manifest, tools: {"observe": True}
        self.initial = RuntimeState(
            skill_name=self.skill, profile="simulation", status="running",
            flow_name="regression", gateway_url="http://127.0.0.1:1",
            active_invocations=("accepted-action",),
            active_task_bindings=("task-binding",),
        )
        self.store.save(self.initial)

    def test_healthy_poll_does_not_rewrite_state_or_timestamp(self):
        before = self.store.path_for(self.skill).read_bytes()
        self.assertTrue(self.manager.status(self.skill).ready)
        self.assertEqual(before, self.store.path_for(self.skill).read_bytes())

    def test_release_during_gateway_probe_is_not_resurrected(self):
        independent_store = RuntimeStateStore(self.store.root)
        def probe(manifest):
            PersistentInvocationSet(self.skill, independent_store).discard("accepted-action")
            PersistentTaskBindingSet(self.skill, independent_store).discard("task-binding")
            return {"ok": True}
        self.manager._gateway_snapshot = probe
        self.assertTrue(self.manager.status(self.skill).ready)
        latest = self.store.load(self.skill)
        self.assertEqual((), latest.active_invocations)
        self.assertEqual((), latest.active_task_bindings)

    def test_admission_during_context_probe_is_not_lost(self):
        independent_store = RuntimeStateStore(self.store.root)
        def probe(manifest, tools):
            PersistentInvocationSet(self.skill, independent_store).add("new-action")
            return {"observe": True}
        self.manager._tool_context_readiness = probe
        self.assertTrue(self.manager.status(self.skill).ready)
        self.assertEqual(
            {"accepted-action", "new-action"},
            set(self.store.load(self.skill).active_invocations),
        )

    def test_unhealthy_runtime_still_transitions_to_failed_with_ownership(self):
        self.manager._flow_running = lambda flow: False
        self.manager._gateway_snapshot = lambda manifest: None
        report = self.manager.status(self.skill)
        self.assertFalse(report.ready)
        latest = self.store.load(self.skill)
        self.assertEqual("failed", latest.status)
        self.assertEqual(("accepted-action",), latest.active_invocations)
        self.assertEqual(("task-binding",), latest.active_task_bindings)

    def test_unknown_or_active_reference_still_blocks_normal_stop(self):
        with self.assertRaisesRegex(RuntimeManagerError, "non-terminal"):
            self.manager.stop(self.skill)
        self.assertEqual(self.initial, self.store.load(self.skill))


if __name__ == "__main__":
    unittest.main()
