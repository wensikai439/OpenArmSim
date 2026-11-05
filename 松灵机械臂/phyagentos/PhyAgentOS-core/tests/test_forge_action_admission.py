"""Focused regression tests for confirmed Action admission versus uncertain execution."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

from PhyAgentOS.agent.tools.forge_tool_api import ForgeToolContextTool, ForgeToolStartActionTool
from PhyAgentOS.config.schema import ForgeConfig
from PhyAgentOS.forge.task import AgentTaskCoordinator, AgentTaskError
from PhyAgentOS.forge.tool_client import ForgeToolAPIError, ForgeToolAdmissionError, ForgeToolClient
from PhyAgentOS.skill_runtime.integration import _gateway_invocation_limits
from PhyAgentOS.verification.contracts import TaskVerificationContract, VerificationEvidencePolicy


REJECTION = "timeout_ms must not exceed configured tools.invoke_timeout_ms maximum 23000"


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    def make_system(self, handler):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        client = ForgeToolClient("http://gateway.test", transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.close)
        tracked = set()
        coordinator = AgentTaskCoordinator(workspace=temporary.name, config=ForgeConfig(),
                                           client=client, runtime_invocation_ids=tracked)
        async def no_capture(task_id):
            return None
        coordinator._capture_before = no_capture
        task = coordinator.create_task(task_description="test admission", verification=TaskVerificationContract(
            mode="off", evidence_policy=VerificationEvidencePolicy(required_kinds=[])))
        return client, coordinator, task, tracked

    async def test_exact_gateway_rejection_allows_corrected_request(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(400, json={"ok": False, "msg": REJECTION})
            return httpx.Response(202, json={"ok": True, "data": {
                "invocation_id": "admitted", "attempt_id": "attempt"}})
        _, coordinator, task, tracked = self.make_system(handler)
        tool = ForgeToolStartActionTool(coordinator)
        failed = json.loads(await tool.execute(task.task_id, "demo.action", {}, timeout_ms=24000))
        self.assertEqual(failed["error"]["remote_state"], "not_admitted")
        self.assertEqual(failed["error"]["status_code"], 400)
        record = coordinator.store.get(task.task_id).execution_records[0]
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.response, {"ok": False, "msg": REJECTION})
        self.assertIsNone(record.invocation_id)
        self.assertFalse(tracked)
        accepted = json.loads(await tool.execute(task.task_id, "demo.action", {}, timeout_ms=23000))
        self.assertEqual(accepted["data"]["invocation_id"], "admitted")
        self.assertEqual(tracked, {"admitted"})
        self.assertEqual(len(requests), 2)

    async def test_concurrency_rejection_is_not_admitted_and_binding_can_finish(self):
        calls = []
        payload = {"ok": False, "msg": "operation concurrency 1 is exhausted"}
        def handler(request):
            calls.append(request)
            return httpx.Response(503, json=payload)
        _, coordinator, task, tracked = self.make_system(handler)
        with self.assertRaises(ForgeToolAdmissionError):
            await coordinator.start_action(task.task_id, "demo.action", {})
        record = coordinator.store.get(task.task_id).execution_records[0]
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error["remote_state"], "not_admitted")
        self.assertEqual(record.response, payload)
        self.assertIsNone(record.invocation_id)
        self.assertIsNone(record.attempt_id)
        self.assertFalse(tracked)
        finalized = await coordinator.finalize_task(task.task_id)
        self.assertEqual(finalized.status.value, "failed")
        self.assertEqual(len(calls), 1)

    async def test_concurrency_message_with_identity_or_wrong_envelope_stays_unknown(self):
        message = "operation concurrency 1 is exhausted"
        cases = [
            (400, {"ok": False, "msg": message}),
            (503, {"ok": False, "msg": "Service unavailable"}),
            (503, {"ok": False, "msg": message, "data": {"invocation_id": "remote"}}),
        ]
        for status, payload in cases:
            with self.subTest(status=status, payload=payload):
                _, coordinator, task, tracked = self.make_system(
                    lambda request: httpx.Response(status, json=payload))
                with self.assertRaises(ForgeToolAPIError) as raised:
                    await coordinator.start_action(task.task_id, "demo.action", {})
                self.assertNotIsInstance(raised.exception, ForgeToolAdmissionError)
                self.assertEqual(coordinator.store.get(task.task_id).execution_records[0].status, "unknown")
                if "data" in payload:
                    self.assertEqual(tracked, {"remote"})

    async def test_local_invalid_timeout_does_not_submit_or_poison_action(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(202, json={"ok": True, "data": {
                "invocation_id": "valid", "attempt_id": "attempt"}})
        _, coordinator, task, _ = self.make_system(handler)
        with self.assertRaises(ForgeToolAdmissionError):
            await coordinator.start_action(task.task_id, "demo.action", {}, timeout_ms=0)
        self.assertFalse(requests)
        self.assertEqual(coordinator.store.get(task.task_id).execution_records[0].status, "failed")
        await coordinator.start_action(task.task_id, "demo.action", {}, timeout_ms=1000)
        self.assertEqual(len(requests), 1)

    async def test_ambiguous_http_rejections_retain_unknown_and_dedup(self):
        cases = [
            (400, {"ok": False, "msg": "other validation error"}, 24000),
            (422, {"ok": False, "msg": REJECTION}, 24000),
            (400, {"ok": False, "msg": REJECTION}, 23000),
            (400, {"ok": False, "msg": REJECTION, "data": {"invocation_id": "remote"}}, 24000),
        ]
        for status, payload, timeout in cases:
            with self.subTest(status=status, payload=payload, timeout=timeout):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(status, json=payload)
                _, coordinator, task, tracked = self.make_system(handler)
                with self.assertRaises(ForgeToolAPIError) as raised:
                    await coordinator.start_action(task.task_id, "demo.action", {}, timeout_ms=timeout)
                self.assertNotIsInstance(raised.exception, ForgeToolAdmissionError)
                self.assertEqual(coordinator.store.get(task.task_id).execution_records[0].status, "unknown")
                with self.assertRaises(AgentTaskError):
                    await coordinator.start_action(task.task_id, "demo.action", {}, timeout_ms=1000)
                self.assertEqual(len(calls), 1)
                if "data" in payload:
                    self.assertEqual(tracked, {"remote"})

    async def test_transport_timeout_keeps_unknown_and_dedup(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("no conclusive response", request=request)
        _, coordinator, task, _ = self.make_system(handler)
        result = json.loads(await ForgeToolStartActionTool(coordinator).execute(task.task_id, "demo.action", {}))
        self.assertEqual(result["error"]["remote_state"], "unknown")
        with self.assertRaises(AgentTaskError):
            await coordinator.start_action(task.task_id, "demo.action", {})
        self.assertEqual(len(calls), 1)

    async def test_agent_context_exposes_actual_profile_limit_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "flow.yaml").write_text("nodes:\n- id: gateway\n  args: --config actual.yaml\n")
            (root / "actual.yaml").write_text("tools:\n  invoke_timeout_ms: 23000\n")
            manifest = SimpleNamespace(name="test", version="1.2", bundle_root=root,
                profiles={"local": SimpleNamespace(dataflow=Path("flow.yaml"))},
                artifacts=SimpleNamespace(nodes={"gateway": SimpleNamespace(entrypoint="gateway")}),
                resolve_bundle_path=lambda path: root / path)
            state = SimpleNamespace(profile="local", runtime_instance_id="runtime-test", gateway_identity="gateway-test")
            limits = _gateway_invocation_limits(manifest, state)
            self.assertEqual(limits["max_action_timeout_ms"], 23000)
            self.assertEqual(limits["source"]["profile"], "local")
            self.assertEqual(limits["source"]["path"], str(root / "actual.yaml"))
            client = ForgeToolClient("http://gateway.test", invocation_limits=limits,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True, "data": {"ready": True}})))
            try:
                result = json.loads(await ForgeToolContextTool(client).execute("demo.action"))
            finally:
                await client.close()
            self.assertEqual(result["data"]["context"]["runtime_invocation_limits"], limits)
            (root / "actual.yaml").write_text("tools: {}\n")
            self.assertIsNone(_gateway_invocation_limits(manifest, state))


if __name__ == "__main__":
    unittest.main()
