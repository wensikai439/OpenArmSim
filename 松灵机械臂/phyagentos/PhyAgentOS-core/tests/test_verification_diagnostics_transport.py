from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib import error, request
from unittest.mock import patch

from PhyAgentOS.verification.engine import VerificationEngine
from PhyAgentOS.verification import service


class VerificationDiagnosticsTransportTests(unittest.TestCase):
    def test_http_payload_and_content_cannot_select_diagnostics_path(self):
        with tempfile.TemporaryDirectory() as directory:
            trusted = Path(directory) / "workspace" / ".paos" / "verification" / "diagnostics"
            untrusted = Path(directory) / "untrusted-model-or-http-path"
            async def complete(**kwargs):
                return SimpleNamespace(content="PRIVATE_RAW_CONTENT", finish_reason="stop", usage={"total_tokens":1})
            engine = VerificationEngine(
                provider=SimpleNamespace(chat_with_retry=complete), model="offline-check",
                diagnostics_dir=trusted,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), service._handler(engine, "test-token"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                payload = {
                    "version": "forge_verification_request_v1",
                    "content": [{"type":"text","text":str(untrusted)}],
                    "diagnostics_dir":str(untrusted),
                }
                req = request.Request(
                    f"http://127.0.0.1:{server.server_port}/v1/verify-task",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json","X-PAOS-Admin-Token":"test-token"},
                    method="POST",
                )
                opener = request.build_opener(request.ProxyHandler({}))
                with self.assertRaises(error.HTTPError) as raised:
                    opener.open(req, timeout=3)
                self.assertEqual(raised.exception.code, 500)
                message = json.loads(raised.exception.read())["error"]
                self.assertIn("diagnostics_file=verification-json-", message)
                self.assertNotIn("PRIVATE_RAW_CONTENT", message)
                self.assertNotIn(str(trusted), message)
                self.assertFalse(untrusted.exists())
                files = list(trusted.glob("verification-json-*.json"))
                self.assertEqual(len(files), 1)
                self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_child_uses_only_trusted_environment_diagnostics_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            trusted = str(Path(directory) / "workspace" / ".paos" / "verification" / "diagnostics")
            settings = {
                "provider":{"provider_name":"test", "model":"offline-check"},
                "timeout_s":3, "host":"127.0.0.1", "port":1,
                "session_token":"test-token", "diagnostics_dir":trusted,
            }
            captured = []
            with patch.dict("os.environ", {"PAOS_VERIFICATION_SERVICE_CONFIG":json.dumps(settings)}), \
                 patch.object(service, "_provider", return_value=object()), \
                 patch.object(service, "serve_verification_service", side_effect=lambda engine,*args:captured.append(engine)):
                self.assertEqual(service.main(), 0)
            self.assertEqual(captured[0].diagnostics_dir, Path(trusted).resolve())


if __name__ == "__main__":
    unittest.main()
