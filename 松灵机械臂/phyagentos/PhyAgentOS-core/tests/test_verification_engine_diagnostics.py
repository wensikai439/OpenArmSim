from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PhyAgentOS.verification.engine import VerificationEngine


class VerificationEngineDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, content, finish_reason="stop", diagnostics_dir=None):
        calls = []
        async def complete(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=content, finish_reason=finish_reason, usage={"total_tokens": 7})
        return VerificationEngine(provider=SimpleNamespace(chat_with_retry=complete), model="offline-check", diagnostics_dir=diagnostics_dir), calls

    async def test_plain_object_is_returned_without_extra_call(self):
        engine, calls = self.engine('{"verdict":"inconclusive"}')
        self.assertEqual(await engine.complete(system_prompt="check", content=[]), {"verdict":"inconclusive"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["temperature"], 0.0)

    async def test_prose_or_truncated_json_is_not_repaired_and_raw_is_not_leaked(self):
        for content, finish in [('Here is JSON: {"secret":"PRIVATE_SENTINEL"}', "stop"), ('{"secret":"PRIVATE_SENTINEL"', "length")]:
            with self.subTest(finish_reason=finish):
                engine, calls = self.engine(content, finish)
                with self.assertRaisesRegex(ValueError, "verifier response is not valid JSON") as raised:
                    await engine.complete(system_prompt="check", content=[])
                message = str(raised.exception)
                self.assertIn("finish_reason=" + repr(finish), message)
                self.assertIn("content_chars=" + str(len(content)), message)
                self.assertIn("position=", message)
                self.assertNotIn("PRIVATE_SENTINEL", message)
                self.assertEqual(len(calls), 1)

    async def test_non_object_json_is_still_rejected(self):
        engine, calls = self.engine('[]')
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            await engine.complete(system_prompt="check", content=[])
        self.assertEqual(len(calls), 1)

    async def test_empty_or_provider_error_is_not_parsed(self):
        for content, finish in [("", "stop"), ("Error: provider unavailable", "error")]:
            with self.subTest(finish_reason=finish):
                engine, calls = self.engine(content, finish)
                with self.assertRaises(RuntimeError):
                    await engine.complete(system_prompt="check", content=[])
                self.assertEqual(len(calls), 1)


    async def test_complete_json_fence_only_is_unwrapped_and_raw_saved_once(self):
        raw = '```json\n{"verdict":"failure"}\n```'
        with tempfile.TemporaryDirectory() as directory:
            engine, calls = self.engine(raw, diagnostics_dir=directory)
            self.assertEqual(await engine.complete(system_prompt="check", content=[]), {"verdict":"failure"})
            paths = list(Path(directory).glob("verification-json-*.json"))
            self.assertEqual(len(paths), 1)
            saved = json.loads(paths[0].read_text())
            self.assertEqual(saved, {"content":raw,"finish_reason":"stop","usage":{"total_tokens":7}})
            self.assertEqual(paths[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(calls), 1)

    async def test_fence_with_prose_multiple_blocks_or_malformed_body_is_rejected(self):
        fenced = '```json\n{"verdict":"failure"}\n```'
        for raw in ("explanation\n"+fenced, fenced+"\nexplanation", fenced+"\n"+fenced,
                    '```json\n{"verdict":\n```', '```json\n{"verdict":"failure"}'):
            with self.subTest(raw=raw):
                engine, calls = self.engine(raw)
                with self.assertRaisesRegex(ValueError, "not valid JSON"):
                    await engine.complete(system_prompt="check", content=[])
                self.assertEqual(len(calls), 1)

    async def test_malformed_responses_have_unique_private_files_and_safe_error_names(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, calls = self.engine("PRIVATE_SENTINEL", diagnostics_dir=directory)
            for _ in range(2):
                with self.assertRaises(ValueError) as raised:
                    await engine.complete(system_prompt="check", content=[])
                self.assertIn("diagnostics_file=verification-json-", str(raised.exception))
                self.assertNotIn(directory, str(raised.exception))
                self.assertNotIn("PRIVATE_SENTINEL", str(raised.exception))
            paths=list(Path(directory).glob("verification-json-*.json"))
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in paths))

    async def test_valid_json_is_archived_before_later_schema_rejection(self):
        from pydantic import ValidationError
        from PhyAgentOS.verification.contracts import VerificationVerdict
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"diagnostics"
            raw = '{ "verdict" : "failure" }'
            engine, calls = self.engine(raw, diagnostics_dir=target)
            parsed = await engine.complete(system_prompt="check", content=[])
            self.assertEqual(parsed, {"verdict": "failure"})
            with self.assertRaises(ValidationError):
                VerificationVerdict.model_validate(parsed)
            paths = list(target.glob("verification-json-*.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].stat().st_mode & 0o777, 0o600)
            archived = json.loads(paths[0].read_text())
            self.assertEqual(archived["content"], raw)
            self.assertEqual(archived["finish_reason"], "stop")
            self.assertEqual(archived["usage"], {"total_tokens": 7})
            self.assertEqual(len(calls), 1)

    async def test_null_criterion_note_preserves_raw_and_semantic_fields(self):
        from PhyAgentOS.verification.contracts import VerificationVerdict
        prompt = "You are the semantic verifier for a physical Agent task."
        expected = {"verdict": "failure", "criteria": [
            {"criterion": "real task criterion", "status": "unsatisfied", "evidence_refs": ["tool:real-query"]}],
            "evidence_refs": ["tool:real-query"], "reason": "Actual goal unmet.", "lesson": "Require actual evidence."}
        raw_object = json.loads(json.dumps(expected))
        raw_object["criteria"][0]["evidence_refs_note"] = None
        raw = json.dumps(raw_object)
        with tempfile.TemporaryDirectory() as directory:
            engine, calls = self.engine(raw, diagnostics_dir=directory)
            value = await engine.complete(system_prompt=prompt, content=[])
            self.assertEqual(value, expected)
            self.assertEqual(VerificationVerdict.model_validate(value).verdict, "failure")
            paths = list(Path(directory).glob("verification-json-*.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(json.loads(paths[0].read_text())["content"], raw)
            self.assertEqual(len(calls), 1)

    async def test_nonempty_or_unrelated_extra_fields_remain_invalid(self):
        from PhyAgentOS.verification.contracts import VerificationVerdict
        prompt = "You are the semantic verifier for a physical Agent task."
        for extra in ({"evidence_refs_note": "claim"}, {"other_extra": None}):
            data = {"verdict": "failure", "criteria": [
                {"criterion": "criterion", "status": "unsatisfied", "evidence_refs": [], **extra}],
                "evidence_refs": [], "reason": "Actual goal unmet.", "lesson": "Require evidence."}
            engine, calls = self.engine(json.dumps(data))
            if "evidence_refs_note" in extra:
                with self.assertRaisesRegex(ValueError, "requires a saved raw response"):
                    await engine.complete(system_prompt=prompt, content=[])
            else:
                value = await engine.complete(system_prompt=prompt, content=[])
                with self.assertRaises(ValueError):
                    VerificationVerdict.model_validate(value)
            self.assertEqual(len(calls), 1)

    async def test_text_criterion_note_is_preserved_in_reason_with_raw_archive(self):
        from PhyAgentOS.verification.contracts import VerificationVerdict
        payload = {"verdict": "failure", "criteria": [
            {"criterion": "physical transfer", "status": "unsatisfied", "evidence_refs": ["tool:query"],
             "evidence_refs_note": "No new contact transfer was observed."}],
            "evidence_refs": ["tool:query"], "reason": "The observed task is incomplete.",
            "lesson": "Require real transfer evidence.", "recovery_context": None}
        raw = json.dumps(payload)
        with tempfile.TemporaryDirectory() as directory:
            engine, calls = self.engine(raw, diagnostics_dir=directory)
            value = await engine.complete(system_prompt="You are the semantic verifier for a physical Agent task.", content=[])
            verdict = VerificationVerdict.model_validate(value)
            self.assertEqual(verdict.verdict, payload["verdict"])
            self.assertEqual(value["criteria"], [{key: item for key, item in payload["criteria"][0].items()
                                                  if key != "evidence_refs_note"}])
            self.assertEqual(value["evidence_refs"], payload["evidence_refs"])
            self.assertEqual(value["lesson"], payload["lesson"])
            self.assertEqual(value["recovery_context"], payload["recovery_context"])
            self.assertTrue(value["reason"].startswith(payload["reason"]))
            self.assertIn(json.dumps(payload["criteria"][0]["evidence_refs_note"]), value["reason"])
            raw_saved = list(Path(directory).glob("verification-json-*.json"))
            self.assertEqual(len(raw_saved), 1)
            self.assertEqual(json.loads(raw_saved[0].read_text())["content"], raw)
            self.assertEqual(len(calls), 1)

    async def test_structured_note_and_unrelated_extra_still_rejected(self):
        from PhyAgentOS.verification.contracts import VerificationVerdict
        for extra in ({"evidence_refs_note": {"status": "success"}}, {"other_extra": "claim"}):
            payload = {"verdict": "failure", "criteria": [
                {"criterion": "physical criterion", "status": "unsatisfied", "evidence_refs": [], **extra}],
                "evidence_refs": [], "reason": "Goal unmet.", "lesson": "Use actual evidence."}
            with tempfile.TemporaryDirectory() as directory:
                engine, calls = self.engine(json.dumps(payload), diagnostics_dir=directory)
                value = await engine.complete(system_prompt="You are the semantic verifier for a physical Agent task.", content=[])
                with self.assertRaises(ValueError):
                    VerificationVerdict.model_validate(value)
                self.assertEqual(len(calls), 1)

    async def test_unwritable_diagnostics_does_not_replace_parse_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"not-a-directory"
            target.write_text("occupied")
            engine, calls = self.engine("PRIVATE_SENTINEL", diagnostics_dir=target)
            with self.assertRaisesRegex(ValueError, "not valid JSON") as raised:
                await engine.complete(system_prompt="check", content=[])
            self.assertIn("diagnostics_write_error=FileExistsError",str(raised.exception))
            self.assertNotIn("PRIVATE_SENTINEL",str(raised.exception))


if __name__ == "__main__":
    unittest.main()
