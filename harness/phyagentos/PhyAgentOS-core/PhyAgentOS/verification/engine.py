"""Stateless model-backed semantic verification."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class VerificationEngine:
    def __init__(
        self, *, provider: Any, model: str, timeout_s: float = 180.0,
        diagnostics_dir: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        # Only trusted constructor configuration supplies this path, never HTTP/model content.
        self.diagnostics_dir = (
            Path(diagnostics_dir).expanduser().resolve() if diagnostics_dir is not None else None
        )

    async def complete(self, *, system_prompt: str, content: list[dict[str, Any]]) -> dict:
        response = await asyncio.wait_for(
            self.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                model=self.model,
                tools=None,
                temperature=0.0,
            ),
            timeout=self.timeout_s,
        )
        if response.finish_reason == "error" or not response.content:
            raise RuntimeError(response.content or "verifier model returned no content")
        # Preserve the exact reply before JSON and later strict verdict validation.
        # Reuse this private audit for compatibility; never rewrite a model decision.
        diagnostic = self._record_json_failure(response)
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError as exc:
            failure = ValueError(
                f"verifier response is not valid JSON: {exc.msg} "
                f"(finish_reason={response.finish_reason!r}, "
                f"content_chars={len(response.content)}, line={exc.lineno}, "
                f"column={exc.colno}, position={exc.pos}{diagnostic})"
            )
            # MiMo can wrap an otherwise strict object in one complete JSON code fence.
            # Do not extract substrings from prose or repair malformed/truncated JSON.
            fenced = re.fullmatch(
                r"```json[ \t]*\r?\n(.*?)\r?\n```", response.content.strip(), re.DOTALL
            )
            if fenced is None:
                raise failure from exc
            try:
                value = json.loads(fenced.group(1))
            except json.JSONDecodeError as fenced_exc:
                raise failure from fenced_exc
        if not isinstance(value, dict):
            raise ValueError("verifier response must be a JSON object")
        # Flash sometimes adds a criterion explanation outside the public schema.
        # Archive the raw reply first. Null notes are empty metadata; textual notes
        # are preserved verbatim in reason. Decisions and references never change.
        criteria = value.get("criteria")
        if (
            system_prompt.startswith("You are the semantic verifier for a physical Agent task.")
            and isinstance(criteria, list)
            and any(isinstance(item, dict) and "evidence_refs_note" in item
                    and (item["evidence_refs_note"] is None
                         or isinstance(item["evidence_refs_note"], str))
                    for item in criteria)
        ):
            audit = diagnostic
            if not audit.startswith(", diagnostics_file="):
                raise ValueError("criterion-note compatibility requires a saved raw response")
            value = dict(value)
            notes = []
            revised = []
            for index, item in enumerate(criteria):
                if not isinstance(item, dict):
                    revised.append(item)
                    continue
                note = item.get("evidence_refs_note")
                compatible = note is None or (isinstance(note, str) and isinstance(value.get("reason"), str))
                revised.append({key: item_value for key, item_value in item.items()
                                if not (key == "evidence_refs_note" and compatible)})
                if isinstance(note, str) and compatible and note:
                    notes.append(f"Criterion {index + 1} model explanation: " + json.dumps(note, ensure_ascii=False))
            value["criteria"] = revised
            if notes:
                value["reason"] += "\n\n" + "\n".join(notes)
        return value


    def _record_json_failure(self, response: Any) -> str:
        if self.diagnostics_dir is None:
            return ""
        try:
            self.diagnostics_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix="verification-json-", suffix=".json", dir=self.diagnostics_dir
            )
            # mkstemp creates an exclusive file with mode 0600 before content is written.
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                        "usage": getattr(response, "usage", {}),
                    },
                    handle, ensure_ascii=False, indent=2,
                )
                handle.write("\n")
            return f", diagnostics_file={Path(name).name}"
        except OSError as exc:
            # Saving diagnostics must not leak content or replace the parsing failure.
            return f", diagnostics_write_error={type(exc).__name__}"
