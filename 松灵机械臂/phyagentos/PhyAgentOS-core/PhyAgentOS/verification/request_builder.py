"""Build a model request from Forge public contracts and immutable artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PhyAgentOS.verification.contracts import (
    EvidenceBundle,
    ForgeSessionRecord,
    VerificationEvidencePolicy,
)

if TYPE_CHECKING:
    from PhyAgentOS.forge.task import AgentTaskRecord


class VerificationEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class VerificationRequest:
    content: list[dict[str, Any]]
    artifact_paths: tuple[Path, ...]
    valid_evidence_refs: frozenset[str]
    evidence: EvidenceBundle


@dataclass(frozen=True)
class _ValidatedEvidence:
    evidence: EvidenceBundle
    artifact_paths: tuple[Path, ...]
    images: tuple[tuple[str, str, bytes], ...]
    structured: dict[str, Any]
    artifact_ids: frozenset[str]


class VerificationRequestBuilder:
    def __init__(self, workspace: str | Path, *, max_image_bytes: int = 16 * 1024 * 1024):
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_image_bytes = max(1, int(max_image_bytes))

    def build(
        self,
        record: ForgeSessionRecord,
        *,
        history: list[dict[str, Any]],
        lessons: str,
    ) -> VerificationRequest:
        reference = record.verification.bundle_ref
        if not reference:
            raise VerificationEvidenceError("Forge session has no Evidence Bundle")
        evidence_path, evidence = self._load_evidence(
            reference,
            expected_session_id=record.session_id,
            expected_command_id=record.command_id,
            identity_name="session",
        )
        if record.execution is None:
            raise VerificationEvidenceError("Forge session has no Execution Record")
        validated = self._validate_evidence(
            evidence_path,
            evidence,
            policy=record.request.verification.evidence_policy,
        )

        context = {
            "task_verification_contract": record.request.verification.model_dump(mode="json"),
            "execution_record": record.execution.model_dump(mode="json"),
            "evidence_bundle": validated.evidence.model_dump(mode="json"),
            "structured_evidence": validated.structured,
            "lineage_history": history,
            "lessons": lessons,
            "valid_evidence_refs": sorted(validated.artifact_ids),
        }
        return self._build_request(
            context=context,
            validated=validated,
            valid_evidence_refs=validated.artifact_ids,
        )

    def build_agent_task(
        self,
        task: AgentTaskRecord,
        *,
        events: list[dict[str, Any]],
        lessons: str,
    ) -> VerificationRequest:
        """Build one strict request for the AgentTask aggregate and its evidence."""
        reference = task.evidence_bundle_ref
        if not reference:
            raise VerificationEvidenceError("AgentTask has no Evidence Bundle")
        if not task.execution_records:
            raise VerificationEvidenceError("AgentTask has no ToolExecutionRecords")
        evidence_path, evidence = self._load_evidence(
            reference,
            expected_session_id=task.task_id,
            expected_command_id="agent_task",
            identity_name="AgentTask",
        )
        validated = self._validate_evidence(
            evidence_path,
            evidence,
            policy=task.verification.evidence_policy,
        )
        execution_refs = self._validate_agent_task_lineage(task)
        overlap = validated.artifact_ids.intersection(execution_refs)
        if overlap:
            raise VerificationEvidenceError(
                "evidence references collide across execution facts and artifacts: "
                + ", ".join(sorted(overlap))
            )
        valid_refs = validated.artifact_ids | execution_refs
        records = task.execution_records
        context = {
            "agent_task": {
                "task_id": task.task_id,
                "task_description": task.task_description,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "runtime_snapshot_ref": task.runtime_snapshot_ref,
            },
            "goal": task.verification.goal,
            "criteria": task.verification.success_criteria,
            "constraints": task.verification.constraints,
            "task_verification_contract": task.verification.model_dump(mode="json"),
            "frozen_skill_binding": (
                task.primary_skill_binding.model_dump(mode="json")
                if task.primary_skill_binding is not None
                else None
            ),
            "supporting_skill_bindings": [
                item.model_dump(mode="json") for item in task.supporting_skill_bindings
            ],
            "plan_revisions": [
                {
                    **item.model_dump(mode="json", exclude={"execution_records"}),
                    "execution_record_ids": [record.record_id for record in item.execution_records],
                }
                for item in task.revisions
            ],
            "tool_execution_records": [item.model_dump(mode="json") for item in records],
            "gateway_terminal_results": [
                {
                    "record_id": item.record_id,
                    "tool_id": item.tool_id,
                    "semantics": item.semantics,
                    "status": item.status,
                    "invocation_id": item.invocation_id,
                    "attempt_id": item.attempt_id,
                    # The full response is retained once in tool_execution_records.
                    "response_record_id": item.record_id,
                    "error": item.error,
                    "evidence_refs": item.evidence_refs,
                }
                for item in records
                if item.terminal
            ],
            "evidence_bundle": validated.evidence.model_dump(mode="json"),
            "structured_evidence": validated.structured,
            "evidence_errors": task.evidence_errors,
            "events": events,
            "lessons": lessons,
            "valid_evidence_refs": sorted(valid_refs),
        }
        # A long SIM task can contain hundreds of Queries and Actions. Present
        # each real Action's terminal facts together without dropping its receipt.
        # These are extracted facts, not a verdict or a substitute for evidence.
        action_facts = []
        for record in context["tool_execution_records"]:
            if record["semantics"] != "action":
                continue
            native = (record.get("response") or {}).get("data", {}).get("result", {}).get("outputs", {}).get("native_result", {})
            action_facts.append({
                "record_id": record["record_id"],
                "tool_id": record["tool_id"],
                "status": record["status"],
                "invocation_id": record["invocation_id"],
                "primitive_completed": native.get("primitive_completed"),
                "verified": native.get("verified"),
                "error": record.get("error") or native.get("error"),
                "evidence_refs": record["evidence_refs"],
            })
        context["action_terminal_facts"] = {
            "source": "exact per-Action extraction from tool_execution_records; full receipts retained",
            "action_count": len(action_facts),
            "records": action_facts,
        }
        # Native observations repeat the complete cumulative SIM action history.
        # Individual Task execution records retain each Action's actual receipt;
        # remove only this duplicate list from the model request, never the store.
        if "composable_actions" not in json.dumps(task.verification.model_dump(mode="json")):
            def omit_history(value):
                if isinstance(value, dict):
                    return {key: omit_history(item) for key, item in value.items()
                            if not (key == "composable_actions" and isinstance(item, list))}
                if isinstance(value, list):
                    return [omit_history(item) for item in value]
                return value
            context = omit_history(context)
            context["request_projection"] = {
                "omitted_field": "repeated native observation.composable_actions",
                "reason": "Cumulative history duplicates individually retained ToolExecutionRecords.",
                "raw_Task_and_native_artifacts_retained": True,
                "all_individual_records_and_other_observation_fields_retained": True,
            }
        # This fixed joint handoff contract asks for terminal Action facts,
        # one measured transfer, and the captured final state. Repeating every
        # full SIM observation obscures time/phase and can exceed a million tokens.
        # Preserve every record, argument, error and evidence reference; project
        # only repeated successful SIM response state in the model request.
        joint_criteria = [
            "All same-task native Actions complete with primitive_completed=true and verified=true; no failed, unknown or unresolved Action.",
            "A new same-episode composable_transfers entry has donor=right, receiver=left, at least two receiver contact geoms, qpos_change=0, table_contact=false, and bottle base z>0.78m.",
            "Final same-task observe reports bucket_bottom_contact=true, bucket_xy_error_m<0.055, bottle_speed_norm<0.05, both assist_active=false, failed=false, collision_stop=null, and numerical_warnings all zero.",
            "Final physics_step and sim_time_s advance from initial same-task observe with actual before/after robot state and both camera evidence.",
        ]
        binding = context.get("frozen_skill_binding") or {}
        if (binding.get("skill_name") == "rpent-piper-sim"
                and binding.get("runtime_profile") == "joint"
                and task.verification.success_criteria == joint_criteria):
            state_fields = (
                "episode_id", "control_step", "physics_step", "sim_time_s",
                "assist_active", "finger_contact_now", "gripper_command",
                "gripper_command_mode", "binary_gripper_commands",
                "gripper_feedback", "bottle_world_m", "bucket_world_m",
                "composable_transfers", "bucket_bottom_contact", "bucket_xy_error_m",
                "bottle_speed_norm", "bottle_worktable_contact", "failed", "reset_required", "collision_stop",
                "numerical_warnings", "task_success", "handoff_verified",
                "air_handoff_verified", "controller_mode", "b2_frame_bindings",
            )
            projected = []
            for record in context["tool_execution_records"]:
                item = dict(record)
                response = record.get("response") or {}
                data = response.get("data") or {}
                if record["status"] == "succeeded":
                    if record["semantics"] == "action":
                        result = data.get("result") or {}
                        outputs = result.get("outputs") or {}
                        native = outputs.get("native_result") or {}
                        relay = native.get("relay_state")
                        if isinstance(relay, dict):
                            item["response"] = {**response, "data": {
                                **data, "result": {**result, "outputs": {
                                    **outputs, "native_result": {
                                        **native, "relay_state": {
                                            key: relay[key] for key in state_fields if key in relay
                                        }
                                    }
                                }}
                            }}
                    elif record["tool_id"] == "rpent.observe":
                        query = data.get("response") or {}
                        result = query.get("result") or {}
                        outputs = result.get("outputs") or {}
                        observation = outputs.get("observation")
                        if isinstance(observation, dict):
                            item["response"] = {**response, "data": {
                                **data, "response": {**query, "result": {
                                    **result, "outputs": {
                                        **outputs, "observation": {
                                            key: observation[key] for key in state_fields if key in observation
                                        }
                                    }
                                }}
                            }}
                projected.append(item)
            context["tool_execution_records"] = projected
            context["request_projection"]["joint_handoff_state_fields"] = list(state_fields)
            context["request_projection"]["scope"] = "exact four-criterion joint handoff contract only"
            context["request_projection"]["raw_task_store_unchanged"] = True
        return self._build_request(
            context=context,
            validated=validated,
            valid_evidence_refs=valid_refs,
        )

    def _load_evidence(
        self,
        reference: str,
        *,
        expected_session_id: str,
        expected_command_id: str,
        identity_name: str,
    ) -> tuple[Path, EvidenceBundle]:
        evidence_path = self._workspace_path(reference)
        try:
            evidence = EvidenceBundle.model_validate_json(
                evidence_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise VerificationEvidenceError(f"invalid Evidence Bundle: {reference}") from exc
        if (
            evidence.session_id != expected_session_id
            or evidence.command_id != expected_command_id
        ):
            raise VerificationEvidenceError(
                f"Evidence Bundle identity does not match {identity_name}"
            )
        return evidence_path, evidence

    def _validate_evidence(
        self,
        evidence_path: Path,
        evidence: EvidenceBundle,
        *,
        policy: VerificationEvidencePolicy,
    ) -> _ValidatedEvidence:
        if (
            policy.minimum_association == "authoritative"
            and evidence.quality.association_quality != "authoritative"
        ):
            raise VerificationEvidenceError("evidence association is below task policy")
        if not evidence.quality.complete:
            raise VerificationEvidenceError(
                "Evidence Bundle is incomplete: "
                + ", ".join(evidence.quality.missing_requirements or ["unknown"])
            )
        self._validate_capture_window(evidence)
        self._validate_requirements(policy, evidence)

        paths: list[Path] = [evidence_path]
        images: list[tuple[str, str, bytes]] = []
        structured: dict[str, Any] = {}
        artifact_ids: set[str] = set()
        for artifact in evidence.artifacts:
            if artifact.artifact_id in artifact_ids:
                raise VerificationEvidenceError("evidence artifact IDs must be unique")
            artifact_ids.add(artifact.artifact_id)
            if not artifact.retained:
                raise VerificationEvidenceError(
                    f"required artifact was removed by retention: {artifact.artifact_id}"
                )
            path = self._workspace_path(artifact.uri)
            if not path.is_file():
                raise VerificationEvidenceError(f"evidence artifact is missing: {artifact.uri}")
            data = path.read_bytes()
            if len(data) != artifact.byte_size:
                raise VerificationEvidenceError(
                    f"evidence artifact size mismatch: {artifact.artifact_id}"
                )
            if hashlib.sha256(data).hexdigest() != artifact.sha256:
                raise VerificationEvidenceError(
                    f"evidence artifact digest mismatch: {artifact.artifact_id}"
                )
            paths.append(path)
            if artifact.media_type.startswith("image/"):
                if not data or len(data) > self.max_image_bytes:
                    raise VerificationEvidenceError(
                        f"verification image exceeds size limit: {artifact.artifact_id}"
                    )
                if not self._matches_media_type(data, artifact.media_type):
                    raise VerificationEvidenceError(
                        f"verification image media type mismatch: {artifact.artifact_id}"
                    )
                images.append((artifact.artifact_id, artifact.media_type, data))
            elif artifact.media_type == "application/json":
                try:
                    structured[artifact.artifact_id] = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise VerificationEvidenceError(
                        f"verification JSON is invalid: {artifact.artifact_id}"
                    ) from exc
        return _ValidatedEvidence(
            evidence=evidence,
            artifact_paths=tuple(paths),
            images=tuple(images),
            structured=structured,
            artifact_ids=frozenset(artifact_ids),
        )

    @staticmethod
    def _validate_agent_task_lineage(task: AgentTaskRecord) -> frozenset[str]:
        expected_binding_id = (
            task.primary_skill_binding.binding_id
            if task.primary_skill_binding is not None
            else None
        )
        revision_ids: set[str] = set()
        record_ids: set[str] = set()
        evidence_refs: set[str] = set()
        for revision in task.revisions:
            if revision.revision_id in revision_ids:
                raise VerificationEvidenceError("AgentTask PlanRevision IDs must be unique")
            revision_ids.add(revision.revision_id)
            if revision.skill_binding_id != expected_binding_id:
                raise VerificationEvidenceError(
                    f"PlanRevision binding does not match frozen AgentTask binding: "
                    f"{revision.revision_id}"
                )
            for record in revision.execution_records:
                if record.record_id in record_ids:
                    raise VerificationEvidenceError(
                        "AgentTask ToolExecutionRecord IDs must be unique"
                    )
                record_ids.add(record.record_id)
                if record.revision_id != revision.revision_id:
                    raise VerificationEvidenceError(
                        f"ToolExecutionRecord revision mismatch: {record.record_id}"
                    )
                if record.skill_binding_id != expected_binding_id:
                    raise VerificationEvidenceError(
                        f"ToolExecutionRecord binding mismatch: {record.record_id}"
                    )
                for reference in record.evidence_refs:
                    if not reference.strip():
                        raise VerificationEvidenceError(
                            f"ToolExecutionRecord has a blank evidence reference: "
                            f"{record.record_id}"
                        )
                    evidence_refs.add(reference)
        return frozenset(evidence_refs)

    def _build_request(
        self,
        *,
        context: dict[str, Any],
        validated: _ValidatedEvidence,
        valid_evidence_refs: frozenset[str],
    ) -> VerificationRequest:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Determine whether every task success criterion is semantically satisfied. "
                    "Use only the supplied task contract, execution facts, and evidence to make "
                    "that decision. Lessons are advisory workflow context only; they are not "
                    "evidence and cannot establish any criterion status.\n\n"
                    + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
                ),
            }
        ]
        # Present exact captured SIM state beside the image blocks. Historical
        # contact observations must not be confused with the terminal snapshot.
        captured_facts = []
        fields = ("episode_id", "control_step", "physics_step", "sim_time_s",
                  "assist_active", "finger_contact_now", "gripper_command",
                  "gripper_feedback", "bottle_world_m", "bucket_world_m",
                  "bucket_bottom_contact", "bucket_xy_error_m", "bottle_speed_norm", "bottle_worktable_contact",
                  "failed", "reset_required", "collision_stop", "numerical_warnings",
                  "b2_frame_bindings", "composable_transfers", "gripper_command_mode",
                  "binary_gripper_commands", "controller_mode", "task_success",
                  "handoff_verified", "air_handoff_verified")
        artifacts = {item.artifact_id: item for item in validated.evidence.artifacts}
        for item in validated.evidence.artifacts:
            value = validated.structured.get(item.artifact_id)
            if not isinstance(value, dict):
                continue
            runtime = value.get("runtime")
            sim_status = runtime.get("sim_status") if isinstance(runtime, dict) else None
            observation = sim_status.get("observation") if isinstance(sim_status, dict) else None
            if isinstance(observation, dict):
                captured_facts.append({
                    "artifact_id": item.artifact_id, "phase": item.phase,
                    "source_id": item.source_id,
                    "observation": {key: observation[key] for key in fields if key in observation},
                })
        if captured_facts:
            content.append({"type": "text", "text": "CAPTURED_ROBOT_STATE_FACTS: "
                            "Exact field projections of the retained captured artifacts; "
                            "not a verdict. Full structured evidence remains above.\n"
                            + json.dumps(captured_facts, ensure_ascii=False, separators=(",", ":"))})
        for artifact_id, media_type, data in validated.images:
            content.extend(
                [
                    {"type": "text", "text": (
                        f"EVIDENCE_ARTIFACT: {artifact_id}; "
                        f"phase={artifacts[artifact_id].phase}; "
                        f"source_id={artifacts[artifact_id].source_id}; "
                        f"captured_at={artifacts[artifact_id].captured_at}"
                    )},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,"
                            + base64.b64encode(data).decode("ascii")
                        },
                    },
                ]
            )
        # Keep the schema/visual-consistency check adjacent to the actual images
        # for this fixed SIM contract; never derive or coerce a verdict in code.
        projection = context.get("request_projection") or {}
        if projection.get("scope") == "exact four-criterion joint handoff contract only":
            content.append({"type": "text", "text": (
                "FINAL RESPONSE CONSISTENCY CHECK: Assess every supplied criterion independently "
                "against the retained before/after artifacts and actual execution facts. "
                "A cropped or occluded image cannot by itself establish that a bottle is outside "
                "the bucket or on the table. A claimed visual contradiction must identify the "
                "specific artifact and the affected criterion; use unknown for that criterion "
                "and inconclusive if the supplied facts cannot resolve it. Check the captured "
                "bottle_worktable_contact and bucket_bottom_contact facts as well as the images. "
                "The overall verdict and reason must agree with the criterion statuses. "
                "The strict official schema rejects failure when every criterion is satisfied. "
                "Do not copy the stored Task status, do not invent object locations or references, "
                "and return exactly the schema required by the system prompt. This checklist is "
                "not a success assessment; all criteria still require the actual supplied evidence."
            )})
        return VerificationRequest(
            content,
            validated.artifact_paths,
            valid_evidence_refs,
            validated.evidence,
        )

    @staticmethod
    def _validate_requirements(
        policy: VerificationEvidencePolicy, evidence: EvidenceBundle
    ) -> None:
        for kind in policy.required_kinds:
            for phase in ("before", "after"):
                candidates = [
                    item
                    for item in evidence.artifacts
                    if item.kind == kind and item.phase == phase
                ]
                if not candidates:
                    raise VerificationEvidenceError(
                        f"required evidence is unavailable: {phase}:{kind}"
                    )
                if "image" in kind:
                    for source in policy.required_sources:
                        if not any(item.source_id == source for item in candidates):
                            raise VerificationEvidenceError(
                                f"required evidence source is unavailable: {phase}:{kind}:{source}"
                            )

    @staticmethod
    def _validate_capture_window(evidence: EvidenceBundle) -> None:
        window = evidence.capture_window
        if (
            window.before_command_at is None
            or window.command_terminal_at is None
            or window.after_command_at is None
        ):
            raise VerificationEvidenceError("evidence capture window is incomplete")
        if not (
            window.before_command_at <= window.command_terminal_at <= window.after_command_at
        ):
            raise VerificationEvidenceError("evidence capture window ordering is invalid")

    def _workspace_path(self, relative: str) -> Path:
        path = (self.workspace / relative).resolve()
        if not path.is_relative_to(self.workspace):
            raise VerificationEvidenceError(f"artifact path escapes workspace: {relative}")
        return path

    @staticmethod
    def _matches_media_type(data: bytes, media_type: str) -> bool:
        normalized = media_type.lower().split(";", 1)[0].strip()
        if normalized in {"image/jpeg", "image/jpg"}:
            return data.startswith(b"\xff\xd8\xff")
        if normalized == "image/png":
            return data.startswith(b"\x89PNG\r\n\x1a\n")
        if normalized == "image/webp":
            return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
        return False
