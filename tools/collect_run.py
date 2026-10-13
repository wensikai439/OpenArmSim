"""Read-only collection of one OpenArm harness run.

This tool never submits, cancels, resets or resumes a task. It reads the local
workbench API and SQLite journal, and stores a compact, linked experiment record.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEFAULT_DB = ROOT / "harness/state/objects.sqlite3"
REQUIRED_TABLES = {"turns", "trace", "plans", "experiences"}
REQUIRED_COLUMNS = {
    "turns": {"id", "parent", "role", "epoch", "prompt", "answer", "status", "model", "vision", "created"},
    "trace": {"id", "turn_id", "kind", "name", "status", "detail", "started", "ended"},
    "plans": {"turn_id", "body"},
    "experiences": {"id", "role", "epoch", "tool", "arguments", "evidence_kind", "feedback", "evidence", "outcome"},
}
MOTION_TOOLS = {"move_to_pose": "absolute", "move_relative": "relative"}


def inside(path, parent, label):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"{label} must be inside {parent}")
    return path


def validate_database(path):
    path = inside(path, ROOT, "database")
    if not path.is_file():
        raise ValueError(f"database does not exist: {path.relative_to(ROOT)}")
    uri = "file:" + path.as_posix() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = REQUIRED_TABLES - tables
        if missing:
            raise ValueError(f"database schema is missing tables: {sorted(missing)}")
        for table, required in REQUIRED_COLUMNS.items():
            columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
            if not required <= columns:
                raise ValueError(f"{table} schema is missing columns: {sorted(required-columns)}")
    return path


def fetch_json(port, route, timeout=3):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/{route}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def task_rows(db, root_id):
    rows = db.execute("""
        WITH RECURSIVE tree(id, depth) AS (
            SELECT id, 0 FROM turns WHERE id=?
            UNION ALL
            SELECT t.id, tree.depth+1 FROM turns t JOIN tree ON t.parent=tree.id
        )
        SELECT turns.* FROM turns JOIN tree ON tree.id=turns.id ORDER BY turns.created, tree.depth
    """, (root_id,)).fetchall()
    return [dict(r) for r in rows]


def decode_json(value, default):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def read_task(database, root_id):
    uri = "file:" + database.as_posix() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2) as db:
        db.row_factory = sqlite3.Row
        turns = task_rows(db, root_id)
        if not turns or turns[0]["id"] != root_id or turns[0]["parent"] is not None:
            raise ValueError("turn_id must identify an existing root task")
        ids = [t["id"] for t in turns]
        marks = ",".join("?" for _ in ids)
        traces = [dict(r) for r in db.execute(
            f"SELECT * FROM trace WHERE turn_id IN ({marks}) ORDER BY id", ids).fetchall()]
        for event in traces:
            event["detail"] = decode_json(event.pop("detail"), {})
        plan_row = db.execute("SELECT body FROM plans WHERE turn_id=?", (root_id,)).fetchone()
        plan = decode_json(plan_row[0], {}) if plan_row else None
        execution_ids = []
        for execution in (plan or {}).get("execution_tools", []):
            ident = execution.get("id")
            if isinstance(ident, str) and ident not in execution_ids:
                execution_ids.append(ident)
        receipts = []
        if execution_ids:
            placeholders = ",".join("?" for _ in execution_ids)
            rows = db.execute(f"SELECT * FROM experiences WHERE id IN ({placeholders})", execution_ids).fetchall()
            by_id = {r["id"]: dict(r) for r in rows}
            for ident in execution_ids:
                row = by_id.get(ident)
                if row is None:
                    receipts.append({"execution_id": ident, "missing": True})
                    continue
                for key in ("arguments", "feedback", "evidence", "context"):
                    row[key] = decode_json(row.get(key), {})
                receipts.append(row)
    return turns, traces, plan, receipts


def event_summary(event):
    detail = event["detail"]
    elapsed = None
    if isinstance(event.get("started"), (int, float)) and isinstance(event.get("ended"), (int, float)):
        elapsed = max(0.0, event["ended"]-event["started"])
    result = {"turn_id": event["turn_id"], "kind": event["kind"], "name": event["name"],
              "status": event["status"], "wall_seconds": elapsed}
    if event["kind"] == "api":
        result.update({k: detail.get(k) for k in ("model", "response_model", "request_number", "tool_names", "usage") if k in detail})
    elif event["kind"] == "tool":
        args = detail.get("arguments") or detail.get("dispatch_arguments") or {}
        result["arguments"] = args
        tool_result = detail.get("result") or {}
        result["execution_id"] = tool_result.get("execution_id") if isinstance(tool_result, dict) else None
        result["error"] = tool_result.get("error") if isinstance(tool_result, dict) else None
    elif event["kind"] == "timing":
        keep = ("mode", "method", "waiting_action", "status", "run_id", "run_path", "start_sim_seconds",
                "end_sim_seconds", "integrated_seconds", "wall_seconds", "task_failed", "error")
        result["wait"] = {k: detail[k] for k in keep if k in detail}
    return result


def motion_modes(events, receipts):
    names = {e["name"] for e in events if e["kind"] == "tool"}
    names.update(r.get("tool") for r in receipts if r.get("tool"))
    return sorted({MOTION_TOOLS[name] for name in names if name in MOTION_TOOLS})


def summarize(root_id, turns, traces, plan, receipts, service, database):
    api_events = [e for e in traces if e["kind"] == "api"]
    usage = {}
    for event in api_events:
        for key, value in (event["detail"].get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    api_seconds = sum(max(0.0, (e.get("ended") or e["started"])-e["started"]) for e in api_events)
    tools = [event_summary(e) for e in traces if e["kind"] in {"api", "tool", "timing"}]
    motion = [e for e in tools if e["kind"] == "tool" and e["name"] in MOTION_TOOLS]
    receipt_rows = []
    for row in receipts:
        if row.get("missing"):
            receipt_rows.append(row)
            continue
        feedback = row.get("feedback") or {}
        evidence = row.get("evidence") or {}
        run_id = evidence.get("run_id")
        run_dir = (RUNS / run_id).resolve() if isinstance(run_id, str) else None
        manifest = run_dir / "manifest.json" if run_dir and run_dir.is_relative_to(RUNS.resolve()) else None
        receipt_rows.append({
            "execution_id": row["id"], "skill": row.get("tool"), "arguments": row.get("arguments"),
            "outcome": row.get("outcome"), "evidence_kind": row.get("evidence_kind"),
            "primitive_success": feedback.get("primitive_success"),
            "verification_source": feedback.get("verification_source"),
            "failure_code": feedback.get("failure_code") or feedback.get("error_code"),
            "message": feedback.get("message") or feedback.get("failure_reason"),
            "planner": feedback.get("planner"), "scene_source": feedback.get("scene_source"),
            "object_id": feedback.get("object_id"), "arm": feedback.get("arm") or feedback.get("held_by"),
            "position_error_m": feedback.get("position_error_m"), "run_id": run_id,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest and manifest.is_file() else None,
            "summary_url": evidence.get("summary_url"),
            "raw_record": f"SQLite experiences.id={row['id']}",
        })
    stages = [{k: step.get(k) for k in ("id", "title", "skill", "status", "checkpoint") if k in step}
              for step in (plan or {}).get("steps", [])]
    finish = next((e for e in reversed(traces) if e["kind"] == "tool" and e["name"] == "finish"), None)
    finish_args = (finish or {}).get("detail", {}).get("arguments", {})
    independent = (plan or {}).get("goal_assessment") or {}
    observation = service.get("observation") or {}
    job = service.get("job")
    last_after = next(((r.get("evidence") or {}).get("after") for r in reversed(receipts)
                       if isinstance((r.get("evidence") or {}).get("after"), dict)), {})
    root = turns[0]
    native_waits = [e["wait"] for e in tools if e["kind"] == "timing" and e.get("wait", {}).get("run_id")]
    return {
        "schema": "openarm-run-record/2", "turn_id": root_id,
        "task": {"goal": root["prompt"], "model": root["model"], "vision": bool(root["vision"]),
                 "source": (plan or {}).get("source"), "declared_verdict": (plan or {}).get("verdict"),
                 "finish_verdict": finish_args.get("verdict"), "finish_reason": finish_args.get("reason"),
                 "independent_goal_assessment": independent,
                 "terminal_observation_id": finish_args.get("observation_id") or last_after.get("observation_id"),
                 "terminal_sim_time_s": last_after.get("sim_time"),
                 "time_budget_s": (plan or {}).get("time_budget_s"), "timing_mode": (plan or {}).get("timing_mode"),
                 "placement_goal": (plan or {}).get("placement_goal"),
                 "object_goals": (plan or {}).get("object_goals"),
                 "start_grasp_admission": (plan or {}).get("start_grasp_admission"),
                 "stages": stages, "stage_count": len(stages),
                 "stages_done": sum(step.get("status") == "done" for step in stages)},
        "result": {"turn_statuses": {t["id"]: t["status"] for t in turns},
                   "all_turns_terminal": bool(turns) and all(t["status"] not in ("running", "waiting") for t in turns),
                   "api_calls": len(api_events), "api_wall_seconds": round(api_seconds, 6), "token_usage": usage,
                   "estimated_cost": None, "cost_note": "provider pricing is not configured; token counts are recorded",
                   "motion_modes": motion_modes(traces, receipts),
                   "absolute_moves": sum(e["name"] == "move_to_pose" for e in motion),
                   "relative_moves": sum(e["name"] == "move_relative" for e in motion),
                   "native_waits": native_waits,
                   "skill_successes": sum(r.get("primitive_success") is True for r in receipt_rows),
                   "skill_failures": sum(r.get("primitive_success") is False for r in receipt_rows),
                   "skill_outcomes_unknown": sum(r.get("primitive_success") is None for r in receipt_rows),
                   "missing_receipts": sum(r.get("missing") is True for r in receipt_rows)},
        "events": tools, "receipts": receipt_rows,
        "live_service_at_collection": {"state": service.get("state"), "backend": service.get("backend"),
            "observation_id": service.get("observation_id"), "sim_time": observation.get("sim_time"),
            "task": service.get("task"), "active_job": ({k: job.get(k) for k in ("execution_id", "kind", "state") if k in job}
                if isinstance(job, dict) else None), "transfer_ready": service.get("transfer_ready")},
        "source": {"collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "database": str(database.relative_to(ROOT)), "raw_robot_transitions": "runs/<run_id>/transitions.jsonl",
                   "credential_values_saved": False},
        "hardware_access": False, "transfer_ready": False,
    }


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn-id", required=True, help="existing root task id; collection does not start a task")
    parser.add_argument("--output", required=True, help="new experiment directory inside runs/")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--database", default=str(DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--timeout-s", type=float, default=600, help="read-only wait for this task to finish")
    parser.add_argument("--poll-s", type=float, default=.5)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", args.turn_id):
        raise ValueError("turn-id must be 8-64 letters, digits, '_' or '-'")
    if not 1 <= args.port <= 65535 or not 0 <= args.timeout_s <= 3600 or not .1 <= args.poll_s <= 10:
        raise ValueError("port, timeout or polling interval is outside its safe range")
    database = validate_database(ROOT / args.database)
    output = inside(args.output, RUNS, "output")
    if output == RUNS.resolve() or output.exists():
        raise ValueError("output must be a new directory inside runs/")
    turns, _, _, _ = read_task(database, args.turn_id)  # Fail before creating output on a typo.
    if not turns:
        raise ValueError("no matching task was found")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    previous = None
    try:
        while True:
            service = fetch_json(args.port, "api/status")
            health = fetch_json(args.port, "health")
            turns, traces, plan, receipts = read_task(database, args.turn_id)
            live_plan = (service.get("agent") or {}).get("plan") or {}
            owns_live = live_plan.get("turn_id") == args.turn_id
            active = any(t["status"] in ("running", "waiting") for t in turns)
            if owns_live:
                active = active or bool(service.get("job")) or bool((service.get("agent") or {}).get("busy"))
            state = summarize(args.turn_id, turns, traces,
                              (live_plan if owns_live else plan), receipts, service, database)
            state["live_service_at_collection"]["health"] = health
            revision = ((service.get("agent") or {}).get("revision"), service.get("observation_id"),
                        len(traces), tuple(state["result"]["turn_statuses"].values()))
            if revision != previous:
                write_json(output / "progress.json", {"turn_id": args.turn_id, "elapsed_s": round(time.monotonic()-started, 3),
                                                        "active": active, "task_statuses": state["result"]["turn_statuses"],
                                                        "api_calls": state["result"]["api_calls"],
                                                        "native_receipts": len(receipts),
                                                        "observation_id": service.get("observation_id")})
                previous = revision
            if not active and state["result"]["all_turns_terminal"]:
                write_json(output / "result.json", state)
                print(json.dumps({"collected": True, "turn_id": args.turn_id, "api_calls": state["result"]["api_calls"],
                                  "motion_modes": state["result"]["motion_modes"],
                                  "skill_successes": state["result"]["skill_successes"],
                                  "skill_failures": state["result"]["skill_failures"],
                                  "output": str(output.relative_to(ROOT))}, ensure_ascii=False))
                return
            if time.monotonic()-started >= args.timeout_s:
                write_json(output / "collector-timeout.json", {"turn_id": args.turn_id, "active": active,
                    "task_was_not_stopped": True, "latest_observation_id": service.get("observation_id"),
                    "latest_task_statuses": state["result"]["turn_statuses"]})
                raise TimeoutError("collector timed out; task was not stopped or restarted")
            time.sleep(args.poll_s)
    except Exception as exc:
        if not (output / "collector-timeout.json").exists():
            write_json(output / "collector-error.json", {"turn_id": args.turn_id, "error": type(exc).__name__ + ": " + str(exc),
                "task_was_not_stopped": True})
        raise


if __name__ == "__main__":
    main()
