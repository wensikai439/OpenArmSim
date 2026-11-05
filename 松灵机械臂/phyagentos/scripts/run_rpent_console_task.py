#!/usr/bin/env python3
"""Run one recorded RPent console task; never resubmit an uncertain request."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("label", help="unique run label")
parser.add_argument("--prompt", default="runs/rpent-forge/task-enforce.txt")
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label):
    parser.error("label must use letters, digits, '-' or '_'")
out = root / "runs" / args.label
out.mkdir(exist_ok=False)
base = "http://127.0.0.1:19081"
session = "console:" + args.label

def save(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        save("http-error.json", {"path": path, "status": error.code, "body": error.read().decode(errors="replace")})
        raise

state = call("/api/state")
assert state["skill"] == "rpent-piper-sim" and state["profile"] == "rpent" and state["agent_mode"] == "simulation"
assert not any(state.get(key) for key in ("job", "active_task", "reflection_busy", "background_busy", "control_busy"))
save("before-state.json", state)
if args.reset:
    result = call("/api/runtime", {"action": "reset"})
    save("reset.json", result)
    assert result.get("completed")
body = {"session": session, "request_id": args.label, "content": (root / args.prompt).read_text()}
save("request.json", body)
save("accepted.json", call("/api/message", body))
print("Submitted once:", session, flush=True)
last = None
for _ in range(180):
    state = call("/api/state")
    tasks = [task for task in state["tasks"] if task.get("session") == session]
    progress = [(task["task_id"], task["status"]) for task in tasks]
    if progress != last:
        print(progress, flush=True)
        last = progress
    if not state["job"] and not state["reflection_busy"]:
        save("final-state.json", state)
        save("session.json", call("/api/session?key=" + session))
        save("experience.json", call("/api/experience?skill=rpent-piper-sim"))
        for task in tasks:
            save(task["task_id"] + ".json", call("/api/tasks/" + task["task_id"]))
        print("Evidence:", out, flush=True)
        if not tasks or any(task["status"] != "succeeded" for task in tasks):
            raise SystemExit("Task did not succeed; inspect evidence. No automatic replay.")
        break
    time.sleep(5)
else:
    save("timeout-state.json", state)
    raise SystemExit("Wait expired; request may still be running. Inspect it before any new action.")
