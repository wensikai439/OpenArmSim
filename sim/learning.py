"""Local, allowlisted learning jobs with persisted state and process ownership."""
from __future__ import annotations

import copy
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"sim.dataset", "sim.train", "sim.evaluate", "sim.calibration", "sim.pi05_server", "sim.timing_study", "sim.wrist"}
ACTIVE = {"starting", "running", "stopping"}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime(module, arguments):
    if module not in MODULES:
        raise ValueError("learning module is not allowlisted")
    if module in {"sim.calibration", "sim.timing_study", "sim.wrist"}:
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH") if k in os.environ}
        env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl")
        return [str(ROOT / ".venv/bin/python"), "-B", "-m", module, *map(str, arguments)], env
    config = json.loads((ROOT / "sim/learning.json").read_text())
    python, source = Path(config["python"]), Path(config["source"])
    if not python.is_file() or not source.is_dir():
        raise ValueError("learning runtime unavailable; check sim/learning.json")
    revision = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    changes = subprocess.check_output(["git", "-C", str(source.parent), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=10)
    if revision != config["source_revision"] or changes.strip():
        raise ValueError("official learning runtime differs from its pinned clean revision")
    # Do not inherit LLM keys, SSH agents or unrelated service credentials.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES") if k in os.environ}
    env.update({"PYTHONPATH": str(source) + os.pathsep + str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    return [str(python), "-B", "-m", module, *map(str, arguments)], env


def terminate_owned(process):
    """Only a process group created by us, never PID recovery from an old file."""
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def run_module(module, arguments):
    command, env = runtime(module, arguments)
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        return process.wait(timeout=7200)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        terminate_owned(process)
        return 130


class LearningJobs:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.thread = None
        self.current = None
        self.cancel = threading.Event()
        existing = sorted((ROOT / "runs").glob("*/job.json"), reverse=True)
        self.previous_id = existing[0].parent.name if existing else None

    def run_path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", run_id):
            raise ValueError("use a run ID from list_experiments, not a path")
        path = (ROOT / "runs" / run_id).resolve()
        if path.parent != (ROOT / "runs").resolve() or not path.is_dir():
            raise ValueError("unknown local run")
        return path

    @property
    def busy(self):
        with self.lock:
            return bool(self.current and self.current["status"] in ACTIVE)

    def status(self, job_id=None):
        with self.lock:
            if self.current and (job_id is None or job_id == self.current["job_id"]):
                result = copy.deepcopy(self.current)
                metrics = ROOT / "runs" / result["job_id"] / "artifacts/metrics.jsonl"
                if metrics.is_file():
                    lines = metrics.read_text().splitlines()
                    if lines:
                        rows = []
                        for line in lines:
                            try:
                                rows.append(json.loads(line))
                            except ValueError:
                                pass
                        if rows:
                            stride = max(1, (len(rows) + 249) // 250)
                            result["metrics"] = rows[::stride]
                            if result["metrics"][-1]["step"] != rows[-1]["step"]:
                                result["metrics"].append(rows[-1])
                        try:
                            result["latest_metric"] = json.loads(lines[-1])
                        except ValueError:
                            pass  # A partial line is not a terminal process.
                return self._result_preview(result)
        if job_id is None:
            return self.status(self.previous_id) if self.previous_id else None
        job_path = self.run_path(job_id) / "job.json"
        if not job_path.is_file():
            raise ValueError("this run is not a learning job")
        result = json.loads(job_path.read_text())
        if result["status"] in ACTIVE:
            result = {**result, "status": "unknown", "note": "No live process handle in this server. Inspect before retrying; no automatic restart."}
        return self._result_preview(result)

    def _result_preview(self, result):
        folder = ROOT / "runs" / result["job_id"]
        if result.get("kind") == "wrist":
            if (folder / "progress.json").is_file():
                result["wrist_progress"] = json.loads((folder / "progress.json").read_text())
            if (folder / "wrist.json").is_file():
                study = json.loads((folder / "wrist.json").read_text())
                result["wrist_result"] = {"samples": len(study["samples"]), "calibrated": False,
                                          "report_url": "/runs/" + folder.name + "/index.html"}
        if result.get("kind") == "timing":
            progress = folder / "artifacts/progress.json"
            if progress.is_file():
                try:
                    data = json.loads(progress.read_text())
                    result["timing_progress"] = {k: data[k] for k in ("completed", "total")}
                except (ValueError, KeyError):
                    pass
        if result.get("kind") == "calibrate":
            progress = folder / "artifacts/progress.json"
            if progress.is_file():
                try:
                    result["calibration_progress"] = json.loads(progress.read_text())
                except ValueError:
                    pass
            study = folder / "artifacts/calibration.json"
            if study.is_file():
                data = json.loads(study.read_text())
                result["calibration_result"] = {k: data[k] for k in ("selected", "holdout_error_ratio", "at_parameter_bound", "applied_to_runtime")}
        if result.get("status") == "completed" and result.get("kind") == "evaluate":
            path = folder / "artifacts/evaluation.json"
            if path.is_file():
                evaluation = json.loads(path.read_text())
                result["offline_metrics"] = {k: evaluation[k] for k in ("frames", "joint_rmse_rad", "gripper_rmse_command_units")}
        plot = folder / ("artifacts/timing.png" if result.get("kind") == "timing" else "artifacts/calibration.png" if result.get("kind") == "calibrate" else "artifacts/learning.png")
        if plot.is_file():
            result["plot_url"] = "/runs/" + str(plot.relative_to(ROOT / "runs"))
        return result

    def catalogue(self):
        rows = []
        for path in sorted((ROOT / "runs").glob("*"), reverse=True):
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                if (path / "job.json").is_file():
                    job = self.status(path.name)
                    rows.append({"run_id": path.name, "kind": job["kind"], "status": job["status"], "scope": job["scope"]})
                elif (path / "manifest.json").is_file():
                    manifest = json.loads((path / "manifest.json").read_text())
                    summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").exists() else {}
                    rows.append({"run_id": path.name, "kind": manifest.get("kind", manifest.get("schema")), "status": summary.get("status", "incomplete"),
                                 "dataset_available": (path / "lerobot/validation.json").is_file(), "scope": manifest.get("scope", "simulation")})
            except (OSError, ValueError, KeyError):
                continue
            if len(rows) >= 60:
                break
        return rows

    def evidence(self, run_id):
        path = self.run_path(run_id)
        result = {"run_id": run_id, "privileged": True, "transfer_ready": False}
        for name in ("manifest", "summary", "validation"):
            p = path / (name + ".json")
            if p.exists():
                result[name] = json.loads(p.read_text())
        if (path / "job.json").exists():
            result["job"] = self.status(run_id)
            for name in ("summary", "validation", "evaluation", "calibration"):
                p = path / "artifacts" / (name + ".json")
                if p.exists():
                    result[name] = json.loads(p.read_text())
        return result

    def start(self, kind, source_run_id=None, *, purpose="diagnostic", recipe="act_smoke", steps=5, seed=42, training_job_id=None, frames=300):
        with self.lock:
            if self.busy:
                raise ValueError("a learning job is already running")
            if (self.status() or {}).get("status") == "unknown":
                raise ValueError("previous job liveness is unknown; inspect before starting another")
            source = self.run_path(source_run_id) if kind not in {"calibrate", "timing", "wrist"} else None
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("invalid seed")
            job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + kind + "-" + uuid.uuid4().hex[:6]
            folder = ROOT / "runs" / job_id
            output = folder / "artifacts"
            scope = "pipeline_validation_only" if recipe == "act_smoke" else "candidate_requires_evaluation"
            if kind == "calibrate":
                module, arguments = "sim.calibration", ["--output", output]
                result_path, budget, scope = output / "calibration.json", 660, "heldout_controller_study"
            elif kind == "wrist":
                module, arguments = "sim.wrist", ["--output", folder]
                result_path, budget, scope = folder / "wrist.json", 420, "development_wrist_projection_candidate"
            elif kind == "timing":
                module, arguments = "sim.timing_study", ["--output", output]
                result_path, budget, scope = output / "summary.json", 600, "recorded_command_delay_diagnostic"
            elif kind == "export":
                if purpose not in {"diagnostic", "imitation"}:
                    raise ValueError("invalid dataset purpose")
                module, arguments = "sim.dataset", [source, "--purpose", purpose]
                result_path = source / "lerobot/validation.json"
                budget, scope = 1800, purpose
            elif kind == "train":
                if recipe not in {"act_smoke", "act_baseline"} or type(steps) is not int or not 1 <= steps <= (10 if recipe == "act_smoke" else 10000):
                    raise ValueError("invalid training recipe or step budget")
                if not (source / "lerobot/validation.json").is_file():
                    raise ValueError("export and validate this run before training")
                provenance = json.loads((source / "lerobot/provenance.json").read_text())
                if recipe != "act_smoke" and not provenance.get("eligible_as_imitation_demonstration"):
                    raise ValueError("diagnostic data only supports the smoke recipe")
                module, arguments = "sim.train", [source / "lerobot", "--recipe", recipe, "--steps", steps, "--seed", seed, "--output", output]
                result_path, budget = output / "summary.json", (300 if recipe == "act_smoke" else 7200)
            elif kind == "evaluate":
                if type(frames) is not int or not 1 <= frames <= 3000:
                    raise ValueError("evaluation frame budget must be 1..3000")
                trained = self.run_path(training_job_id)
                job = self.status(training_job_id)
                if job["kind"] != "train" or job["status"] != "completed":
                    raise ValueError("training job is not completed")
                module, arguments = "sim.evaluate", [trained / "artifacts", source / "lerobot", "--frames", frames, "--output", output]
                result_path, budget, scope = output / "evaluation.json", 900, "offline_action_prediction_only"
            else:
                raise ValueError("unsupported learning job")
            command, env = runtime(module, arguments)
            folder.mkdir(parents=True, exist_ok=False)
            self.cancel.clear()
            self.current = {"job_id": job_id, "kind": kind, "status": "starting", "source_run_id": source_run_id,
                            "training_job_id": training_job_id, "scope": scope, "recipe": recipe if kind == "train" else None,
                            "steps": steps if kind == "train" else None, "seed": seed, "wall_budget_seconds": budget,
                            "created_at": time.time(), "transfer_ready": False, "task_performance_measured": False,
                            "result_url": "/runs/" + str(result_path.relative_to(ROOT / "runs")), "error": None}
            write_json(folder / "job.json", self.current)
            self.thread = threading.Thread(target=self._work, args=(folder, command, env, budget, result_path), name="openarm-learning", daemon=True)
            self.thread.start()
            return copy.deepcopy(self.current)

    def _work(self, folder, command, env, budget, result_path):
        started = time.monotonic()
        outcome, error, code, process = "failed", None, None, None
        try:
            with (folder / "worker.log").open("w") as log:
                with self.lock:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.process = process
                    self.current["status"] = "running"
                    write_json(folder / "job.json", self.current)
                outcome = None
                while process.poll() is None:
                    if self.cancel.wait(.2):
                        outcome = "cancelled"
                        break
                    if time.monotonic() - started > budget:
                        outcome = "timed_out"
                        break
                if outcome:
                    terminate_owned(process)
                code = process.wait()
                if outcome is None:
                    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                    okay = code == 0 and (result.get("result") == "PASS" or result.get("status") in {"validation_passed", "training_completed", "evaluated", "studied"}
                                          or (self.current["kind"] == "timing" and result.get("status") == "completed" and len(result.get("trials", [])) == 9)
                                          or (self.current["kind"] == "wrist" and result.get("status") == "completed" and len(result.get("samples", [])) == 24 and result.get("applied") is False))
                    outcome = "completed" if okay else "failed"
                error = None if outcome in {"completed", "cancelled"} else "Job did not pass; inspect this job's worker.log and result."
        except Exception as exc:
            if process:
                terminate_owned(process)
            outcome, error = "failed", type(exc).__name__ + ": " + str(exc)
        finally:
            with self.lock:
                self.current.update(status=outcome, exit_code=code, elapsed_seconds=time.monotonic()-started,
                                    error=error, finished_at=time.time())
                write_json(folder / "job.json", self.current)
                write_json(folder / "summary.json", self.current)
                self.process = None

    def stop(self, job_id):
        with self.lock:
            if not self.current or self.current["job_id"] != job_id:
                raise ValueError("cannot cancel a job without its live owned process handle")
            if self.current["status"] not in ACTIVE:
                return self.status(job_id)
            self.cancel.set()
            self.current["status"] = "stopping"
            return self.status(job_id)

    def close(self):
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=12)
