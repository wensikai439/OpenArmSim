"""Execute one accepted simulation job on its existing physics-owner thread.

The service owns the queue, rendering cache, cancellation events and lifetime.
This runner owns planning/action iteration and terminal evidence. It never
creates a second simulator, starts a worker thread or changes control clocks.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from queue import Queue
from threading import Event
import time
from typing import Any, Callable, Protocol
import uuid

from harness.execution_contracts import (
    ExecutionKind, completion_record, execution_lifecycle, planner_provenance,
)
from .tamp import PlanningFailure


@dataclass(frozen=True)
class ExecutionProviders:
    """Resolve replaceable providers once per job, without global installation."""
    load_policy_config: Callable
    policy_client: Callable
    policy_timing: Callable
    replay: Callable
    planning_timing: Callable


class ExecutionHost(Protocol):
    """Owner services needed by execution; HTTP and agent internals stay out."""
    env: Any
    epoch: str
    job: dict | None
    last_run: dict | None
    keyframes: list[dict]
    policy_config: dict | None
    completed: Queue
    stop_job: Event
    shutdown: Event

    def _publish(self) -> None: ...
    def _capture(self, label, recorder=None, *, sensor_evidence=False, camera=None): ...
    def snapshot(self) -> dict: ...
    def _recorder_class(self): ...
    def _tamp_planner_factory(self): ...
    def _relocation_planner_factory(self): ...
    def _curobo_planner_factory(self): ...


def run_execution(host: ExecutionHost, kind: ExecutionKind, args: dict,
                  providers: ExecutionProviders) -> None:
    execution_id=(host.job or {}).get('execution_id')
    if host.job:
        host.job.update(status='planning' if kind=='tamp' else 'executing',
                        phase='planning' if kind=='tamp' else 'executing')
        host._publish()
    recorder = client = timing = replay = planner = planning_timing = native_source = None
    status, error, metrics, start_failure_code = "incomplete", None, {}, None
    try:
        episode, seed = int(args.get("episode", 0)), int(args.get("seed", 0))
        if kind == 'tamp':
            planning_timing = providers.planning_timing(args.get('timing_mode', 'paused'))
            if args.get('observation_id') != f'{host.epoch}:{host.env.tick}':
                raise ValueError('场景已变化，TAMP 需要重新观察')
            episode, seed = host.env.episode, int(host.env.np_random_seed)
            # Accepted planning attempts need their own evidence even
            # when preconditions/IK fail before the first actuator step.
            host.keyframes = []
            selected=(args.get('parameters') or {}).get('planner','tamp')
            if selected not in ('tamp','curobo'):raise ValueError('unknown planning backend')
            if args.get('task') == 'relocate_object':
                planner_type=host._relocation_planner_factory()
            elif selected == 'curobo':
                planner_type=host._curobo_planner_factory()
            else:
                planner_type=host._tamp_planner_factory()
            recorder = host._recorder_class()(host.env, kind, record_images=bool(args.get('record_images', False)), seed=seed,
                details={'episode':episode,'execution_id':execution_id,
                    'initiator':args.get('initiator',{'type':'direct-command'}),
                    'tamp':{**planner_type.CONTRACT,'task':args.get('task'),'skill':args.get('skill'),
                            'parameters':args.get('parameters', {})},
                    'planning_timing':planning_timing.contract()})
            host.job['run_id'] = recorder.path.name
            host._capture('规划起点', recorder, sensor_evidence=True)
            def feedback(value):
                host.job.update(value)
                if value.get('composite_step_complete'):
                    titles={'pick':'抓取确认','handover':'交接确认','move_relative':'末端到位','place':'放置确认'}
                    host._capture(titles[value['composite_step_complete']],recorder,sensor_evidence=True)
                host._publish()
            # Include construction and all generator resumes in the same
            # timing model. No planner thread may read live MjData.
            def native_actions():
                nonlocal planner
                planner = planner_type(host.env, args.get('task'), parameters=args.get('parameters'),
                    cancel=lambda:host.stop_job.is_set() or host.shutdown.is_set(), feedback=feedback)
                recorder.event({'type':'planner_runtime',**planner_provenance(
                    planner_type,planner,selected,args.get('task'))})
                yield from planner.actions()
            steps = min(3600,max(1,int(host.env.task_window()['remaining_seconds']*host.env.profile['control_hz']))) if args.get('task')=='relocate_object' else 1200
        if kind == "policy":
            if host.env._task_budget!=host.env.profile['episode_seconds']:
                raise ValueError('学习策略保持原 40 秒评测契约；请重置标准场景后运行')
            if args.get("observation_id") != f"{host.epoch}:{host.env.tick}":
                raise ValueError("场景在工具排队后发生变化，旧观测已失效")
            if host.env.tick != 0 or host.env._done:
                raise ValueError("先重置场景并重新观察，再运行整段 VLA；不能在观察后偷偷更换场景")
            episode, seed = host.env.episode, int(host.env.np_random_seed)
            path = os.environ.get("OPENARM_POLICY_CONFIG")
            if not path:
                raise ValueError("尚未配置 OpenArm VLA；不会用脚本替代模型执行")
            config = providers.load_policy_config(path)
            if config != host.policy_config:
                raise ValueError('策略配置与服务启动时不同，请重启 ./run 后重试')
            steps = config["max_steps"]
            timing = providers.policy_timing(config["timing_mode"])
        if kind == "replay":
            host.env.reset(seed=seed, options={"episode": episode})
            host.epoch = uuid.uuid4().hex[:12]
            replay = providers.replay(host.env, episode)
            steps = replay.intervals
            host.job["clock_mode"] = replay.clock
        if recorder is None:
            host.keyframes = []
            recorder = host._recorder_class()(host.env, kind, record_images=bool(args.get("record_images", kind == "policy")),
                            seed=seed, details={"episode": episode, 'execution_id':execution_id, 'initiator':args.get('initiator',{'type':'direct-command'}), **({"policy": config, "timing": timing.contract()} if kind == "policy" else
                                {"tamp":{**planner.CONTRACT,'task':planner.task, 'skill':args.get('skill'), 'parameters':args.get('parameters', {})}} if kind == 'tamp' else {"replay_clock": replay.contract})})
            host.job["run_id"] = recorder.path.name
            host._capture("本次操作起点" if host.env.tick else "初始场景", recorder, sensor_evidence=True)
        event_count, wall_start, last_publish = len(host.env.events), time.monotonic(), 0.
        sim_start = float(host.env.data.time)
        before = recorder.observation(host.env)
        actions = iter(())
        if kind == "policy":
            client = providers.policy_client(config, seed=seed).__enter__()
            recorder.event({"type": "policy_connected", "metadata": client.metadata})
            actions = timing.actions(host.env, client, config, recorder.event,
                                     lambda: host.stop_job.is_set() or host.shutdown.is_set())
            host.job["timing_mode"] = timing.mode
        elif kind == 'tamp':
            native_source = native_actions()
            actions = planning_timing.actions(host.env, native_source, recorder.event,
                lambda:host.stop_job.is_set() or host.shutdown.is_set())
            host.job['timing_mode'] = planning_timing.mode
        # Delay-hold transitions consume physical time, not policy command
        # budget. The environment's independent 40 s ceiling still applies.
        frame = -1
        while kind == "policy" or (planning_timing.commands if planning_timing else frame + 1) < steps:
            frame += 1
            if host.stop_job.is_set() or host.shutdown.is_set():
                status = "cancelled"
                break
            if kind == "replay":
                action, (_, _, terminal, truncated, info), diagnostic = replay.advance()
            elif kind == 'tamp':
                try:
                    action, diagnostic, is_hold = next(actions)
                except StopIteration:
                    status = 'primitive_completed'
                    break
                _, _, terminal, truncated, info = (host.env.hold_tamp(action,diagnostic['planning_timing']['duration_s'])
                    if is_hold else host.env.step_tamp(action))
            else:
                try:
                    action, diagnostic, is_hold = next(actions)
                except StopIteration:
                    status = "cancelled" if host.stop_job.is_set() or host.shutdown.is_set() else "budget_exhausted"
                    break
            if host.job:
                host.job.update(status='executing',phase='executing')
            if kind == "policy" and host.stop_job.is_set():
                status = "cancelled"
                break
            if kind == "policy":
                requested = action.copy()
                if is_hold:
                    controller_report = {"mode": "exact previous actuator target"}
                else:
                    action, controller_report = host.env.prepare_policy_action(requested)
                diagnostic.update({"requested_action": requested.tolist(), "controller": controller_report})
                _, _, terminal, truncated, info = host.env.step(action)
            after = recorder.observation(host.env)
            recorder.append(before, action, after, info, terminated=terminal, truncated=truncated, diagnostics=diagnostic)
            if timing:
                timing.committed(is_hold, before["sim_time"], after["sim_time"])
            if planning_timing:
                planning_timing.committed(is_hold, before['sim_time'], after['sim_time'])
            before = after
            for event in info["events"][event_count:]:
                recorder.event(event)
                host._capture(event["name"], recorder)
            event_count = len(info["events"])
            host.job["progress"] = (timing.commands if timing else planning_timing.commands if planning_timing else frame + 1) / steps
            if planning_timing:
                host.job['planning_timing'] = planning_timing.summary()
                if is_hold:host.job['phase'] = '规划等待 · 保持上一指令'
            if timing:
                host.job["phase"] = "等待期间保持目标" if is_hold else "执行动作"
                host.job["sim_seconds"] = float(host.env.data.time)
            if time.monotonic() - last_publish > .15:
                host._publish()
                last_publish = time.monotonic()
            if terminal or truncated:
                status = "success" if info["evaluation"]["success"] else "invalid" if info["evaluation"]["invalid"] else "truncated"
                break
            if args.get("realtime", True) and (not timing or timing.mode == "paused"):
                host.stop_job.wait(max(0., float(host.env.data.time) - sim_start - (time.monotonic() - wall_start)))
        else:
            status = "reference_exhausted" if kind == "replay" else "budget_exhausted"
    except Exception as exc:
        status, error = ('cancelled' if isinstance(exc,PlanningFailure) and exc.code=='cancelled' else "error"), str(exc)
        start_failure_code = exc.code if isinstance(exc,PlanningFailure) else 'start_rejected'
        if planner:
            planner.result.update(status=status,primitive_success=False,
                failure_code=exc.code if isinstance(exc,PlanningFailure) else 'execution_error',message=error)
            if hasattr(exc,'revalidation'):planner.result['planning_revalidation']=exc.revalidation
    finally:
        try:
            if host.job:
                host.job.update(status='verifying',phase='verifying')
                host._publish()
            if native_source:
                native_source.close()
                if planner:
                    for row in planner.result.get('substeps',[]):
                        if row.get('status')=='aborted':
                            row.update(failure_code=planner.result.get('failure_code',status),
                                       message=error or '执行在返回下一动作前终止')
            if client:
                try:
                    client.__exit__()
                except Exception as close_error:
                    if recorder:
                        recorder.event({"type": "policy_close_failed", "error": str(close_error)})
            if recorder:
                if kind == 'tamp' and planner is None:
                    metrics['tamp'] = {**host._tamp_planner_factory().CONTRACT,'status':status,'primitive_success':False,
                        'failure_code':start_failure_code or 'start_rejected','message':error,
                        'phase':'preflight','motion_started':False}
                if planner:
                    metrics['tamp'] = {**planner.result, 'stats':planner.stats, 'phase':planner.phase}
                    if status=='success':
                        metrics['tamp'].update(status='completed',primitive_success=True)
                        if planner.task=='place_bottle' and host.env.evaluation.get('placement',{}).get('confirmed'):
                            metrics['tamp'].update(verification_source='native_support_release_and_stability',
                                placement=dict(host.env.evaluation['placement']),held_by=None,
                                actual_bottle_position_m=host.env.data.xpos[host.env.bottle_id].tolist())
                    elif status != 'primitive_completed':
                        metrics['tamp'].update(status=status,primitive_success=False)
                        metrics['tamp'].setdefault('failure_code',status)
                        metrics['tamp'].setdefault('message',error or '; '.join(host.env.evaluation['failure_reasons']) or '原语未在预算内完成')
                if planning_timing:
                    metrics['planning_timing'] = planning_timing.summary()
                    metrics['tamp']['planning_timing'] = planning_timing.summary()
                if replay:
                    metrics.update(replay.metrics())
                if timing:
                    metrics["timing"] = timing.summary()
                try:
                    host._capture("结束 · " + status, recorder, sensor_evidence=True)
                except Exception as capture_error:
                    status, error = "error", "final keyframe failed: " + str(capture_error)
                recorder.finish(status, host.env.info(), error=error, metrics=metrics)
                lifecycle = execution_lifecycle(kind,status,metrics)
                host.last_run = {"run_id": recorder.path.name, "kind":kind, "status": status, "error": error,
                                 "lifecycle_status":lifecycle,
                                 "execution_id":execution_id, "initiator":args.get('initiator',{'type':'direct-command'}),
                                 "summary_url": f"/runs/{recorder.path.name}/summary.json", "metrics": metrics}
            else:
                host.last_run = {"kind":kind,"status": status, "error": error, "execution_id":execution_id,
                    'lifecycle_status':execution_lifecycle(kind,status,{}),
                    'metrics':{'tamp':{'status':status,'primitive_success':False,'message':error,
                        'scene_source':'simulator_ground_truth','failure_code':start_failure_code or 'start_rejected'}} if kind=='tamp' else {}}
        finally:
            host.job = None
            host._publish()
            terminal = host.snapshot()
            host.completed.put(completion_record(
                epoch=host.epoch, execution_id=execution_id, last_run=host.last_run,
                evaluation=host.env.evaluation, keyframes=host.keyframes,
                terminal=terminal, recorded=bool(recorder)))
