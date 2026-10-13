"""Account for inference delay with the last actuator target held.

This is discrete-event simulation, not a wall-clock real-time controller:
network inference completes first, then its elapsed time is integrated before
any returned action can be applied. There are no concurrent MuJoCo readers.
Waiting is rounded UP to control ticks; the rounding error is logged explicitly.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .environment import TASK

MODES = ("paused", "measured_delay")
SOURCE = {"host": "data-cs", "inspection": "read-only source inspection, 2026-10-01; no robot execution",
          "path": "/home/sj/Desktop/wuji/lerobot_20260116/openarm_deploy/model_launchers/models/pi05/deploy_pi05_dual_joint.py",
          "sha256": "31a68f693ac472fbcf1b18f625fea07e5637d86c65d2520feb22e685a85e8173",
          "lines": "728-829",
          "observed": "synchronous select_action; refresh joint feedback after long inference; limit and send targets; sleep remaining cycle time",
          "not_equivalent": "our 30 Hz wait/command grid is a conservative quantized sensitivity model; native loop can send immediately after a long inference, so exact send cadence is not reproduced"}


def hold_ticks(seconds, hz):
    if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(hz) or hz <= 0:
        raise ValueError("invalid inference duration or control frequency")
    return max(0, math.ceil(seconds * hz - 1e-10))


class PolicyTiming:
    def __init__(self, mode="paused", *, fixed_delay=None):
        if mode not in MODES:
            raise ValueError("unsupported timing mode")
        if fixed_delay is not None and (not math.isfinite(fixed_delay) or not 0 <= fixed_delay <= 5):
            raise ValueError("diagnostic delay must be 0..5 seconds")
        self.mode, self.fixed_delay = mode, fixed_delay
        self.requests = self.commands = self.wait_ticks = 0
        self.network_seconds = self.modeled_seconds = self.rounding_seconds = 0.
        self.completed_wait_seconds = 0.

    def contract(self):
        return {"mode": self.mode,
                "delay_source": "fixed diagnostic injection; NOT a measured policy" if self.fixed_delay is not None else "client render/encode + local policy round trip + validation wall duration",
                "fixed_delay_seconds": self.fixed_delay,
                "method": "discrete-event integration after response, before applying any new target",
                "waiting_action": "hold exact previous applied target; do not re-limit it against changing feedback",
                "quantization": "round each wait up to 30 Hz control ticks; <1 control period overhead per request",
                "deployment_evidence": SOURCE,
                "omissions": ["real camera capture and synchronization delay", "CAN send/feedback delay", "motor watchdog semantics", "sub-control-period send phase after a long inference"],
                "real_time_validated": False, "transfer_ready": False}

    def committed(self, is_hold, before_seconds, after_seconds):
        """Count only transitions actually stepped and recorded by the owner."""
        if is_hold:
            self.wait_ticks += 1
            self.completed_wait_seconds += after_seconds - before_seconds
        else:
            self.commands += 1

    def summary(self):
        return {**self.contract(), "requests": self.requests, "policy_commands": self.commands,
                "hold_transitions": self.wait_ticks, "policy_roundtrip_seconds": self.network_seconds,
                "requested_delay_seconds": self.modeled_seconds, "planned_rounding_seconds": self.rounding_seconds,
                "integrated_hold_seconds": self.completed_wait_seconds}

    def actions(self, env, client, config, event, cancelled=lambda: False, *, clock=time.monotonic):
        """Yield (command, diagnostics, is_hold); caller owns step/record/stop.

        Every yielded wait is a real physics transition. A caller must stop on
        environment termination; commands from that pending response are then
        never executed. Errors also account for elapsed waiting before raising.
        """
        hz = env.profile["control_hz"]
        while self.commands < config["max_steps"] and not cancelled():
            observation_tick, observation_time = env.tick, float(env.data.time)
            started, failure, chunk = clock(), None, None
            try:
                chunk = client.infer(env, TASK)
            except Exception as exc:
                failure = exc
            elapsed = max(0., clock() - started)
            duration = elapsed if self.fixed_delay is None else self.fixed_delay
            waits = hold_ticks(duration, hz) if self.mode == "measured_delay" else 0
            request_id = self.requests
            self.requests += 1
            self.network_seconds += elapsed
            self.modeled_seconds += duration if waits else 0.
            self.rounding_seconds += waits / hz - duration if waits else 0.
            base = {"request_id": request_id, "observation_tick": observation_tick,
                    "observation_sim_time": observation_time, "roundtrip_wall_seconds": elapsed,
                    "delay_seconds": duration, "planned_hold_ticks": waits, "timing_mode": self.mode}
            event({"type": "policy_inference", **base, "candidate_prefix": chunk,
                   "response_valid": failure is None,
                   "delay_source": "fixed diagnostic" if self.fixed_delay is not None else "measured local round trip"})
            held_target = env.last_action.copy()
            for index in range(waits):
                if cancelled():
                    return
                yield held_target.copy(), {"timing": {**base, "action_source": "inference_hold",
                                                      "wait_index": index}}, True
            if cancelled():
                return
            if failure is not None:
                raise failure
            for chunk_index, action in enumerate(chunk):
                if self.commands >= config["max_steps"] or cancelled():
                    return
                diagnostics = {"timing": {**base, "action_source": "policy",
                                          "chunk_index": chunk_index, "command_index": self.commands,
                                          "observation_age_sim_seconds": float(env.data.time) - observation_time}}
                yield np.asarray(action, np.float32).copy(), diagnostics, False
