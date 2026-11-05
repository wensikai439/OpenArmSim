"""Execution facts shared by the simulator and its agent-facing receipts.

This module has no simulator, model, HTTP, or device dependencies. A worker
termination and a verified task outcome remain different facts.
"""
from __future__ import annotations

import copy
from typing import Any, Literal


ExecutionKind = Literal['tamp', 'policy', 'replay']
LifecycleStatus = Literal['cancelled', 'failed', 'succeeded', 'unverified']


def execution_lifecycle(kind: str, status: str, metrics: dict) -> LifecycleStatus:
    """Preserve the existing evidence rule; policy exit never proves success."""
    if status == 'cancelled':
        return 'cancelled'
    if status in {'error', 'invalid', 'truncated', 'budget_exhausted', 'reference_exhausted'}:
        return 'failed'
    if kind == 'tamp':
        return 'succeeded' if (metrics.get('tamp') or {}).get('primitive_success') is True else 'failed'
    return 'unverified'


def planner_provenance(factory, instance, backend, task) -> dict:
    """Record both the requested router and the concrete implementation."""
    def identity(value):
        cls = value if isinstance(value, type) else type(value)
        return {'module': cls.__module__, 'class': cls.__qualname__}
    return {
        'requested_factory': identity(factory), 'implementation': identity(instance),
        'backend': backend, 'task': task,
        'contract': copy.deepcopy(getattr(instance, 'CONTRACT', {})),
        'scope': 'implementation provenance only; not a performance or success claim',
    }


def completion_record(*, epoch: str, execution_id: str | None,
                      last_run: dict | None, evaluation: dict,
                      keyframes: list[dict], terminal: dict, recorded: bool) -> dict[str, Any]:
    """Snapshot the existing completion envelope before another job can mutate it.

    Privileged evaluation is kept in its existing separate field. The robot
    observation and camera reference fields retain their current wire format.
    This function records evidence; it does not mark an agent plan complete.
    """
    camera_keys = ('observation_id', 'sim_time', 'sensor_urls', 'url', 'label', 'tick', 'camera')
    return {
        'epoch': epoch, 'execution_id': execution_id,
        'run_id': (last_run or {}).get('run_id'),
        'result': copy.deepcopy(last_run), 'evaluation': copy.deepcopy(evaluation),
        'camera_evidence': [
            {key: copy.deepcopy(frame[key]) for key in camera_keys if key in frame}
            for frame in keyframes if frame.get('sensor_urls')
        ] if recorded else [],
        'observation': {
            'observation_id': terminal['observation_id'], **terminal['observation'],
            'source': 'robot state and forward kinematics; no object truth',
        },
    }
