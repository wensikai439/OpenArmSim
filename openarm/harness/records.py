"""Read-only archive summaries for the UI, without a second success verdict."""
from __future__ import annotations

import math


def _number(value):
    return value if (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(value) and value >= 0) else None


def archive_overview(run_id: str, manifest: dict, summary: dict) -> dict:
    """Project one already scoped archive; never open files or infer success.

    The caller owns directory selection and access control. Terminal summary
    facts take precedence over a manifest that may still say running.
    """
    if manifest.get('run_id', run_id) != run_id:
        raise ValueError('archive identity mismatch')
    details = manifest.get('details')
    details = details if isinstance(details, dict) else {}
    tamp = details.get('tamp')
    tamp = tamp if isinstance(tamp, dict) else {}
    parameters = tamp.get('parameters')
    parameters = parameters if isinstance(parameters, dict) else {}
    metrics = summary.get('metrics')
    measured = metrics.get('tamp') if isinstance(metrics, dict) else None
    measured = measured if isinstance(measured, dict) else {}
    initiator = details.get('initiator')
    initiator = initiator if isinstance(initiator, dict) else {}
    return {
        'run_id': run_id,
        'kind': manifest.get('kind', manifest.get('schema', 'unknown')),
        'status': summary.get('status', manifest.get('status', 'incomplete')),
        'scope': manifest.get('scope', 'simulation'),
        'task': tamp.get('task') or details.get('task') or measured.get('task') or measured.get('skill') or manifest.get('kind'),
        'parameters': {key: parameters[key] for key in ('arm', 'delta_m', 'target') if key in parameters},
        'started_at': _number(manifest.get('started_at_unix')),
        'wall_seconds': _number(summary.get('wall_seconds')),
        'sim_seconds': _number(summary.get('duration_sim_seconds')),
        'transitions': _number(summary.get('transitions')),
        'initiator': initiator.get('type'),
        'error': str(summary['error'])[:500] if summary.get('error') else None,
    }
