"""Separate format/refresh/contract failures without relaxing any motion guard."""
from sim.decision_timing import ObservationChanged


class CallFailures:
    LIMITS = {'protocol': 3, 'observation_refresh': 3, 'contract': 3}

    def __init__(self):
        self.signatures = {}
        self.counts = {kind: 0 for kind in self.LIMITS}

    def repeated(self, signature):
        return self.signatures.get(signature, 0)>0

    def record(self, signature, exception, *, malformed=False):
        kind = ('observation_refresh' if isinstance(exception, ObservationChanged) else
                'protocol' if malformed else 'contract')
        self.signatures[signature] = self.signatures.get(signature, 0)+1
        self.counts[kind] += 1
        result = {'tool_failed': True, 'error': str(exception), 'failure_kind': kind,
                  'motion_started': False if kind in {'protocol','observation_refresh'} else None,
                  'failure_budget': {'used': dict(self.counts), 'limits': dict(self.LIMITS)}}
        if isinstance(exception, ObservationChanged):
            result['revalidation'] = exception.feedback
            result['next_step'] = 'Use the supplied current observation and sensors. Preserve the original goal; do not reset or repeat the rejected arguments.'
        return result

    @property
    def exhausted(self):
        return any(self.counts[kind]>=limit for kind, limit in self.LIMITS.items())
