"""Explicit, process-local TAMP service adapter for the SourceMIT candidate.

Importing this module does nothing. The existing owner, recorder, clocks,
limits, cancellation and policy/replay dispatch stay in charge. This is an
optional controller design, not an identified physical controller.
"""
import json
from .recording import serial
from .planning_timing import PlanningTiming as OriginalTiming
from .tamp import PlanningFailure
from .waypoint_control import commands


class ReferenceTiming(OriginalTiming):
    def contract(self):
        return {**super().contract(), 'waypoint_dispatch': {
            'input': 'nominal TAMP joint reference',
            'output': 'native motor target recorded by the unchanged service',
            'pending_reference': 'next planner point blocked until full dispatch',
            'waits': 'exact previous native target, never remapped',
            'extra_commands': 'counted by original native command and task budgets',
            'conversion_timing': 'local conversion outside planner duration; no real-time guarantee',
            'policy_and_replay_changed': False,
            'physical_calibration': False, 'transfer_ready': False}}

    def actions(self, env, source, event, cancelled=lambda: False):
        # Conversion happens after original planning waits and fresh-state
        # revalidation, so it uses the current controller history.
        for action, diagnostic, is_hold in super().actions(env, source, event, cancelled):
            if env.source_controller is None:
                yield action, diagnostic, is_hold
                continue
            if not is_hold and diagnostic.get('hold_validation') == 'native_contact_and_relative_pose':
                # HoldAudit takes last_action, which is already a motor target
                # under SourceMIT. Keep the current geometric reference rather
                # than accidentally treating that native target as a new pose.
                action = env.waypoint_adapter.reference.copy()
                diagnostic = {**diagnostic, 'hold_reference_source': 'current nominal waypoint; not last motor target'}
            for motor, detail, hold in commands(env, action, diagnostic, is_hold):
                if cancelled():
                    raise PlanningFailure('cancelled', 'Pending reference stopped before native dispatch')
                yield motor, detail, hold


class ReceiptPlanner:
    """Convert terminal diagnostics at the service boundary, after computation."""
    def __init__(self, planner):
        self.planner = planner

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def actions(self):
        try:
            yield from self.planner.actions()
        finally:
            # Numerical proposal data is private in-process until it becomes a
            # terminal receipt. Preserve all fields and values as JSON, without
            # changing solver inputs, robot state or task-success flags.
            payload = json.loads(json.dumps(self.planner.result, default=serial, allow_nan=False))
            self.planner.result.clear()
            self.planner.result.update(payload)


def receipt_provider(provider):
    class Provider:
        CONTRACT = {**provider.CONTRACT, 'receipt_diagnostics': 'JSON values; numerical solver data preserved'}
        def __new__(cls, env, task, **kwargs):
            return ReceiptPlanner(provider(env, task, **kwargs))
    return Provider


def install_isolated():
    """Call only in a separate process before constructing its service."""
    import sim.service as service
    import sim.predictive_place as placement
    from .waypoint_control import install_isolated as controls
    from .waypoint_ik import install_isolated as poses
    from .waypoint_preview import RetimedPreview
    if service.PlanningTiming is ReferenceTiming:
        return
    if service.PlanningTiming is not OriginalTiming:
        raise ValueError('Unknown service timing adapter; activation rejected')
    controls()
    poses()
    placement.NativePreview = RetimedPreview
    service.PlanningTiming = ReferenceTiming
    for name in ('TampPlanner', 'CuRoboPlanner', 'RelocationPlanner'):
        setattr(service, name, receipt_provider(getattr(service, name)))
