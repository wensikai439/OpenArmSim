"""Opt-in native PD settling before an explicitly requested complete audit."""
from sim.hold_settling import HoldSettling
from sim.tamp import PlanningFailure


class SettledPlanner:
    def __init__(self, planner): self.planner = planner
    def __getattr__(self, name): return getattr(self.planner, name)

    def actions(self):
        yield from self.planner.actions()
        if not self.result.get('primitive_success'):
            raise PlanningFailure('grasp_not_confirmed', 'Settling cannot establish an unconfirmed grasp')
        side = self.result.get('held_by')
        self.result.update(status='pre_hold_settling', primitive_success=False)
        settling = HoldSettling(self.env, side)
        try: yield from settling.actions()
        finally: self.result['pre_hold_settling'] = settling.summary()
        self.result.update(status='completed', primitive_success=True)


def compose_planner(previous):
    """Return a local TAMP router with the optional settling behavior composed in."""
    from sim.bottle_routing import ContinuousHoldAudit
    if getattr(previous, 'PRE_HOLD_SETTLING', False):
        return previous
    class Router:
        PRE_HOLD_SETTLING = True
        CONTRACT = {**previous.CONTRACT, 'pre_hold_settling': 'PD only; extra bounded native settling before explicitly requested audit'}
        def __new__(cls, env, task, **kwargs):
            planner = previous(env, task, **kwargs)
            if (env.source_controller is None and isinstance(planner, ContinuousHoldAudit)
                and (kwargs.get('parameters') or {}).get('hold_seconds') is not None):
                planner.planner = SettledPlanner(planner.planner)
            return planner
    Router.__module__ = __name__
    return Router


def install_isolated(factory):
    """Compatibility installer for standalone research scripts."""
    import sim.service as service
    service.TampPlanner = compose_planner(service.TampPlanner)
    return factory
