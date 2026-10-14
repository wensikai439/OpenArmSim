"""Isolated empty-scene corridor provider; all occupied routes unchanged."""


def compose_planner(previous):
    """Compose empty-scene routes without replacing the module default."""
    from sim.free_transport import FreeBottlePick, FreeBottleHandover
    from sim.bottle_routing import ContinuousHoldAudit
    from sim.oriented_grasp import HoldAudit
    if getattr(previous,'FREE_TRANSPORT_ROUTER',False):return previous

    class Router:
        FREE_TRANSPORT_ROUTER=True
        CONTRACT = {**previous.CONTRACT, 'empty_scene_corridors': 'current geometry and robot anchors; native paths'}

        def __new__(cls, env, task, **kwargs):
            params = kwargs.get('parameters') or {}
            empty = not any(ident != 'bottle' for ident in env.registry)
            upright = env.data.xmat[env.bottle_id].reshape(3, 3)[2, 2] >= .98
            if env.object_id == 'bottle' and empty and upright:
                if task == 'pick_bottle' and params.get('arm') == 'left' and params.get('next_handover', {}).get('to_arm') == 'right':
                    planner = FreeBottlePick(env, task, **kwargs)
                    return HoldAudit(planner, params['hold_seconds']) if 'hold_seconds' in params else planner
                if task in ('handover_object', 'handover_bottle') and params.get('from_arm') == 'left' and params.get('to_arm') == 'right':
                    planner = FreeBottleHandover(env, task, **kwargs)
                    return ContinuousHoldAudit(planner, params['hold_seconds']) if 'hold_seconds' in params else planner
            return previous(env, task, **kwargs)

    Router.__module__ = __name__
    return Router


def install_isolated(factory):
    """Compatibility installer for standalone research scripts."""
    import sim.service as service
    service.TampPlanner = compose_planner(service.TampPlanner)
    return factory
