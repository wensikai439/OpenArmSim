"""Isolated empty-scene corridor provider; all occupied routes unchanged."""


def install_isolated(factory):
    import sim.service as service
    from sim.free_transport import FreeBottlePick, FreeBottleHandover
    from sim.bottle_routing import ContinuousHoldAudit
    from sim.oriented_grasp import HoldAudit
    previous = service.TampPlanner

    class Router:
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

    service.TampPlanner = Router
    return factory
