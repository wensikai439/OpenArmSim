"""Isolated candidate: bounded bottle grasp refinement and efficient relay.

Preserve the existing provider boundary, legacy demo and all other objects.
This is a proposal/compute improvement, not calibrated perception or hardware.
"""


def compose_planner(previous):
    """Compose the margin provider without replacing the module default."""
    from sim.margin_grasp import MarginBottlePick
    from sim.fast_handover import FastBottleHandover, ForwardBottleHandover
    from sim.bottle_routing import ContinuousHoldAudit
    from sim.oriented_grasp import HoldAudit
    if getattr(previous, 'MARGIN_GRASP_ROUTER', False):
        return previous

    class Router:
        MARGIN_GRASP_ROUTER = True
        CONTRACT={**previous.CONTRACT,
            'bottle_grasp_refinement':'bounded scratch pose proposals; original native gates',
            'bottle_relay_IK':'minimal native kinematics; full native collision acceptance'}

        def __new__(cls,env,task,**kwargs):
            params=kwargs.get('parameters') or {}
            upright=env.data.xmat[env.bottle_id].reshape(3,3)[2,2]>=.98
            if env.object_id=='bottle' and upright:
                if (task == 'handover_bottle' and params.get('from_arm', 'right') == 'right'
                        and params.get('to_arm', 'left') == 'left'):
                    planner = ForwardBottleHandover(env, task, **kwargs)
                    return ContinuousHoldAudit(planner, params['hold_seconds']) if 'hold_seconds' in params else planner
                if (task=='pick_bottle' and (params.get('bottle_single_arm_placement') is True
                    or params.get('arm')=='left' and params.get('next_handover',{}).get('to_arm')=='right')):
                    planner=MarginBottlePick(env,task,**kwargs)
                    return HoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
                if (task in ('handover_object','handover_bottle')
                    and params.get('from_arm')=='left' and params.get('to_arm')=='right'):
                    planner=FastBottleHandover(env,task,**kwargs)
                    return ContinuousHoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
            return previous(env,task,**kwargs)

    Router.__module__ = __name__
    return Router


def install_isolated(factory):
    """Compatibility installer for standalone research scripts."""
    import sim.service as service
    service.TampPlanner = compose_planner(service.TampPlanner)
    return factory
