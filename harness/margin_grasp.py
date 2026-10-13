"""Isolated candidate: bounded bottle grasp refinement and efficient relay.

Preserve the existing provider boundary, legacy demo and all other objects.
This is a proposal/compute improvement, not calibrated perception or hardware.
"""


def install_isolated(factory):
    import sim.service as service
    from sim.margin_grasp import MarginBottlePick
    from sim.fast_handover import FastBottleHandover
    from sim.bottle_routing import ContinuousHoldAudit
    from sim.oriented_grasp import HoldAudit
    previous=service.TampPlanner

    class Router:
        CONTRACT={**previous.CONTRACT,
            'bottle_grasp_refinement':'bounded scratch pose proposals; original native gates',
            'bottle_relay_IK':'minimal native kinematics; full native collision acceptance'}

        def __new__(cls,env,task,**kwargs):
            params=kwargs.get('parameters') or {}
            upright=env.data.xmat[env.bottle_id].reshape(3,3)[2,2]>=.98
            if env.object_id=='bottle' and upright:
                if (task=='pick_bottle' and (params.get('bottle_single_arm_placement') is True
                    or params.get('arm')=='left' and params.get('next_handover',{}).get('to_arm')=='right')):
                    planner=MarginBottlePick(env,task,**kwargs)
                    return HoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
                if (task in ('handover_object','handover_bottle')
                    and params.get('from_arm')=='left' and params.get('to_arm')=='right'):
                    planner=FastBottleHandover(env,task,**kwargs)
                    return ContinuousHoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
            return previous(env,task,**kwargs)

    service.TampPlanner=Router
    return factory
