"""Optional efficient IK for the already connected bottle pick provider."""

def install_isolated(factory):
    import sim.service as service
    from sim.adaptive_grasp import FastBottlePick
    from sim.oriented_grasp import HoldAudit
    previous = service.TampPlanner

    class Router:
        CONTRACT = {**previous.CONTRACT, 'bottle_pick_ik': 'minimal native kinematics; full native collision acceptance'}

        def __new__(cls, env, task, **kwargs):
            params = kwargs.get('parameters') or {}
            # Exactly the existing directed/table-goal provider boundary. The
            # demo, other shapes, non-upright route and ordinary picks keep
            # their own provider, with no new goals or hold semantics.
            if (env.object_id == 'bottle' and task == 'pick_bottle'
                    and (params.get('bottle_single_arm_placement') is True
                        or params.get('arm') == 'left' and params.get('next_handover', {}).get('to_arm') == 'right')
                    and env.data.xmat[env.bottle_id].reshape(3, 3)[2, 2] >= .98):
                planner = FastBottlePick(env, task, **kwargs)
                return HoldAudit(planner, params['hold_seconds']) if 'hold_seconds' in params else planner
            return previous(env, task, **kwargs)

    service.TampPlanner = Router
    return factory
