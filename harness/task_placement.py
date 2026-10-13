"""Optional task-pose/loaded-control routing, preserving all existing tools."""
import copy


def install_isolated(factory, *, deployed=False):
    from . import agent
    import sim.service as service
    from sim.task_grasp import TaskBottlePick
    from sim.predictive_place import PredictiveBottlePlacement
    from sim.bottle_routing import ContinuousHoldAudit
    previous = service.TampPlanner

    class Router:
        CONTRACT = {**previous.CONTRACT,
            'bottle_task_placement': 'current grasp baseline + continuous-yaw fallback + bounded native-preview control'}

        def __new__(cls, env, task, **kwargs):
            params = kwargs.get('parameters') or {}
            if env.object_id == 'bottle' and env.data.xmat[env.bottle_id].reshape(3,3)[2,2] >= .98:
                if task == 'pick_bottle' and params.get('bottle_single_arm_placement') is True:
                    planner = TaskBottlePick(env, task, **kwargs)
                    return ContinuousHoldAudit(planner, params['hold_seconds']) if 'hold_seconds' in params else planner
                if task == 'place_bottle' and params.get('receptacle_id') == 'table':
                    planner = PredictiveBottlePlacement(env, task, **kwargs)
                    planner.result['preview_servo_contract']['not_deployed'] = not deployed
                    return planner
            return previous(env, task, **kwargs)

    service.TampPlanner = Router
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    for item in tools:
        value = item['function']
        if value['name'] == 'pick':
            value['description'] += (' A declared upright bottle single-arm table goal retains the current grasp provider first; '
                'after exhausted placement lookahead it may search continuous upright yaw with native checks. '
                'Requested hold_seconds checks every complete native audit window before returning.')
        elif value['name'] == 'place':
            value['description'] += (' Upright bottle table placement may recover stopped loaded tracking with bounded '
                'actuator increments checked by short native dynamics preview. This never changes the original goal, '
                'model parameters or stable-release requirement; preview is not measured hardware safety.')
    agent.SKILL_TOOLS = tools
    previous_catalogue = agent.catalogue
    def catalogue(snapshot, policy_type='pi05'):
        result = previous_catalogue(snapshot, policy_type)
        for skill in result['skills']:
            tool = next((t['function'] for t in tools if t['function']['name'] == skill['id']), None)
            if tool:
                skill.update(description=tool['description'], parameters=tool['parameters'])
        result['bottle_task_placement'] = {'provider': Router.CONTRACT['bottle_task_placement'],
            'deployed': deployed, 'original_demo_and_other_objects_preserved': True,
            'scope': 'registered upright bottle on table; known simulator geometry',
            'validation_difference': 'small biased actuator increments checked using150ms original native dynamics',
            'physics_and_final_task_scores_unchanged': True, 'hardware_access': False, 'transfer_ready': False}
        return result
    agent.catalogue = catalogue
    return factory
