"""Isolated table-placement routing; original bottle demo remains available."""
import copy


def install_isolated(factory):
    from . import agent
    from sim.connected_place import ConnectedBottlePlacement
    import sim.service as service
    previous = service.TampPlanner

    class Router:
        CONTRACT = {**previous.CONTRACT, 'bottle_table_placement': 'connected upright loaded-path proposals'}

        def __new__(cls, env, task, **kwargs):
            params = kwargs.get('parameters') or {}
            if env.object_id == 'bottle' and task == 'place_bottle' and params.get('receptacle_id') == 'table':
                return ConnectedBottlePlacement(env, task, **kwargs)
            return previous(env, task, **kwargs)

    service.TampPlanner = Router
    controls = copy.deepcopy(agent.EXECUTION_TOOLS)
    plan = next(t['function'] for t in controls if t['function']['name'] == 'set_plan')
    fields = plan['parameters']['properties']
    fields['time_budget_s']['description'] = (
        'Declare once before motion. Geometric tasks default to120s when omitted; '
        'you may explicitly choose a supported40/60/90/120s budget suited to this task. '
        'Keep the original declared budget through replanning; never extend after an attempt. '
        'Learned policies and the legacy benchmark retain their original40s contract.')
    fields['timing_mode']['description'] = (
        'Declare on the FIRST plan. Geometric tasks default to measured_delay when omitted; '
        'later API waits and native TAMP/cuRobo computation consume the original budget while '
        'holding the prior target; stale proposals are rejected. Initial planning is excluded. '
        'paused is an explicitly labelled action-only diagnostic. VLA keeps its own timing contract.')
    agent.EXECUTION_TOOLS = controls
    tools = copy.deepcopy(agent.SKILL_TOOLS)
    for tool in tools:
        value = tool['function']
        if value['name'] == 'place':
            value['description'] += (' For registered upright bottle table placement, yaw is selected only after '
                'checking a connected upright approach and collision-free loaded lowering. '
                'Native object-goal, release and stable-support verification remains required. '
                'This is an experimental simulation capability, not open-object or real-robot transfer.')
    agent.SKILL_TOOLS = tools
    catalogue = agent.catalogue

    def describe(snapshot, policy_type='pi05'):
        value = catalogue(snapshot, policy_type)
        for skill in value['skills']:
            tool = next((t['function'] for t in agent.SKILL_TOOLS if t['function']['name'] == skill['id']), None)
            if tool:
                skill.update(description=tool['description'], parameters=tool['parameters'])
        value['bottle_table_placement'] = {'provider': 'connected upright loaded-path proposals',
            'original_demo_preserved': True, 'scope': 'registered synthetic native scene',
            'physical_calibration': False, 'transfer_ready': False}
        return value

    agent.catalogue = describe
    return factory
