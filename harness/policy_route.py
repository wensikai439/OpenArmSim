"""Admission for the reviewed whole-task policy, never completion evidence.

The atomic skill includes right pick, right→left transfer and left placement.
Every lower policy identity, freshness, scene, budget and native guard remains.
"""
import copy

from .intent import same, progress

ROUTE=({'skill':'pick','object_id':'bottle','arm':'right'},
    {'skill':'handover','object_id':'bottle','from_arm':'right','to_arm':'left'},
    {'skill':'place','object_id':'bottle','arm':'left'})


def admit(contract, state):
    if 'bottle' in contract['protected_objects']:
        raise ValueError('原始任务要求瓶子保持原位；整项学习策略未启动')
    route=list(ROUTE);remaining=copy.deepcopy(state['remaining']);cursor=0
    for requirement in remaining:
        while cursor<len(route) and not same(requirement,route[cursor]):cursor+=1
        if cursor==len(route):
            raise ValueError('原始用户要求与整项学习策略（右手抓瓶→右交左→左手放入白盒）的手部/对象顺序不兼容；未启动运动')
        cursor+=1
    return {'admission_only':True,'declared_route':copy.deepcopy(ROUTE),
        'remaining_requirements_unchanged':remaining,'completion_verified':False}


def install_isolated(intent_base):
    class PolicyRouteHarness(intent_base):
        def _invoke(self,name,args,role,context=None):
            if role=='execute' and name in ('run_policy','run_pi05_vla'):
                pair=self._intent(context)
                if pair:
                    root,contract=pair
                    # Admission is for the bound reviewed local capability,
                    # not for an arbitrary backend name supplied by the LLM.
                    config=self.service.policy_config or {}
                    if config.get('capability')=='atomic_right_to_left_handover':
                        state=progress(contract,self._role_records(root))
                        admission=admit(contract,state)
                        # Skip only IntentHarness's incomplete two-step preview.
                        # Act, Stage, Bottle and all native dispatch guards run.
                        result=super(intent_base,self)._invoke(name,args,role,context)
                        result['policy_role_admission']=admission
                        return result
            return super()._invoke(name,args,role,context)
    return PolicyRouteHarness
