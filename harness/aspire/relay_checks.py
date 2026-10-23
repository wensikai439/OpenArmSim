"""Meaningful guards for resumable skills, model variants and legacy routing."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from .runtime import ROOT, ProgramExecutor, ToolRegistry, ExecutionResult
from .variants import validate_variant, variant_environment


def execute(held=None, *, upright=True, fail=None, stale=None):
    state={'held_by':held or [],'upright':upright};calls=[]
    def observe():return dict(state)
    def motion(name, holder, **kwargs):
        calls.append((name,kwargs))
        if name==fail:return ExecutionResult(ok=False,steps_executed=1,diagnostics={'failure_code':'rejected'})
        if name!=stale:state['held_by']=holder
        return ExecutionResult(ok=True,steps_executed=1)
    registry=ToolRegistry(shared_public_names=('get_observation','pick','handover','place'),public_extension_allowlist=())
    functions={'get_observation':observe,'pick':lambda **kw:motion('pick',['right'],**kw),
        'handover':lambda **kw:motion('handover',['left'],**kw),'place':lambda **kw:motion('place',[],**kw)}
    for name,fn in functions.items():registry.register(fn,name=name,public=True,layer='shared_high_level',capability='mock-only')
    program=Path(os.environ.get('OPENARM_SKILL_PROGRAM',ROOT/'skills/bottle_relay/program.py')).read_text()
    return ProgramExecutor(registry).execute_program(program),calls


class RelayChecks(unittest.TestCase):
    def test_failure_never_advances_to_release(self):
        for stage,count in (('pick',1),('handover',2),('place',3)):
            result,calls=execute(fail=stage)
            self.assertFalse(result.ok);self.assertEqual(len(calls),count)

    def test_receipt_without_fresh_holder_does_not_advance(self):
        result,calls=execute(stale='pick')
        self.assertFalse(result.ok);self.assertEqual([c[0] for c in calls],['pick'])

    def test_unsafe_initial_state_never_moves(self):
        for kwargs in ({'upright':False},{'held':['left','right']},{'held':['unknown']}):
            result,calls=execute(**kwargs)
            self.assertFalse(result.ok);self.assertEqual(calls,[])

    def test_resume_keeps_current_holder_and_declares_downstream_goal(self):
        for held,names in (([],['pick','handover','place']),(['right'],['handover','place']),(['left'],['place'])):
            result,calls=execute(held)
            self.assertTrue(result.ok,result.error);self.assertEqual([c[0] for c in calls],names)
            self.assertTrue(calls[-1][1]['shape_clearance'])
            if 'handover' in names:self.assertTrue(next(c[1] for c in calls if c[0]=='handover')['placement_aware'])

    def test_legacy_routes_are_not_replaced(self):
        from sim.relay_place import compose_planner
        sentinel=object()
        class Previous:
            CONTRACT={}
            def __new__(cls,*a,**kw):return sentinel
        route=compose_planner(Previous)
        env=SimpleNamespace(object_id='bottle')
        for task,params in (('handover_to_basket',{}),('place_bottle',{'receptacle_id':'basket'}),
                            ('handover_bottle',{'from_arm':'right','to_arm':'left'})):
            self.assertIs(route(env,task,parameters=params),sentinel)

    def test_variant_audit_stays_enabled_and_reset_is_idempotent(self):
        from sim.scene_library import LibraryEnv
        case={'schema':'relay-fixture/2','seed':1,'bottle_xy_m':[.51,-.18],
            'bottle_yaw_rad':.8,'basket_offset_m':[.01,-.01],
            'bottle_mass_scale':1.1,'bottle_friction_scale':.9}
        env=variant_environment(LibraryEnv,case)(images=False,selection=[])
        try:
            env.reset(seed=1);env.require_current_sources()
            mass=float(env.model.body_mass[env.bottle_id]);pos=env.model.body_pos[env.box_id].copy()
            env.step(env.last_action.copy())
            env.reset(seed=1);env.require_current_sources()
            self.assertEqual(mass,env.model.body_mass[env.bottle_id])
            np.testing.assert_array_equal(pos,env.model.body_pos[env.box_id])
            env.model.body_mass[env.bottle_id]*=1.01
            with self.assertRaises(ValueError):env.require_current_sources()
        finally:env.close()
        for update in ({'bottle_mass_scale':2},{'basket_offset_m':[.1,0]}, {'bottle_yaw_rad':float('nan')}):
            with self.assertRaises(ValueError):validate_variant({**case,**update})


if __name__=='__main__':unittest.main()
