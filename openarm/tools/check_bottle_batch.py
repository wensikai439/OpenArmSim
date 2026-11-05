"""Declared geometry cases preserve old fixtures and reject ambiguous input."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.check_bottle_routes import CASES,load_cases,execute_gpu_with_recovery
from sim.tamp import PlanningFailure


class BatchContract(unittest.TestCase):
    def load(self,cases,selected=None):
        with tempfile.TemporaryDirectory(prefix='openarm-batch-check-') as folder:
            path=Path(folder)/'cases.json'
            path.write_text(json.dumps({'schema':'openarm-bottle-batch/1','cases':cases}))
            return load_cases(path,selected)

    def test_default_fixtures_and_returned_inputs_do_not_alias(self):
        original=copy.deepcopy(CASES);values,meta=load_cases()
        self.assertEqual(values,original);self.assertIsNone(meta)
        values[0]['target_xy_m'][0]=999;self.assertEqual(CASES,original)

    def test_no_silent_case_substitution_or_directory_traversal(self):
        with self.assertRaises(ValueError):self.load(CASES,'missing-case')
        rows=copy.deepcopy(CASES);rows[0]['name']='../../outside'
        with self.assertRaises(ValueError):self.load(rows)
        rows=copy.deepcopy(CASES);rows[1]['name']=rows[0]['name']
        with self.assertRaises(ValueError):self.load(rows)

    def test_bad_coordinates_and_unconnected_routes_are_rejected(self):
        for field,value in [('target_xy_m',[True,0]),('origin_xy_m',[float('nan'),0]),
            ('target_xy_m',[1]),('receiver','left')]:
            rows=copy.deepcopy(CASES);rows[0][field]=value
            with self.assertRaises(ValueError):self.load(rows)

    def test_all_declared_new_cases_are_kept_without_collision_based_filtering(self):
        rows=copy.deepcopy(CASES);rows[0]['origin_xy_m']=[.49,.075]
        values,metadata=self.load(rows)
        self.assertEqual(values,rows);self.assertTrue(metadata['cases_declared_before_execution'])
        self.assertEqual(len(metadata['sha256']),64)


class GPURecoveryContract(unittest.TestCase):
    """Protocol-only fixtures, not native motion or GPU performance evidence."""
    def setUp(self):
        self.env=SimpleNamespace(_done=False,data=SimpleNamespace(time=81.),
            evaluation={'held':{'right':True}},last_action=np.arange(16,dtype=float),
            profile={'control_hz':30},model=SimpleNamespace(opt=SimpleNamespace(timestep=.001)))
        self.motion={'arm':'right','position_m':[.57,-.09,1.25],
            'quaternion_xyzw':[0.,0.,0.,1.],'object_id':'bottle'}
        self.saved=[];self.commands=[];self.calls=[]
    def save(self,value):self.saved=copy.deepcopy(value)
    def settle(self,action,diagnostic,is_hold):
        seconds=diagnostic['planning_timing']['duration_s']
        self.assertGreaterEqual(seconds,.001);self.assertLessEqual(seconds,1./30+.001)
        self.assertAlmostEqual(seconds/.001,round(seconds/.001))
        self.commands.append((action.copy(),copy.deepcopy(diagnostic),is_hold))
        self.env.data.time+=seconds
    def run_recovery(self,execute,retries=2,settle=None):
        return execute_gpu_with_recovery(self.env,execute,self.motion,retries,
            settle or self.settle,lambda:{'partial_commands':5},self.save)
    def test_fresh_attempt_keeps_locked_pose_partial_failure_and_original_clock(self):
        original=copy.deepcopy(self.motion)
        def execute(cls,task,args):
            self.calls.append(copy.deepcopy(args))
            if len(self.calls)==1:
                args['position_m'][0]=999
                raise PlanningFailure('planning_state_changed','injected stopped correction')
        result=self.run_recovery(execute)
        self.assertEqual(self.calls,[original,original]);self.assertEqual(self.motion,original)
        self.assertAlmostEqual(self.env.data.time,82.);self.assertFalse(self.env._done)
        self.assertEqual(len(result),1);self.assertEqual(result,self.saved)
        self.assertEqual(result[0]['feedback']['partial_commands'],5)
        self.assertNotIn('primitive_success',result[0]);self.assertEqual(len(self.commands),31)
        np.testing.assert_array_equal(self.commands[0][0],self.env.last_action)
        self.assertTrue(self.commands[0][2])
    def test_zero_default_or_exhausted_retries_never_keep_trying(self):
        for retries in (0,2):
            self.calls=[];self.commands=[];self.saved=[]
            def execute(cls,task,args):
                self.calls.append(copy.deepcopy(args))
                raise PlanningFailure('planning_state_changed','injected stale pose')
            with self.assertRaises(PlanningFailure):self.run_recovery(execute,retries)
            self.assertEqual(len(self.calls),retries+1)
            self.assertEqual(len(self.commands),31*retries);self.assertEqual(len(self.saved),retries+1)
    def test_other_failures_are_not_retried_or_settled(self):
        def execute(cls,task,args):
            self.calls.append(args);raise PlanningFailure('carried_object_lost','injected contact loss')
        with self.assertRaises(PlanningFailure):self.run_recovery(execute)
        self.assertEqual(len(self.calls),1);self.assertFalse(self.commands);self.assertFalse(self.saved)
    def test_ended_task_or_lost_hold_does_not_issue_a_fresh_motion(self):
        for state in ('ended','lost-before-settle','lost-during-settle'):
            self.setUp()
            if state=='ended':self.env._done=True
            if state=='lost-before-settle':self.env.evaluation['held']['right']=False
            def execute(cls,task,args):
                self.calls.append(args);raise PlanningFailure('planning_state_changed','injected stopped proposal')
            def settle(action,diagnostic,is_hold):
                self.settle(action,diagnostic,is_hold);self.env.evaluation['held']['right']=False
            with self.assertRaises(RuntimeError):self.run_recovery(execute,settle=settle)
            self.assertEqual(len(self.calls),1);self.assertEqual(len(self.saved),1)
            self.assertEqual(len(self.commands),int(state=='lost-during-settle'))


if __name__=='__main__':unittest.main()
