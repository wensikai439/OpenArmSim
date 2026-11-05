"""Native bounded pose gates, proposal provenance and no live-state changes."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sim.environment import ROOT
from sim.margin_grasp import MarginBottlePick, refine
from sim.adaptive_grasp import FastBottlePick
from sim.tamp import PlanningFailure
from harness.object_handover import ProtectedObjectEnv
from tools.check_bottle_routes import initialize


class MarginChecks(unittest.TestCase):
    def setUp(self):
        self.env=ProtectedObjectEnv(images=False);self.addCleanup(self.env.close)
        data=json.loads((ROOT/'runs/20261009-grasp/refined-cross/results.json').read_text())
        self.case=next(row for row in data['cases'] if row['case']['name']=='cross-relay-a')
        rejected=self.case['failed_feedback']['grasp_search']['rejected']
        self.original=next(row['candidate'] for row in rejected if row['index']==25)
        self.neighbor=next(row['candidate'] for row in rejected if row['candidate'].get('axial_shift_m')==.0032)
        self.seed=np.asarray(self.neighbor['_seed_q'])
        initialize(self.env,self.case['case'])
        self.before=(self.env.data.qpos.copy(),self.env.data.qvel.copy(),self.env.data.ctrl.copy(),self.env.data.time,
            self.env.manifest()['model_evidence'])

    def planner(self,**kwargs):
        p=MarginBottlePick(self.env,'pick_object',parameters={'object_id':'bottle','arm':'left','next_handover':{'to_arm':'right'}},**kwargs)
        p._arm('left');p.planning_qpos=self.env.data.qpos.copy()
        for f in self.env.fingers['left']:p.planning_qpos[f.qposadr[0]]=max(f.range)
        p._sync();return p

    def check_live(self):
        for old,new in zip(self.before[:3],(self.env.data.qpos,self.env.data.qvel,self.env.data.ctrl)):
            np.testing.assert_array_equal(old,new)
        self.assertEqual(self.env.data.time,self.before[3]);self.assertEqual(self.env.manifest()['model_evidence'],self.before[4])

    def solve(self,p,value):
        return refine(p,np.asarray(value['grasp_m']),np.asarray(value['rotation']),self.seed,True,.0015)

    def test_dimension_derived_neighbor_has_native_margin_pose_and_full_collision_checks(self):
        p=self.planner();q,r=self.solve(p,self.neighbor)
        self.assertIsNotNone(q);self.assertTrue(r['accepted']);self.assertTrue(r['native_collision_checked']);self.assertFalse(r['collision'])
        self.assertLess(r['position_error_m'],.0015);self.assertLess(r['rotation_error_rad'],.018)
        self.assertGreaterEqual(r['minimum_joint_slack_rad'],.020)
        p._at(q);point,rotation=p._pose();self.assertLess(np.linalg.norm(point-np.asarray(self.neighbor['grasp_m'])),.0015)
        self.assertFalse(p._collision(q,True));self.assertNotIn('primitive_success',r)
        self.assertEqual(len(p.result['private_margin_refinements']),1);self.check_live()

    def test_solver_success_cannot_override_original_position_threshold(self):
        p=self.planner();q,r=self.solve(p,self.original)
        self.assertTrue(r['solver_reported_success']);self.assertIsNone(q);self.assertFalse(r['accepted'])
        self.assertGreaterEqual(r['position_error_m'],.0015);self.assertFalse(r['native_collision_checked']);self.check_live()

    def test_nonfinite_solver_output_rejected_and_feedback_remains_serializable(self):
        p=self.planner()
        value=SimpleNamespace(x=np.full(7,np.nan),status=0,message='injected nonfinite success',success=True)
        # Native kinematic calls on NaN are avoided by injecting the solver
        # result only after the tested residual, then checking finite first.
        with patch('sim.margin_grasp.minimize',return_value=value):
            q,r=self.solve(p,self.neighbor)
        self.assertIsNone(q);self.assertFalse(r['accepted']);self.assertIsNone(r['q_rad'])
        json.dumps(r,allow_nan=False);self.check_live()

    def test_cancellation_preserves_trace_flags_and_live_model(self):
        p=self.planner(cancel=lambda:True)
        with self.assertRaises(PlanningFailure) as error:self.solve(p,self.neighbor)
        self.assertEqual(error.exception.code,'cancelled');self.assertFalse(p._kinematics_only)
        r=p.result['private_margin_refinements'][-1];self.assertEqual(r['interrupted_code'],'cancelled');self.assertFalse(r['accepted'])
        self.check_live()

    def test_ordinary_empty_hand_queries_retain_original_solver_result(self):
        env=self.env;values=[]
        for cls in (FastBottlePick,MarginBottlePick):
            p=cls(env,'pick_object',parameters={'object_id':'bottle','arm':'left'});p._arm('left');p._sync()
            point,rotation=p._pose();p.rotation=rotation.copy()
            with patch('sim.margin_grasp.refine',side_effect=AssertionError('ordinary query must not refine')):
                values.append(p._ik(point+[0,0,.008],False))
        np.testing.assert_array_equal(*values);self.check_live()

    def test_single_arm_table_goal_retains_original_selected_grasp_and_joint_solutions(self):
        env=ProtectedObjectEnv(images=False)
        try:
            initialize(env,{'name':'left-forward','arm':'left','receiver':None,
                'origin_xy_m':[.44,.18],'target_xy_m':[.52,.18]})
            before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),env.data.time,
                env.manifest()['model_evidence'])
            results=[]
            for cls in (FastBottlePick,MarginBottlePick):
                p=cls(env,'pick_object',parameters={'object_id':'bottle','arm':'left',
                    'next_placement':{'surface':'table','target_xy_m':[.52,.18],'position_tolerance_m':.01}})
                p._arm('left');p._sync()
                points=p._choose_pick_pose(None)
                results.append((points,p.result['grasp_search']))
                self.assertNotIn('private_margin_refinements',p.result)
            for a,b in zip(results[0][0],results[1][0]):np.testing.assert_array_equal(a,b)
            self.assertEqual(results[0][1],results[1][1])
            for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):
                np.testing.assert_array_equal(old,new)
            self.assertEqual(before[3],env.data.time);self.assertEqual(before[4],env.manifest()['model_evidence'])
        finally:env.close()


if __name__=='__main__':unittest.main()
