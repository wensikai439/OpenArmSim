"""Native state preservation and unchanged acceptance around oriented proposals."""
import copy
import unittest

import numpy as np

from harness.object_handover import ProtectedObjectEnv
from sim.oriented_grasp import OrientedPlanner, candidates, radial_frame, initial_conflicts
from sim.object_planner import ObjectPlanner
from sim.tamp import PlanningFailure
from tools.check_pose_grasp import initialize, CASES


class OrientedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env = ProtectedObjectEnv(images=False)
    @classmethod
    def tearDownClass(cls): cls.env.close()

    def test_frames_cover_horizontal_vertical_and_antipolar_axes_without_singularities(self):
        for axis in ([1.,0,0], [-1.,0,0], [0,1.,0], [0,-1.,0], [0,0,1.], [0,0,-1.], [1,2,3]):
            for angle in np.linspace(-np.pi,np.pi,17):
                for pitch in (-.3,0.,.3):
                    rotation = radial_frame(axis, angle, pitch)
                    self.assertTrue(np.isfinite(rotation).all())
                    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(rotation),1.,places=12)
        axis=np.array([2.,3.,4.]);before=axis.copy();radial_frame(axis,.3);np.testing.assert_array_equal(axis,before)

    def test_proposal_frames_and_native_primitive_widths_are_read_only(self):
        env=self.env
        for case in CASES:
            initialize(env,case);before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time))
            transforms=env.data.geom_xmat.copy();rows=list(candidates(env,env.data,case['arm']));self.assertTrue(rows)
            opening=sum(max(f.range)-min(f.range) for f in env.fingers[case['arm']])
            self.assertTrue(all(row['required_opening_m']<=opening and row['required_opening_m']>0 for row in rows))
            self.assertLess(rows[0]['rotation'][2,2],-.9)
            for row in rows:
                np.testing.assert_allclose(row['pregrasp_m']+.10*row['rotation'][:,2],row['grasp_m'],atol=1e-12)
            for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(old,new)
            self.assertEqual(before[3],float(env.data.time))
            np.testing.assert_array_equal(transforms,env.data.geom_xmat)

    def test_oriented_context_construction_never_executes_its_cartesian_initializer(self):
        env=self.env;initialize(env,CASES[0]);before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time))
        model=copy.deepcopy(env.manifest()['model_evidence']);contacts=copy.deepcopy(env.contacts.summary())
        planner=OrientedPlanner(env,'pick_object',parameters={'object_id':'can','arm':'right'})
        self.assertEqual(planner.task,'pick_bottle');self.assertTrue(planner.strict_orientation)
        self.assertEqual(planner.result['grasp_start']['orientation_class'],'non_upright')
        for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(old,new)
        self.assertEqual(before[3],float(env.data.time));self.assertEqual(contacts,env.contacts.summary())
        self.assertEqual(model,env.manifest()['model_evidence'])

    def test_legacy_upright_control_sequence_is_bit_identical(self):
        with ProtectedObjectEnv(images=False) as other:
            env=self.env;env.reset(seed=0);other.reset(seed=0);env.select_object('can');other.select_object('can')
            p=OrientedPlanner(env,'pick_object',parameters={'object_id':'can','arm':'right'})
            q=ObjectPlanner(other,'pick_object',parameters={'object_id':'can','arm':'right'})
            self.assertNotIn('oriented_grasp_contract',p.result)
            a,b=p.actions(),q.actions()
            for _ in range(32):
                left,_=next(a);right,_=next(b);np.testing.assert_array_equal(left,right)
                env.step_tamp(left);other.step_tamp(right)
                for one,two in ((env.data.qpos,other.data.qpos),(env.data.qvel,other.data.qvel),(env.data.ctrl,other.data.ctrl)):
                    np.testing.assert_array_equal(one,two)
                self.assertEqual(float(env.data.time),float(other.data.time))

    def test_finished_invalid_contact_and_unknown_identity_still_reject_without_motion(self):
        env=self.env
        for condition,code in (('finished','episode_finished'),('invalid','unsupported_start'),('touching','already_holding'),('unknown','unknown_object')):
            initialize(env,CASES[0]);params={'object_id':'can','arm':'right'}
            if condition=='finished':env._done=True
            elif condition=='invalid':env.evaluation['invalid']=True
            elif condition=='touching':env.evaluation['touching']['right']=True
            else:params['object_id']='unknown'
            before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time))
            with self.assertRaises(PlanningFailure) as exc:OrientedPlanner(env,'pick_object',parameters=params)
            self.assertEqual(exc.exception.code,code)
            for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(old,new)
            self.assertEqual(before[3],float(env.data.time))

    def test_original_invalid_fixture_is_rejected_without_repositioning(self):
        env = self.env; initialize(env, CASES[1])
        before = [env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time)]
        conflicts = initial_conflicts(env); self.assertTrue(conflicts)
        self.assertGreater(max(row['penetration_m'] for row in conflicts), .024)
        with self.assertRaises(PlanningFailure) as error:
            OrientedPlanner(env,'pick_object',parameters={'object_id':'can','arm':'left'})
        self.assertEqual(error.exception.code, 'initial_object_overlap')
        for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)): np.testing.assert_array_equal(old,new)
        self.assertEqual(before[3],float(env.data.time))


if __name__=='__main__':unittest.main()
