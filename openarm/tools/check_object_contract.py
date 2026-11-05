"""Native object identity, scratch attachment, stage and memory guards."""
import copy
import json
import unittest

import mujoco
import numpy as np

from sim.environment import ROOT
from sim.objects import ObjectEnv
from sim.object_planner import ObjectPlanner, candidates
from harness.objects import ObjectWorkspace, resolve_goal, bind_goal, consolidate_objects, object_state


class ObjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env = ObjectEnv(images=False)
    @classmethod
    def tearDownClass(cls): cls.env.close()
    def setUp(self): self.env.reset(seed=0)

    def test_registered_identity_selection_never_changes_live_physics(self):
        env = self.env; before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time)
        selected = env.select_object('block')
        self.assertEqual(selected['free_joint'], 'block_free')
        self.assertNotEqual(env.registry['block']['qposadr'], env.registry['bottle']['qposadr'])
        for left, right in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)): np.testing.assert_array_equal(left, right)
        self.assertEqual(before[3], env.data.time)
        with self.assertRaises(ValueError): env.select_object('unknown')
        self.assertEqual(env.object_id, 'block')

    def test_shape_bounds_are_box_extents_not_a_cylinder_substitute(self):
        env = self.env; env.select_object('block'); low, high = env._bottle_bounds()
        np.testing.assert_allclose(low, [.466, .054, 1.005], atol=1e-12)
        np.testing.assert_allclose(high, [.514, .096, 1.185], atol=1e-12)
        proposals = list(candidates(env, env.data))
        self.assertTrue(proposals); self.assertEqual({p['shape'] for p in proposals}, {'mjGEOM_BOX'})
        for proposal in proposals:
            np.testing.assert_allclose(proposal['rotation'].T @ proposal['rotation'], np.eye(3), atol=1e-12)
            self.assertLess(proposal['required_opening_m'], .09)

    def test_scratch_carry_attaches_selected_joint_and_never_moves_bottle_or_live_state(self):
        env = self.env; env.select_object('block')
        planner = ObjectPlanner(env, 'move_relative', parameters={'object_id':'block','arm':'left','frame':'world','delta_m':[0,0,.02]})
        planner.carried = (np.array([0., 0., .10]), np.eye(3))
        live = env.data.qpos.copy(); original_bottle = env.registry['bottle']['qposadr']; selected = env.registry['block']['qposadr']
        before = planner.data.qpos.copy(); planner._at(env.data.qpos[env.qids['left']])
        np.testing.assert_array_equal(live, env.data.qpos)
        np.testing.assert_array_equal(before[original_bottle:original_bottle+7], planner.data.qpos[original_bottle:original_bottle+7])
        self.assertGreater(np.linalg.norm(before[selected:selected+3] - planner.data.qpos[selected:selected+3]), .01)

    def test_selected_velocity_drives_evaluation_not_other_free_joint(self):
        env = self.env; env.select_object('block')
        env.data.qvel[env.registry['block']['dofadr']] = .123
        env.data.qvel[env.registry['bottle']['dofadr']] = 8.
        value = env._evaluate(0.)
        self.assertAlmostEqual(value['linear_speed_m_s'], .123)
        self.assertFalse(value['legacy_score_applicable']); self.assertFalse(value['success'])

    def test_real_contact_prevents_object_switch_and_does_not_reset_failures(self):
        env = self.env; env.select_object('block'); env.set_task_budget(120)
        result = json.loads((ROOT/'runs/20261009-objects/block-left/result.json').read_text())
        path = ROOT/result['native_episode']/'transitions.jsonl'
        count = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['diagnostics']['phase'] == '10s-hold': break
            env.step_tamp(np.asarray(row['action'])); count += 1
            np.testing.assert_array_equal(env.state(), np.asarray(row['next_observation']['state'], dtype=np.float32))
        self.assertGreater(count, 0); self.assertTrue(env.evaluation['held']['left'])
        q = env.data.qpos.copy(); t = env.data.time
        with self.assertRaises(ValueError): env.select_object('can')
        self.assertEqual(env.object_id, 'block'); self.assertEqual(t, env.data.time)
        np.testing.assert_array_equal(q, env.data.qpos)

    def test_new_shape_handover_is_not_silently_routed_to_bottle_receiver_template(self):
        env = self.env
        with self.assertRaisesRegex(Exception, '接收抓取'):
            ObjectPlanner(env, 'handover_bottle', parameters={'object_id':'block','from_arm':'left','to_arm':'right'})
        self.assertEqual(env.object_id, 'bottle')


class ObjectMemoryChecks(unittest.TestCase):
    def receipts(self):
        return [{'id': str(i), 'tool':'pick','outcome':'success','arguments':{'object_id':ident,'arm':'left'},
            'feedback':{'object_id':ident,'primitive_success':True,'held_by':'left','verification_source':'native_two_finger_contact_and_lift'},
            'context':'same-cell','evidence_kind':'simulator_oracle','created':float(i)}
            for i,ident in enumerate(['block','block','can'])]

    def test_repeated_object_methods_merge_but_different_shapes_keep_separate_scope(self):
        rows = self.receipts(); saved = copy.deepcopy(rows); lessons = consolidate_objects(rows)
        self.assertEqual(rows, saved); self.assertEqual(len(lessons), 2)
        self.assertEqual({x['object_id']:x['repeat_count'] for x in lessons}, {'block':2,'can':1})
        self.assertEqual({x['context'] for x in lessons}, {'same-cell'})
        block = next(x for x in lessons if x['object_id']=='block')
        self.assertIn('长方体', block['title'])
        self.assertEqual(len(consolidate_objects(rows, [block['id']])), 1)

    def test_wrong_object_receipt_cannot_complete_a_grasp_or_placement_stage(self):
        goal = {'object_id':'block','surface':'table','target_xy_m':[.57,.075],'tolerance_m':.010}
        good = self.receipts()[0]['feedback']
        step = {'checkpoint':'grasp_confirmed','skill':'pick'}
        self.assertTrue(ObjectWorkspace._step_done(step,good,{'placement_goal':goal}))
        self.assertFalse(ObjectWorkspace._step_done(step,{**good,'object_id':'can'},{'placement_goal':goal}))
        self.assertEqual(object_state('pick',good)['object_id'],'block')

    def test_goal_anchor_and_identity_remain_locked_across_recovery(self):
        scene = {'object_id':'block','position_m':[.49,.075,1.005],
            'world_bounds':{'min_m':[.466,.054,1.005],'max_m':[.514,.096,1.185]},
            'surfaces':{'table':{'xy_min':[0,-.55],'xy_max':[.82,.55]}}}
        snapshot = {'observation_id':'epoch:0','diagnostics':{'manipulation_scene':scene}}
        spec = {'object_id':'block','surface':'table','offset_xy_m':[.08,0]}
        goal = resolve_goal(spec,snapshot)
        np.testing.assert_allclose(goal['target_xy_m'],[.57,.075])
        moved = copy.deepcopy(snapshot); moved['diagnostics']['manipulation_scene']['position_m'][0] += .03
        self.assertEqual(resolve_goal(spec,moved,goal,snapshot),goal)
        with self.assertRaises(ValueError): resolve_goal({**spec,'offset_xy_m':[.04,0]},moved,goal,snapshot)
        with self.assertRaises(ValueError): bind_goal({'object_id':'can','receptacle_id':'table'},goal)
        self.assertEqual(bind_goal({'object_id':'block','receptacle_id':'table'},goal)['target_xy_m'],goal['target_xy_m'])


if __name__ == '__main__': unittest.main(verbosity=2)
