"""Initial geometric plan reuses native inspection, without motor motion."""
import time
import unittest

import numpy as np


class PlanChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.check_task_queue import setup
        from harness.poses import install_isolated
        from tools.check_pose_api import fixture_service
        setup(); install_isolated(); cls.factory = staticmethod(fixture_service())

    def setUp(self):
        from harness.poses import PoseHarness, PoseWorkspace
        self.service = self.factory(); self.service.call('_initial_pose_fixture', name='can-left-oblique')
        self.workspace = PoseWorkspace(); self.harness = PoseHarness(self.service, workspace=self.workspace)
        self.addCleanup(self.workspace.db.close); self.addCleanup(self.service.close); self.addCleanup(self.harness.close)
        snapshot = self.service.snapshot(); epoch = snapshot['observation_id'].split(':')[0]
        root = self.workspace.start('execute', epoch, 'native contract fixture', {'model':'offline-native','vision':False})
        self.context = {'type':'llm', 'turn_id':root, 'plan_turn_id':root, 'model':'offline-native'}
        self.harness.sessions[root] = {'actions':0, 'selected_object':'can', 'initial_snapshot':snapshot}
        self.harness.invoke('observe', {}, context=self.context)
        self.args = {'observation_id':snapshot['observation_id'],
            'placement_goal':{'object_id':'can','surface':'table','target_xy_m':[.42,.18]},
            'time_budget_s':120, 'timing_mode':'measured_delay',
            'steps':[{'id':'pick','title':'抓起圆柱','skill':'pick','checkpoint':'grasp_confirmed'},
                {'id':'place','title':'放到原目标','skill':'place','checkpoint':'placement_confirmed'}]}

    def test_first_plan_checks_native_conditions_without_control_or_time_change(self):
        env = self.service.env
        before = [env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time)]
        result = self.harness.invoke('set_plan', self.args, context=self.context)
        self.assertTrue(result['native_preconditions_checked']); self.assertTrue(result['contains_privileged_object_state'])
        self.assertEqual(self.harness.privileged_epoch, self.args['observation_id'].split(':')[0])
        self.assertEqual(self.workspace.active_plan['placement_goal']['target_xy_m'], [.42,.18])
        for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)): np.testing.assert_array_equal(old,new)
        self.assertEqual(before[3],float(env.data.time)); self.assertFalse(self.service.snapshot().get('job'))
        self.assertFalse(result['start_grasp_admission']['valid'])
        with self.assertRaisesRegex(ValueError, '起点存在物体穿模'):
            self.harness.invoke('pick', {'observation_id':self.args['observation_id'], 'object_id':'can', 'arm':'left'}, context=self.context)
        self.assertFalse(self.service.snapshot().get('job')); self.assertEqual(before[3],float(env.data.time))

    def test_stale_initial_plan_does_not_perform_oracle_inspection(self):
        self.args['observation_id'] = 'stale:0'
        with self.assertRaisesRegex(ValueError,'先读取当前观测'):
            self.harness.invoke('set_plan', self.args, context=self.context)
        self.assertIsNone(self.harness.privileged_epoch); self.assertIsNone(self.workspace.active_plan)
        self.assertEqual(self.service.snapshot()['observation']['sim_time'],0.)

    def test_trusted_invalid_start_can_finish_failure_without_fabricating_a_run(self):
        self.harness.invoke('set_plan', self.args, context=self.context)
        args = {'observation_id':self.args['observation_id'], 'verdict':'success', 'reason':'fixture rejected'}
        with self.assertRaisesRegex(ValueError, '不能提交success'):
            self.harness.invoke('finish',args,context=self.context)
        args['verdict']='failure';result=self.harness.invoke('finish',args,context=self.context)
        self.assertTrue(result['assessment_recorded'])
        self.assertTrue(result['report_evidence']['no_execution_receipt_created'])
        self.assertEqual(self.workspace.active_plan['verdict'],'failure')
        self.assertEqual(self.workspace.active_plan.get('execution_tools',[]),[])
        self.assertIsNone(self.service.snapshot()['last_run']);self.assertFalse(self.service.snapshot().get('job'))

    def test_catalogue_matches_actual_model_tool_schemas(self):
        from harness.agent import tools_for
        catalogue = self.harness.invoke('list_skills',{},context=self.context)
        definitions = {tool['function']['name']:tool['function'] for tool in tools_for('execute')}
        for skill in catalogue['skills']:
            self.assertEqual(skill['parameters'],definitions[skill['id']]['parameters'])
        pick = next(skill for skill in catalogue['skills'] if skill['id']=='pick')
        self.assertEqual(pick['parameters']['properties']['hold_seconds']['minimum'],2)
        self.assertEqual(pick['parameters']['properties']['object_id']['enum'],['bottle','block','can'])


if __name__ == '__main__': unittest.main()
