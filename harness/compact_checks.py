"""Model-packet projection: binding, failure and payload-preservation checks."""
import copy
import json
from pathlib import Path
from unittest import TestCase,main

from .compact import adapt,project
from sim.environment import ROOT


class ProjectionChecks(TestCase):
    def setUp(self):
        self.feedback={'planner':'native','primitive_success':False,'failure_code':'goal_collision',
            'verification_source':'native','scene_source':'simulator_ground_truth','hardware_access':False,
            'parameters':{'arm':'left','position_m':[.4,.2,1.1],'path':[[i]*7 for i in range(10)]},
            'planning_diagnostics':{'blocked_pairs':[{'geoms':['bottle','block'],'depth_m':.005}]},
            'hold_audit':{'requested_seconds':10,'actual_seconds':9.7,'continuous_full_windows_stable':False},
            'private_search_progress':{'large':list(range(500))},
            'grasp_search':{'selected':{'q':[.1]*7},'rejected':[{'code':'ik_unreachable','index':i} for i in range(100)]}}
        self.result={'run_id':'bound-run','observation_id':'epoch:12','tool_feedback':self.feedback,
            'task_progress':{'turn_id':'original','placement_goal':{'xy':[.43,.12]},'all_planned_robot_stages_done':False}}

    def test_raw_record_not_mutated(self):
        raw=copy.deepcopy(self.result);value,omitted=project(self.result)
        self.assertEqual(raw,self.result);self.assertTrue(omitted);self.assertNotIn('private_search_progress',value['tool_feedback'])

    def test_failure_and_collision_do_not_become_success(self):
        value,_=project(self.result);f=value['tool_feedback']
        for key in ['primitive_success','failure_code','planning_diagnostics','hold_audit']:
            self.assertEqual(f[key],self.feedback[key])

    def test_parameters_state_goal_and_binding_exact(self):
        value,_=project(self.result)
        for key in ['run_id','observation_id','task_progress']:self.assertEqual(value[key],self.result[key])
        self.assertEqual(value['tool_feedback']['parameters'],self.feedback['parameters'])

    def test_rejections_keep_count_codes_and_recent_blockers(self):
        f=project(self.result)[0]['tool_feedback']['grasp_search']
        self.assertEqual(f['rejected_summary']['count'],100)
        self.assertEqual(f['rejected_summary']['failure_codes'],{'ik_unreachable':100})
        self.assertEqual(f['rejected_summary']['last'],self.feedback['grasp_search']['rejected'][-3:])

    def test_images_system_user_and_assistant_calls_are_exact(self):
        payload={'model':'same-model','temperature':0,'max_tokens':1600,'tools':[{'schema':'unchanged'}],
            'messages':[{'role':'system','content':'original rules'},
                {'role':'user','content':[{'type':'text','text':'Original goal with private_note: keep this'},
                    {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,unchanged'}}]},
                {'role':'assistant','tool_calls':[{'id':'tool-id','function':{'name':'move_to_pose','arguments':'{"position_m":[0.4,0.2,1.1]}'}}]},
                {'role':'tool','tool_call_id':'tool-id','content':json.dumps(self.result)}]}
        before=copy.deepcopy(payload);after,meta=adapt(payload)
        self.assertEqual(payload,before)
        for key in ['model','temperature','max_tokens','tools']:self.assertEqual(after[key],payload[key])
        self.assertEqual(after['messages'][:3],payload['messages'][:3])
        self.assertEqual(after['messages'][3]['tool_call_id'],'tool-id');self.assertEqual(meta['changed_tool_results'],1)

    def test_unrecognized_and_invalid_results_are_exact(self):
        payload={'messages':[{'role':'tool','content':'{bad'},
            {'role':'tool','content':' { "private_user_preference": "keep verbatim" } '}]}
        after,meta=adapt(payload);self.assertEqual(after,payload);self.assertEqual(meta['changed_tool_results'],0)

    def test_latest_trajectory_endpoints_preserved_with_no_new_command(self):
        f=copy.deepcopy(self.feedback);f['trajectory']=[[i/100]*7 for i in range(100)]
        result=project({'tool_feedback':f})[0]['tool_feedback']['trajectory_summary']
        self.assertEqual(result['first'],f['trajectory'][0]);self.assertEqual(result['last'],f['trajectory'][-1])
        self.assertEqual(result['points'],100);self.assertIn('not a new command',result['scope'])

    def test_actual_archived_receipts_keep_native_scalar_gates(self):
        for name in ['api-sideways','workbench']:
            record=json.loads((ROOT/'runs/20261009-taskpose'/name/'result.json').read_text())
            for receipt in record['receipts']:
                original=receipt['feedback'];projected=project({'tool_feedback':original})[0]['tool_feedback']
                for key in ['primitive_success','failure_code','verification_source','object_id','held_by',
                    'position_error_m','rotation_error_rad','placement','hold_audit','parameters']:
                    if key in original:self.assertEqual(projected[key],original[key],(name,key))


if __name__=='__main__':main()
