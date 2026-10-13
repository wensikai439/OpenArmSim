"""Optional routing/read-only geometry and continuous hold contract checks.

The hold unit fixtures test command/protocol logic only; actual manipulation
evidence lives in the separate native/API experiments.
"""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harness.object_handover import ProtectedObjectEnv
from harness.bottle import BottleHarness,service_factory
from sim.bottle_routing import BottlePickPlanner,BottlePlacementPlanner,bypass_origins,ContinuousHoldAudit
from sim.oriented_grasp import HoldAudit
from harness.object_handover import RelayPlanner


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=ProtectedObjectEnv(images=False)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):self.env.reset(seed=0)

    def test_bottle_provider_does_not_mutate_native_physics_or_start(self):
        env=self.env;before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),env.data.time,
            copy.deepcopy(env.manifest()['model_evidence']))
        BottlePickPlanner(env,'pick_bottle',parameters={'object_id':'bottle','arm':'left','next_handover':{'to_arm':'right'}})
        for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(old,new)
        self.assertEqual(before[3],env.data.time);self.assertEqual(before[4],env.manifest()['model_evidence'])

    def test_obstacle_proposals_do_not_alias_or_mutate_native_transforms(self):
        env=self.env;before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.xpos.copy(),env.data.geom_xpos.copy(),env.data.time)
        proposals=bypass_origins(env,env.data);self.assertTrue(proposals)
        for p in proposals:
            self.assertFalse(np.shares_memory(p['origin_m'],env.data.xpos))
            self.assertGreater(p['origin_m'][0],.1);self.assertLess(p['origin_m'][0],.9)
        for old,new in zip(before[:4],(env.data.qpos,env.data.qvel,env.data.xpos,env.data.geom_xpos)):np.testing.assert_array_equal(old,new)
        self.assertEqual(before[4],env.data.time)

    def test_original_right_pick_and_full_demo_routing_unchanged(self):
        import sim.service as service
        before=service.TampPlanner;env_class=service.OpenArmEnv
        self.addCleanup(setattr,service,'TampPlanner',before);self.addCleanup(setattr,service,'OpenArmEnv',env_class)
        service_factory();env=self.env
        params={'object_id':'bottle','arm':'right'}
        actual=service.TampPlanner(env,'pick_bottle',parameters=params)
        original=RelayPlanner(env,'pick_bottle',parameters=params)
        self.assertIs(type(actual),type(original))
        a=actual.actions();b=original.actions()
        for _ in range(32):np.testing.assert_array_equal(next(a)[0],next(b)[0])
        a.close();b.close()
        self.assertIs(type(service.TampPlanner(env,'handover_to_basket')),type(RelayPlanner(env,'handover_to_basket')))

    def test_explicit_left_relay_pick_is_optional_and_hold_is_typed(self):
        import sim.service as service
        before=service.TampPlanner;env_class=service.OpenArmEnv
        self.addCleanup(setattr,service,'TampPlanner',before);self.addCleanup(setattr,service,'OpenArmEnv',env_class)
        service_factory();env=self.env
        params={'object_id':'bottle','arm':'left','next_handover':{'to_arm':'right'}}
        self.assertIsInstance(service.TampPlanner(env,'pick_bottle',parameters=params),BottlePickPlanner)
        params['hold_seconds']=2
        self.assertIsInstance(service.TampPlanner(env,'pick_bottle',parameters=params),HoldAudit)
        params['hold_seconds']=1
        with self.assertRaises(ValueError):service.TampPlanner(env,'pick_bottle',parameters=params)

    def test_plan_hint_preserves_inputs_and_does_not_start_a_second_action(self):
        harness=BottleHarness.__new__(BottleHarness)
        harness.workspace=SimpleNamespace(active_plan={'turn_id':'this-task','steps':[{'id':'relay','skill':'handover','status':'pending','object_id':'bottle'}],
            'placement_goal':{'object_id':'bottle','surface':'table','target_xy_m':[.56,-.13]}})
        args={'object_id':'bottle','arm':'left'};original=copy.deepcopy(args)
        with patch('harness.poses.PoseHarness._invoke',side_effect=lambda name,args,role,context=None:args):
            returned=harness._invoke('pick',args,'execute',{'turn_id':'this-task'})
        self.assertEqual(args,original);self.assertEqual(returned['next_handover'],{'to_arm':'right'})
        harness.workspace.active_plan['steps'][0]['object_id']='block'
        with patch('harness.poses.PoseHarness._invoke',side_effect=lambda name,args,role,context=None:args):
            self.assertEqual(harness._invoke('pick',args,'execute'),original)

    def test_single_arm_hint_is_bound_to_current_plan_and_never_aliases_inputs(self):
        harness=BottleHarness.__new__(BottleHarness)
        harness.workspace=SimpleNamespace(active_plan={'turn_id':'goal','steps':[],
            'placement_goal':{'object_id':'bottle','surface':'table','target_xy_m':[.52,.18]}})
        args={'object_id':'bottle','arm':'left'};context={'turn_id':'child','plan_turn_id':'goal'}
        before=copy.deepcopy((args,context))
        with patch('harness.poses.PoseHarness._invoke',side_effect=lambda n,a,r,c=None:(a,c)):
            result=harness._invoke('pick',args,'execute',context)
            self.assertTrue(result[1]['bottle_single_arm_placement'])
            for ctx in (None,{'turn_id':'old-goal'}):
                self.assertNotIn('bottle_single_arm_placement',(harness._invoke('pick',args,'execute',ctx)[1] or {}))
            harness.workspace.active_plan['steps']=[{'skill':'handover','status':'pending','object_id':'bottle','id':'relay'}]
            self.assertNotIn('bottle_single_arm_placement',harness._invoke('pick',args,'execute',context)[1])
        self.assertEqual((args,context),before)

    def test_new_geometric_plan_defaults_include_waits_without_changing_explicit_or_VLA_mode(self):
        harness=BottleHarness.__new__(BottleHarness);harness.workspace=SimpleNamespace(active_plan=None)
        context={'turn_id':'goal'};args={'steps':[{'skill':'pick'}],'time_budget_s':120}
        before=copy.deepcopy(args)
        with patch('harness.poses.PoseHarness._invoke',side_effect=lambda n,a,r,c=None:a):
            self.assertEqual(harness._invoke('set_plan',args,'execute',context)['timing_mode'],'measured_delay')
            omitted={'steps':[{'skill':'pick'}]}
            self.assertEqual(harness._invoke('set_plan',omitted,'execute',context)['time_budget_s'],120)
            self.assertEqual(harness._invoke('set_plan',{**args,'timing_mode':'paused'},'execute',context)['timing_mode'],'paused')
            policy={'steps':[{'skill':'run_policy'}]}
            self.assertEqual(harness._invoke('set_plan',policy,'execute',context),policy)
            harness.workspace.active_plan={'turn_id':'goal'}
            self.assertEqual(harness._invoke('set_plan',args,'execute',context),args)
        self.assertEqual(args,before)

    def test_motion_cannot_silently_diverge_from_its_declared_stage(self):
        harness=BottleHarness.__new__(BottleHarness)
        harness.workspace=SimpleNamespace(active_plan={'turn_id':'goal','steps':[
            {'id':'transport','skill':'move_to_pose','status':'pending'}]})
        with patch('harness.poses.PoseHarness._invoke',return_value={}) as dispatch:
            with self.assertRaisesRegex(ValueError,'运动工具与本任务阶段不一致'):
                harness._invoke('move_relative',{},'execute',{'turn_id':'goal'})
            dispatch.assert_not_called()
            harness._invoke('move_to_pose',{},'execute',{'turn_id':'goal'})
            harness._invoke('move_relative',{},'execute',None)  # explicit manual motion remains independent
        self.assertEqual(dispatch.call_count,2)

    def test_pristine_and_active_goal_hide_redundant_prepare_without_mutating_catalogue(self):
        harness=BottleHarness.__new__(BottleHarness);harness.active_session='goal';harness.sessions={'goal':{}}
        harness.workspace=SimpleNamespace(active_plan=None)
        state={'observation':{'tick':0,'sim_time':0,'episode_finished':False},'diagnostics':{'evaluation':{'invalid':False}}}
        harness.service=SimpleNamespace(snapshot=lambda:state)
        tools=[{'function':{'name':n}} for n in ['observe','prepare_simulation','set_plan']];before=copy.deepcopy(tools)
        visible,meta=harness.tools_for_request(tools)
        self.assertEqual([t['function']['name'] for t in visible],['observe','set_plan']);self.assertEqual(tools,before)
        state['observation'].update(tick=100,sim_time=3)
        harness.workspace.active_plan={'turn_id':'goal'}
        self.assertNotIn('prepare_simulation',[t['function']['name'] for t in harness.tools_for_request(tools)[0]])

    def test_new_goal_in_an_advanced_scene_keeps_prepare_available(self):
        harness=BottleHarness.__new__(BottleHarness);harness.active_session='new-goal';harness.sessions={'new-goal':{}}
        harness.workspace=SimpleNamespace(active_plan={'turn_id':'old-goal'})
        harness.service=SimpleNamespace(snapshot=lambda:{'observation':{'tick':100,'sim_time':3}})
        tools=[{'function':{'name':'prepare_simulation'}}]
        self.assertEqual(harness.tools_for_request(tools)[0],tools)

    def test_tool_filter_preserves_recipient_headers_messages_and_budget(self):
        import json,urllib.request
        from harness.bottle import VisibleRequests
        requests=[];events=[]
        base=SimpleNamespace(urlopen=lambda request,*a,**kw:requests.append(request))
        owner=SimpleNamespace(active_session='goal',workspace=SimpleNamespace(event=lambda *a:events.append(a)),
            tools_for_request=lambda tools:([t for t in tools if t['function']['name']!='prepare_simulation'],{'hidden_tools':['prepare_simulation']}))
        payload={'model':'test-model','messages':[{'role':'user','content':'test'}],'max_tokens':1600,'temperature':0,
            'tools':[{'function':{'name':n}} for n in ['observe','prepare_simulation','set_plan']]}
        request=urllib.request.Request('https://example.test/chat/completions',data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer test-placeholder','Content-Type':'application/json'})
        VisibleRequests(base,owner).urlopen(request)
        sent=requests[0];body=json.loads(sent.data)
        self.assertEqual(sent.full_url,request.full_url);self.assertEqual(sent.header_items(),request.header_items())
        for key in ['model','messages','max_tokens','temperature']:self.assertEqual(body[key],payload[key])
        self.assertEqual(json.loads(request.data),payload);self.assertNotIn('Authorization',json.dumps(events))


class HoldContracts(unittest.TestCase):
    @staticmethod
    def collect(cls,unstable=False):
        command=np.arange(16,dtype=np.float64)/20
        env=SimpleNamespace(data=SimpleNamespace(time=0.),last_action=command.copy(),
            profile={'control_hz':30},evaluation={'held':{'left':False,'right':True}})
        class Planner:
            result={'primitive_success':True,'held_by':'right'}
            def __init__(self):self.env=env;self.result=copy.deepcopy(self.result)
            def _check(self):pass
            def actions(self):return iter(())
        class Audit:
            def __init__(self,env):self.env=env
            def read(self):pass
            def summary(self):
                status='unstable' if unstable and 2.1<=self.env.data.time<=2.6 else 'stable_within_audit_window'
                return {'arms':{'right':{'status':status}}}
        planner=Planner();adapter=cls(planner,4);commands=[]
        with patch('sim.decision_timing.GraspStability',Audit):
            for action,diag in adapter.actions():commands.append(action.copy());env.data.time+=1/30
        return commands,adapter.result,command

    def test_continuous_audit_preserves_exact_original_commands(self):
        original,_,expected=self.collect(HoldAudit)
        actual,result,_=self.collect(ContinuousHoldAudit)
        self.assertEqual(len(actual),120);self.assertEqual(len(original),len(actual))
        for a,b in zip(actual,original):np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,expected)
        self.assertTrue(result['hold_audit']['continuous_full_windows_stable'])

    def test_failed_earlier_window_cannot_be_hidden_by_a_stable_final_window(self):
        self.assertTrue(self.collect(HoldAudit,unstable=True)[1]['primitive_success'])
        from sim.tamp import PlanningFailure
        with self.assertRaises(PlanningFailure):self.collect(ContinuousHoldAudit,unstable=True)

    def test_early_contact_loss_preserves_continuous_audit_type_and_failed_result(self):
        from sim.tamp import PlanningFailure
        env=SimpleNamespace(data=SimpleNamespace(time=0.),last_action=np.zeros(16),
            profile={'control_hz':30},evaluation={'held':{'right':True}})
        planner=SimpleNamespace(env=env,result={'primitive_success':True,'held_by':'right'},
            _check=lambda:None,actions=lambda:iter(()))
        class Audit:
            def __init__(self,env):pass
            def read(self):pass
            def summary(self):return {'arms':{'right':{'status':'stable_within_audit_window'}}}
        adapter=ContinuousHoldAudit(planner,4)
        with patch('sim.decision_timing.GraspStability',Audit):
            with self.assertRaises(PlanningFailure):
                for action,diag in adapter.actions():
                    env.data.time+=1/30
                    if env.data.time>.3:env.evaluation['held']['right']=False
        audit=adapter.result['hold_audit']
        self.assertFalse(adapter.result['primitive_success']);self.assertFalse(audit['audit_completed'])
        self.assertFalse(audit['continuous_full_windows_stable']);self.assertEqual(audit['validation_method'],'all_full_windows')
        self.assertEqual(audit['full_window_checks'],0)


class PlacementContracts(unittest.TestCase):
    """Capture routing logic only; native task outcomes are scored separately."""
    @staticmethod
    def planner():
        p=BottlePlacementPlanner.__new__(BottlePlacementPlanner)
        p.env=SimpleNamespace(object_id='bottle',data=SimpleNamespace(qpos=np.zeros(7),xpos=np.array([[.54,-.13,1.01]])),
            bottle_id=0,evaluation={'held':{'right':True},'invalid':False,'non_target_object_guard':{'passed':True}},
            placement_surface=lambda name:{'top_z':1.005},_bottle_bounds=lambda:(np.array([0,0,1.01]),None))
        p.task='place_bottle';p.placement_surface='table';p.phase='place';p.carried=(np.zeros(3),np.eye(3))
        p.result={};p.side='right';p.qids=np.arange(7);p.placement_xy=[.56,-.13]
        return p

    def test_collision_failure_never_enters_object_capture(self):
        from sim.tamp import PlanningFailure
        p=self.planner()
        def failed(owner,*a,**kw):
            owner.result['tracking_feedback']={'position_error_m':.02}
            raise PlanningFailure('goal_collision','checked path collision')
            yield
        with patch('sim.object_planner.ObjectPlanner._move',failed):
            with self.assertRaises(PlanningFailure):list(p._move(np.zeros(3),True,position_tolerance=.003))
        self.assertNotIn('lowering_capture',p.result)

    def test_initial_IK_failure_never_enters_object_capture(self):
        from sim.tamp import PlanningFailure
        p=self.planner()
        def failed(*a,**kw):
            raise PlanningFailure('ik_unreachable','no completed native segment')
            yield
        with patch('sim.object_planner.ObjectPlanner._move',failed):
            with self.assertRaises(PlanningFailure):list(p._move(np.zeros(3),True,position_tolerance=.003))
        self.assertNotIn('lowering_capture',p.result)

    def test_completed_capture_preserves_commands_and_rejects_lost_hold(self):
        from sim.tamp import PlanningFailure
        command=np.arange(16,dtype=float)
        def completed(owner,*a,**kw):
            yield command.copy(),{'settling':True}
            owner.result['tracking_feedback']={'position_error_m':.02}
            raise PlanningFailure('ik_unreachable','loaded compensation target')
        p=self.planner();before=p.env.data.qpos.copy()
        with patch('sim.object_planner.ObjectPlanner._move',completed),patch.object(p,'_collision',return_value=False):
            rows=list(p._move(np.zeros(3),True,position_tolerance=.003))
        np.testing.assert_array_equal(rows[0][0],command);np.testing.assert_array_equal(before,p.env.data.qpos)
        self.assertTrue(p.result['lowering_capture']['original_release_and_placement_gates_unchanged'])
        p=self.planner();p.env.evaluation['held']['right']=False
        with patch('sim.object_planner.ObjectPlanner._move',completed):
            with self.assertRaises(PlanningFailure):list(p._move(np.zeros(3),True,position_tolerance=.003))
        self.assertNotIn('lowering_capture',p.result)

    def test_alignment_rejects_non_upright_prediction_and_restores_attitude(self):
        from sim.tamp import PlanningFailure
        p=self.planner();p.result={'lowering_capture':{'native_segment_completed':True}}
        attitude=np.eye(3);p.rotation=attitude
        tilted=np.eye(3);tilted[2,2]=.97;p.data=SimpleNamespace(xmat=tilted.reshape(1,9))
        with patch('sim.object_planner.ObjectPlanner._ik',return_value=np.zeros(7)),patch.object(p,'_at'):
            with self.assertRaises(PlanningFailure):p._ik(np.zeros(3),True,position_tolerance=.0002)
        self.assertIs(p.rotation,attitude);self.assertNotIn('object_alignment_position_IK',p.result)

    def test_full_pose_targets_keep_the_original_attitude_constraint(self):
        p=self.planner();p.result={'lowering_capture':{'native_segment_completed':True}}
        attitude=np.eye(3);p.rotation=attitude
        def native(owner,*a,**kw):
            self.assertIs(owner.rotation,attitude);return np.zeros(7)
        with patch('sim.object_planner.ObjectPlanner._ik',native):p._ik(np.zeros(3),True,position_tolerance=.0015)

    def test_pre_release_never_opens_a_moving_bottle(self):
        from sim.tamp import PlanningFailure
        p=self.planner();p.phase='release_left';p.result={'lowering_capture':{'native_segment_completed':True}}
        p.env.data.xpos[0,:2]=[.56,-.13];p.env.data.qvel=np.array([.03,0,0,0,0,0])
        p.env.placement_target={'tolerance_m':.01};p.env.registry={'bottle':{'dofadr':0}}
        p.env._bottle_bounds=lambda:(np.array([0,0,1.007]),None)
        p.base_action=np.zeros(16);p.base_action[15]=.05;p._check=lambda:None
        commands=[]
        with patch('sim.object_planner.ObjectPlanner._grip') as opening:
            with self.assertRaises(PlanningFailure):
                for action,_ in p._grip('right',-.9):commands.append(action.copy())
            opening.assert_not_called()
        self.assertEqual(len(commands),120)
        for action in commands:np.testing.assert_array_equal(action,p.base_action)


class MemoryContracts(unittest.TestCase):
    def test_grasp_purpose_groups_compatible_methods_without_raw_target_pose(self):
        import json
        from harness.bottle import consolidate_hold
        from harness.object_memory import index_entry
        rows=[]
        for i,target in enumerate([None,[.52,.18],[.60,.15]]):
            feedback={'object_id':'bottle','primitive_success':True}
            if target:feedback['future_placement']={'target_xy_m':target,'yaw_rad':0.}
            rows.append({'id':str(i),'tool':'pick','outcome':'success','evidence_kind':'simulator_oracle',
                'arguments':{'object_id':'bottle','arm':'left'},'feedback':feedback,
                'context':'same','created':float(i),'evidence':{'run_id':'raw-'+str(i)}})
        before=copy.deepcopy(rows);lessons=consolidate_hold(rows)
        self.assertEqual(rows,before);self.assertEqual(len(lessons),2)
        conditional=next(x for x in lessons if x.get('conditions',{}).get('grasp_intent'))
        self.assertEqual(conditional['repeat_count'],2)
        index=json.dumps(index_entry(conditional));self.assertNotIn('target_xy_m',index);self.assertNotIn('receipt_id',index)
        self.assertEqual(len(consolidate_hold(rows,[conditional['id']])),1)

    def test_duration_and_audit_strength_do_not_merge_or_resurrect_forgotten_lessons(self):
        from harness.bottle import consolidate_hold
        from harness.poses import consolidate as original
        rows=[]
        for i,(seconds,continuous) in enumerate([(2,True),(10,True),(10,True),(10,False)]):
            audit={'requested_seconds':seconds,'actual_seconds':seconds}
            if continuous:audit['continuous_full_windows_stable']=True
            rows.append({'id':str(i),'tool':'handover','outcome':'success','evidence_kind':'simulator_oracle',
                'arguments':{'object_id':'bottle','from_arm':'left','to_arm':'right','hold_seconds':seconds},
                'feedback':{'object_id':'bottle','primitive_success':True,'hold_audit':audit},
                'context':'same-cell','created':float(i),'evidence':{'run_id':'raw-'+str(i)}})
        before=copy.deepcopy(rows);lessons=consolidate_hold(rows)
        self.assertEqual(rows,before);self.assertEqual(len(lessons),3)
        self.assertEqual(sorted(x['repeat_count'] for x in lessons),[1,1,2])
        self.assertEqual({x['conditions']['hold_validation'] for x in lessons},{'all_full_windows','final_window'})
        deleted=next(x['id'] for x in lessons if x['repeat_count']==2)
        self.assertEqual(len(consolidate_hold(rows,[deleted])),2)
        old=original(rows)[0]['id'];self.assertEqual(consolidate_hold(rows,[old]),[])


class APIEvidenceContracts(unittest.TestCase):
    def test_valid_relative_GPU_transport_is_distinct_from_empty_or_failed_motion(self):
        from tools.check_bottle_api import gpu_evidence
        move={'execution':{'skill':'move_relative'},'arguments':{'planner':'curobo','arm':'left'},
            'feedback':{'primitive_success':True,'planner':'official cuRobo V2 GPU IK/trajectory optimization',
                'payload_state':'single_hold','native_path_revalidation':True,'fallback':False,'worker_port':8004,
                'backend_metadata':{'ready':True,'worker_source_sha256':'frozen-worker'},'plans':[{'native':True}],
                'verification_source':'robot_forward_kinematics','position_error_m':.004,'rotation_error_rad':.01}}
        self.assertTrue(gpu_evidence({'receipts':[move]},'frozen-worker'))
        for field,value in [('payload_state','empty'),('primitive_success',False),('fallback',True),('position_error_m',.013)]:
            bad=copy.deepcopy(move);bad['feedback'][field]=value
            self.assertFalse(gpu_evidence({'receipts':[bad]},'frozen-worker'))

    def test_stale_proposal_is_archived_retry_but_failed_native_motion_cannot_pass(self):
        from tools.check_bottle_api import route_evidence
        rows=[{'execution':{'skill':'pick'},'feedback':{'primitive_success':True,'held_by':'left',
            'verification_source':'native_two_finger_contact_and_lift'}},
            {'execution':{'skill':'move_relative'},'arguments':{'arm':'left'},'feedback':{'object_id':'bottle'}},
            {'execution':{'skill':'place'},'feedback':{}}]
        self.assertTrue(route_evidence(rows,False,'left'))
        rows[1]['arguments']['arm']='right';self.assertFalse(route_evidence(rows,False,'left'))
        rows[1]['arguments']['arm']='left';rows.insert(1,copy.deepcopy(rows[0]))
        self.assertFalse(route_evidence(rows,False,'left'))


if __name__=='__main__':unittest.main(verbosity=2)
