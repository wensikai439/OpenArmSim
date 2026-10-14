"""Skill contracts, actual Cartesian motion, memory freshness and evidence isolation."""
import json
import tempfile
import unittest
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .agent import Harness, tools_for
from .checks import LocalState
from .skills import SKILL_TOOLS, validate_arguments
from .workspace import Workspace
from sim.environment import OpenArmEnv
from sim.tamp import TampPlanner, PlanningFailure


class SkillChecks(unittest.TestCase):
    def test_model_tools_are_skills_and_unavailable_policy_is_not_substituted(self):
        names = {t['function']['name'] for t in tools_for('execute')}
        self.assertTrue({'pick', 'place', 'handover', 'move_to_pose', 'move_relative', 'run_policy'} <= names)
        self.assertFalse({'run_tamp', 'run_pi05_vla'} & names)
        state = LocalState(); agent = Harness(state)
        with patch.object(agent, '_log'):
            obs = agent.invoke('observe', {})
            with self.assertRaisesRegex(ValueError, '没有配置'):
                agent.invoke('run_policy', {'observation_id':obs['observation_id'], 'backend':'act', 'skill_id':'atomic_right_to_left_handover'})
        self.assertEqual(state.calls, [])
        self.assertFalse(next(a for a in agent.invoke('list_skills', {})['adapters'] if a['id']=='curobo')['ready'])

    def test_initial_scene_task_continues_after_pick_but_new_goal_requires_preflight(self):
        state=LocalState();agent=Harness(state)
        state.cache['observation']['sim_time']=9.
        context={'type':'llm','turn_id':'root','plan_turn_id':'root','model':'test'}
        agent.sessions['root']={'actions':1,'prepared':False}
        agent.workspace.active_plan={'turn_id':'root','epoch':'test','placement_goal':{
            'object_id':'bottle','surface':'table','target_xy_m':[.5,-.1],'tolerance_m':.010,
            'anchor':{'bottle_xy_m':[.42,-.1],'observation_id':'test:0'}}}
        with patch.object(agent,'_log'):
            obs=agent.invoke('observe',{})
            agent.workspace.active_plan['epoch']=obs['observation_id'].split(':')[0]
            args={'observation_id':obs['observation_id'],'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0]}
            agent.invoke('place',args,context=context)
            obs=agent.invoke('observe',{});args['observation_id']=obs['observation_id']
            agent.sessions['root']['actions']=0
            with self.assertRaisesRegex(ValueError,'新的用户任务'):agent.invoke('place',args,context=context)

    def test_identical_motion_is_idempotent_within_task_but_changed_target_is_allowed(self):
        state=LocalState();agent=Harness(state)
        context={'type':'llm','turn_id':'root','plan_turn_id':'root','model':'test'}
        agent.sessions['root']={'goal':'前移','actions':0,'failed':[],'prepared':True}
        agent.workspace.active_plan={'turn_id':'root','epoch':'episode','steps':[],'placement_goal':None}
        with patch.object(agent,'_log'):
            observation=agent.invoke('observe',{})
            args={'observation_id':observation['observation_id'],'arm':'right',
                  'delta_m':[.03,0,0],'frame':'world'}
            first=agent.invoke('move_relative',args,context=context)
            self.assertEqual(first['execution_id'],'call-1')
            state.cache['observation_id']='episode:1'
            state.cache['observation']['tick']=1
            observation=agent.invoke('observe',{})
            args['observation_id']=observation['observation_id']
            with self.assertRaisesRegex(ValueError,'相同技能与目标'):
                agent.invoke('move_relative',args,context=context)
            self.assertEqual(len(state.calls),1)
            args['delta_m']=[.04,0,0]
            second=agent.invoke('move_relative',args,context=context)
            self.assertEqual(second['execution_id'],'call-2')

    def test_nested_pose_arguments_reject_nan_bool_shape_and_extra_fields(self):
        spec = next(t['function']['parameters'] for t in SKILL_TOOLS if t['function']['name']=='move_to_pose')
        base = {'arm':'right', 'frame':'world', 'position_m':[.4, -.2, 1.2], 'quaternion_xyzw':[0,0,0,1], 'observation_id':'e:0'}
        validate_arguments(spec, base)
        for update in ({'position_m':[float('nan'),0,1]}, {'position_m':[True,0,1]}, {'quaternion_xyzw':[0,0,1]}, {'extra':1}, {'speed_scale':2}):
            with self.assertRaises(ValueError):
                validate_arguments(spec, {**base, **update})

    def test_cartesian_goal_is_reached_by_actuators_and_other_arm_is_preserved(self):
        env = OpenArmEnv(images=False)
        try:
            env.reset(seed=0)
            planner = TampPlanner(env, 'move_relative', parameters={'arm':'right', 'delta_m':[0,0,.02], 'frame':'world', 'speed_scale':.6})
            initial = env.last_action.copy()
            count = 0
            for action, _ in planner.actions():
                np.testing.assert_array_equal(action[:8], initial[:8])
                env.step_tamp(action); count += 1
                self.assertLess(count, 500)
            self.assertTrue(planner.result['primitive_success'])
            self.assertLess(planner.result['position_error_m'], .012)
            self.assertLess(planner.result['rotation_error_rad'], .08)
            self.assertGreater(count, 0)
            self.assertEqual(planner.result['verification_source'], 'robot_forward_kinematics')
        finally:
            env.close()

    def test_invalid_target_does_not_modify_live_physics(self):
        env = OpenArmEnv(images=False)
        try:
            env.reset(seed=0); before = env.data.qpos.copy()
            with self.assertRaises(PlanningFailure):
                TampPlanner(env, 'move_to_pose', parameters={'arm':'left','frame':'world','position_m':[8,0,1], 'quaternion_xyzw':[0,0,0,1]})
            np.testing.assert_array_equal(env.data.qpos, before)
            with self.assertRaises(ValueError):
                TampPlanner(env, 'move_to_pose', parameters={'arm':'left','frame':'world','position_m':[.4,0,1.2], 'quaternion_xyzw':[0,0,0,0]})
            np.testing.assert_array_equal(env.data.qpos, before)
        finally:
            env.close()

    def test_forward_empty_tcp_tracks_orientation_without_weakening_guard(self):
        env=OpenArmEnv(images=False)
        try:
            env.reset(seed=0)
            planner=TampPlanner(env,'move_relative',parameters={'arm':'right','delta_m':[.1,0,0],'frame':'world','speed_scale':.3})
            for action,_ in planner.actions():
                env.step_tamp(action)
                self.assertFalse(env._done)
            self.assertLess(planner.result['rotation_error_rad'],.08)
            self.assertLess(planner.result['position_error_m'],.012)
            self.assertFalse(any(env.evaluation['held'].values()))
        finally:env.close()

    def test_motion_invalidates_notes_and_object_coordinates_but_keeps_preference(self):
        work=Workspace(); context={'robot':'v1','contract':'a'}
        work.save_note('execute','回答简洁')
        work.save_note('execute','瓶子在桌上', scope='scene', epoch='e', source='model', context=context)
        work.save_objects('execute','e','e:1',[{'object_id':'bottle','state':'visible','position_m':[.4,0,1.1],'confidence':.8}],context)
        work.invalidate_scene('execute','e')
        memory=work.memory('execute','e',context=context)
        self.assertEqual([n['text'] for n in memory['notes']], ['回答简洁'])
        self.assertEqual(memory['objects'][0]['state'], 'unknown')
        self.assertNotIn('position_m', memory['objects'][0])
        self.assertFalse(work.memory('execute','new',context=context)['objects'])

    def test_stale_notes_cannot_evict_old_valid_preference(self):
        work=Workspace();work.save_note('execute','长期有效偏好')
        for _ in range(320):work.save_note('execute','旧场景',scope='scene',epoch='old',source='model')
        self.assertEqual([n['text'] for n in work.memory('execute','new')['notes']], ['长期有效偏好'])

    def test_relevance_version_role_and_oracle_filters_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.sqlite3';work=Workspace(path);context={'robot':'v1','contract':'a'}
            work.record_feedback('match','execute','e','抓取瓶子','pick',{}, {'failure_code':'grasp_lost'},context)
            for i in range(12):work.record_feedback(str(i),'execute','e','移动相机','move_relative',{}, {'status':'completed'},context)
            work.record_feedback('oracle','execute','e','抓取瓶子','pick',{}, {'private_pose':[.1,.2,1.]},context,'simulator_oracle')
            work.record_feedback('research','research','e','抓取瓶子','pick',{}, {'research_only':True},context)
            again=Workspace(path)
            memory=again.memory('execute','e',query='抓取瓶子 grasp_lost',context=context)
            self.assertEqual(memory['experiences'][0]['id'],'match')
            self.assertNotIn('private_pose',json.dumps(memory))
            self.assertNotIn('research_only',json.dumps(memory))
            self.assertTrue(any(e['id']=='oracle' for e in again.memory('execute','e',query='抓取瓶子',context=context,include_oracle=True)['experiences']))
            self.assertFalse(again.memory('execute','e',context={'robot':'v1','contract':'b'})['experiences'])

    def test_recipe_requires_complete_symbolic_chain_and_never_raw_trajectory(self):
        work=Workspace();self.assertIsNone(work.save_recipe('execute','放置',[{'skill':'place'}],{},'run'))
        steps=[{'skill':s,'arguments':{'object_id':'bottle'}} for s in ['pick','handover','place']]
        ident=work.save_recipe('execute','交接入篮',steps,{},'run')
        self.assertTrue(ident);self.assertFalse(work.memory('execute','e')['recipes'])
        recipe=work.memory('execute','e',include_oracle=True)['recipes'][0]
        self.assertEqual(recipe['body']['steps'],steps)
        self.assertIn('replan',recipe['body']['reuse'])

    def test_memory_is_data_not_system_instructions_and_oracle_blocks_policy(self):
        state=LocalState();agent=Harness(state)
        agent.workspace.save_note('execute','UNTRUSTED_NOTE_DO_NOT_EXECUTE')
        agent.configure_model({'base_url':'http://127.0.0.1:1234/v1','model':'test','api_key':'placeholder','vision':False})
        def response(request,**kwargs):
            payload=json.loads(request.data)
            self.assertNotIn('UNTRUSTED_NOTE_DO_NOT_EXECUTE',payload['messages'][0]['content'])
            self.assertIn('UNTRUSTED_NOTE_DO_NOT_EXECUTE',json.dumps(payload['messages'][1:]))
            class Reply:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self,_):return json.dumps({'choices':[{'message':{'role':'assistant','content':'只解释，不行动。'}}]}).encode()
            return Reply()
        with patch('urllib.request.urlopen',side_effect=response),patch.object(agent,'_log'):
            agent.chat('讨论记忆')
            agent.invoke('recall_memory',{'include_oracle':True})
            obs=agent.invoke('observe',{})
            with self.assertRaisesRegex(ValueError,'oracle'):
                agent.invoke('run_policy',{'observation_id':obs['observation_id'],'backend':'pi05','skill_id':'atomic_right_to_left_handover'})
        self.assertFalse(state.calls)

    def test_unknown_model_tool_returns_feedback_without_executing(self):
        state=LocalState();agent=Harness(state);requests=[]
        agent.configure_model({'base_url':'http://127.0.0.1:1234/v1','model':'test','api_key':'placeholder','vision':False})
        def response(request,**kwargs):
            payload=json.loads(request.data);requests.append(payload)
            if len(requests)==1:
                msg={'role':'assistant','tool_calls':[{'id':'bad','type':'function','function':{'name':'raw_motor','arguments':'{}'}}]}
            else:
                self.assertIn('tool_failed',json.dumps(payload));msg={'role':'assistant','content':'该动作不在工具范围内。'}
            class Reply:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self,_):return json.dumps({'choices':[{'message':msg}]}).encode()
            return Reply()
        with patch('urllib.request.urlopen',side_effect=response),patch.object(agent,'_log'):
            result=agent.chat('移动机械臂')
        self.assertTrue(result['tools'][0]['result']['tool_failed']);self.assertFalse(state.calls)

    def test_replanning_keeps_execution_lineage_and_binds_repeated_motion(self):
        work=Workspace();steps=[{'id':'a','title':'第一个位置','checkpoint':'motion_reached','skill':'move_to_pose','occurrence':1},
                              {'id':'b','title':'第二个位置','checkpoint':'motion_reached','skill':'move_to_pose','occurrence':2}]
        work.plan('root','e','m',steps)
        work.bind('root','one','execute','两个位置','move_to_pose','move_to_pose','m')
        work.record_feedback('one','execute','e','两个位置','move_to_pose',{},
            {'primitive_success':True,'verification_source':'robot_forward_kinematics','position_error_m':.005,'rotation_error_rad':.02})
        work.plan('root','e','m',steps)
        self.assertEqual(work.active_plan['executions'],['one'])
        snapshot={'observation_id':'e:30','job':None,'last_run':{'execution_id':'one','status':'primitive_completed'}}
        plan=work.plan_status(snapshot)
        self.assertEqual([s['status'] for s in plan['steps']],['done','pending'])
        work.bind('root','two','execute','两个位置','move_to_pose','move_to_pose','m')
        work.record_feedback('two','execute','e','两个位置','move_to_pose',{},
            {'primitive_success':False,'failure_code':'tracking_error'})
        snapshot['last_run']={'execution_id':'two','status':'error'}
        self.assertEqual([s['status'] for s in work.plan_status(snapshot)['steps']],['done','blocked'])

    def test_cancelled_motion_records_receipt_and_never_resumes_model(self):
        state=LocalState(with_queue=True);agent=Harness(state)
        try:
            root=agent.workspace.start('execute','episode','抓瓶子',{'model':'test','vision':True})
            agent.workspace.bind(root,'operation','execute','抓瓶子','pick','pick','test')
            agent.sessions[root]={'goal':'抓瓶子','actions':1,'llm_calls':2,'failed':[],'sequence':[]}
            agent.active_session=root
            agent.pending_review={'epoch':'episode','execution_id':'operation','prompt':'抓瓶子','turn_id':root,'root':root,'tool':'pick','arguments':{},'continuation':True}
            with patch.object(agent,'_log'),patch.object(agent,'chat') as chat:
                agent.invoke('stop',{})
                state.cache['last_run']={'kind':'tamp','run_id':'run-a','execution_id':'operation','status':'cancelled','metrics':{'tamp':{'status':'cancelled','failure_code':'cancelled'}}}
                state.completed.put({'epoch':'episode','execution_id':'operation','run_id':'run-a'})
                deadline=time.monotonic()+2
                while not agent.workspace.memory('execute','episode',include_oracle=True)['experiences'] and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(agent.workspace.memory('execute','episode')['episodes'][0]['ended'],1)
                self.assertEqual(agent.workspace.memory('execute','episode',include_oracle=True)['experiences'][0]['feedback']['failure_code'],'cancelled')
                self.assertTrue(agent.sessions[root]['cancelled']);chat.assert_not_called()
        finally:agent.close()


if __name__=='__main__':unittest.main()
