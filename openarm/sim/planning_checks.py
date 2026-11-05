"""Native latency integration and stale-proposal rejection, no model API."""
import tempfile
import threading
import queue
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .environment import OpenArmEnv, ROOT
from .planning_timing import PlanningTiming, state, revalidate
from .tamp import PlanningFailure
from .service import SimulationService
from .recording import Recorder
from .relocation import RelocationPlanner
from harness.workspace import Workspace


class Clock:
    value = 0.
    def __call__(self):return self.value


class PlanningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env = OpenArmEnv(images=False)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):
        self.env.reset(seed=0)
        self.clock = Clock()
        self.clock.value = 0.

    def drive(self, source, *, mode='measured_delay', cancelled=lambda:False):
        timing = PlanningTiming(mode, clock=self.clock)
        rows, events = [], []
        self.timing, self.rows, self.events = timing, rows, events
        for action, diagnostic, held in timing.actions(self.env, source, events.append, cancelled):
            before = float(self.env.data.time)
            _, _, term, trunc, info = (self.env.hold_tamp(action,diagnostic['planning_timing']['duration_s'])
                if held else self.env.step_tamp(action))
            timing.committed(held, before, float(self.env.data.time))
            rows.append((action.copy(), diagnostic, held))
            if term or trunc:break
        return rows, events

    def test_previous_target_held_before_new_command_and_wait_not_command_budget(self):
        target = self.env.last_action.copy()
        # A candidate changes only the right gripper. It must not reach physics
        # during the declared synthetic planner delay.
        candidate = target.copy();candidate[15] = -.5
        def source():
            self.clock.value += .1
            yield candidate, {'phase':'test'}
        rows, _ = self.drive(source())
        self.assertEqual((self.timing.wait_ticks,self.timing.commands), (4,1))
        self.assertTrue(all(np.array_equal(q,target) for q,_,held in rows if held))
        self.assertTrue(np.array_equal(rows[4][0],candidate))
        self.assertAlmostEqual(self.timing.integrated_seconds,.1,places=8)
        self.assertAlmostEqual(rows[4][1]['planning_timing']['source_age_sim_seconds'],.1,places=8)
        self.assertAlmostEqual(float(self.env.data.time),.133,places=8)

    def test_cumulative_quantization_does_not_invent_one_tick_per_cheap_resume(self):
        def source():
            for _ in range(30):
                self.clock.value += .0001
                yield self.env.last_action.copy(), {}
        self.drive(source())
        self.assertEqual(self.timing.commands,30)
        self.assertEqual(self.timing.wait_units,3)
        self.assertAlmostEqual(self.timing.wall_seconds,.003)
        self.assertLess(self.timing.summary()['rounding_overhead_seconds'],.001001)

    def test_failed_computation_still_consumes_time_without_new_target(self):
        target = self.env.last_action.copy()
        def source():
            self.clock.value += .1
            raise PlanningFailure('path_not_found','declared test failure')
            yield
        with self.assertRaisesRegex(PlanningFailure,'declared test failure'):self.drive(source())
        self.assertEqual((self.timing.wait_ticks,self.timing.commands),(4,0))
        self.assertTrue(all(np.array_equal(row[0],target) for row in self.rows))
        self.assertFalse(self.events[0]['response_valid'])

    def test_cancel_during_computation_discards_candidate(self):
        cancelled=[False]
        def source():
            self.clock.value += .2;cancelled[0]=True
            yield self.env.last_action.copy(), {}
        with self.assertRaisesRegex(PlanningFailure,'停止'):self.drive(source(),cancelled=lambda:cancelled[0])
        self.assertEqual((self.env.tick,self.timing.commands,self.timing.wait_ticks),(0,0,0))

    def test_declared_short_fixture_budget_stops_wait_without_applying_candidate(self):
        # Explicit shortened unit fixture only; production budgets unchanged.
        self.env._task_budget=.1
        self.env.task_termination='skill'
        def source():
            self.clock.value += .5
            candidate=self.env.last_action.copy();candidate[15]=.05
            yield candidate, {}
        self.drive(source())
        self.assertTrue(self.env._done)
        self.assertEqual(self.timing.commands,0)
        self.assertLessEqual(self.timing.integrated_seconds,.134)
        self.assertTrue(all(row[2] for row in self.rows))

    def test_paused_baseline_records_wall_cost_without_integrating_it(self):
        def source():
            self.clock.value += .2
            yield self.env.last_action.copy(), {}
        self.drive(source(),mode='paused')
        self.assertEqual((self.timing.wait_ticks,self.timing.commands),(0,1))
        self.assertAlmostEqual(self.timing.wall_seconds,.2)

    def test_partial_wait_preserves_command_cadence_and_reset_clears_offset(self):
        target=self.env.last_action.copy()
        self.env.step_tamp(target);stamp=float(self.env.data.time)
        self.env.hold_tamp(target,.002)
        self.assertAlmostEqual(float(self.env.data.time)-stamp,.002,places=8)
        before=float(self.env.data.time);self.env.step_tamp(target)
        self.assertAlmostEqual(float(self.env.data.time)-before,.034,places=8)
        wrong=target.copy();wrong[15]=-.5
        with self.assertRaisesRegex(ValueError,'exact previous'):self.env.hold_tamp(wrong,.001)
        self.env.reset(seed=0);self.env.step_tamp(self.env.last_action.copy())
        self.assertAlmostEqual(float(self.env.data.time),.033,places=8)

    def test_stale_guard_rejects_object_translation_rotation_and_hold_change(self):
        before=state(self.env)
        # Mutate the copied evidence, never live physical state.
        before['object'][0] += .004
        self.assertEqual(revalidate(self.env,before)['reason'],'scene_drift')
        before=state(self.env);before['held']['right']=True
        self.assertEqual(revalidate(self.env,before)['reason'],'holding_state_changed')
        before=state(self.env);theta=.06;c,s=np.cos(theta),np.sin(theta)
        before['object_rotation']=np.array([[c,-s,0],[s,c,0],[0,0,1]])@before['object_rotation']
        self.assertEqual(revalidate(self.env,before)['reason'],'scene_drift')

    def test_legacy_receipt_is_not_counted_as_new_planner_timing_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='.planning-check-') as out:
            workspace=Workspace(Path(out)/'test.sqlite3')
            try:
                context={'robot':'fixture','contract':'fixture'}
                for key,planning in [('old',None),('new',{'schema':'planning-timing/1','mode':'measured_delay'})]:
                    feedback={'primitive_success':True,'decision_timing_mode':'measured_delay'}
                    if planning:feedback['planning_timing']=planning
                    workspace.record_feedback(key,'execute','test','fixture','pick',{'arm':'right'},feedback,
                        context,evidence_kind='simulator_oracle')
                groups=workspace.capability_profile('execute',context=context,include_oracle=True)
                self.assertEqual({g['planning_timing'] for g in groups}, {'legacy_unmodeled','planning-timing/1:measured_delay'})
                self.assertTrue(all(g['counts']['success']==1 for g in groups))
            finally:workspace.db.close()

    def test_service_records_constructor_latency_and_every_hold_before_new_command(self):
        env=self.env
        class ControlledPlanner:
            CONTRACT={'planner':'controlled service contract fixture, not a benchmark'}
            def __init__(self,*args,**kwargs):
                time.sleep(.055)
                self.task='move_relative';self.stats={};self.phase='fixture'
                self.result={'primitive_success':False}
            def actions(self):
                yield env.last_action.copy(), {'phase':'fixture'}
                self.result.update(primitive_success=True,status='completed')
        service=SimulationService.__new__(SimulationService)
        service.env=env;service.epoch='fixture';service.keyframes=[]
        service.tamp_planner_factory=ControlledPlanner
        service.job={'kind':'tamp','execution_id':'fixture'}
        service.stop_job=threading.Event();service.shutdown=threading.Event();service.completed=queue.Queue()
        service._publish=lambda:None;service._capture=lambda *args,**kwargs:None
        service.snapshot=lambda:{'observation_id':'fixture:'+str(env.tick), 'observation':{'tick':env.tick}}
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='.planning-service-') as out:
            def recorder(*args,**kwargs):return Recorder(*args,root=out,**kwargs)
            class UnexpectedGlobalPlanner:
                CONTRACT={'planner':'module-global planner must not be used'}
                def __init__(self,*args,**kwargs):raise AssertionError('service ignored its injected TAMP router')
            with patch('sim.service.TampPlanner',UnexpectedGlobalPlanner),patch('sim.service.Recorder',side_effect=recorder):
                service._run('tamp',{'observation_id':'fixture:0','task':'move_relative',
                    'timing_mode':'measured_delay','realtime':False})
            summary=json.loads(next(Path(out).glob('*/summary.json')).read_text())
            rows=[json.loads(line) for line in next(Path(out).glob('*/transitions.jsonl')).read_text().splitlines()]
            metric=summary['metrics']['planning_timing']
            self.assertEqual(summary['status'],'primitive_completed')
            self.assertEqual(metric['native_commands'],1)
            self.assertGreaterEqual(metric['hold_transitions'],2)
            self.assertEqual(len(rows),metric['native_commands']+metric['hold_transitions'])
            self.assertEqual(rows[0]['diagnostics']['planning_timing']['action_source'],'planning_hold')
            self.assertTrue(all(row['action']==rows[0]['action'] for row in rows))
            self.assertIsNone(service.job)
            self.assertEqual(service.completed.get_nowait()['execution_id'],'fixture')

    def test_interrupted_composite_preserves_child_feedback_without_claiming_completion(self):
        env=self.env
        class Child:
            def __init__(self,*args,**kwargs):
                self.deadline=time.monotonic()+90;self.stats={'ik_attempts':2}
                self.result={'plans':[{'success':True,'path_not_yet_executed':True}]}
            def actions(self):yield env.last_action.copy(), {}
        planner=RelocationPlanner(env,'relocate_object',parameters={
            'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0.]})
        source=planner._skill('move_relative',{'arm':'right'},Child)
        next(source);source.close()
        row=planner.result['substeps'][0]
        self.assertEqual(row['status'],'aborted')
        self.assertTrue(row['feedback']['plans'][0]['path_not_yet_executed'])
        self.assertEqual(planner.stats['ik_attempts'],2)
        self.assertFalse(planner.result['primitive_success'])
