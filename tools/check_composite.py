"""Adversarial child-receipt binding checks; physical/API checks are separate."""
import copy
import unittest

from harness.composite import CompositeWorkspace,validate_composite_plan,preserved_anchor


def row(skill='pick',**updates):
    f={'primitive_success':True,'verification_source':'native_two_finger_contact_and_lift','held_by':'right'}
    r={'skill':skill,'status':'completed','parameters':{'arm':'right'},'feedback':f};r.update(updates);return r


class CompositeChecks(unittest.TestCase):
    def test_fine_plan_keeps_parent_identity_and_does_not_edit_model_input(self):
        steps=[{'id':'p','title':'右手抓取','checkpoint':'grasp_confirmed','skill':'relocate'}];old=copy.deepcopy(steps)
        result=validate_composite_plan({"steps":steps})
        self.assertEqual(steps,old);self.assertEqual(result[0]['skill'],'relocate');self.assertEqual(result[0]['subskill'],'pick')

    def test_incompatible_child_and_bool_occurrence_are_rejected(self):
        for extras in ({'subskill':'handover'},{'sub_occurrence':True},{'sub_occurrence':0}):
            with self.assertRaises(ValueError):validate_composite_plan({'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate',**extras}]})

    def test_parent_success_cannot_invent_missing_child(self):
        step=validate_composite_plan({'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate'}]})[0]
        self.assertFalse(CompositeWorkspace._step_done(step,{'primitive_success':True},{}))
        self.assertTrue(CompositeWorkspace._step_done(step,{'substeps':[row()]},{}))

    def test_wrong_native_source_status_or_receiver_never_passes(self):
        step=validate_composite_plan({'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate'}]})[0]
        for r in [row(status='planning'),row(feedback={'primitive_success':True,'held_by':'right','verification_source':'robot_forward_kinematics'}),
                  row(parameters={'arm':'left'})]:
            self.assertFalse(CompositeWorkspace._step_done(step,{'substeps':[r]},{}))

    def test_repeated_children_require_explicit_matching_occurrence(self):
        step=validate_composite_plan({'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate'}]})[0]
        feedback={'substeps':[row(),row(status='failed')]}
        self.assertFalse(CompositeWorkspace._step_done(step,feedback,{}))
        self.assertTrue(CompositeWorkspace._step_done({**step,'sub_occurrence':1},feedback,{}))
        self.assertFalse(CompositeWorkspace._step_done({**step,'sub_occurrence':2},feedback,{}))

    def test_handover_requires_actual_receiver_and_correct_source(self):
        step=validate_composite_plan({'steps':[{'id':'h','title':'交接','checkpoint':'handover_confirmed','skill':'relocate'}]})[0]
        r=row('handover',parameters={'from_arm':'left','to_arm':'right'},
              feedback={'primitive_success':True,'held_by':'right','verification_source':'native_receiver_contact_then_donor_release'})
        self.assertTrue(CompositeWorkspace._step_done(step,{'substeps':[r]},{}))
        r['feedback']['held_by']='left';self.assertFalse(CompositeWorkspace._step_done(step,{'substeps':[r]},{}))

    def test_live_receipt_updates_only_bound_current_outer_execution(self):
        ws=CompositeWorkspace();epoch='now'
        turn=ws.start('execute',epoch,'抓取',{'model':'unit','vision':False},False,None)
        steps=validate_composite_plan({'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate'}]})
        ws.plan(turn,epoch,'unit',steps);ws.bind(turn,'current','execute','抓取','relocate','relocate_object','unit')
        snapshot={'observation_id':epoch+':9','job':{'execution_id':'current','completed_substeps':[row()]},'diagnostics':{'events':[]}}
        self.assertEqual(ws.plan_status(snapshot)['steps'][0]['status'],'done')
        snapshot['job']['execution_id']='older';self.assertIsNone(ws.plan_status(snapshot))

    def test_all_children_and_placement_bind_the_declared_outer_occurrence(self):
        ws=CompositeWorkspace();turn=ws.start('execute','now','搬运',{'model':'unit','vision':False},False,None)
        steps=validate_composite_plan({'steps':[
            {'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate'},
            {'id':'m','title':'移动','checkpoint':'motion_reached','skill':'relocate'},
            {'id':'z','title':'放置','checkpoint':'placement_confirmed','skill':'relocate'},
            {'id':'retry','title':'第二次搬运','checkpoint':'placement_confirmed','skill':'relocate','occurrence':2}]})
        ws.plan(turn,'now','unit',steps);ws.bind(turn,'one','execute','搬运','relocate','relocate_object','unit')
        self.assertEqual(ws.active_plan['stage_bindings'],{'p':'one','m':'one','z':'one'})
        ws.bind(turn,'two','execute','搬运','relocate','relocate_object','unit')
        self.assertEqual(ws.active_plan['stage_bindings'],{'p':'one','m':'one','z':'one','retry':'two'})

    def test_child_motion_still_requires_position_and_orientation_tolerance(self):
        step=validate_composite_plan({'steps':[{'id':'m','title':'移动','checkpoint':'motion_reached','skill':'relocate'}]})[0]
        r=row('move_relative',feedback={'primitive_success':True,'verification_source':'robot_forward_kinematics',
              'position_error_m':.005,'rotation_error_rad':.04})
        self.assertTrue(CompositeWorkspace._step_done(step,{'substeps':[r]},{}))
        r['feedback']['rotation_error_rad']=.09
        self.assertFalse(CompositeWorkspace._step_done(step,{'substeps':[r]},{}))

    def test_malformed_stage_and_unbound_child_fields_are_rejected(self):
        for values in ({'steps':[None]},{'steps':[]},
            {'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate','subskill':[]}]},
            {'steps':[{'id':'p','title':'抓取','checkpoint':'grasp_confirmed','skill':'pick','subskill':'pick'}]}):
            with self.assertRaises(ValueError):validate_composite_plan(values)

    def test_native_preserve_rebases_identity_without_replacing_original_object_target(self):
        scene={'object_id':'bottle','position_m':[.5,-.1,1.0],'held_by':['left'],'upright':True}
        before={'observation_id':'old:7','observation':{'state':[0]*16,'tcp':{}},
                'diagnostics':{'manipulation_scene':scene}}
        after=copy.deepcopy(before);after['observation_id']='new:7'
        initial=copy.deepcopy(before);initial['observation_id']='old:0'
        initial['diagnostics']['manipulation_scene']['position_m']=[.4,-.2,1.0]
        proof={'mode':'preserve','physical_state_preserved':True,'observation_id':'new:7'}
        rebased=preserved_anchor(initial,before,after,proof)
        self.assertEqual(rebased['observation_id'],'new:0')
        self.assertEqual(rebased['diagnostics']['manipulation_scene']['position_m'],[.4,-.2,1.0])
        self.assertEqual(initial['observation_id'],'old:0')
        self.assertIs(preserved_anchor(initial,before,after,{'mode':'reset'}),initial)

    def test_changed_object_arm_or_unrelated_epoch_cannot_rebase_anchor(self):
        before={'observation_id':'old:7','observation':{'state':[0]*16,'tcp':{}},
            'diagnostics':{'manipulation_scene':{'object_id':'bottle','position_m':[.5,-.1,1.0],'held_by':['left'],'upright':True}}}
        proof={'mode':'preserve','physical_state_preserved':True,'observation_id':'new:7'}
        for change in ('object','arm','epoch','missing'):
            after=copy.deepcopy(before);after['observation_id']='new:7';initial=copy.deepcopy(before)
            if change=='object':after['diagnostics']['manipulation_scene']['position_m'][0]+=.001
            if change=='arm':after['observation']['state'][0]+=.001
            if change=='epoch':initial['observation_id']='unrelated:0'
            if change=='missing':del after['diagnostics']['manipulation_scene']['held_by']
            with self.assertRaises(ValueError):preserved_anchor(initial,before,after,proof)


if __name__=='__main__':unittest.main()
