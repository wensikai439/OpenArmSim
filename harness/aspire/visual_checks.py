"""Sensor geometry, missing-evidence rejection, and preserved legacy checks."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageDraw

from .visual import COLORS, MARKERS, RGBRig, bottle_feature, tool_feature
from .visual_control import RobotWorld
from .runtime import ROOT, ProgramExecutor, ToolRegistry


class VisualChecks(unittest.TestCase):
    def test_projective_tool_fixture(self):
        # Independent image construction: camera matrices are not passed to
        # the tracker. Two arbitrary projective views of the same planar tool.
        for hom in (np.array([[1000,70,450],[10,900,420],[.12,.2,1.]]),
                    np.array([[800,-90,380],[40,1100,390],[-.15,.1,1.]])):
            im=Image.new('RGB',(1280,960),'#777777');draw=ImageDraw.Draw(im)
            for xy,color in zip(MARKERS,COLORS):
                q=hom@np.r_[xy,1];u,v=q[:2]/q[2]
                draw.ellipse((u-5,v-5,u+5,v+5),fill=tuple(round(c*255) for c in color[:3]))
            measured,_=tool_feature(np.array(im))
            expected=hom[:2,2]/hom[2,2]
            self.assertLess(np.linalg.norm(measured-expected),4.)

    def test_occluded_marker_rejected(self):
        with self.assertRaisesRegex(ValueError,'not visible'):
            tool_feature(np.full((960,1280,3),200,np.uint8))

    def test_bottle_translation_changes_goal(self):
        goals=[]
        for dx,dy in ((0,0),(35,-20)):
            a=np.full((960,1280,3),240,np.uint8)
            a[400+dy:550+dy,650+dx:690+dx]=10
            goal,_=bottle_feature(a);goals.append(goal)
        np.testing.assert_allclose(goals[1]-goals[0],[35,-20],atol=.1)

    def test_hold_evidence_requires_visible_bottle_motion(self):
        import tempfile
        rig=RGBRig.__new__(RGBRig)
        rig.detections=[{'bbox':[650,400,689,549]}]*2
        with tempfile.TemporaryDirectory() as folder:
            rig.folder=Path(folder)
            for up in (0,50):
                a=np.full((960,1280,3),240,np.uint8)
                a[400-up:550-up,650:690]=10
                rig.image=lambda view,dots=False:a
                shifts=rig.object_motion()
                np.testing.assert_allclose(shifts,[up,up])
            rig.image=lambda view,dots=False:np.full((960,1280,3),240,np.uint8)
            with self.assertRaisesRegex(ValueError,'not visible'):
                rig.object_motion()

    def test_unsafe_commands_rejected_before_motion(self):
        world=RobotWorld.__new__(RobotWorld);world._guard=lambda:None
        for delta in ([1,0,0],[.1805,0,0],[float('nan'),0,0],[0,0]):
            with self.assertRaises(ValueError):world.move_relative(delta_m=delta)
        for args in ({'offset_m':[.3,0,0]}, {'step_m':.1}, {'tolerance_px':100},
                     {'target_pixels':[0,0,0,0]}):
            with self.assertRaises(ValueError):world.visual_align(**args)

    def test_no_task_macros_in_new_controller(self):
        forbidden={'pick','handover','place','step_tamp','TampPlanner','StableWorld','SegmentPlanner'}
        for source in ('harness/aspire/visual_control.py','skills/bottle_visual/program.py'):
            tree=ast.parse((ROOT/source).read_text())
            calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
                   for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
            self.assertFalse(calls & forbidden)

    def test_lost_close_retry_and_fail_before_lift(self):
        # Two distinct failure responses: one lost command is recoverable;
        # two lost commands must prevent lift. Command endpoints are absent.
        for failures,expected_ok in ((1,True),(2,False)):
            counts={'close':0,'lift':0};actual={'motor':-.89947}
            def observe():
                return {'image_targets':{'bottle_lower':[600,500,650,500]},
                        'robot':{'right':{'gripper_motor_rad':actual['motor']}}}
            def gripper(arm,opened):
                if not opened:
                    counts['close']+=1
                    if counts['close']>failures:actual['motor']=-.35
                return {'motor_rad':actual['motor']}
            def move(**kwargs):counts['lift']+=1;return {'ok':True}
            functions={'observe':observe,'set_gripper':gripper,'move_relative':move,
                       **{n:(lambda **kw:{'ok':True}) for n in ('orient','visual_align','wait','check_grasp')}}
            registry=ToolRegistry(shared_public_names=tuple(functions),public_extension_allowlist=())
            for name,fn in functions.items():registry.register(fn,name=name,public=True,layer='shared_atomic',capability='test response')
            result=ProgramExecutor(registry).execute_program((ROOT/'skills/bottle_visual/program.py').read_text())
            self.assertEqual(result.ok,expected_ok,str(result.error))
            self.assertEqual(counts['close'],2)
            self.assertEqual(counts['lift'],int(expected_ok))

    def test_old_runtime_unchanged(self):
        manifest=json.loads((ROOT/'skills/bottle_recovery/manifest.json').read_text())
        for name,expected in manifest['runtime_sources'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)


if __name__=='__main__':unittest.main()
