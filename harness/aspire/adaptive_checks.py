"""Adaptive skill contract, evidence gates, and RGB-only cache checks."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from .runtime import ROOT, ProgramExecutor, ToolRegistry
from .visual_adaptive import AdaptiveWorld, CachedRig


class AdaptiveChecks(unittest.TestCase):
    def test_receiver_search_rejects_unreachable_middle_not_only_endpoints(self):
        with TemporaryDirectory() as tmp:
            world=object.__new__(AdaptiveWorld);world._guard=lambda:None;world.output=Path(tmp)
            world.planning_seed={};world.approach_checks=[];checked=[]
            def solve(arm,target,rotation):
                yaw=np.degrees(np.arctan2(rotation[1,2],rotation[0,2]));checked.append((yaw,float(target[1])))
                if yaw<-80 and .025<target[1]<.045:raise ValueError('middle corridor blocked')
                return np.zeros(7)
            world._solve=solve
            result=world.approach_options('left',[.5,0,1.3],[-85,-75],pitch_deg=-5)
            self.assertTrue(result['ok']);self.assertEqual(result['yaw_deg'],-75)
            self.assertEqual(result['rejected'][0]['yaw_deg'],-85)
            self.assertTrue(result['corridor_checked']);self.assertEqual(world.planning_seed,{})

    def simulate(self,*,receiver=True,drop=False,position=(.50,-.21,1.06)):
        state={'left':[.39,.36,1.24],'right':[.33,-.41,1.20]};motors={'left':-1.26,'right':-1.26}
        moves=[];grips=[];close_count=0
        def observe():
            return {'image_targets':{'bottle_lower':[600,500,650,500]},'robot':{
                s:{'tcp_m':list(state[s]),'rotation':[[0,0,1],[0,-1,0],[1,0,0]],'gripper_motor_rad':motors[s]} for s in state}}
        def move(arm,position_m,**kwargs):state[arm]=list(position_m);moves.append((arm,list(position_m)));return {'ok':True}
        def relative(arm,delta_m):state[arm]=(np.array(state[arm])+delta_m).tolist();return {'ok':True}
        def grip(arm,opened):
            nonlocal close_count
            grips.append((arm,opened))
            if arm=='right' and not opened:
                close_count+=1;motors[arm]=-1.26 if drop and close_count==1 else -.7
            else:motors[arm]=-1.26 if opened else -.7
            return {'motor_rad':motors[arm]}
        functions={'observe':observe,'move_tcp':move,'move_relative':relative,'set_gripper':grip,
            'locate_pixels':lambda **kw:{'estimated_position_m':list(position)},
            'locate_basket':lambda:{'estimated_position_m':[.50,.21,1.15]},
            'grip_evidence':lambda arm:{'blocked_closure':receiver if arm=='left' else True},
            'approach_options':lambda **kw:{'ok':True,'pre_m':kw['target_m'],'target_m':kw['target_m'],'yaw_deg':-85,'pitch_deg':-5},
            **{n:(lambda **kw:{'ok':True}) for n in ('label','orient','calibrate_views','visual_align','wait','check_grasp')}}
        registry=ToolRegistry(shared_public_names=tuple(functions),public_extension_allowlist=())
        for name,fn in functions.items():registry.register(fn,name=name,public=True,layer='shared_atomic',capability='measured mock')
        result=ProgramExecutor(registry).execute_program((ROOT/'skills/bottle_visual/adaptive/program.py').read_text())
        return result,moves,grips

    def test_receiver_evidence_required_before_release(self):
        result,_,grips=self.simulate(receiver=False)
        self.assertFalse(result.ok)
        self.assertEqual([opened for arm,opened in grips if arm=='right'],[True,False])

    def test_dropped_close_retried_once(self):
        result,_,grips=self.simulate(drop=True)
        self.assertTrue(result.ok)
        self.assertEqual(sum(arm=='right' and not opened for arm,opened in grips),2)

    def test_same_program_recomputes_coarse_targets_from_perception(self):
        first,m1,_=self.simulate(position=(.48,-.24,1.06))
        second,m2,_=self.simulate(position=(.52,-.18,1.06))
        self.assertTrue(first.ok and second.ok)
        np.testing.assert_allclose(np.array(m2[0][1])-m1[0][1],[.04,.06,0],atol=1e-10)
        np.testing.assert_allclose(np.array(m2[1][1])-m1[1][1],[.04,.06,0],atol=1e-10)

    def test_renderer_cache_reuses_only_the_same_physics_tick_and_view(self):
        rig=object.__new__(CachedRig);rig.cache={};rig.cache_tick=None;rig.env=type('Env',(),{'tick':0})()
        rig.exposure=1.;frame=np.full((4,4,3),100,np.uint8)
        with patch('harness.aspire.visual_geometry.GeometryRig.image',return_value=frame) as render:
            rig.image(0);rig.image(0);self.assertEqual(render.call_count,1)
            rig.image(1);self.assertEqual(render.call_count,2)
            rig.env.tick=1;rig.image(0);self.assertEqual(render.call_count,3)

    def test_no_legacy_macros_or_simulator_camera_truth(self):
        for name in ('harness/aspire/visual_adaptive.py','skills/bottle_visual/adaptive/program.py'):
            tree=ast.parse((ROOT/name).read_text());attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
            self.assertFalse(attrs&{'cam_xpos','cam_xmat','bottle_id','geom_xpos','geom_xmat','step_tamp'})
            calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
            self.assertFalse(calls&{'pick','handover','place','reset','eval','exec'})
        self.assertIn('calibrate_views',AdaptiveWorld.PUBLIC)


if __name__=='__main__':unittest.main()
