"""Cache boundaries only; real CUDA precision/performance is tested separately."""
import copy
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np

from .curobo_cache import cache_key,robot_config,kinematic_models,copy_kinematics
from .curobo_server import Planner,validate_scene
from .environment import OpenArmEnv
from .relocation import RelocationPlanner
from .tamp import PlanningFailure


class Tensor:
    def __init__(self,shape=(2,3),pointer=1):self.shape=shape;self.pointer=pointer
    def data_ptr(self):return self.pointer


def params():
    return types.SimpleNamespace(joint_names=['a'],tool_frames=['right_candidate_tcp'],
        **{field:Tensor() for field in ('fixed_transforms','joint_map','link_spheres','link_sphere_idx_map','link_map')})


def model(p=None):
    return types.SimpleNamespace(config=types.SimpleNamespace(kinematics_config=p or params()),
                                 update_kinematics_config=Mock())


class CacheChecks(unittest.TestCase):
    def test_world_or_topology_change_never_reuses_old_geometry(self):
        world={'cuboid':{'table':{'dims':[1.,1.,.1],'pose':[0,0,0,1,0,0,0]}}}
        original=cache_key('right',{'left_joint':.3},world,True)
        self.assertEqual(original,cache_key('right',{'left_joint':.30123456789},world,True))
        moved=copy.deepcopy(world);moved['cuboid']['table']['pose'][0]+=.001
        for key in [cache_key('right',{'left_joint':.3},moved,True),
                    cache_key('left',{'left_joint':.3},world,True),
                    cache_key('right',{'left_joint':.3},world,False),
                    cache_key('right',{'another_joint':.3},world,True)]:self.assertNotEqual(key,original)

    def test_fresh_robot_config_preserves_exact_lock_values_and_source_geometry(self):
        base={'kinematics':{'collision_link_names':['hand'],'self_collision_ignore':{},'self_collision_buffer':{}}}
        original=copy.deepcopy(base);value=.30123456789
        config=robot_config(base,'right',{'left_joint':value},True)
        self.assertEqual(config['kinematics']['lock_joints']['left_joint'],value)
        self.assertEqual(base,original)
        self.assertEqual(config['kinematics']['extra_collision_spheres']['carried_bottle'],32)

    def test_all_solver_rollouts_are_included_and_aliases_deduplicated(self):
        a,b,c=model(),model(),model()
        rollout=lambda m:types.SimpleNamespace(transition_model=types.SimpleNamespace(robot_model=m))
        solver=lambda ms:types.SimpleNamespace(get_all_rollout_instances=lambda:[rollout(m) for m in ms])
        planner=types.SimpleNamespace(ik_solver=solver([a,b]),trajopt_solver=solver([b]),graph_planner=solver([c,a]))
        self.assertEqual(kinematic_models(planner),[a,b,c])

    def test_shape_failure_occurs_before_any_in_place_model_update(self):
        a,b=model(),model();b.config.kinematics_config.link_spheres=Tensor((9,4))
        with self.assertRaisesRegex(ValueError,'shape'):copy_kinematics([a,b],params())
        a.update_kinematics_config.assert_not_called();b.update_kinematics_config.assert_not_called()

    def test_pointer_replacement_is_rejected_before_using_cached_graph(self):
        a=model()
        a.update_kinematics_config.side_effect=lambda _:setattr(a.config.kinematics_config,'fixed_transforms',Tensor(pointer=2))
        with self.assertRaisesRegex(ValueError,'captured buffers'):copy_kinematics([a],params())

    def test_malformed_preparation_cannot_build_or_warm_solver(self):
        worker=Planner.__new__(Planner);worker.state=Mock();worker._solver=Mock()
        with self.assertRaises(ValueError):worker.prepare({'arm':'right','joints':{},'scene':{},'carrying':'true'})
        worker.state.assert_not_called();worker._solver.assert_not_called()
        with self.assertRaises(ValueError):validate_scene({'cuboid':{'bad':{'dims':[-1,1,1],'pose':[0,0,0,1,0,0,0]}}})

    def test_failed_backend_preparation_never_starts_grasp_or_fallback(self):
        env=OpenArmEnv(images=False);env.reset(seed=0);before=env.data.qpos.copy()
        try:
            with patch('sim.relocation.metadata',return_value={'ready':True}),patch('sim.relocation.call',return_value={'prepared':False}) as remote:
                planner=RelocationPlanner(env,'relocate_object',parameters={'object_id':'bottle','arm':'right',
                    'receptacle_id':'table','offset_xy_m':[.08,0],'transport_planner':'curobo','transport_delta_m':[.015,0,0]})
                with self.assertRaises(PlanningFailure) as error:next(planner.actions())
                self.assertEqual(error.exception.code,'curobo_preparation_failed')
                self.assertEqual(remote.call_args.args[0],'/prepare')
            self.assertEqual(env.tick,0);np.testing.assert_array_equal(env.data.qpos,before)
            self.assertEqual(planner.result['substeps'],[])
        finally:env.close()
