"""Guard contracts for real local backends; no GPU/model/hardware required."""
import json
import unittest
from unittest.mock import patch,Mock
import numpy as np
from .environment import OpenArmEnv
from .curobo import CuRoboPlanner, attachment, scene
from .tamp import PlanningFailure
from .policy import LeRobotClient
from .policy_checks import observation,metadata
from .policy_wire import TASK
from harness.skills import catalogue
from .cache_checks import CacheChecks


class BackendChecks(unittest.TestCase):
    def test_carried_geometry_covers_entire_declared_cylinder_cross_sections(self):
        env=OpenArmEnv(images=False);env.reset(seed=0)
        try:
            spheres=np.array(attachment(env,'right')['spheres_tcp'])
            hand=env.data.xmat[env.ee_ids['right']].reshape(3,3)
            tcp=env.data.xpos[env.ee_ids['right']]+hand@np.array([0,.006,.075])
            for g in env.bottle_geoms:
                radius,half=env.model.geom_size[g,:2];rot=env.data.geom_xmat[g].reshape(3,3)
                for z in np.linspace(-half,half,31):
                    for theta in np.linspace(0,2*np.pi,41):
                        local=np.array([radius*np.cos(theta),radius*np.sin(theta),z])
                        point=hand.T@(env.data.geom_xpos[g]+rot@local-tcp)
                        self.assertLessEqual(np.min(np.linalg.norm(spheres[:,:3]-point,axis=1)-spheres[:,3]),1e-10)
            world=scene(env,carrying=True)['cuboid']
            self.assertFalse(set(world)&{'bottle_body','bottle_neck','bottle_cap'})
            self.assertTrue({'bottle_body','bottle_neck','bottle_cap'}<=set(scene(env)['cuboid']))
        finally:env.close()

    def test_other_hand_hold_is_rejected_before_GPU_request(self):
        env=OpenArmEnv(images=False);env.reset(seed=0)
        try:
            # Contract fixture only; not evidence of a native grasp.
            env.evaluation['held']['left']=True
            with patch('sim.curobo.metadata') as meta:
                with self.assertRaises(PlanningFailure) as error:
                    CuRoboPlanner(env,'move_relative',parameters={'arm':'right','delta_m':[0,0,.02],'frame':'world'})
                self.assertEqual(error.exception.code,'curobo_other_arm_hold_not_supported')
                meta.assert_not_called();self.assertEqual(env.tick,0)
        finally:env.close()

    def test_GPU_must_acknowledge_carried_object_before_any_motion(self):
        env=OpenArmEnv(images=False);env.reset(seed=0);q=env.data.qpos.copy()
        try:
            env.evaluation['held']['right']=True  # isolated protocol fixture
            with patch('sim.curobo.metadata',return_value={'ready':True}):
                planner=CuRoboPlanner(env,'move_relative',parameters={'arm':'right','delta_m':[0,0,.02],'frame':'world'})
            def remote(path,data):
                if path=='/fk':
                    point,_=planner._pose()
                    return {'tool_frames':['left_candidate_tcp','right_candidate_tcp'],
                            'position':[[0,0,0],point.tolist()]}
                self.assertIn('attachment',data)
                return {'success':True,'path':[], 'joint_names':[]}
            with patch('sim.curobo.call',side_effect=remote):
                with self.assertRaises(PlanningFailure) as error:next(planner.actions())
                self.assertEqual(error.exception.code,'curobo_attachment_missing')
            np.testing.assert_array_equal(env.data.qpos,q);self.assertEqual(env.tick,0)
        finally:env.close()

    def test_act_accepts_32_chunk_and_executes_16_only(self):
        data=observation();env=Mock();env.state.return_value=data['state'];env.render.side_effect=lambda n:data[n]
        client=LeRobotClient({'policy_type':'act','chunk_size':32,'action_key':'actions','execute_horizon':16})
        client.session='a'*32;client.request=Mock(return_value={'sequence':0,'actions':np.zeros((32,16)).tolist()})
        self.assertEqual(client.infer(env,TASK).shape,(16,16))
        client.request=Mock(return_value={'sequence':1,'actions':np.zeros((50,16)).tolist()})
        with self.assertRaises(ValueError):client.infer(env,TASK)

    def test_act_cannot_connect_to_pi05_endpoint(self):
        config={**metadata(),'policy_type':'act','chunk_size':32}
        client=LeRobotClient(config);client.request=Mock(return_value=metadata())
        with self.assertRaises(ValueError):client.__enter__()
        client.request.assert_called_once_with('/metadata')

    def test_backend_skill_membership_is_explicit(self):
        cat=catalogue({'policy_configured':True,'curobo':{'ready':True}},'act')
        entries={x['id']:x for x in cat['skills']}
        self.assertEqual(entries['pick']['backends'],['tamp'])
        self.assertEqual(entries['move_relative']['backends'],['tamp','curobo'])
        self.assertEqual(entries['run_policy']['backends'],['pi05','act'])

    def test_curobo_fk_mismatch_never_requests_plan_or_executes(self):
        env=OpenArmEnv(images=False);env.reset(seed=0);q=env.data.qpos.copy()
        try:
            with patch('sim.curobo.metadata',return_value={'ready':True}),patch('sim.curobo.call',return_value={'tool_frames':['left_candidate_tcp','right_candidate_tcp'],'position':[[[[0,0,0],[0,0,0]]]]}) as remote:
                planner=CuRoboPlanner(env,'move_relative',parameters={'arm':'right','delta_m':[0,0,.02],'frame':'world'})
                with self.assertRaises(PlanningFailure) as error:next(planner.actions())
                self.assertEqual(error.exception.code,'curobo_fk_mismatch')
                self.assertEqual(remote.call_args.args[0],'/fk')
            np.testing.assert_array_equal(q,env.data.qpos);self.assertEqual(env.tick,0)
        finally:env.close()


if __name__=='__main__':unittest.main()
