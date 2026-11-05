"""Scene cache contract checks; GPU correctness is a separate experiment."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sim.scene_cache import topology_key,SceneCacheMixin


class SceneCacheChecks(unittest.TestCase):
    def setUp(self):self.scene={'cuboid':{'box':{'pose':[.5,0,1,1,0,0,0],'dims':[.1,.2,.3]}}}
    def test_numeric_scene_and_lock_changes_keep_topology_without_mutating_inputs(self):
        original=copy.deepcopy(self.scene);other=copy.deepcopy(self.scene)
        other['cuboid']['box']['pose'][0]+=.012;other['cuboid']['box']['dims'][2]+=.0001
        a=topology_key('right',{'left_joint1':.2},self.scene,True)
        b=topology_key('right',{'left_joint1':.21},other,True)
        self.assertEqual(a,b);self.assertEqual(self.scene,original)
    def test_topology_arm_lock_inventory_and_attachment_changes_force_new_key(self):
        baseline=topology_key('right',{'j1':1},self.scene,True)
        other=copy.deepcopy(self.scene);other['cuboid']['new']=copy.deepcopy(other['cuboid']['box'])
        for side,locks,scene,carry in [('left',{'j1':1},self.scene,True),
            ('right',{'j2':1},self.scene,True),('right',{'j1':1},other,True),('right',{'j1':1},self.scene,False)]:
            self.assertNotEqual(baseline,topology_key(side,locks,scene,carry))
    def test_invalid_or_unknown_world_cannot_be_hidden_by_a_reduced_key(self):
        for change in ('nan','negative','extra','mesh'):
            value=copy.deepcopy(self.scene)
            if change=='nan':value['cuboid']['box']['pose'][0]=float('nan')
            if change=='negative':value['cuboid']['box']['dims'][0]=-.1
            if change=='extra':value['cuboid']['box']['ignored']=True
            if change=='mesh':value['mesh']={}
            with self.assertRaises(ValueError):topology_key('right',{},value,False)
    def test_key_function_restored_and_failed_world_update_invalidates_solver(self):
        import sim.curobo_server as base
        original=base.cache_key;destroyed=[]
        planner=SimpleNamespace(destroy=lambda:destroyed.append(True))
        class Parent:
            def _solver(self,*a):
                self.assert_key=base.cache_key is topology_key
                if self.fail_build:raise ValueError('injected build failure')
                return planner,{'reused':True}
        class Candidate(SceneCacheMixin,Parent):pass
        candidate=Candidate();candidate.fail_build=True
        with self.assertRaises(ValueError):candidate._solver('right',{},self.scene,False)
        self.assertIs(base.cache_key,original);self.assertTrue(candidate.assert_key)
        candidate.fail_build=False;candidate.torch=None;candidate.cached=('key',planner)
        candidate.cached_locks={};candidate.refresh_fk=object();candidate.warmed_generation=1
        with patch('sim.scene_cache.update_exact_world',side_effect=ValueError('injected stale scene guard')):
            with self.assertRaises(ValueError):candidate._solver('right',{},self.scene,False)
        self.assertIs(base.cache_key,original);self.assertEqual(destroyed,[True])
        self.assertIsNone(candidate.cached);self.assertIsNone(candidate.refresh_fk)
        self.assertEqual(candidate.warmed_generation,0)


if __name__=='__main__':unittest.main()
