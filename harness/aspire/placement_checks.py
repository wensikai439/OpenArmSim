"""A yaw replan must keep object center and upright axis, not just TCP."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .placement import yaw_alternative


class PlacementChecks(unittest.TestCase):
    def test_object_center_and_axis_are_preserved_for_every_proposed_yaw(self):
        obj=Rotation.from_euler('xyz',[.02,-.015,.7]).as_matrix()
        relative=Rotation.from_euler('xyz',[1.5,.2,-.4]).as_matrix()
        offset=np.array([.03,-.02,.08]);point=np.array([0.,.006,.075]);goal=np.array([.5,.3,1.04])
        for degrees in (5,-5,10,-10,20,-20,30,-30):
            hand,tcp,target=yaw_alternative(obj,relative,offset,point,goal,np.deg2rad(degrees))
            np.testing.assert_allclose(tcp+hand@(offset-point),goal,atol=1e-12)
            np.testing.assert_allclose(target[:,2],obj[:,2],atol=1e-12)
            np.testing.assert_allclose(hand@relative,target,atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(hand)),1.)


if __name__=='__main__':unittest.main()
