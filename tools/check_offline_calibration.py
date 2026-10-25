"""Device-free numerical and evidence checks for offline calibration.

OpenCV is an independent projection reference, not used by the fitter.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import offline_calibration as study
from tools import fit_carry_camera as carry


class OfflineChecks(unittest.TestCase):
    def setUp(self):
        self.x=np.array([.08,-.02,.93,.05,-.03,.02,np.log(300.),622.,477.,.015])
        rng=np.random.default_rng(7)
        self.xyz=rng.uniform([.0,-.4,.1],[.8,.4,.6],(48,3))

    def reference(self,model):
        rotation=Rotation.from_rotvec(self.x[3:6]).as_matrix()@study.R0
        rvec=cv2.Rodrigues(rotation)[0];t=-rotation@self.x[:3]
        k=np.array([[300.,0.,622.],[0.,300.,477.],[0.,0.,1.]])
        if model=='fisheye_k1':
            uv,_=cv2.fisheye.projectPoints(self.xyz.reshape(-1,1,3),rvec,t,k,
                np.array([self.x[9],0.,0.,0.]))
        else:uv,_=cv2.projectPoints(self.xyz,rvec,t,k,np.zeros(5))
        return uv.reshape(-1,2)

    def test_fisheye_matches_opencv_optical_convention(self):
        uv,z=study.project(self.x,self.xyz,'fisheye_k1')
        self.assertGreater(z.min(),0)
        np.testing.assert_allclose(uv,self.reference('fisheye_k1'),atol=1e-9,rtol=0)

    def test_pinhole_matches_opencv(self):
        uv,_=study.project(self.x[:9],self.xyz,'pinhole')
        np.testing.assert_allclose(uv,self.reference('pinhole'),atol=1e-9,rtol=0)

    def test_recovers_known_metric_camera_from_independent_projections(self):
        row=study.solve(self.xyz,self.reference('fisheye_k1'),'fisheye_k1',starts=2)
        self.assertLess(row['fit_rmse_px'],1e-5)
        np.testing.assert_allclose(row['parameters'],self.x,atol=1e-5,rtol=0)

    def test_source_tampering_refused_before_correspondence_or_fit(self):
        with tempfile.TemporaryDirectory(prefix='offline-check-',dir=ROOT/'runs') as temporary:
            out=Path(temporary)
            (out/'data.json').write_text('{}\n')
            protocol={'data_sha256':'wrong','urdf_sha256':study.digest(ROOT/'reference/source/inputs/robot.urdf')}
            study.write(out/'protocol.json',protocol)
            study.write(out/'selected.json',{'protocol_sha256':study.digest(out/'protocol.json')})
            with patch.object(study,'points',side_effect=AssertionError('must not read points')):
                with self.assertRaisesRegex(ValueError,'source changed'):
                    study.evaluate(out,Path('not-read.json'))

    def test_frozen_selection_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='offline-check-',dir=ROOT/'runs') as temporary:
            out=Path(temporary);(out/'selected.json').write_text('{"frozen":true}\n')
            with self.assertRaisesRegex(ValueError,'selection is frozen'):
                study.fit(out,Path('not-read.json'))
            self.assertEqual(json.loads((out/'selected.json').read_text()),{'frozen':True})

    def test_closed_command_is_not_promoted_to_grasp_evidence(self):
        frames=[{'timestamp':f'2026-01-01T00:00:{i:02d}',
                 'gripper':[-.5,-.5],'gripper_command':[.05,.05]} for i in range(3)]
        data={'episodes':{str(i):{'frames':frames} for i in range(4)}}
        result=study.grippers(data)
        self.assertFalse(result['contact_threshold_accepted'])
        self.assertIsNone(result['contact_torque_range'])
        self.assertGreater(result['sides']['left']['fit'][0]['command_closed']['n'],0)

    def test_carried_point_triangulates_known_wrist_attachment(self):
        rng=np.random.default_rng(21);attachment=np.array([.02,-.03,.15]);rows=[];world=[]
        for i in range(8):
            t=np.eye(4);t[:3,:3]=Rotation.from_rotvec(rng.normal(0,.3,3)).as_matrix()
            t[:3,3]=rng.uniform([.2,-.3,.1],[.6,.3,.35]);world.append(t[:3,:3]@attachment+t[:3,3])
            rows.append({'T_base_wrist':t.tolist()})
        rotation=Rotation.from_rotvec(self.x[3:6]).as_matrix()@study.R0
        k=np.array([[300.,0.,622.],[0.,300.,477.],[0.,0.,1.]])
        uv=cv2.fisheye.projectPoints(np.array(world).reshape(-1,1,3),cv2.Rodrigues(rotation)[0],
            -rotation@self.x[:3],k,np.array([self.x[9],0.,0.,0.]))[0].reshape(-1,2)
        for r,u in zip(rows,uv):r['uv']=u.tolist()
        point,condition=carry.local_point(self.x,rows)
        np.testing.assert_allclose(point,attachment,atol=1e-9,rtol=0)
        self.assertTrue(carry.registration_valid(point,condition))

    def test_static_views_or_impossible_attachment_are_rejected(self):
        rows=[{'T_base_wrist':np.eye(4).tolist(),'uv':[700.,600.]}]*3
        point,condition=carry.local_point(self.x,rows)
        self.assertFalse(carry.registration_valid(point,condition))
        self.assertFalse(carry.registration_valid([1.,0.,0.],1.))
        self.assertFalse(carry.registration_valid([float('nan'),0.,0.],1.))

    def test_fit_split_does_not_require_or_read_validation_images(self):
        with tempfile.TemporaryDirectory(prefix='offline-split-',dir=ROOT/'runs') as temporary:
            out=Path(temporary);(out/'fit.jpg').write_bytes(b'not decoded by loader')
            fit={'episode':0,'group':'ep0-right','frame':0,'order':0,
                 'image':'fit.jpg','sha256':study.digest(out/'fit.jpg'),'T_base_wrist':np.eye(4).tolist()}
            validation={**fit,'episode':3,'group':'ep3-right','image':'MUST_NOT_READ.jpg','sha256':'wrong'}
            study.write(out/'data.json',{'urdf_sha256':study.digest(ROOT/'reference/source/inputs/robot.urdf'),
                'rows':[fit,validation]})
            study.write(out/'points.json',{'points':[{'group':'ep0-right','frame':0,'uv':[600,500]}]})
            self.assertEqual(len(carry.samples(out,[0,1])),1)

    def test_failed_registrations_keep_failed_coverage_and_report_missing_scores(self):
        with tempfile.TemporaryDirectory(prefix='offline-coverage-',dir=ROOT/'runs') as temporary:
            base=Path(temporary);out=base/'carry';out.mkdir()
            (out/'images').mkdir()
            from PIL import Image
            Image.new('RGB',(16,16),'white').save(out/'pixel.jpg')
            study.write(out/'data.json',{})
            study.write(out/'protocol.json',{'fit_rows':[],'data_sha256':study.digest(out/'data.json')})
            study.write(out/'selected.json',{'protocol_sha256':study.digest(out/'protocol.json'),
                'parameters':self.x.tolist(),'attachments':{},'at_bounds':[],'jacobian_condition':1.})
            rows=[{'group':f'ep{ep}-{side}','episode':ep,'frame':i,'order':i,
                'T_base_wrist':np.eye(4).tolist(),'uv':[600,500],'image':'pixel.jpg'} for ep in [2,3]
                for side in ['left','right'] for i in range(8)]
            with patch.object(carry,'samples',return_value=rows),patch.object(carry,'local_point',return_value=(np.array([1.,0.,0.]),1.)):
                carry.evaluate(out)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(len(result['failed_registration_groups']),4)
            self.assertEqual(result['scored_withheld_frames'],0)
            self.assertEqual(result['expected_withheld_frames'],20)
            self.assertIsNone(result['metrics']['held_out']['rmse_px'])
            self.assertFalse(result['physical_trial_passed'])
            study.write(out/'timing-result.json',{**result,'offset_records':0})
            proxy={'metrics':{n:{'n':0} for n in ['fit','development_validation','new_validation']},'rows':[],'reasons':[]}
            study.report(base,proxy,{'model':'unavailable'},{})
            self.assertIn('不可评分',(base/'index.html').read_text())


if __name__=='__main__':
    unittest.main(verbosity=2)
