"""Image/state association and diagnostic/physical separation."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .alignment import load_samples, recorded_pose, sha
from .environment import ROOT, OpenArmEnv
from harness.agent import Harness
from harness.checks import LocalState
from .reference_phases import landmarks


class AlignmentChecks(unittest.TestCase):
    def test_command_landmarks_require_ordered_transitions_and_do_not_label_success(self):
        left, right = np.full(80,-.9), np.full(80,-.9)
        right[10:40], left[30:61] = .05, .05
        plan = landmarks({'left_action_gripper':left,'right_action_gripper':right})
        self.assertEqual(plan['events'], {'right_close':10,'left_close':30,'right_open':40,'left_open':61})
        self.assertIn(79,plan['samples'])
        self.assertTrue(all(0 <= f < 80 for f in plan['samples']))
        self.assertNotIn('success',plan)
        # A right hand released before left-close cannot invent a new opening.
        right[20:] = -.9
        with self.assertRaises(ValueError):
            landmarks({'left_action_gripper':left,'right_action_gripper':right})

    def test_audit_rejects_shifted_encoders_timestamps_and_swapped_images(self):
        source = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'reference/frames').mkdir(parents=True)
            for name in ('acquisition.json', 'real_trajectory.json'):
                shutil.copyfile(ROOT/'reference'/name, root/'reference'/name)
            manifest = copy.deepcopy(source)
            manifest['samples'] = manifest['samples'][:1]
            for item in manifest['samples'][0]['images'].values():
                dest = root/'reference/frames'/item['path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/'reference/frames'/item['path'], dest)
            file = root/'reference/frames/manifest.json'
            file.write_text(json.dumps(manifest))
            self.assertEqual(len(load_samples(root)['samples']), 1)
            for change in ('state', 'time', 'image', 'landmark'):
                bad = copy.deepcopy(manifest)
                if change == 'state':
                    bad['samples'][0]['state'][3] += .1
                elif change == 'time':
                    bad['samples'][0]['recorded_seconds'] += .033
                elif change == 'image':
                    bad['samples'][0]['images']['left'] = bad['samples'][0]['images']['right']
                else:
                    bad['samples'][0]['command_landmarks'] = [{'id':'success','label':'任务成功'}]
                file.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_samples(root)

    def test_recorded_pose_uses_separate_model_and_never_advances_physics(self):
        physics, geometry = OpenArmEnv(images=False), OpenArmEnv(images=False)
        try:
            physics.reset(seed=0)
            geometry.reset(seed=0)
            physics.step(physics.reference_action(0, 0))
            q, v, t, tick = physics.data.qpos.copy(), physics.data.qvel.copy(), float(physics.data.time), physics.tick
            sample = load_samples()['samples'][2]
            recorded_pose(geometry, sample['state'])
            for i, side in enumerate(('left','right')):
                np.testing.assert_allclose(geometry.data.qpos[geometry.qids[side]], sample['state'][8*i:8*i+7], atol=1e-12)
            self.assertEqual(float(geometry.data.time), 0.)
            np.testing.assert_array_equal(physics.data.qpos, q)
            np.testing.assert_array_equal(physics.data.qvel, v)
            self.assertEqual((float(physics.data.time), physics.tick), (t, tick))
        finally:
            physics.close()
            geometry.close()

    def test_archived_alignment_is_research_only_and_images_are_not_logged(self):
        agent = Harness(LocalState(), learning=object())
        with self.assertRaises(ValueError):
            agent.invoke('inspect_alignment', {}, role='execute')
        value = {'evidence_id': 'archive:0:0:head', 'evidence_kind': 'recorded_alignment_not_live_observation',
                 'images': {'real': 'private-image'}, 'transfer_ready': False}
        with patch('sim.alignment.inspect', return_value=value), patch.object(agent, '_log') as log:
            result = agent.invoke('inspect_alignment', {'episode':0,'frame':868,'camera':'head'}, role='research')
            self.assertIn('images', result)
            self.assertNotIn('images', log.call_args.args[2])
            self.assertNotIn('observation_id', result)


if __name__ == '__main__':
    unittest.main()
