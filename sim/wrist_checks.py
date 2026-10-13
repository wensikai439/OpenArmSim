"""Wrist evidence must not become live observations or silently change."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .environment import ROOT
from .wrist import latest, inspect
from .alignment import sha
from harness.agent import Harness
from harness.checks import LocalState


class WristChecks(unittest.TestCase):
    def test_selected_parameters_are_bound_to_native_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'runs/test-wrist-study';path.mkdir(parents=True)
            (path/'selected.json').write_text('{}')
            (path/'wrist.json').write_text(json.dumps({'status':'completed','selected_sha256':'wrong'}))
            (path/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed after verification'):
                latest(root)

    def test_archive_image_tamper_is_rejected(self):
        data=latest()
        self.assertEqual(data['status'],'completed')
        bad=copy.deepcopy(data)
        row=next(r for r in bad['samples'] if r['episode']==0 and r['frame']==0 and r['side']=='right')
        row['images']['candidate']['sha256']='changed'
        with patch('sim.wrist.latest',return_value=bad):
            with self.assertRaisesRegex(ValueError,'image changed'):
                inspect()

    def test_wrist_tools_are_research_only_and_do_not_refresh_observations(self):
        agent=Harness(LocalState(),learning=object())
        before=copy.deepcopy(agent.observed)
        for name in ('inspect_wrist','fit_wrist_camera'):
            with self.assertRaises(ValueError):agent.invoke(name,{},role='execute')
        with patch.object(agent,'_log') as log:
            result=agent.invoke('inspect_wrist',{},role='research')
        self.assertEqual(set(result['images']),{'real','baseline','candidate','overlay'})
        self.assertFalse(result['applied'])
        self.assertNotIn('observation_id',result)
        self.assertNotIn('images',log.call_args.args[2])
        self.assertEqual(agent.observed,before)


if __name__=='__main__':
    unittest.main()
