"""Deployment entry-point boundaries, without importing hardware modules."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .confined import command,environment


class ConfinedChecks(unittest.TestCase):
    def test_motion_services_global_keyring_and_arbitrary_code_are_not_commands(self):
        for args in [['start'],['serve'],['real','execute'],['aspire','configure'],
                     ['aspire','train'],['-c','pass'],['objects']]:
            with self.assertRaises(ValueError):command(args)

    def test_expected_offline_commands(self):
        self.assertEqual(command(['real','demo']),['-m','harness.real','demo'])
        self.assertEqual(command(['aspire','recover','--fault','receiver_close_loss']),
                         ['-m','harness.aspire','recover','--fault','receiver_close_loss'])

    def test_cache_and_temp_locations_stay_inside_root(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,clear=True):
            root=Path(temp).resolve();environment(root)
            for key in ('TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME',
                        'XDG_STATE_HOME','PIP_CACHE_DIR','TORCH_HOME','HF_HOME',
                        'CUDA_CACHE_PATH','__GL_SHADER_DISK_CACHE_PATH','PYTHONUSERBASE'):
                self.assertTrue(Path(os.environ[key]).is_relative_to(root),key)
            self.assertEqual(os.environ['PYTHONNOUSERSITE'],'1')

    def test_symlink_cache_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root=Path(temp);(root/'.runtime').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):environment(root)


if __name__=='__main__':unittest.main()
