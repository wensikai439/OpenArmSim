"""Runtime builder guard and layer-failure diagnostics."""
import copy
import unittest

from tools import object_workbench


class RuntimeCompositionChecks(unittest.TestCase):
    def setUp(self):
        self.saved=copy.deepcopy(object_workbench._RUNTIME_TRACE)

    def tearDown(self):
        object_workbench._RUNTIME_TRACE.clear()
        object_workbench._RUNTIME_TRACE.update(self.saved)

    def test_builder_allows_one_process_local_installation_only(self):
        object_workbench._RUNTIME_TRACE.update(attempted=False,complete=False,profile=None,
            global_layers=[],factory_layers=[],failed_layer=None)
        object_workbench._claim_backend_build()
        with self.assertRaisesRegex(RuntimeError,'全新进程'):
            object_workbench._claim_backend_build()

    def test_failed_layer_names_the_partially_modified_process(self):
        object_workbench._RUNTIME_TRACE.update(attempted=True,complete=False,profile=None,
            global_layers=[],factory_layers=[],failed_layer=None)
        def fail(): raise ValueError('fixture')
        with self.assertRaisesRegex(RuntimeError,'fixture.layer'):
            object_workbench._run_layer('global_layers','fixture.layer',fail)
        self.assertEqual(object_workbench._RUNTIME_TRACE['failed_layer'],'fixture.layer')


if __name__=='__main__':unittest.main()
