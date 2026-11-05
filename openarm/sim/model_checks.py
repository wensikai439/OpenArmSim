"""Actual VFS bytes, mesh-drift invalidation, and compiled parameter identity."""
import os
from pathlib import Path
import tempfile
import unittest

import mujoco

from .model_evidence import ModelInputs,ParameterAudit,compiled_evidence
from .service import memory_context


MESH='v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nv 0 0 0.01\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n'


class ModelChecks(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder);(root/'shape.obj').write_text(MESH)
        (root/'model.xml').write_text('<mujoco><compiler meshdir="'+str(root)+'"/>'
            '<asset><mesh name="shape" file="shape.obj"/></asset><worldbody>'
            '<geom type="mesh" mesh="shape"/></worldbody></mujoco>')
        return root,ModelInputs(root,root/'model.xml')

    def test_mesh_loaded_from_frozen_bytes_even_if_disk_changed_before_compile(self):
        with tempfile.TemporaryDirectory() as folder:
            root,inputs=self.fixture(folder)
            (root/'shape.obj').write_text(MESH.replace('0.01','0.02'))
            spec=mujoco.MjSpec.from_string(inputs.xml(root/'model.xml'),assets=inputs.assets)
            frozen=spec.compile()
            current=mujoco.MjModel.from_xml_path(str(root/'model.xml'))
            self.assertNotEqual(compiled_evidence(frozen)['sha256'],compiled_evidence(current)['sha256'])
            self.assertEqual(inputs.changed(),['shape.obj'])
            self.assertEqual(inputs.describe(frozen)['compiled_resources'][0]['source'],'shape.obj')
            self.assertEqual(len(inputs.describe(frozen)['files']),2)

    def test_same_size_mesh_edit_is_detected_despite_restored_mtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root,inputs=self.fixture(folder);p=root/'shape.obj';before=p.stat()
            self.assertEqual(inputs.changed(),[])
            p.write_text(MESH.replace('0.01','0.02'))
            os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertEqual(inputs.changed(),['shape.obj'])
            p.write_text(MESH)
            self.assertEqual(inputs.changed(),[])
            p.unlink()
            self.assertEqual(inputs.changed(),['shape.obj'])

    def test_recursive_xml_and_ambiguous_asset_basename_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'a').mkdir();(root/'b').mkdir()
            (root/'a/shape.obj').write_text(MESH)
            (root/'b/shape.obj').write_text(MESH.replace('0.01','0.02'))
            (root/'child.xml').write_text('<mujoco><asset><mesh name="b" file="b/shape.obj"/></asset></mujoco>')
            (root/'model.xml').write_text('<mujoco><include file="child.xml"/>'
                                         '<asset><mesh name="a" file="a/shape.obj"/></asset></mujoco>')
            with self.assertRaisesRegex(ValueError,'ambiguous'):ModelInputs(root,root/'model.xml')
            (root/'b/shape.obj').write_text(MESH)
            inputs=ModelInputs(root,root/'model.xml')
            self.assertEqual(set(inputs.hashes()),{'model.xml','child.xml','a/shape.obj','b/shape.obj'})

    def test_compiled_identity_tracks_contact_inertia_constraint_and_camera_parameters(self):
        xml='<mujoco><worldbody><body><freejoint/><geom size=".02" mass=".5"/>'
        xml+='<camera name="eye" pos="0 0 1"/></body></worldbody></mujoco>'
        original=mujoco.MjModel.from_xml_string(xml)
        expected=compiled_evidence(original)['sha256']
        self.assertEqual(expected,compiled_evidence(mujoco.MjModel.from_xml_string(xml))['sha256'])
        for field in ('geom_friction','body_mass','cam_fovy'):
            model=mujoco.MjModel.from_xml_string(xml)
            getattr(model,field)[-1]*=1.1
            self.assertNotEqual(expected,compiled_evidence(model)['sha256'])
        model=mujoco.MjModel.from_xml_string(xml);model.opt.noslip_iterations=2
        self.assertNotEqual(expected,compiled_evidence(model)['sha256'])
        manifest={'hashes':{},'model_evidence':{'inputs':{'content_sha256':'files'},
                                              'compiled':{'sha256':expected}}}
        before=memory_context(manifest,{},'v1')
        manifest['model_evidence']['compiled']['sha256']='changed'
        self.assertNotEqual(before['contract'],memory_context(manifest,{},'v1')['contract'])

    def test_missing_or_external_assets_cannot_claim_complete_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'model.xml').write_text('<mujoco><asset><mesh file="missing.obj"/></asset></mujoco>')
            with self.assertRaises(OSError):ModelInputs(root,root/'model.xml')
            with self.assertRaisesRegex(ValueError,'outside'):ModelInputs(root,root.parent/'model.xml')

    def test_runtime_gain_and_solver_changes_are_visible_but_feedforward_is_dynamic(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/>'
                '<geom size=".02"/></body></worldbody><actuator><position joint="j"/></actuator></mujoco>')
        audit=ParameterAudit(model);before=audit.summary()
        model.actuator_biasprm[:,0]=2
        self.assertEqual(before,audit.summary())
        model.actuator_gainprm[:,0]*=2
        model.opt.noslip_iterations=3
        after=audit.summary()
        self.assertNotEqual(before['sha256'],after['sha256'])
        self.assertEqual(after['changed_fields'],['actuator_gainprm','option.noslip_iterations'])

    def test_native_experiment_guard_rejects_changed_contact_parameters(self):
        from .environment import OpenArmEnv
        env=OpenArmEnv(images=False)
        try:
            env.reset(seed=0);env.step_tamp(env.last_action)
            env.require_current_sources()
            original=env.data.qpos.copy()
            gid=env.model.geom('bottle_body').id
            env.model.geom_friction[gid,0]*=1.1
            with self.assertRaisesRegex(ValueError,'模型参数'):env.require_current_sources()
            self.assertTrue((env.data.qpos==original).all())
        finally:env.close()
