from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from PhyAgentOS.agent.skills import SkillsLoader
from PhyAgentOS.agent.experience.activation import SkillActivationManager
from PhyAgentOS.agent.experience.store import ExperienceStore
from PhyAgentOS.forge.binding import ForgeSkillBindingError, ForgeSkillBindingResolver
from PhyAgentOS.skill_runtime.catalog import SkillCatalog


class QueryCatalogFixture:
    async def get_tool(self, tool_id):
        return {"ok": True, "data": {"tool_id": tool_id, "semantics": "query"}}

    async def get_tool_context(self, tool_id):
        return {"ok": True, "data": {"ready": True, "binding_error": None}}


class WorkflowOverlayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.workspace = root / "workspace"
        installed = root / "installed"
        bundle = installed / "example-forge"
        (bundle / "profiles/local").mkdir(parents=True)
        self.document = bundle / "SKILL.md"
        self.document.write_text("---\nname: example-forge\ndescription: Example workflow.\n---\n# Example workflow\n")
        (bundle / "profiles/local/dataflow.yaml").write_text("nodes: []\n")
        (bundle / "skill.yaml").write_text(yaml.safe_dump({
            "manifest_version": 2, "name": "example-forge", "version": "1.2.3",
            "description": "Example workflow", "skill_document": "SKILL.md",
            "gateway_url": "http://127.0.0.1:9", "required_tools": ["demo.query"],
            "profiles": {"local": {"dataflow": "profiles/local/dataflow.yaml"}},
        }))
        self.overlay = self.workspace / "skills/example-forge/SKILL.md"
        self.overlay.parent.mkdir(parents=True)
        self.overlay.write_text(self.document.read_text() + "\n## Learned Workflow (revision 2)\nUse fresh observations.\n")
        runtime = SimpleNamespace(
            skill_name="example-forge", skill_version="1.2.3", profile="local",
            runtime_instance_id="runtime-unit", gateway_url="http://127.0.0.1:9",
            gateway_identity="gateway-unit", client=QueryCatalogFixture(),
        )
        registry = SimpleNamespace(current=lambda: runtime)
        self.resolver = ForgeSkillBindingResolver(
            registry, catalog=SkillCatalog(installed),
            skills_loader=SkillsLoader(self.workspace, installed_skills_dir=installed),
        )
        self.activation = SkillActivationManager(
            workspace=self.workspace, store=ExperienceStore(root / "experience"),
            runtime_availability_provider=lambda name: True, binding_resolver=self.resolver,
        )
        self.activation.skills.installed_skills = installed
        self.activation.begin_turn("unit:test", "use the saved workflow")

    async def activate(self):
        return await self.activation.activate(
            session_key="unit:test", name="example-forge", role="primary",
        )

    async def test_workspace_guide_binds_installed_metadata_and_both_hashes(self):
        self.assertFalse(self.overlay.with_name("skill.yaml").exists())
        activated, content, _ = await self.activate()
        self.assertEqual(activated.skill_version, "1.2.3")
        self.assertIn("revision 2", content)
        self.assertTrue(activated.binding_candidate_id)
        binding = await self.resolver.freeze(activated.binding_candidate_id, task_id="unit-task")
        self.assertEqual(binding.skill_document_sha256, hashlib.sha256(self.document.read_bytes()).hexdigest())
        self.assertEqual(binding.workflow_document_sha256, hashlib.sha256(self.overlay.read_bytes()).hexdigest())
        self.assertEqual(activated.content_sha256, binding.workflow_document_sha256)
        self.assertNotEqual(binding.skill_document_sha256, binding.workflow_document_sha256)

    async def test_workflow_change_after_activation_freezes_activated_hash(self):
        activated, _, _ = await self.activate()
        self.overlay.write_text(self.overlay.read_text() + "\nA later revision.\n")
        binding = await self.resolver.freeze(
            activated.binding_candidate_id, task_id="unit-task",
            activated_document_sha256=activated.content_sha256,
        )
        self.assertEqual(binding.workflow_document_sha256, activated.content_sha256)
        self.assertNotEqual(binding.workflow_document_sha256, hashlib.sha256(self.overlay.read_bytes()).hexdigest())

    async def test_overlay_cannot_silently_use_bundle_document_hash(self):
        self.resolver.skills_loader = None
        with self.assertRaisesRegex(ValueError, "workflow binding preview"):
            await self.activate()


if __name__ == "__main__":
    unittest.main()
