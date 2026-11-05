"""Run the real management branch with a mocked CLI; never install a bundle."""
import ast, tempfile, unittest
from pathlib import Path
from unittest.mock import AsyncMock

class BundlePathMigrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        old = self.base / 'phyagentos'
        (old / 'bundles').mkdir(parents=True)
        (old / 'bundles/local.zip').write_bytes(b'fixture')
        self.root = self.base / 'agents/phyagentos'
        self.root.parent.mkdir()
        old.rename(self.root)
        old.symlink_to(self.root, target_is_directory=True)
        self.alias = old
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'console/management.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'skills_post')
        self.cli = AsyncMock(return_value='mocked')
        namespace = {'Path': Path, 'ROOT': self.alias, 'cli': self.cli}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<management.skills_post>', 'exec'), namespace)
        self.call = namespace['skills_post']

    async def test_local_bundle_accepts_legacy_and_canonical_roots(self):
        for bundle in [self.alias / 'bundles/local.zip', self.root / 'bundles/local.zip']:
            with self.subTest(bundle=str(bundle)):
                result = await self.call(object(), {'action': 'install', 'local': True, 'name': str(bundle)})
                self.assertTrue(result['ok'])
                self.assertEqual(self.cli.await_args.args[1], ['install', str(bundle.resolve()), '--yes', '--local'])

    async def test_outside_bundle_rejected_before_cli(self):
        outside = self.base / 'outside.zip'
        outside.write_bytes(b'fixture')
        with self.assertRaises(ValueError):
            await self.call(object(), {'action': 'install', 'local': True, 'name': str(outside)})
        self.cli.assert_not_awaited()

    async def test_bundle_symlink_escape_rejected_before_cli(self):
        outside = self.base / 'outside.zip'
        outside.write_bytes(b'fixture')
        escape = self.root / 'bundles/escape.zip'
        escape.symlink_to(outside)
        with self.assertRaises(ValueError):
            await self.call(object(), {'action': 'install', 'local': True, 'name': str(escape)})
        self.cli.assert_not_awaited()

    async def test_missing_bundle_rejected_before_cli(self):
        with self.assertRaises(ValueError):
            await self.call(object(), {'action': 'install', 'local': True,
                                      'name': str(self.root / 'bundles/missing.zip')})
        self.cli.assert_not_awaited()

if __name__ == '__main__':
    unittest.main()
