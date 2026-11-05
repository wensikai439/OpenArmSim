"""Archive schema compatibility and tamper detection, without robot access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import verify_handoff


class PackageChecks(unittest.TestCase):
    def fixture(self,root,version):
        data=b'offline fixture';(root/'payload.txt').write_bytes(data)
        manifest={'schema':'openarmsim-colleague-package/'+version,'files':1,
            'entries':[{'path':'payload.txt','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
        (root/'PACKAGE-MANIFEST.json').write_text(json.dumps(manifest))

    def test_both_published_manifest_schemas(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            for version in ('2','3'):
                self.fixture(root,version)
                with patch.object(verify_handoff,'ROOT',root):self.assertEqual(verify_handoff.verify_manifest(),1)

    def test_unknown_schema_does_not_silently_pass(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);self.fixture(root,'999')
            with patch.object(verify_handoff,'ROOT',root),self.assertRaisesRegex(RuntimeError,'schema'):
                verify_handoff.verify_manifest()

    def test_modified_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);self.fixture(root,'3');(root/'payload.txt').write_text('changed')
            with patch.object(verify_handoff,'ROOT',root),self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                verify_handoff.verify_manifest()


if __name__=='__main__':unittest.main()
