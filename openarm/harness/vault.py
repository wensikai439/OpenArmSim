"""Scoped Secret Service storage; no credentials in files, argv or logs."""
import json
from pathlib import Path
import subprocess


class SecretStore:
    def __init__(self, namespace=None):
        self.namespace=namespace or str(Path(__file__).resolve().parents[1])

    def _call(self, operation, value=None):
        try:
            process=subprocess.run(['/usr/bin/python3','-B',str(Path(__file__).resolve()),operation,self.namespace],
                input=json.dumps(value) if value is not None else '', text=True,
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=15,check=False)
            if process.returncode:
                raise RuntimeError()
            return json.loads(process.stdout)
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired):
            raise RuntimeError('系统密钥环暂不可用，请解锁本机登录密钥环后重试；现有配置未被替换') from None

    def load(self):return self._call('read')
    def save(self, value):self._call('write',value)


if __name__=='__main__':
    import sys
    try:
        import gi
        gi.require_version('Secret','1')
        from gi.repository import Secret
        schema=Secret.Schema.new('org.openarmsim.Model',Secret.SchemaFlags.NONE,
                                  {'workspace':Secret.SchemaAttributeType.STRING})
        attrs={'workspace':sys.argv[2]}
        if sys.argv[1]=='read':
            raw=Secret.password_lookup_sync(schema,attrs,None)
            result=json.loads(raw) if raw else None
        elif sys.argv[1]=='write':
            raw=sys.stdin.read(16384)
            json.loads(raw)
            if not Secret.password_store_sync(schema,attrs,Secret.COLLECTION_DEFAULT,'OpenArm Lab · API 配置',raw,None):
                raise RuntimeError()
            result=True
        elif sys.argv[1]=='remove-test' and attrs['workspace'].startswith('openarm-vault-test-'):
            result=Secret.password_clear_sync(schema,attrs,None)
        else:
            raise ValueError()
        print(json.dumps(result))
    except Exception:
        sys.exit(1)
