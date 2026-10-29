"""Export the simulation and reviewed ACT weights, without credentials or raw runs."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'handoff/4090/platform.tar.gz'


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for directory in ('sim', 'harness', 'ui', 'task', 'tools', 'reference',
                      'openarm_mujoco/v1', 'openarm_mujoco/v2', 'deploy/server', 'models/act'):
        for path in sorted((ROOT / directory).rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.is_symlink(): continue
            if any(part in ('__pycache__', 'state', '.git', 'node_modules', 'history') for part in relative.parts): continue
            if path.suffix in ('.pyc', '.log', '.db', '.sqlite', '.sqlite3'): continue
            if path.name.startswith('.env') or path.name in ('api.json', 'llm.json'): continue
            files.append(path)
    files += [ROOT / name for name in ('server.py', 'calibrate.json', 'source.json')]
    manifest = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': {}, 'credentials_included': False, 'raw_runs_included': False}
    with tarfile.open(OUTPUT, 'w:gz') as archive:
        for path in sorted(set(files)):
            name = path.relative_to(ROOT).as_posix()
            archive.add(path, arcname=name, recursive=False)
            manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
        info = tarfile.TarInfo('SERVER-MANIFEST.json'); info.size = len(data); info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix('.sha256').write_text(digest + '  ' + OUTPUT.name + '\n')
    print(json.dumps({'package': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
                      'files': len(manifest['files']), 'sha256': digest}))


if __name__ == '__main__': main()
