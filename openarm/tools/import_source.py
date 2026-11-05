"""Snapshot a small allowlist of existing robot sources over read-only SSH.

Remote code uses only the standard library and read_bytes(); it never imports
robot modules, writes a file, touches devices, or builds an archive on disk.
"""
from pathlib import Path
import base64
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = '/home/sj/Documents/moqi_workspace_20251119_211400'
FILES = {
    'robot.urdf': 'openarm/openarm_description/urdf/robot/openarm_bimanual.urdf',
    'robot-original.urdf': 'openarm/openarm_description/urdf/robot/openarm_bimanual_origin.urdf',
    'robot.yaml': 'pyroki/config/robot.yaml',
    'solver.py': 'pyroki/robot_ik_solver.py',
    'controller.py': 'openarm/openarm_controller.py',
    'recording-v4.py': 'pyroki/main_v4_record_velocity.py',
    'recording-v5.py': 'pyroki/main_v5_record_velocity.py',
    'camera.py': 'pyroki/realsense_camera.py',
}


def main():
    # No filename or shell command comes from a model/tool argument.
    remote = ("from pathlib import Path\nimport base64,hashlib,json\n"
              f"base=Path({BASE!r})\nfiles={FILES!r}\nrows={{}}\n"
              "for name,relative in files.items():\n"
              " p=(base/relative).resolve()\n"
              " assert p.is_relative_to(base) and p.is_file() and p.stat().st_size<262144\n"
              " raw=p.read_bytes()\n"
              " rows[name]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),"
              "'bytes':len(raw),'content':base64.b64encode(raw).decode()}\n"
              "print(json.dumps(rows))\n")
    process = subprocess.run([
        'ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
        '-T', 'data-cs', 'python3 -B -'], input=remote, text=True,
        capture_output=True, timeout=60, check=True)
    if len(process.stdout) > 3 * 1024 * 1024:
        raise ValueError('unexpected source snapshot size')
    rows = json.loads(process.stdout)
    if set(rows) != set(FILES):
        raise ValueError('source allowlist mismatch')
    contents = {}
    for name, row in rows.items():
        raw = base64.b64decode(row.pop('content'), validate=True)
        if (row['path'] != BASE + '/' + FILES[name] or len(raw) != row['bytes']
                or hashlib.sha256(raw).hexdigest() != row['sha256']):
            raise ValueError('source identity mismatch')
        contents[name] = raw
    output = ROOT / 'reference' / 'source'
    (output / 'inputs').mkdir(parents=True, exist_ok=False)
    for name, raw in contents.items():
        (output / 'inputs' / name).write_bytes(raw)
    manifest = {
        'schema': 'openarm-source-audit/1', 'kind': 'source-audit', 'status': 'snapshot',
        'run_id': output.name, 'captured_at_unix': time.time(), 'host': 'data-cs',
        'files': rows, 'importer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'current on-disk source; capture-time source revision is not established',
        'remote_access': 'read-only allowlist; no hardware modules imported or devices accessed',
        'transfer_ready': False,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(output), 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows.values())}))


if __name__ == '__main__':
    main()
