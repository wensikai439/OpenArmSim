#!/usr/bin/env python3
"""Read-only integrity and native-MuJoCo smoke check for an exported package."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_manifest():
    manifest_path = ROOT / 'PACKAGE-MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'openarmsim-colleague-package/2':
        raise RuntimeError('unsupported handoff manifest schema')
    entries = manifest.get('entries')
    if not isinstance(entries, list) or manifest.get('files') != len(entries):
        raise RuntimeError('invalid handoff manifest entry count')
    root = ROOT.resolve()
    for entry in entries:
        rel = PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError(f'unsafe manifest path: {entry["path"]}')
        path = (ROOT / Path(*rel.parts)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'missing or unsafe package file: {entry["path"]}')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry['bytes'] or digest != entry['sha256']:
            raise RuntimeError(f'package hash mismatch: {entry["path"]}')
    return len(entries)


def verify_simulation():
    import numpy as np
    from sim.alignment import load_samples
    from sim.environment import OpenArmEnv
    from sim.replay import ReferenceReplay

    samples = load_samples()
    env = OpenArmEnv(images=False)
    try:
        env.reset(seed=0, options={'episode': 0})
        replay = ReferenceReplay(env, 0)
        replay.advance()
        state = np.asarray(env.state(), dtype=float)
        if env.tick != 1 or state.shape != (16,) or not np.isfinite(state).all():
            raise RuntimeError('simulator smoke transition failed')
        return {
            'reference_image_samples': len(samples['samples']),
            'replay_steps': env.tick,
            'state_dimension': int(state.size),
            'physics_time_seconds': float(env.data.time),
        }
    finally:
        env.close()


def main():
    file_count = verify_manifest()
    simulation = verify_simulation()
    print(json.dumps({
        'status': 'ok',
        'manifest_files_verified': file_count,
        **simulation,
        'scope': 'offline package integrity and MuJoCo smoke only; not physical calibration or sim-to-real validation',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
