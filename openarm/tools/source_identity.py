"""Small runtime source fingerprint for the managed simulation workbench."""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time


EXTENSIONS = {'.py', '.html', '.css', '.js', '.json', '.xml', '.urdf', '.yaml', '.yml', '.toml', '.obj', '.stl', '.dae', '.mjb'}
RUNTIME_TREES = ('harness', 'sim', 'ui', 'task/planners', 'openarm_mujoco/v2')
RUNTIME_FILES = (
    'run', 'server.py', 'calibrate.json', 'sim/learning.json', 'harness/policy.json',
    'tools/object_workbench.py', 'tools/platform_service.py', 'tools/source_identity.py',
    'tools/openarm-sim.service', 'tools/openarm-act.service', 'tools/openarm-curobo.service',
    'tools/contact_workbench.py', 'tools/scene_gpu.py', 'tools/scene_cache.py', 'tools/object_gpu.py',
)
EXCLUDED_DIRS = {'.git', '__pycache__', 'state', 'cache', 'logs'}


def source_files(root: Path) -> list[Path]:
    root = Path(root).resolve()
    found = {root / name for name in RUNTIME_FILES if (root / name).is_file()}
    for name in RUNTIME_TREES:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            relative = path.relative_to(root)
            if (path.is_file() and path.suffix.lower() in EXTENSIONS
                    and not EXCLUDED_DIRS.intersection(relative.parts)):
                found.add(path)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def fingerprint(root: Path) -> tuple[str, int]:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    paths = source_files(root)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, 'big'))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest(), len(paths)


class SourceIdentity:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.revision, self.file_count = fingerprint(self.root)
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._current_revision = self.revision
        self._current_count = self.file_count

    def status(self, *, force: bool = False) -> dict:
        with self._lock:
            now = time.monotonic()
            if force or now - self._last_check >= 1.0:
                self._current_revision, self._current_count = fingerprint(self.root)
                self._last_check = now
            current = self._current_revision == self.revision
            return {
                'workbench_id': 'openarm-objects',
                'entrypoint': 'tools.object_workbench',
                'started_at': self.started_at,
                'source_revision_sha256': self.revision,
                'source_file_count': self.file_count,
                'source_current': current,
                'source_change_detected': not current,
                'source_restart_required': not current,
            }

    def assert_current(self) -> None:
        if not self.status(force=True)['source_current']:
            raise RuntimeError('运行源码或仿真配置已变化；为避免新旧版本混跑，请先停止并重新启动仿真平台。')
