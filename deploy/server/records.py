"""Keep simulation archives and workspace state with their browser owner."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from deploy.server.profiles import PROFILE, profile_id, results_dir


class OwnedQueue(queue.Queue):
    """Propagate the request owner to the physics and completion threads."""
    def put(self, item, block=True, timeout=None):
        return super().put((item, PROFILE.get()), block, timeout)

    def get(self, block=True, timeout=None):
        item, profile = super().get(block, timeout)
        PROFILE.set(profile)
        return item


def install_service():
    import sim.service as service
    from sim.recording import Recorder
    from urllib.parse import quote
    service.queue = SimpleNamespace(Queue=OwnedQueue, Empty=queue.Empty)

    def record_url(_service, path):
        relative = Path(path).relative_to(results_dir())
        return '/runs/' + quote(relative.as_posix(), safe='/')
    service.SimulationService._record_url = record_url

    class OwnedRecorder(Recorder):
        def __init__(self, *args, root=None, **kwargs):
            self.archive_root = results_dir()
            super().__init__(*args, root=root or self.archive_root, **kwargs)

    base_wait = service.DecisionWait
    class OwnedDecisionWait(base_wait):
        def __init__(self, env, root, request_id, observation_id, **kwargs):
            if kwargs.get('output') is None:
                kwargs['output'] = results_dir(create=True) / (time.strftime('%Y%m%d') + '-agent-' + root[:8]) / 'waits'
            super().__init__(env, root, request_id, observation_id, **kwargs)

    service.Recorder = OwnedRecorder
    service.DecisionWait = OwnedDecisionWait


def workspace_factory(base):
    class BrowserWorkspace:
        def __init__(self, _path=None):
            object.__setattr__(self, '_lock', threading.RLock())
            object.__setattr__(self, '_stores', {})
            object.__setattr__(self, '_empty', base(None))

        def selected(self):
            profile = profile_id(PROFILE.get())
            if not profile: return self._empty
            try:
                folder = results_dir(profile, create=True)
            except ValueError:
                return self._empty
            key = str(folder)
            with self._lock:
                if key not in self._stores:
                    for name in ('workspace.sqlite3', 'workspace.sqlite3-wal', 'workspace.sqlite3-shm'):
                        if (folder / name).is_symlink():
                            raise ValueError('工作区文件不能是符号链接')
                    self._stores[key] = base(folder / 'workspace.sqlite3')
                return self._stores[key]

        def __getattr__(self, name):
            return getattr(self.selected(), name)

        def __setattr__(self, name, value):
            if name.startswith('_'): object.__setattr__(self, name, value)
            else: setattr(self.selected(), name, value)

    return BrowserWorkspace
