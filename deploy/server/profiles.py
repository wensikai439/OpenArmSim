"""Browser-owned model configuration and task-bound automatic-review credentials."""
from contextvars import ContextVar
from collections import deque
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ContextVar('openarm_api_profile', default=None)


def profile_id(value):
    return value if isinstance(value, str) and re.fullmatch(r'[a-f0-9]{48}', value) else None


def folder_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[\w-]{1,40}', value, re.UNICODE):
        raise ValueError('目录名请使用 1–40 个中文、字母、数字、下划线或短横线')
    return value


def results_dir(profile=None, *, create=False):
    profile = profile_id(profile or PROFILE.get())
    if not profile: raise ValueError('请先登录并设置个人实验目录')
    values = ProfileStore(profile).load() or {}
    name = folder_name(values.get('results_folder'))
    root = ROOT / 'runs'
    parent = root / 'users'
    path = parent / (name + '-' + profile[:16])
    for item in (root, parent, path):
        if item.is_symlink(): raise ValueError('个人实验目录不能使用符号链接')
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('个人实验目录超出允许范围')
    if create: path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


class ProfileStore:
    def __init__(self, profile):
        if not profile_id(profile): raise ValueError('无效的使用者标识')
        self.path = ROOT / '.private/profiles' / (profile + '.json')
        self.metadata = {}

    def load(self):
        if not self.path.exists(): return None
        if self.path.is_symlink() or self.path.stat().st_mode & 0o077:
            raise RuntimeError('使用者配置权限应为 600')
        return json.loads(self.path.read_text())

    def save(self, value):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        value = {**value, **self.metadata}
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix='profile-', dir=self.path.parent)
            with os.fdopen(fd, 'w') as stream: json.dump(value, stream)
            os.replace(temporary, self.path)
        finally:
            if temporary and Path(temporary).exists(): Path(temporary).unlink()


def model_settings(base):
    class Settings:
        def __init__(self, store=None):
            self.lock = threading.RLock()
            self.profiles = {}

        def selected(self):
            profile = profile_id(PROFILE.get())
            if not profile: return None
            with self.lock:
                if profile not in self.profiles:
                    store = ProfileStore(profile)
                    self.profiles[profile] = base(store=store)
                return self.profiles[profile]

        def snapshot(self):
            selected = self.selected()
            values = selected.snapshot() if selected else {}
            if not values.get('confirmed'):
                return {'base_url': 'https://zzq.si/v1', 'model': '', 'api_key': '', 'vision': True,
                        'confirmed': False, 'display_name': '', 'results_folder': ''}
            return values

        def public(self):
            values = self.snapshot()
            return {key: value for key, value in values.items() if key != 'api_key'} | {
                'configured': bool(values['api_key']) and values.get('confirmed', False) and bool(values.get('results_folder')),
                'storage': 'private_browser_profile', 'storage_error': None}

        def configure(self, values):
            selected = self.selected()
            if selected is None: raise ValueError('请先登录')
            values = dict(values)
            confirmed = values.pop('confirmed', False)
            name = values.pop('display_name', '')
            folder = folder_name(values.pop('results_folder', ''))
            if confirmed is not True: raise ValueError('请先确认本人的请求地址、模型与 Key')
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
                raise ValueError('请输入姓名或昵称')
            selected.store.metadata = {'confirmed': True, 'display_name': name.strip(), 'results_folder': folder}
            selected.configure(values)
            # ModelSettings keeps its in-memory override separately from Store.
            selected.override.update(selected.store.metadata)
            results_dir(create=True)
            return self.public()

        def clear(self):
            selected = self.selected()
            if selected is None: raise ValueError('请先登录')
            selected.store.metadata = {'confirmed': False, 'display_name': '', 'results_folder': ''}
            selected.clear(); selected.override.update(selected.store.metadata)
            return self.public()

        def test(self):
            selected = self.selected()
            if selected is None or not self.public()['configured']:
                raise ValueError('请先确认并保存本人的 API 配置')
            return selected.test()
    return Settings


class ProfileSessions(dict):
    def setdefault(self, key, default=None):
        if key not in self and isinstance(default, dict):
            default['_api_profile'] = PROFILE.get()
        return super().setdefault(key, default)


class ProfileHistory:
    def __init__(self):
        self.histories = {}
        self.lock = threading.Lock()

    def selected(self):
        profile = profile_id(PROFILE.get())
        with self.lock:
            return self.histories.setdefault(profile, {'execute': deque(maxlen=12), 'research': deque(maxlen=12)})

    def __getitem__(self, role): return self.selected()[role]
    def values(self): return self.selected().values()


def bind_harness(base):
    class BrowserHarness(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sessions = ProfileSessions(self.sessions)
            if hasattr(self, 'history'): self.history = ProfileHistory()

        def chat(self, *args, **kwargs):
            root = kwargs.get('session_id')
            profile = self.sessions.get(root, {}).get('_api_profile') if root else PROFILE.get()
            if not profile_id(profile): raise ValueError('本次任务缺少已确认的使用者配置')
            token = PROFILE.set(profile)
            try: return super().chat(*args, **kwargs)
            finally: PROFILE.reset(token)

        def _log(self, name, args, result):
            profile = profile_id(PROFILE.get())
            if not profile:
                root = getattr(self, 'active_session', None)
                profile = self.sessions.get(root, {}).get('_api_profile') if root else None
            path = results_dir(profile, create=True)
            snapshot = self.service.snapshot()
            active = snapshot.get('job')
            run_id = active.get('run_id') if active else (snapshot.get('last_run') or {}).get('run_id')
            with (path / 'harness.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': time.time(), 'run_id': run_id, 'tool': name,
                                         'arguments': args, 'result': result}, ensure_ascii=False) + '\n')
    return BrowserHarness
