"""Browser-owned model configuration and task-bound automatic-review credentials."""
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ContextVar('openarm_api_profile', default=None)


def profile_id(value):
    return value if isinstance(value, str) and re.fullmatch(r'[a-f0-9]{48}', value) else None


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
                        'confirmed': False, 'display_name': ''}
            return values

        def public(self):
            values = self.snapshot()
            return {key: value for key, value in values.items() if key != 'api_key'} | {
                'configured': bool(values['api_key']) and values.get('confirmed', False),
                'storage': 'private_browser_profile', 'storage_error': None}

        def configure(self, values):
            selected = self.selected()
            if selected is None: raise ValueError('请先登录')
            values = dict(values)
            confirmed = values.pop('confirmed', False)
            name = values.pop('display_name', '')
            if confirmed is not True: raise ValueError('请先确认本人的请求地址、模型与 Key')
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
                raise ValueError('请输入姓名或昵称')
            selected.store.metadata = {'confirmed': True, 'display_name': name.strip()}
            selected.configure(values)
            # ModelSettings keeps its in-memory override separately from Store.
            selected.override.update(selected.store.metadata)
            return self.public()

        def clear(self):
            selected = self.selected()
            if selected is None: raise ValueError('请先登录')
            selected.store.metadata = {'confirmed': False, 'display_name': ''}
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


def bind_harness(base):
    class BrowserHarness(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sessions = ProfileSessions(self.sessions)

        def chat(self, *args, **kwargs):
            root = kwargs.get('session_id')
            profile = self.sessions.get(root, {}).get('_api_profile') if root else PROFILE.get()
            if not profile_id(profile): raise ValueError('本次任务缺少已确认的使用者配置')
            token = PROFILE.set(profile)
            try: return super().chat(*args, **kwargs)
            finally: PROFILE.reset(token)
    return BrowserHarness
