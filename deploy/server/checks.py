"""Verify gateway access and persistent authentication, without simulation/API calls."""
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import secrets
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('gateway', Path(__file__).with_name('gateway.py'))
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class GatewayChecks(unittest.TestCase):
    def test_access_and_persistent_login(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'auth.json'
            password = secrets.token_urlsafe(16)
            salt = secrets.token_bytes(16)
            path.write_text(json.dumps({'salt': salt.hex(), 'password_hash': hashlib.scrypt(
                password.encode(), salt=salt, n=16384, r=8, p=1).hex(), 'session_key': secrets.token_hex(32)}))
            path.chmod(0o600)
            class Backend(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200); self.end_headers(); self.wfile.write(b'protected')
                def log_message(self, *_): pass
            backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
            server = ThreadingHTTPServer(('127.0.0.1', 0), gateway.Gateway)
            server.auth = gateway.Auth(path)
            server.backend_port = backend.server_port
            for item in (backend, server):
                threading.Thread(target=item.serve_forever, daemon=True).start()
            def request(method, route, body=None, headers=None):
                connection = HTTPConnection('127.0.0.1', server.server_port)
                connection.request(method, route, body, headers or {})
                response = connection.getresponse()
                headers = dict(response.getheaders())
                cookies = [value for key, value in response.getheaders() if key.lower() == 'set-cookie']
                if cookies: headers['Set-Cookie'] = next((value for value in cookies if value.startswith(gateway.COOKIE + '=')), cookies[0])
                result = response.status, headers, response.read()
                connection.close()
                return result
            try:
                self.assertEqual(request('GET', '/')[0], 303)
                for route in ('/health', '/api/status', '/api/frame', '/runs/private/result.json'):
                    self.assertEqual(request('GET', route)[0], 401)
                self.assertEqual(request('POST', '/api/chat', '{}')[0], 401)
                self.assertEqual(request('POST', '/login', 'password=wrong')[0], 401)
                status, headers, _ = request('POST', '/login', 'password=' + password)
                self.assertEqual(status, 303)
                cookie = headers['Set-Cookie'].split(';')[0]
                self.assertIn('HttpOnly', headers['Set-Cookie'])
                self.assertIn('Max-Age=31536000', headers['Set-Cookie'])
                self.assertTrue(gateway.Auth(path).valid(cookie))
                profile = server.auth.profile(cookie)
                self.assertEqual(server.auth.profile(gateway.COOKIE + '=' + server.auth.issue(profile)), profile)
                device = gateway.DEVICE + '=' + server.auth.issue(profile, name=gateway.DEVICE)
                self.assertFalse(server.auth.valid(device.replace(gateway.DEVICE, gateway.COOKIE)))
                self.assertFalse(gateway.Auth(path).valid(cookie[:-2] + 'xx'))
                self.assertEqual(request('GET', '/api/frame', headers={'Cookie': cookie})[2], b'protected')
                self.assertEqual(request('POST', '/api/chat', '{}', {'Cookie': cookie, 'Origin': 'https://unrelated.example'})[0], 403)
                self.assertIn('Max-Age=0', request('POST', '/logout', headers={'Cookie': cookie})[1]['Set-Cookie'])
            finally:
                for item in (server, backend): item.shutdown(); item.server_close()

    def test_independent_profiles_and_review_owner(self):
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
            return result
        profiles = module('profiles_test', Path(__file__).with_name('profiles.py'))
        llm = module('llm_test', Path(__file__).resolve().parents[2] / 'harness/llm.py')
        with tempfile.TemporaryDirectory() as temporary:
            profiles.ROOT = Path(temporary)
            settings_type = profiles.model_settings(llm.ModelSettings)
            settings = settings_type()
            a, b = secrets.token_hex(24), secrets.token_hex(24)
            profiles.PROFILE.set(a)
            values = {'base_url': 'https://first.invalid/v1', 'model': 'model-a',
                      'vision': True, 'api_key': secrets.token_hex(24), 'display_name': 'A', 'results_folder': '实验A', 'confirmed': True}
            settings.configure(values)
            self.assertNotIn('api_key', settings.public())
            profiles.PROFILE.set(b)
            self.assertFalse(settings.public()['configured'])
            with self.assertRaises(ValueError): settings.configure({**values, 'confirmed': False})
            values_b = {**values, 'base_url': 'https://second.invalid/v1', 'api_key': secrets.token_hex(24), 'display_name': 'B', 'results_folder': '实验B'}
            settings.configure(values_b)
            folder_b = profiles.results_dir()
            self.assertEqual(settings.snapshot()['api_key'], values_b['api_key'])
            profiles.PROFILE.set(a)
            folder_a = profiles.results_dir()
            self.assertNotEqual(folder_a, folder_b)
            self.assertTrue(folder_a.is_relative_to(profiles.ROOT / 'runs'))
            with self.assertRaises(ValueError): settings.configure({**values, 'results_folder': '../other'})
            self.assertEqual(settings.snapshot()['api_key'], values['api_key'])
            self.assertEqual(settings_type().snapshot()['base_url'], values['base_url'])
            self.assertEqual((profiles.ROOT / '.private/profiles' / (a + '.json')).stat().st_mode & 0o777, 0o600)
            class Base:
                def __init__(self): self.sessions = {}
                def chat(self, *args, **kwargs):
                    root = kwargs.get('session_id', 'task-a')
                    self.sessions.setdefault(root, {})
                    return profiles.PROFILE.get()
            harness = profiles.bind_harness(Base)()
            self.assertEqual(harness.chat('first'), a)
            profiles.PROFILE.set(b)
            self.assertEqual(harness.chat('review', session_id='task-a'), a)
            self.assertEqual(profiles.PROFILE.get(), b)

    def test_queue_and_workspace_keep_owner(self):
        from deploy.server import profiles, records
        old_root = profiles.ROOT
        with tempfile.TemporaryDirectory() as temporary:
            profiles.ROOT = Path(temporary)
            try:
                owners = [secrets.token_hex(24), secrets.token_hex(24)]
                for owner in owners:
                    store = profiles.ProfileStore(owner)
                    store.save({'results_folder': 'team'})
                q = records.OwnedQueue()
                profiles.PROFILE.set(owners[0]); q.put('first')
                profiles.PROFILE.set(owners[1]); q.put('second')
                self.assertEqual(q.get(), 'first')
                self.assertEqual(profiles.PROFILE.get(), owners[0])
                self.assertEqual(q.get(), 'second')
                self.assertEqual(profiles.PROFILE.get(), owners[1])
                class Base:
                    def __init__(self, path): self.path = path; self.busy = False
                workspace = records.workspace_factory(Base)()
                profiles.PROFILE.set(owners[0]); path_a = workspace.path; workspace.busy = True
                profiles.PROFILE.set(owners[1]); path_b = workspace.path
                self.assertNotEqual(path_a, path_b)
                self.assertFalse(workspace.busy)
                profiles.PROFILE.set(owners[0]); self.assertTrue(workspace.busy)
            finally:
                profiles.ROOT = old_root


if __name__ == '__main__': unittest.main()
