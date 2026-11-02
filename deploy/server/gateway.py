"""Password gateway for the existing loopback-only simulation API."""
from __future__ import annotations
import fcntl

import argparse
import base64
import hashlib
import hmac
from http.client import HTTPConnection
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit

COOKIE = 'openarm_session'
DEVICE = 'openarm_device'
YEAR = 365 * 24 * 3600
LOGIN = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>登录 · OpenArm Lab</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f6f5f1;color:#302e29;font:15px system-ui,sans-serif;display:grid;place-items:center;min-height:100dvh}main{width:min(360px,calc(100% - 40px));background:#fffefa;border:1px solid #e4e0d7;border-radius:18px;padding:38px 32px;box-shadow:0 12px 48px #3a322b08}h1{font-size:24px;font-weight:600;margin:0 0 30px}input,button{width:100%;height:46px;border-radius:9px;font:inherit}input{padding:0 13px;border:1px solid #dcd5cb;background:#fff;outline:none}input:focus{border-color:#b98469;box-shadow:0 0 0 3px #b9846915}button{margin-top:16px;border:0;background:#a96d50;color:white;cursor:pointer}button:hover{background:#955d43}.error{color:#a24635;margin:12px 0 0;font-size:14px}.error:empty{display:none}</style>
<main><h1>OpenArm Lab</h1><form method="post" action="/login"><input type="password" name="password" aria-label="密码" placeholder="输入密码" autocomplete="current-password" required autofocus><p class="error" role="alert">{error}</p><button>登录</button></form></main></html>'''


class Auth:
    def __init__(self, path):
        path = Path(path)
        if path.stat().st_mode & 0o077:
            raise RuntimeError('登录配置必须仅由文件所有者读写')
        self.config = json.loads(path.read_text())
        self.key = bytes.fromhex(self.config['session_key'])
        self.lock = threading.Lock()
        self.attempts = {}

    def allowed_attempt(self, address):
        now = time.monotonic()
        with self.lock:
            self.attempts = {key: value for key, value in self.attempts.items() if now - value[0] < 60}
            return self.attempts.get(address, (now, 0))[1] < 10

    def verify(self, password, address):
        with self.lock:
            start, count = self.attempts.get(address, (time.monotonic(), 0))
            self.attempts[address] = (start, count + 1)
        derived = hashlib.scrypt(password.encode(), salt=bytes.fromhex(self.config['salt']),
                                 n=16384, r=8, p=1).hex()
        return hmac.compare_digest(derived, self.config['password_hash'])

    def issue(self, profile=None, *, name=COOKIE):
        payload = f'{int(time.time()) + YEAR}:{profile or secrets.token_hex(24)}'
        message = (('device:' if name == DEVICE else '') + payload).encode()
        signature = hmac.new(self.key, message, hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(f'{payload}:{signature}'.encode()).decode()

    def profile(self, header, *, name=COOKIE):
        try:
            cookie = SimpleCookie(header)
            token = cookie[name].value
            if len(token) > 256:
                return None
            expiry, nonce, signature = base64.urlsafe_b64decode(token).decode().split(':')
            message = (('device:' if name == DEVICE else '') + f'{expiry}:{nonce}').encode()
            expected = hmac.new(self.key, message, hashlib.sha256).hexdigest()
            return nonce if int(expiry) > time.time() and hmac.compare_digest(signature, expected) else None
        except (ValueError, KeyError, UnicodeError, CookieError):
            return None

    def valid(self, header):
        return self.profile(header) is not None


class Gateway(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def reply(self, status, body=b'', content_type='text/html; charset=utf-8', headers=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'same-origin')
        for key, value in (headers or {}).items():
            for item in value if isinstance(value, list) else [value]:
                self.send_header(key, item)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def same_origin(self):
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        return (self.headers.get('Sec-Fetch-Site') != 'cross-site' and
                (not origin or origin in (f'http://{host}', f'https://{host}')))

    def cookie(self, token, *, clear=False, name=COOKIE):
        secure = ('; Secure' if self.client_address[0] in ('127.0.0.1', '::1')
                  and self.headers.get('X-Forwarded-Proto', '').lower() == 'https' else '')
        return f'{name}={token}; Path=/; Max-Age={0 if clear else YEAR}; HttpOnly; SameSite=Strict{secure}'

    def remember(self, profile):
        return [self.cookie(self.server.auth.issue(profile)),
                self.cookie(self.server.auth.issue(profile, name=DEVICE), name=DEVICE)]

    def dispatch(self):
        root = getattr(self.server, 'root', None)
        if root is None or self.command != 'POST' or not urlsplit(self.path).path.startswith('/api/'):
            return self._dispatch()
        folder = root / '.runtime/update'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'admission.lock').open('a') as admission:
            try: fcntl.flock(admission, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return self.reply(503, b'{"error":"platform_updating_retry_later"}', 'application/json')
            try: return self._dispatch()
            finally: fcntl.flock(admission, fcntl.LOCK_UN)

    def _dispatch(self):
        if not self.path.startswith('/') or self.path.startswith('//'):
            return self.reply(400)
        if self.command == 'POST' and not self.same_origin():
            return self.reply(403, b'{"error":"cross_origin_rejected"}', 'application/json')
        path = urlsplit(self.path).path
        if path == '/version' and self.command in ('GET','HEAD'):
            root = getattr(self.server, 'root', None)
            release = json.loads((root / 'DEPLOYMENT.json').read_text()) if root and (root / 'DEPLOYMENT.json').is_file() else {}
            value = {'source_commit':release.get('source_commit'), 'channel':'main'}
            return self.reply(200, json.dumps(value).encode(), 'application/json')
        if path == '/login' and self.command in ('GET', 'HEAD'):
            return self.reply(200, LOGIN.replace('{error}', '').encode())
        if path == '/login' and self.command == 'POST':
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError()
                form = parse_qs(self.rfile.read(size).decode())
                password = form.get('password', [''])[0]
            except (ValueError, UnicodeError):
                return self.reply(400)
            auth = self.server.auth
            if not auth.allowed_attempt(self.client_address[0]):
                return self.reply(429, LOGIN.replace('{error}', '尝试过多，请稍后再试').encode())
            if not auth.verify(password, self.client_address[0]):
                return self.reply(401, LOGIN.replace('{error}', '密码不正确').encode())
            profile = auth.profile(self.headers.get('Cookie', ''), name=DEVICE)
            return self.reply(303, headers={'Location': '/', 'Set-Cookie': self.remember(profile or secrets.token_hex(24))})
        profile = self.server.auth.profile(self.headers.get('Cookie', ''))
        if not profile:
            if path.startswith(('/api/', '/runs/')) or path == '/health':
                return self.reply(401, b'{"error":"login_required"}', 'application/json')
            return self.reply(303, headers={'Location': '/login'})
        if path == '/logout' and self.command == 'POST':
            return self.reply(303, headers={'Location': '/login', 'Set-Cookie': self.cookie('', clear=True)})
        connection = HTTPConnection('127.0.0.1', self.server.backend_port, timeout=180)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 <= size <= 1024 * 1024 or self.headers.get('Transfer-Encoding'):
                return self.reply(413)
            body = self.rfile.read(size) if size else None
            host = f'127.0.0.1:{self.server.backend_port}'
            headers = {'Host': host, 'X-OpenArm-Profile': profile}
            if body is not None:
                headers['Content-Type'] = self.headers.get('Content-Type', 'application/json')
                headers['Origin'] = f'http://{host}'
            connection.request(self.command, self.path, body, headers)
            response = connection.getresponse()
            payload = response.read()
            self.reply(response.status, payload, response.getheader('Content-Type', 'application/octet-stream'),
                       {'Set-Cookie': self.remember(profile)})
        except (OSError, ValueError):
            self.reply(502, b'{"error":"simulation_unavailable"}', 'application/json')
        finally:
            connection.close()

    def do_GET(self): self.dispatch()
    def do_HEAD(self): self.dispatch()
    def do_POST(self): self.dispatch()
    def log_message(self, *_): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--backend-port', type=int, default=18767)
    parser.add_argument('--bind', default='0.0.0.0')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), Gateway)
    server.daemon_threads = True
    server.auth = Auth(args.root / '.private/auth.json')
    server.root = args.root
    server.backend_port = args.backend_port
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__': main()
