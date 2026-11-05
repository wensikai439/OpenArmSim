"""Manage only this project's optional temporary HTTPS tunnel."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / '.runtime/tunnel'
BIN = DIR / 'cloudflared'
STATE = DIR / 'process.json'
LOG = DIR / 'tunnel.log'


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        command = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if str(BIN).encode() not in command or b'tunnel' not in command or b'http://127.0.0.1:8767' not in command:
            return None
        if proc.joinpath('cwd').resolve() != DIR: return None
        return proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError): return None


def state():
    try:
        info = json.loads(STATE.read_text())
        return info if identity(info['pid']) == info['start'] else None
    except (OSError, ValueError, KeyError): return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('start', 'stop', 'status'))
    args = parser.parse_args()
    current = state()
    if args.command == 'stop':
        if current and identity(current['pid']) == current['start']:
            os.kill(current['pid'], signal.SIGTERM)
        print(json.dumps({'stopped': bool(current)})); return
    if args.command == 'start' and not current:
        if not BIN.is_file(): raise RuntimeError('cloudflared 未安装在本项目目录')
        DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        offset = LOG.stat().st_size if LOG.exists() else 0
        with LOG.open('ab') as log:
            process = subprocess.Popen([str(BIN), 'tunnel', '--no-autoupdate', '--protocol', 'http2',
                '--metrics', '127.0.0.1:0', '--url', 'http://127.0.0.1:8767'], cwd=DIR,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        start = identity(process.pid)
        if start is None:
            time.sleep(.2); start = identity(process.pid)
        if start is None: raise RuntimeError('公网通道未能启动，请查看本目录日志')
        STATE.write_text(json.dumps({'pid': process.pid, 'start': start, 'log_offset': offset}))
        current = state()
    url = None
    if current and LOG.exists():
        with LOG.open('rb') as log:
            log.seek(current.get('log_offset', 0)); content = log.read().decode(errors='replace')
        matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', content)
        if matches: url = matches[-1]
    print(json.dumps({'running': bool(current), 'url': url}))


if __name__ == '__main__': main()
