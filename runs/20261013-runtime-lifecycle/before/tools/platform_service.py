#!/usr/bin/env python3
"""On-demand user service; independent of Codex command/session lifetime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UNIT = 'openarm-sim.service'
SOURCE = ROOT / 'tools' / UNIT
URL = 'http://127.0.0.1:8765'


def systemctl(*args, check=True, capture=False):
    return subprocess.run(['systemctl', '--user', *args], check=check,
                          capture_output=capture, text=True)


def install():
    directory = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'systemd' / 'user'
    directory.mkdir(parents=True, exist_ok=True)
    for unit in (UNIT,'openarm-act.service','openarm-curobo.service'):
        target=directory/unit;source=ROOT/'tools'/unit
        if target.is_symlink():
            if target.resolve()!=source.resolve():raise RuntimeError(f'{target} 已属于其他服务，未修改。')
        elif target.exists():raise RuntimeError(f'{target} 已存在，未覆盖。')
        else:target.symlink_to(source)
    systemctl('daemon-reload')


def status_json():
    with urllib.request.urlopen(URL + '/api/status', timeout=2) as response:
        return json.load(response)


def ready():
    try:
        return status_json().get('state') == 'ready'
    except (OSError, urllib.error.URLError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description='按需启动、停止或查看 OpenArm 平台服务')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'logs'])
    args = parser.parse_args()
    if args.command == 'logs':
        return subprocess.call(['journalctl', '--user', '-u', UNIT, '-n', '50', '--no-pager'])
    if args.command == 'status':
        active = systemctl('is-active', UNIT, check=False, capture=True)
        print(f'服务：{active.stdout.strip() or "未安装"}；页面：{URL}/；就绪：{ready()}')
        return 0 if active.returncode == 0 and ready() else 1
    if args.command == 'start':
        overrides = [name for name in ('OPENARM_POLICY_CONFIG', 'OPENARM_LLM_BASE', 'OPENARM_LLM_MODEL',
                                       'OPENARM_LLM_KEY', 'OPENARM_LLM_VISION', 'CUDA_VISIBLE_DEVICES')
                     if name in os.environ]
        if overrides or os.environ.get('MUJOCO_GL', 'egl') != 'egl':
            raise RuntimeError('持久服务不继承终端的模型/渲染环境设置。此类启动请使用 ./run serve；API 可在页面保存。')
    install()
    if args.command == 'stop':
        systemctl('stop', UNIT)
        print('平台已停止，睡眠保护已释放。')
        return 0
    if os.environ.get('OPENARM_PORT', '8765') != '8765':
        raise RuntimeError('持久服务固定使用 8765；自定义端口请使用 OPENARM_PORT=端口 ./run serve。')
    active = systemctl('is-active', UNIT, check=False, capture=True).returncode == 0
    if not active and ready():
        raise RuntimeError('已有前台服务占用 8765；请先结束该服务，再使用 ./run 启动持久服务。')
    systemctl('start', UNIT)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if ready():
            print(f'OpenArm Lab: {URL}/')
            print('持久服务已就绪；运行期间保持系统清醒。停止：./run stop；状态：./run status')
            return 0
        state = systemctl('is-active', UNIT, check=False, capture=True).stdout.strip()
        if state not in {'active', 'activating'}:
            raise RuntimeError('服务未启动成功；运行 ./run logs 查看原因。')
        time.sleep(.3)
    raise RuntimeError('服务仍在启动，未强制重启；使用 ./run status 或 ./run logs 查看状态。')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
