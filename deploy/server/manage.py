"""Own-process service control. No system service or user environment is edited."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.runtime/server.json'
SCRIPT = Path(__file__).resolve()


def environment():
    env = dict(os.environ)
    runtime = ROOT / '.runtime'
    state_root = runtime.resolve().parent
    directories = {
        'XDG_CACHE_HOME': runtime / 'cache', 'XDG_CONFIG_HOME': runtime / 'config',
        'XDG_DATA_HOME': runtime / 'data', 'TMPDIR': runtime / 'cache/tmp',
        'HF_HOME': runtime / 'cache/huggingface', 'TORCH_HOME': runtime / 'cache/torch',
        'TORCH_EXTENSIONS_DIR': runtime / 'cache/torch-extensions',
        'CUDA_CACHE_PATH': runtime / 'cache/cuda', 'TRITON_CACHE_DIR': runtime / 'cache/triton',
        'WARP_CACHE_PATH': runtime / 'cache/warp', 'NUMBA_CACHE_DIR': runtime / 'cache/numba',
        'TORCHINDUCTOR_CACHE_DIR': runtime / 'cache/inductor',
        'MPLCONFIGDIR': runtime / 'cache/matplotlib', 'PYTHONUSERBASE': runtime / 'python-user',
        '__GL_SHADER_DISK_CACHE_PATH': runtime / 'cache/opengl',
    }
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OPENARM_PORT='18767',
               OPENARM_SERVER_DATA=str(state_root),
               OPENARM_POLICY_CONFIG=str(state_root / '.private/policy.json' if (state_root / '.private/policy.json').is_file() else ROOT / 'harness/policy.json'),
               OPENARM_SCENE_OBJECTS='[]',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', LP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='1', MUJOCO_GL='osmesa',
               PYTHONPATH=str(ROOT / 'third_party/lerobot/src'),
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    env['OPENARM_GPU_START'] = '1' if (ROOT / '.runtime/gpu-ready').is_file() else '0'
    library = str(runtime / 'system/usr/lib/x86_64-linux-gnu')
    env['LD_LIBRARY_PATH'] = library + ':' + env.get('LD_LIBRARY_PATH', '')
    if (runtime / 'system/egl.json').is_file():
        env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(runtime / 'system/egl.json')
        env['MUJOCO_EGL_DEVICE_ID'] = '1'
    config = ROOT / '.private/runtime.json'
    if config.exists():
        values = json.loads(config.read_text())
        for key in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_GL'):
            if key in values: env[key] = str(values[key])
    return env


def identity(pid):
    base = Path('/proc') / str(pid)
    try:
        command = base.joinpath('cmdline').read_bytes().split(b'\0')
        if str(SCRIPT).encode() not in command or b'supervise' not in command:
            return None
        if base.joinpath('cwd').resolve() != ROOT:
            return None
        return base.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def state():
    try:
        record = json.loads(STATE.read_text())
        return record if identity(record['pid']) == record['start'] else None
    except (OSError, ValueError, KeyError):
        return None


def health():
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            'http://127.0.0.1:18767/health', timeout=3) as response:
        return json.load(response)


def supervise():
    os.umask(0o077)
    os.nice(10)
    env = environment()
    STATE.write_text(json.dumps({'pid': os.getpid(), 'start': identity(os.getpid())}))
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    children = []
    logs = []
    def launch(name, arguments, *, python=None):
        stream = (ROOT / '.runtime/logs' / (name + '.log')).open('ab')
        logs.append(stream)
        process = subprocess.Popen([str(python or ROOT / '.venv/bin/python'), '-B', *arguments], cwd=ROOT,
            env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        children.append(process)
        return process
    try:
        if env['OPENARM_GPU_START'] == '1':
            launch('act', ['-m', 'sim.act_server'], python=ROOT / '.runtime/curobo/bin/python')
        backend = launch('simulation', [str(ROOT / 'deploy/server/workbench.py')])
        deadline = time.monotonic() + 180
        while not stopped.is_set():
            if backend.poll() is not None:
                raise RuntimeError('仿真进程未能启动，查看 .runtime/logs/simulation.log')
            try:
                if health().get('state') == 'ready': break
            except (OSError, ValueError): pass
            if time.monotonic() >= deadline: raise RuntimeError('仿真启动超时')
            stopped.wait(.5)
        if not stopped.is_set():
            launch('web', [str(ROOT / 'deploy/server/gateway.py'), '--root', str(ROOT)])
        while not stopped.wait(1):
            if any(child.poll() is not None for child in children):
                raise RuntimeError('平台子进程退出，请查看目录内日志')
    finally:
        for child in reversed(children):
            if child.poll() is None: child.send_signal(signal.SIGINT)
        for child in reversed(children):
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # The Popen object still owns this exact live child.
                if child.poll() is None: child.terminate()
        for stream in logs: stream.close()
        if STATE.exists(): STATE.unlink()


def main():
    current = ROOT / '.runtime/update/current'
    if not os.environ.get('OPENARM_MANAGER_DIRECT') and current.is_symlink():
        active = current.resolve()
        releases = (ROOT / '.runtime/update/releases').resolve()
        if active != ROOT and active.parent == releases and (active / 'deploy/server/manage.py').is_file():
            os.execv(sys.executable, [sys.executable, '-B', str(active / 'deploy/server/manage.py'), *sys.argv[1:]])
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['start', 'stop', 'status', 'supervise'])
    args = parser.parse_args()
    if args.command == 'supervise': return supervise()
    record = state()
    if args.command == 'status':
        result = {'running': bool(record), 'root': str(ROOT)}
        if record:
            try:
                data = health()
                result.update({key: data.get(key) for key in ('state', 'policy_ready', 'curobo_ready', 'source_current')})
            except (OSError, ValueError): result['state'] = 'starting'
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == 'stop':
        if not record:
            print('未找到本目录拥有的平台进程，未发送停止信号。')
            return 0
        if identity(record['pid']) != record['start']: raise RuntimeError('进程身份已变化，未停止')
        os.kill(record['pid'], signal.SIGINT)
        print('已通知本目录的平台进程停止。')
        return 0
    if record:
        print('本目录的平台已启动。')
        return 0
    for port in (8767, 18767, 8003, 8005):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'端口 {port} 已占用，未修改或结束已有服务')
    if not (ROOT / '.private/auth.json').exists(): raise RuntimeError('请先配置登录密码')
    (ROOT / '.runtime/logs').mkdir(parents=True, exist_ok=True)
    with (ROOT / '.runtime/logs/platform.log').open('ab') as stream:
        subprocess.Popen([sys.executable, '-B', str(SCRIPT), 'supervise'], cwd=ROOT,
            env=environment(), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    print('平台正在启动；状态：.venv/bin/python -B deploy/server/manage.py status')


if __name__ == '__main__': main()
