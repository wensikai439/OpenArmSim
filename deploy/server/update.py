"""Directory-local main follower with immutable releases and idle-only activation.

No system services, Python environments, model weights or user records are edited.
The container has no init/cron: start this supervisor once after container boot.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SHA = re.compile(r'[0-9a-f]{40}')
LOCKS = ('deploy/server/base-lock.txt', 'deploy/server/gpu-lock.txt', 'requirements.txt', 'pyproject.toml', 'uv.lock')
CHECKS = ('harness.execution_contract_checks', 'harness.execution_receipt_checks',
          'harness.provider_checks', 'harness.records_checks', 'deploy.server.checks', 'deploy.server.update_checks')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except FileNotFoundError: return default


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity(pid, script, root):
    try:
        proc = Path('/proc') / str(pid)
        args = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if str(script).encode() not in args or b'run' not in args or proc.joinpath('cwd').resolve() != root:
            return None
        return proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError): return None


def safe_error(error):
    text = str(error)
    text = re.sub(r'(?i)(?:sk-|gho_|ghp_|github_pat_|bearer\s+)[a-z0-9_.-]+', '[credential omitted]', text)
    return ' '.join(text.split())[:700]


class Updater:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.home = self.root / '.runtime/update'
        self.releases = self.home / 'releases'
        self.current = self.home / 'current'
        self.status_file = self.home / 'status.json'
        self.repo = self.home / 'repository.git'
        self.private = self.root / '.private/update'
        self.config = read_json(self.private / 'config.json', {})

    def report(self, phase, **values):
        previous = read_json(self.status_file, {})
        result = {**previous, 'phase':phase, 'checked_at':time.time(), **values}
        atomic_json(self.status_file, result)
        keys=('phase','active_commit','target_commit','blocked_commit','error')
        if any(previous.get(key)!=result.get(key) for key in keys):
            history=self.home/'events.jsonl'
            if history.exists() and history.stat().st_size>1024*1024:
                os.replace(history,self.home/'events-previous.jsonl')
            with history.open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
        return result

    def command(self, args, *, cwd=None, timeout=120, env=None):
        result = subprocess.run([str(x) for x in args], cwd=cwd or self.root, env=env,
                                capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(safe_error(result.stderr.decode(errors='replace') or result.stdout.decode(errors='replace')))
        return result.stdout

    def git(self, *args):
        identity = self.private / 'id_ed25519'
        hosts = self.private / 'known_hosts'
        if not identity.is_file() or identity.stat().st_mode & 0o077:
            raise RuntimeError('只读部署密钥不存在或权限不是 600')
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1',
                   GIT_SSH_COMMAND=shlex.join(['ssh','-i',str(identity),'-o','IdentitiesOnly=yes',
                       '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',
                       '-o','UserKnownHostsFile='+str(hosts)]))
        return self.command(['git','--git-dir='+str(self.repo),*args], env=env, timeout=120)

    def fetch(self):
        url = self.config.get('repository')
        if url != 'ssh://git@ssh.github.com:443/wensikai439/OpenArmSim.git':
            raise RuntimeError('更新源必须是已配置的 OpenArmSim 仓库')
        if not self.repo.exists():
            self.command(['git','init','--bare',str(self.repo)])
            self.git('remote','add','origin',url)
        # No force: rewritten branch history cannot silently replace a release.
        self.git('fetch','--no-tags','origin','refs/heads/main:refs/remotes/origin/main')
        revision = self.git('rev-parse','refs/remotes/origin/main').decode().strip()
        if not SHA.fullmatch(revision): raise RuntimeError('无效的 main 提交')
        return revision

    def validate_release(self, value):
        path = Path(value)
        if path == self.root: return path
        if path.parent != self.releases or not SHA.fullmatch(path.name) or path.is_symlink():
            raise RuntimeError('发布路径超出本项目管理目录')
        return path

    def active(self):
        if self.current.is_symlink(): return self.validate_release(self.current.resolve())
        if self.current.exists(): raise RuntimeError('current 必须是更新器管理的符号链接')
        return self.root

    def point(self, path):
        path = self.validate_release(path)
        temporary = self.home / 'current-next'
        if temporary.exists() or temporary.is_symlink(): temporary.unlink()
        temporary.symlink_to(path, target_is_directory=True)
        os.replace(temporary, self.current)

    def health(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open('http://127.0.0.1:18767/health', timeout=4) as response:
                return json.load(response)
        except urllib.error.URLError as error:
            if isinstance(error.reason, ConnectionRefusedError): return None
            raise RuntimeError('平台状态未知，未停止现有服务') from None

    @staticmethod
    def busy(health):
        return health is not None and (not all(k in health for k in ('task_active','agent_busy','learning_active'))
            or any(health[k] for k in ('task_active','agent_busy','learning_active')))

    def manage(self, release, action):
        env = dict(os.environ, OPENARM_MANAGER_DIRECT='1')
        if action == 'stop':
            status = json.loads(self.command([self.root / '.venv/bin/python','-B',release / 'deploy/server/manage.py','status'], env=env))
            if not status.get('running'): return
        self.command([self.root / '.venv/bin/python','-B',release / 'deploy/server/manage.py',action], timeout=50, env=env)
        if action == 'stop':
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                status = json.loads(self.command([self.root / '.venv/bin/python','-B',release / 'deploy/server/manage.py','status'], env=env))
                if not status.get('running'): return
                time.sleep(.25)
            raise RuntimeError('平台尚未停止，未切换代码')

    def fingerprint(self, release):
        script = 'import json; from tools.source_identity import fingerprint; from pathlib import Path; print(json.dumps(fingerprint(Path.cwd())))'
        return json.loads(self.command([self.root / '.venv/bin/python','-B','-c',script],cwd=release))[0]

    def wait_ready(self, release):
        expected = self.fingerprint(release)
        deadline = time.monotonic() + self.config.get('health_timeout', 180)
        while time.monotonic() < deadline:
            value = self.health()
            if value and value.get('state')=='ready' and value.get('source_current') is True and value.get('policy_ready') is True and value.get('curobo_ready') is True:
                if value.get('source_revision_sha256') != expected:
                    raise RuntimeError('运行源码指纹与发布目录不一致')
                try:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open('http://127.0.0.1:8767/version', timeout=3) as response:
                        version = json.load(response)
                    if release == self.root or version.get('source_commit') == release.name:
                        return value
                except (OSError, ValueError):
                    pass
            time.sleep(.5)
        raise RuntimeError('新版本启动验证超时')

    def resources(self):
        if shutil.disk_usage(self.home).free < 1024**3: raise RuntimeError('可用空间不足 1 GiB')
        output = self.command(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'])
        if int(output.decode().strip()) < 4096: raise RuntimeError('GPU 1 空闲显存不足 4 GiB，等待资源')

    def stage(self, revision):
        destination = self.releases / revision
        if destination.is_dir():
            marker=read_json(destination / 'DEPLOYMENT.json', {})
            if marker.get('source_commit') != revision or not marker.get('files'):
                raise RuntimeError('发布目录缺少本更新器的身份记录')
            for name,digest in marker['files'].items():
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts:raise RuntimeError('发布校验路径越界')
                target=destination.joinpath(*path.parts)
                if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('发布目录已被修改，保留当前服务：'+name)
            return destination
        self.releases.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='stage-', dir=self.releases) as folder:
            staging = Path(folder)
            body = self.git('archive','--format=tar',revision)
            hashes={}
            with tarfile.open(fileobj=io.BytesIO(body)) as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    if name.is_absolute() or '..' in name.parts or not name.parts:
                        raise RuntimeError('仓库包含越界路径')
                    if name.parts[0] in {'.git','.runtime','.private','runs','handoff','data'}: continue
                    if not member.isdir() and not member.isfile(): raise RuntimeError('发布包不接受链接或设备文件')
                    target = staging.joinpath(*name.parts)
                    if member.isdir(): target.mkdir(parents=True, exist_ok=True); continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
                    hashes[name.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
            for name in ('.venv','.runtime','.private','third_party','runs'):
                source = self.root / name
                if name == 'runs': source.mkdir(exist_ok=True)
                if source.exists(): (staging / name).symlink_to(source, target_is_directory=True)
            # Host-specific interpreter/source paths are deployment data.
            local_runtime = self.root / 'sim/learning.json'
            if local_runtime.is_file():
                (staging / 'sim').mkdir(exist_ok=True)
                target = staging / 'sim/learning.json'
                target.write_bytes(local_runtime.read_bytes())
                hashes['sim/learning.json'] = hashlib.sha256(target.read_bytes()).hexdigest()
            for checkpoint in (self.root / 'models').glob('*/checkpoint'):
                target = staging / checkpoint.relative_to(self.root)
                if not checkpoint.resolve().is_relative_to(self.root): raise RuntimeError('模型路径超出本项目')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(checkpoint, target_is_directory=True)
            atomic_json(staging / 'DEPLOYMENT.json', {'source_commit':revision,'created_at':time.time(),
                'state_root':str(self.root),'files':hashes,
                'scope':'code release; environment, credentials, weights and records are shared state'})
            os.replace(staging, destination)
        return destination

    def prune(self, keep):
        for path in self.releases.iterdir():
            if path.is_symlink() or not path.is_dir() or not SHA.fullmatch(path.name) or path in keep:continue
            marker=read_json(path/'DEPLOYMENT.json',{})
            if marker.get('source_commit')==path.name and marker.get('state_root')==str(self.root):
                shutil.rmtree(path)  # shared runtime/weights are symlinks; rmtree never follows them

    def preflight(self, release, previous):
        # Dependency installation is deliberately outside automatic source updates.
        for name in LOCKS:
            old, new = previous/name, release/name
            if previous == self.root and not old.exists(): continue
            if (old.read_bytes() if old.is_file() else None) != (new.read_bytes() if new.is_file() else None):
                raise RuntimeError('依赖清单已变化，保留当前服务，需先在个人环境验证：'+name)
        temporary = self.root / '.runtime/cache/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TMPDIR=str(temporary), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   XDG_CACHE_HOME=str(self.root / '.runtime/cache'), PYTHONDONTWRITEBYTECODE='1')
        self.command([self.root / '.venv/bin/python','-B','-m','unittest',*CHECKS],cwd=release,timeout=90,env=env)

    def recover(self, previous, candidate, revision, reason):
        self.manage(candidate,'stop')
        self.point(previous)
        self.manage(previous,'start')
        self.wait_ready(previous)
        return self.report('rolled_back', active_commit=previous.name if SHA.fullmatch(previous.name) else None,
                           blocked_commit=revision, error=safe_error(reason), previous_root=None, candidate=None)

    def once(self):
        self.home.mkdir(parents=True, exist_ok=True)
        with exclusive(self.home/'update.lock'):
            state = read_json(self.status_file, {})
            if state.get('previous_root') and state.get('candidate'):
                with exclusive(self.home/'admission.lock'):
                    previous=self.validate_release(state['previous_root'])
                    candidate=self.validate_release(state['candidate'])
                    return self.recover(previous,candidate,candidate.name,'上次发布中断，恢复上一版')
            revision = self.fetch()
            active = self.active()
            health = self.health()
            if self.busy(health): return self.report('waiting_for_idle', target_commit=revision, error=None)
            if state.get('blocked_commit')==revision: return self.report('blocked',target_commit=revision)
            if active != self.root and active.name == revision and health is not None:
                if not all(health.get(key) is True for key in ('source_current','policy_ready','curobo_ready')) or health.get('state')!='ready':
                    raise RuntimeError('当前版本未通过健康检查，保留现场等待检查')
                return self.report('up_to_date', active_commit=revision, target_commit=revision, error=None)
            self.resources()
            if active != self.root:
                self.git('merge-base','--is-ancestor',active.name,revision)
            candidate = self.stage(revision)
            try:self.preflight(candidate,active)
            except Exception as error:
                return self.report('blocked',target_commit=revision,blocked_commit=revision,error=safe_error(error))
            try:
                with exclusive(self.home/'admission.lock'):
                    # Public POSTs take the shared side of this same lock.
                    if self.busy(self.health()): return self.report('waiting_for_idle',target_commit=revision,error=None)
                    self.report('activating',target_commit=revision,previous_root=str(active),candidate=str(candidate),error=None)
                    try:
                        self.manage(active,'stop')
                        self.point(candidate)
                        self.manage(candidate,'start')
                        self.wait_ready(candidate)
                    except Exception as error:
                        return self.recover(active,candidate,revision,error)
                    result=self.report('up_to_date',active_commit=revision,target_commit=revision,
                        previous_commit=active.name if SHA.fullmatch(active.name) else None,
                        previous_root=None,candidate=None,blocked_commit=None,error=None)
                    self.prune({active,candidate})
                    return result
            except BlockingIOError:
                return self.report('waiting_for_requests',target_commit=revision,error=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['start','stop','status','once','run','retry'])
    args=parser.parse_args();os.umask(0o077)
    updater=Updater();updater.home.mkdir(parents=True,exist_ok=True)
    record_path=updater.home/'process.json';script=Path(__file__).resolve()
    record=read_json(record_path,{})
    running=bool(record.get('pid') and record.get('start') and process_identity(record['pid'],script,ROOT)==record['start'])
    if args.command=='status':
        print(json.dumps({'running':running,**read_json(updater.status_file,{})},ensure_ascii=False));return
    if args.command=='stop':
        if running:os.kill(record['pid'],signal.SIGTERM)
        print(json.dumps({'stop_requested':running}));return
    if args.command=='start':
        if not running:
            with (updater.home/'updater.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-B',str(script),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'starting':not running,'already_running':running}));return
    if args.command=='retry':
        with exclusive(updater.home/'update.lock'):
            state=read_json(updater.status_file,{})
            updater.report(state.get('phase','retry_requested'),blocked_commit=None)
        args.command='once'
    if args.command=='once':
        print(json.dumps(updater.once(),ensure_ascii=False));return
    with exclusive(updater.home/'daemon.lock'):
        atomic_json(record_path,{'pid':os.getpid(),'start':process_identity(os.getpid(),script,ROOT)})
        stopped=threading.Event()
        signal.signal(signal.SIGINT,lambda *_:stopped.set())
        signal.signal(signal.SIGTERM,lambda *_:stopped.set())
        while not stopped.is_set():
            try:updater.once()
            except BlockingIOError:pass
            except Exception as error:updater.report('error',error=safe_error(error))
            stopped.wait(max(30,int(updater.config.get('poll_seconds',60))))


if __name__=='__main__':main()
