"""Thin management API over official Core services; never fabricates execution state."""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError
from starlette.responses import JSONResponse
from starlette.routing import Route

from PhyAgentOS.config.loader import get_config_path, load_config
from PhyAgentOS.config.schema import Config
from PhyAgentOS.cron.types import CronSchedule
from PhyAgentOS.cron.service import _compute_next_run, _validate_schedule_for_add
from PhyAgentOS.embodiment_registry import EmbodimentRegistry
from PhyAgentOS.skill_runtime.registry import RegistryClient
from PhyAgentOS.utils.helpers import load_environment_doc

ROOT = Path(__file__).resolve().parents[1]
# These profiles have been inspected and executed as simulation in this project.
SIMULATION_PROFILES = {'piper-dual-sim': {'osc', 'joint'}, 'rpent-piper-sim': {'rpent', 'joint'}, 'move-arm-by-ee': {'mujoco'}}
PRIVATE_KEYS = {'api_key', 'token', 'secret', 'app_secret', 'client_secret', 'bridge_token',
                'access_token', 'bot_token', 'app_token', 'claw_token', 'encrypt_key',
                'verification_token', 'imap_password', 'smtp_password', 'password'}
PRIVATE_MAPS = {'headers', 'extra_headers', 'env', 'environment'}
CHANNEL_REQUIRED = {
    'whatsapp': ['bridge_url'], 'telegram': ['token'], 'discord': ['token'],
    'feishu': ['app_id', 'app_secret'], 'dingtalk': ['client_id', 'client_secret'],
    'email': ['consent_granted', 'imap_host', 'imap_username', 'imap_password',
              'smtp_host', 'smtp_username', 'smtp_password', 'from_address'],
    'slack': ['bot_token', 'app_token'], 'qq': ['app_id', 'secret'],
    'matrix': ['homeserver', 'access_token', 'user_id'], 'wecom': ['bot_id', 'secret'],
    'mochat': ['claw_token', 'agent_user_id'],
}


class BusyError(ValueError):
    pass


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def redact(value):
    if not isinstance(value, dict):
        return [redact(v) for v in value] if isinstance(value, list) else value
    result = {}
    for key, item in value.items():
        if key in PRIVATE_KEYS:
            result[key] = ''
            result[key + '_configured'] = bool(item)
        elif key in PRIVATE_MAPS:
            result[key] = {k: '' for k in item} if isinstance(item, dict) else {}
            result[key + '_configured'] = bool(item)
        else:
            result[key] = redact(item)
    return result


def merge(base, patch):
    """Partial edits preserve omitted values and blank secret fields."""
    result = dict(base)
    for key, value in patch.items():
        if key.endswith('_configured'):
            continue
        if key in PRIVATE_KEYS and value in ('', None):
            continue
        if key in PRIVATE_MAPS and isinstance(value, dict):
            if not value:
                continue
            old = result.get(key) or {}
            result[key] = {**old, **{k: (old.get(k, '') if v in ('', None) else v) for k, v in value.items()}}
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def safe_error(exc, config):
    if isinstance(exc, ValidationError):
        return '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg']
                         for e in exc.errors(include_input=False))
    result = str(exc)
    def secrets(value, private=False):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from secrets(item, private or key in PRIVATE_KEYS or key in PRIVATE_MAPS)
        elif private and isinstance(value, str) and len(value) > 3:
            yield value
    for value in secrets(config.model_dump()):
        result = result.replace(value, '[redacted]')
    return re.sub(r'(?i)(Bearer\s+)[^\s\"\']+', r'\1[redacted]', result)


def reply(value, status=200):
    return JSONResponse(plain(value), status_code=status, headers={'Cache-Control': 'no-store'})


def assert_idle(console):
    controller = getattr(console, 'real_commands', None)
    if controller and controller.configuration_busy():
        raise BusyError('真机任务仍在执行或结果待核对，不能重载配置。')
    reflection = getattr(console, 'reflection_busy', False)
    reflection = reflection() if callable(reflection) else reflection
    if reflection:
        raise BusyError('经验反思正在执行，请等待完成后再修改配置或切换 Skill。')
    subagents = getattr(console.agent, 'subagents', None)
    background = getattr(console, 'background_busy', None)
    background = background() if callable(background) else bool(subagents and subagents.get_running_count())
    if background:
        raise BusyError('后台 Agent 仍在执行，请等待完成后再修改配置或切换 Skill。')
    if (console.job or console.control_busy or getattr(console, 'management_run_busy', False)
            or console.coordinator.store.active()):
        raise BusyError('已有活动 Agent、Task 或管理操作，请等待终态。')
    for manifest in console.manager.catalog.list():
        state = console.manager.state_store.load(manifest.name)
        if state and (state.active_invocations or state.active_sessions or state.active_task_bindings):
            raise BusyError('Runtime 仍持有 invocation、Session 或 Task 绑定，请先等待原生终态。')


def selected(console):
    if getattr(console, 'agent_mode', 'simulation') == 'real':
        return {'name': 'piper-real-runtime', 'profile': 'staged'}
    return {'name': getattr(console, 'skill', 'piper-dual-sim'),
            'profile': getattr(console, 'profile', 'joint')}


async def callback(console, name, *args):
    func = getattr(console, name, None)
    if func is None:
        raise RuntimeError(f'Console lifecycle callback unavailable: {name}')
    result = func(*args)
    return await result if inspect.isawaitable(result) else result


def atomic_text(path, content, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


async def save_configuration(console, data, *, credentials_only=False):
    config = Config.model_validate(data)
    # The knowledge editing surface must never move the application into another project.
    if config.workspace_path.resolve() != console.workspace.resolve():
        raise ValueError('工作区位置由项目部署管理，不能从控制台切换。')
    destination = get_config_path()
    previous = destination.read_bytes() if destination.exists() else None
    atomic_text(destination, json.dumps(config.model_dump(by_alias=True), ensure_ascii=False, indent=2) + '\n')
    try:
        await callback(console, 'reload_credentials' if credentials_only else 'reload_configuration', config)
    except (Exception, SystemExit) as exc:
        if previous is not None:
            atomic_text(destination, previous.decode('utf-8'))
        if isinstance(exc, SystemExit):
            raise ValueError('Provider or integration initialization failed; previous configuration restored') from None
        raise
    return config


async def cli(console, arguments):
    proc = await asyncio.create_subprocess_exec('paos', 'skill', *arguments, cwd=ROOT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    output, _ = await proc.communicate()
    output = safe_error(RuntimeError(output.decode(errors='replace')), console.config)
    if proc.returncode:
        raise RuntimeError(f'paos skill exited {proc.returncode}: {output[-12000:]}')
    return output[-20000:]


async def skills_get(console, params):
    def read():
        items = []
        for manifest in console.manager.catalog.list():
            state = console.manager.state_store.load(manifest.name)
            items.append({'name': manifest.name, 'version': manifest.version,
                'description': manifest.description, 'profiles': sorted(manifest.profiles),
                'allowed_profiles': sorted(set(manifest.profiles) & SIMULATION_PROFILES.get(manifest.name, set())),
                'runtime': plain(state), 'manifest': redact(plain(manifest))})
        value = {'selected': selected(console), 'skills': items, 'errors': console.manager.catalog.errors()}
        if params.get('name'):
            name = params['name']
            value['inspect'] = redact(plain(console.manager.catalog.get(name)))
            value['logs'] = safe_error(RuntimeError(console.manager.read_logs(name, lines=100)), console.config)
        if 'query' in params:
            with RegistryClient() as registry:
                value['search'] = registry.search_skills(params['query'])
        return value
    return await asyncio.to_thread(read)


async def skills_post(console, body):
    action, name = body.get('action'), str(body.get('name', '')).strip()
    if not name or name.startswith('-'):
        raise ValueError('Skill name is required')
    if action == 'switch':
        if getattr(console, 'agent_mode', 'simulation') == 'real':
            raise ValueError('真机命令页固定使用 piper-real-runtime；仿真资产请在仿真工作台切换。')
        profile = str(body.get('profile', ''))
        manifest = console.manager.catalog.get(name)
        if profile not in SIMULATION_PROFILES.get(name, set()) or profile not in manifest.profiles:
            raise ValueError('只允许已登记的仿真 Skill/profile。')
        output = await cli(console, ['switch', name, '--profile', profile])
        await callback(console, 'select_skill', name, profile)
    elif action in {'install', 'update', 'remove'}:
        if action == 'remove' and name in {selected(console)['name'], getattr(console, 'skill', None)}:
            raise ValueError('请先切换到其他 Skill，再移除当前选择。')
        args = [action, name]
        if action in {'install', 'update'}:
            args.append('--yes')
            if body.get('version'):
                args.extend(['--version', str(body['version'])])
        if body.get('local'):
            if action != 'install':
                raise ValueError('local is supported only for install')
            path = Path(name).expanduser().resolve()
            if not path.is_relative_to((ROOT / 'bundles').resolve()) or not path.is_file():
                raise ValueError('本地制品必须位于本项目 bundles 目录。')
            args[1] = str(path)
            args.append('--local')
        output = await cli(console, args)
    else:
        raise ValueError('Unknown Skill action')
    return {'ok': True, 'action': action, 'output': output}


def settings_get(console):
    config = console.config
    from console.providers import active_model
    return {'active_model': active_model(config), 'defaults': plain(config.agents.defaults), 'providers': redact(plain(config.providers)),
            'verification': plain(config.agents.verification), 'evolution': plain(config.agents.evolution),
            'modes': plain(config.agents.modes),
            'agent_mode': getattr(console, 'agent_mode', 'simulation'),
            'active_provider': config.get_provider_name()}


async def settings_post(console, body):
    if body.get('action') == 'switch_model':
        model = str(body.get('model', '')).strip()
        provider = body.get('provider')
        if not model or len(model) > 200 or any(c.isspace() for c in model):
            raise ValueError('请输入有效的模型 ID')
        data = console.config.model_dump()
        if provider not in data['providers']:
            raise ValueError('请选择已登记的 Provider')
        entry = data['providers'][provider]
        if not entry.get('enabled', True) or not (entry.get('api_key') or entry.get('api_base')):
            raise ValueError('该 Provider 尚未配置，请先在高级设置中配置服务')
        data['agents']['defaults'].update(model=model, provider=provider)
        if data['agents']['modes']['enabled']:
            data['agents']['modes']['default_mode'] = 'main'
            data['agents']['modes']['models']['main']['model'] = model
        await save_configuration(console, data)
        return {'ok': True, **settings_get(console)}
    if body.get('action') == 'credentials':
        if set(body) - {'action', 'api_key'} or not isinstance(body.get('api_key', ''), str):
            raise ValueError('此入口仅允许更新当前模型密钥')
        data = console.config.model_dump()
        provider = console.config.get_provider_name()
        if not provider:
            raise ValueError('当前模型没有匹配的 Provider')
        data['providers'][provider] = merge(data['providers'][provider], {'api_key':body.get('api_key', '')})
        await save_configuration(console, data, credentials_only=True)
        return {'ok': True, **settings_get(console)}
    if body.get('action') != 'save':
        raise ValueError('Unknown settings action')
    data = load_config().model_dump()
    for key in ('defaults', 'verification', 'evolution', 'modes'):
        if key in body:
            data['agents'][key] = merge(data['agents'][key], body[key])
    if 'providers' in body:
        data['providers'] = merge(data['providers'], body['providers'])
    if (body.get('defaults', {}).get('model') and
            body['defaults']['model'] != console.config.agents.defaults.model and
            data['agents']['modes']['enabled']):
        data['agents']['modes']['default_mode'] = 'main'
        data['agents']['modes']['models']['main']['model'] = data['agents']['defaults']['model']
    mode = body.get('agent_mode', getattr(console, 'agent_mode', 'simulation'))
    if mode not in {'simulation', 'general', 'real'}:
        raise ValueError('Unknown agent mode')
    await save_configuration(console, data)
    if mode != getattr(console, 'agent_mode', 'simulation'):
        await callback(console, 'set_agent_mode', mode)
    return {'ok': True, **settings_get(console)}


def knowledge_path(console, name):
    relative = Path(name)
    root = console.workspace.resolve()
    path = (root / relative).resolve()
    if (relative.is_absolute() or path.suffix.lower() != '.md' or not path.is_relative_to(root)
            or not relative.parts or any(x.startswith('.') or x in {'artifacts', 'skills', 'sessions'} for x in relative.parts)):
        raise ValueError('只允许当前工作区中的知识 Markdown 文件。')
    return path


def workspace_get(console, params):
    root = console.workspace.resolve()
    files = []
    for path in sorted(root.rglob('*.md')):
        try:
            target = knowledge_path(console, str(path.relative_to(root)))
        except ValueError:
            continue
        if target.is_file():
            files.append({'path': str(path.relative_to(root)), 'bytes': target.stat().st_size})
    registry = EmbodimentRegistry(console.config)
    result = {'files': files, 'embodiments': {'mode': registry.mode,
        'instances': plain(registry.instances()), 'workspace': str(registry.resolve_agent_workspace())},
        'scene_graph': load_environment_doc(registry.resolve_environment_path()),
        'embodiment_config': plain(console.config.embodiments),
        'knowledge_workspace': str(root)}
    if params.get('path'):
        path = knowledge_path(console, params['path'])
        if path.exists() and path.stat().st_size > 1024 * 1024:
            raise ValueError('Markdown exceeds 1 MiB')
        result.update(path=params['path'], content=path.read_text() if path.exists() else '')
    return result


def workspace_post(console, body):
    if body.get('action') != 'save':
        raise ValueError('Unknown workspace action')
    path = knowledge_path(console, str(body.get('path', '')))
    content = str(body.get('content', ''))
    if len(content.encode()) > 1024 * 1024:
        raise ValueError('Markdown exceeds 1 MiB')
    atomic_text(path, content)
    return {'ok': True, 'path': str(path.relative_to(console.workspace.resolve()))}


def scene_post(console, body):
    environment = body.get('environment')
    if not isinstance(environment, dict):
        raise ValueError('Environment must be a JSON object')
    encoded = json.dumps(environment, ensure_ascii=False, indent=2, allow_nan=False)
    if len(encoded.encode()) > 1024 * 1024:
        raise ValueError('Environment exceeds 1 MiB')
    path = knowledge_path(console, 'ENVIRONMENT.md')
    text = path.read_text() if path.exists() else '# Environment knowledge\n\n'
    # Match the same first fenced JSON/YAML document that the official reader consumes.
    from PhyAgentOS.utils.helpers import _YAML_BLOCK_RE
    match = _YAML_BLOCK_RE.search(text)
    block = '```json\n' + encoded + '\n```'
    content = text[:match.start()] + block + text[match.end():] if match else text.rstrip() + '\n\n' + block + '\n'
    atomic_text(path, content)
    actual = load_environment_doc(path)
    if actual != environment:
        raise RuntimeError('Official environment reader did not reproduce the saved document')
    return {'ok': True, 'path': 'ENVIRONMENT.md', 'scene_graph': actual}


async def workspace_update(console, body):
    action = body.get('action')
    if action == 'scene':
        return await asyncio.to_thread(scene_post, console, body)
    if action != 'embodiments':
        return await asyncio.to_thread(workspace_post, console, body)
    config_patch = body.get('config')
    if not isinstance(config_patch, dict):
        raise ValueError('Embodiment config must be a JSON object')
    data = load_config().model_dump()
    current = merge(data['embodiments'], config_patch)
    root = console.workspace.resolve()
    if current.get('mode') == 'fleet':
        shared = Path(current.get('shared_workspace') or str(root)).expanduser()
        shared = (root / shared).resolve() if not shared.is_absolute() else shared.resolve()
        if shared != root:
            raise ValueError('Fleet shared_workspace must remain ' + str(root))
        current['shared_workspace'] = str(root)
    seen = set()
    for item in current.get('instances', []):
        robot_id = item.get('robot_id', '')
        if not robot_id or robot_id in seen or not re.fullmatch(r'[A-Za-z0-9_-]+', robot_id):
            raise ValueError('Robot IDs must be unique letters, digits, underscores or hyphens')
        seen.add(robot_id)
        directory = Path(item.get('workspace', '')).expanduser()
        directory = (root / directory).resolve() if not directory.is_absolute() else directory.resolve()
        relative = directory.relative_to(root) if directory.is_relative_to(root) else None
        if (not relative or str(relative) == '.' or any(x.startswith('.') or x in {'artifacts', 'skills', 'sessions'} for x in relative.parts)):
            raise ValueError('Robot knowledge workspace must be a child knowledge directory of ' + str(root))
        item['workspace'] = str(directory)
        if item.get('shared_environment'):
            shared = Path(item['shared_environment']).expanduser()
            if shared.is_absolute():
                if not shared.resolve().is_relative_to(root):
                    raise ValueError('Shared environment must be in the current knowledge workspace')
                shared = shared.resolve().relative_to(root)
            item['shared_environment'] = str(knowledge_path(console, str(shared)))
        if item.get('profile_name') and (Path(item['profile_name']).name != item['profile_name'] or item['profile_name'].startswith('.')):
            raise ValueError('profile_name must be a knowledge filename, not a path')
    data['embodiments'] = current
    config = await save_configuration(console, data)
    await asyncio.to_thread(EmbodimentRegistry(config).sync_layout)
    return {'ok': True, **await asyncio.to_thread(workspace_get, console, {})}


def schedules_get(console):
    cron = console.cron
    heartbeat = console.config.gateway.heartbeat
    path = console.workspace / 'HEARTBEAT.md'
    return {'jobs': plain(cron.list_jobs(include_disabled=True)), 'status': cron.status(),
        'heartbeat': {**plain(heartbeat), 'running': bool(getattr(getattr(console, 'heartbeat', None), '_running', False))},
        'heartbeat_content': path.read_text() if path.exists() else ''}


def valid_schedule(value):
    schedule = CronSchedule(**value)
    if schedule.kind not in {'at', 'every', 'cron'}:
        raise ValueError('Unknown schedule kind')
    _validate_schedule_for_add(schedule)
    if _compute_next_run(schedule, int(time.time() * 1000)) is None:
        raise ValueError('Schedule has no future execution time')
    return schedule


async def schedules_post(console, body):
    action, cron = body.get('action'), console.cron
    if action == 'heartbeat_run':
        heartbeat = getattr(console, 'heartbeat', None)
        if heartbeat is None:
            raise RuntimeError('Heartbeat service is unavailable')
        provider = heartbeat.provider
        original_chat = provider.chat
        calls = 0
        decision_calls = 0
        decision_error = None
        response = None
        error = None
        async def counted_chat(*args, **kwargs):
            nonlocal calls, decision_calls, decision_error
            calls += 1
            messages = kwargs.get('messages') or (args[0] if args else [])
            is_decision = bool(messages and 'You are a heartbeat agent.' in str(messages[0].get('content', '')))
            if is_decision:
                decision_calls += 1
            try:
                result = await original_chat(*args, **kwargs)
            except Exception as exc:
                if is_decision:
                    decision_error = str(exc)
                raise
            if is_decision:
                decision_error = (result.content or 'Heartbeat model decision failed') if result.finish_reason == 'error' else None
            return result
        provider.chat = counted_chat
        try:
            response = await heartbeat.trigger_now()
            if decision_error:
                raise RuntimeError(decision_error)
        except Exception as exc:
            error = safe_error(exc, console.config)
            raise
        finally:
            provider.chat = original_chat
            atomic_text(ROOT / 'runs/feature-completion-20261002/management/heartbeat-last.json',
                json.dumps({'at_unix': time.time(), 'provider_calls': calls, 'decision_calls': decision_calls,
                    'execution_calls': calls - decision_calls, 'executed': response is not None,
                    'error': error}, ensure_ascii=False, indent=2) + '\n')
        return {'ok': True, **schedules_get(console), 'heartbeat_result': {
            'executed': response is not None, 'decision_calls': decision_calls,
            'provider_calls': calls, 'execution_calls': calls - decision_calls, 'response': response}}
    if action == 'heartbeat':
        data = load_config().model_dump()
        interval = int(body.get('interval_s', data['gateway']['heartbeat']['interval_s']))
        if interval < 60:
            raise ValueError('Heartbeat interval must be at least 60 seconds')
        data['gateway']['heartbeat'] = {'enabled': bool(body.get('enabled', False)), 'interval_s': interval}
        await save_configuration(console, data)
        if 'content' in body:
            workspace_post(console, {'action': 'save', 'path': 'HEARTBEAT.md', 'content': body['content']})
    elif action == 'add':
        if not str(body.get('name', '')).strip() or not str(body.get('message', '')).strip():
            raise ValueError('Name and message are required')
        job = cron.add_job(name=body['name'], schedule=valid_schedule(body['schedule']),
            message=body['message'], deliver=False, channel='console',
            delete_after_run=bool(body.get('delete_after_run', False)))
        if not body.get('enabled', True):
            cron.enable_job(job.id, False)
    elif action in {'remove', 'enable', 'run', 'update'}:
        job_id = str(body.get('id', ''))
        jobs = {j.id: j for j in cron.list_jobs(include_disabled=True)}
        if job_id not in jobs:
            raise ValueError('Cron job not found')
        if action == 'remove':
            cron.remove_job(job_id)
        elif action == 'enable':
            cron.enable_job(job_id, bool(body.get('enabled', True)))
        elif action == 'run':
            if not cron.on_job:
                raise RuntimeError('Cron execution callback is unavailable')
            await cron.run_job(job_id, force=True)
            if jobs[job_id].state.last_status == 'error':
                raise RuntimeError('Cron run failed: ' + str(jobs[job_id].state.last_error))
        else:
            job = jobs[job_id]
            schedule = valid_schedule(body['schedule']) if 'schedule' in body else job.schedule
            name, message = str(body.get('name', job.name)).strip(), str(body.get('message', job.payload.message)).strip()
            if not name or not message:
                raise ValueError('Name and message are required')
            job.name, job.payload.message, job.schedule = name, message, schedule
            job.payload.deliver, job.payload.channel, job.payload.to = False, 'console', None
            # Official service has no public update method; persist the same CronJob and rearm via enable_job.
            cron.enable_job(job_id, bool(body.get('enabled', job.enabled)))
    else:
        raise ValueError('Unknown schedule action')
    return {'ok': True, **schedules_get(console)}


def channel_missing(name, config):
    missing = [key for key in CHANNEL_REQUIRED.get(name, []) if not config.get(key)]
    if not config.get('allow_from'):
        missing.append('allow_from')
    return missing


def integrations_get(console):
    configured = plain(console.config.channels)
    manager = getattr(console, 'channels', None)
    statuses = manager.get_status() if manager and hasattr(manager, 'get_status') else {}
    return {'channels': {name: {'config': redact(value), 'missing': channel_missing(name, value),
        'runtime_enabled': bool(statuses.get(name, {}).get('enabled')),
        'running': bool(statuses.get(name, {}).get('running'))}
        for name, value in configured.items() if isinstance(value, dict)},
        'mcp_servers': {name: redact(plain(value)) for name, value in console.config.tools.mcp_servers.items()},
        'connected_tools': [n for n in console.agent.tools.tool_names if n.startswith('mcp_')],
        'note': '渠道只有显式启用并填齐凭证及 allow_from 后才连接；保存配置本身不发送外部消息。'}


async def integrations_post(console, body):
    if body.get('action') != 'save':
        raise ValueError('Unknown integration action')
    data = load_config().model_dump()
    for name, value in body.get('channels', {}).items():
        if name not in CHANNEL_REQUIRED or not isinstance(value, dict):
            raise ValueError('Unknown channel')
        updated = merge(data['channels'][name], value)
        missing = channel_missing(name, updated)
        if updated['enabled'] and missing:
            raise ValueError(f'{name}: missing ' + ', '.join(missing))
        data['channels'][name] = updated
    if 'mcp_servers' in body:
        old = data['tools']['mcp_servers']
        data['tools']['mcp_servers'] = {name: merge(old.get(name, {}), value)
                                     for name, value in body['mcp_servers'].items()}
        for name, value in data['tools']['mcp_servers'].items():
            if not value.get('command') and not value.get('url'):
                raise ValueError(f'MCP {name}: command or url is required')
    await save_configuration(console, data)
    return {'ok': True, **integrations_get(console)}


def experience_get(console, params):
    skill = params.get('skill', 'current')
    if skill == 'current':
        skill = selected(console)['name']
    path = console.workspace / '.paos/evolution/experience.sqlite3'
    result = {'skill': skill, 'skills': [], 'active_lessons': 0, 'candidates': [],
              'lessons': [], 'jobs': [], 'episodes': [], 'clusters': [], 'failure_observations': [], 'events': []}
    if not path.exists():
        return result
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        def read(table):
            rows = []
            for item in db.execute(f'SELECT * FROM {table} ORDER BY rowid DESC LIMIT 300'):
                value = dict(item)
                if 'record_json' in value:
                    value = {**json.loads(value.pop('record_json')), **value}
                rows.append(value)
            return rows
        episodes = read('episodes')
        bindings = {r['root_task_id']: json.loads(r['binding_json']) for r in db.execute('SELECT * FROM task_bindings')}
        skill_by_task = {key: (value.get('forge_skill_binding') or {}).get('skill_name') for key, value in bindings.items()}
        for episode in episodes:
            episode['skill_name'] = skill_by_task.get(episode['root_task_id']) or next(
                (a.get('skill_name') for a in episode.get('skill_activations', []) if a.get('role') == 'primary'), None)
        result['skills'] = sorted(set(filter(None, skill_by_task.values())))
        result['episodes'] = [e for e in episodes if skill == 'all' or e['skill_name'] == skill]
        for field, table in [('candidates', 'skill_candidates'), ('lessons', 'scoped_lessons'),
                             ('clusters', 'lesson_clusters'), ('failure_observations', 'failure_observations')]:
            result[field] = [r for r in read(table) if skill == 'all' or r.get('skill_name') == skill]
        result['jobs'] = [{**r, 'skill_name': skill_by_task.get(r['root_task_id'])} for r in read('evolution_jobs')
                          if skill == 'all' or skill_by_task.get(r['root_task_id']) == skill]
        subjects = {r.get('root_task_id') for r in result['episodes']} | {r.get('episode_id') for r in result['episodes']}
        for field, key in [('candidates', 'candidate_id'), ('lessons', 'lesson_id'), ('clusters', 'cluster_id')]:
            subjects.update(r.get(key) for r in result[field])
        for row in read('evolution_events'):
            if skill == 'all' or row['subject_id'] in subjects:
                row['payload'] = json.loads(row.pop('payload_json'))
                result['events'].append(row)
        result['active_lessons'] = sum(r['status'] == 'active' for r in result['lessons'])
    result['limit_per_table'] = 300
    return result


# Compatibility for the console's existing /api/experience route.
experience_snapshot = experience_get


def tools_get(console):
    from console.providers import provider_capabilities
    definitions = console.agent.tools.get_definitions()
    provider = getattr(console.agent, 'provider', None)
    return {'agent_mode': getattr(console, 'agent_mode', 'simulation'), 'tools': definitions,
            'registered_count': len(definitions), 'capabilities': provider_capabilities(console.config),
            'model_modes': provider.list_models() if hasattr(provider, 'list_models') else [],
            'default_model_mode': provider.get_default_mode() if hasattr(provider, 'get_default_mode') else None}


def routes(get_console, check_origin):
    async def manage(request):
        console = get_console()
        section = request.path_params['section']
        acquired = False
        run = False
        try:
            if section not in {'skills', 'settings', 'workspace', 'schedules', 'integrations', 'experience', 'tools'}:
                return reply({'error': 'Unknown management section'}, 404)
            if request.method == 'GET':
                if section == 'skills':
                    result = await skills_get(console, request.query_params)
                elif section == 'settings':
                    result = settings_get(console)
                elif section == 'workspace':
                    result = await asyncio.to_thread(workspace_get, console, request.query_params)
                elif section == 'schedules':
                    result = schedules_get(console)
                elif section == 'integrations':
                    result = integrations_get(console)
                elif section == 'experience':
                    result = await asyncio.to_thread(experience_get, console, request.query_params)
                else:
                    result = tools_get(console)
                return reply(result)
            check_origin(request)
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError('JSON object required')
            if section == 'settings' and body.get('action') == 'credentials' and console.coordinator.store.active():
                if (console.control_busy or getattr(console, 'management_run_busy', False)
                        or not console.idle_execution_task()):
                    raise BusyError('Agent 或原生动作仍在执行，暂不能更新密钥。')
            else:
                assert_idle(console)
            run = section == 'schedules' and body.get('action') in {'run', 'heartbeat_run'}
            if run:
                console.management_run_busy = True
            else:
                console.control_busy = True
            acquired = True
            if section == 'skills':
                result = await skills_post(console, body)
            elif section == 'settings':
                result = await settings_post(console, body)
            elif section == 'workspace':
                result = await workspace_update(console, body)
            elif section == 'schedules':
                result = await schedules_post(console, body)
            elif section == 'integrations':
                result = await integrations_post(console, body)
            elif section == 'tools':
                if body.get('action') != 'mode' or body.get('mode') not in {'simulation', 'general'}:
                    raise ValueError('Unknown tools mode')
                await callback(console, 'set_agent_mode', body['mode'])
                result = {'ok': True, **tools_get(console)}
            else:
                raise ValueError('经验只能通过真实任务和上游晋升规则产生，管理接口只读。')
            await console.emit('management', content=f'{section}: {body.get("action")} completed')
            return reply(result)
        except BusyError as exc:
            return reply({'error': str(exc)}, 409)
        except (ValueError, KeyError, TypeError) as exc:
            return reply({'error': safe_error(exc, console.config)}, 400)
        except Exception as exc:
            return reply({'error': safe_error(exc, console.config)}, 500)
        finally:
            if acquired:
                if run:
                    console.management_run_busy = False
                else:
                    console.control_busy = False
    return [Route('/api/manage/{section}', manage, methods=['GET', 'POST'])]
