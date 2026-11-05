"""Local simulation console; official Agent/Forge lifecycle, read-only history."""
from __future__ import annotations
import os
import asyncio, base64, dataclasses, hashlib, json, sqlite3, time, uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, quote
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from loguru import logger
from PhyAgentOS import __version__
from PhyAgentOS.cli.commands import _make_provider, _make_evolution_provider, _make_forge_components
from PhyAgentOS.config.loader import load_config
from PhyAgentOS.config.paths import get_cron_dir
from PhyAgentOS.agent.loop import AgentLoop
from PhyAgentOS.bus.queue import MessageBus
from PhyAgentOS.cron.service import CronService
from PhyAgentOS.heartbeat.service import HeartbeatService
from PhyAgentOS.channels.manager import ChannelManager
from PhyAgentOS.embodiment_registry import EmbodimentRegistry
from PhyAgentOS.utils.helpers import sync_workspace_templates
from PhyAgentOS.forge.observation import ForgeObservationCollector
from PhyAgentOS.skill_runtime.manager import RuntimeManager
from console.providers import make_agent_provider

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/console'
RUN.mkdir(parents=True, exist_ok=True)
SKILL = 'piper-dual-sim'
PROFILE = 'joint'
IMAGE_SOURCES = ['image/overview', 'image/cam_right']
GATEWAY = 'http://127.0.0.1:19013'
TERMINAL = {'succeeded', 'failed', 'cancelled', 'stopped', 'rejected', 'expired'}
logger.add(RUN / 'server.log', rotation='10 MB', retention=4)

def now():
    return datetime.now(timezone.utc).isoformat()

def encode(value):
    return json.dumps(value, ensure_ascii=False, default=str)

def records(task):
    return [e for r in task.get('revisions', []) for e in r.get('execution_records', [])]

def agent_forge_result(name, result):
    """Omit repeated cumulative native actions only from the Agent text view."""
    if not name.startswith('forge_') or not isinstance(result, str):
        return result
    try:
        value = json.loads(result)
    except (ValueError, TypeError):
        return result
    omitted = []
    def compact(item):
        if isinstance(item, dict):
            output = {}
            for key, child in item.items():
                if key == 'composable_actions' and isinstance(child, list) and child:
                    omitted.append(len(child))
                    output[key] = {'omitted_cumulative_history_count': len(child),
                        'source': 'Full original result in console journal and Task execution records; not a verdict.'}
                else:
                    output[key] = compact(child)
            return output
        if isinstance(item, list):
            return [compact(child) for child in item]
        return item
    view = compact(value)
    return json.dumps(view, ensure_ascii=False, separators=(',', ':')) if omitted else result

class Console:
    def __init__(self):
        self.events = deque(maxlen=3000)
        self.seq = 0
        self.seen = set()
        self.job = None
        self.running = None
        self.control_busy = False
        self.snapshot = {}
        self.condition = asyncio.Condition()
        journal = RUN / 'events.jsonl'
        if journal.exists():
            for line in journal.read_text().splitlines():
                event = json.loads(line)
                self.events.append(event)
                self.seq = max(self.seq, event['id'])
                if event['type'] == 'accepted':
                    self.seen.add(event['request_id'])
        prefs = json.loads((RUN/'preferences.json').read_text()) if (RUN/'preferences.json').exists() else {}
        self.skill = prefs.get('skill', SKILL)
        self.profile = prefs.get('profile', PROFILE)
        self.agent_mode = prefs.get('agent_mode', 'simulation')
        self.management_run_busy = False
        self.service_tasks = []
        self.config = load_config()
        self.workspace = self.config.workspace_path
        self.database = self.workspace / '.paos/agent_tasks/tasks.sqlite3'
        self.artifacts = self.workspace / 'artifacts/agent_tasks'
        self.manager = RuntimeManager()
        self.configure_agent(self.config)
        self.configure_observation()
        self.collector = self.new_collector()

    def configure_observation(self):
        if self.skill == 'piper-dual-sim' and self.profile in {'osc', 'joint'}:
            self.gateway = GATEWAY
            self.image_sources = list(IMAGE_SOURCES)
            self.asset_root = str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa"))))
        elif self.skill == 'rpent-piper-sim' and self.profile in {'rpent', 'joint'}:
            self.gateway = 'http://127.0.0.1:19012'
            self.image_sources = list(IMAGE_SOURCES)
            self.asset_root = str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa"))))
        elif (self.skill, self.profile) == ('move-arm-by-ee', 'mujoco'):
            self.gateway = 'http://127.0.0.1:19002'
            self.image_sources = ['image/angle']
            self.asset_root = 'official move-arm-by-ee mujoco bundle'
        else:
            raise ValueError('Only the registered simulation profiles may execute in this console')
        self.config.forge.evidence.required_image_sources = list(self.image_sources)
        self.coordinator.config.evidence.required_image_sources = list(self.image_sources)

    def save_preferences(self):
        path = RUN/'preferences.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(encode({'skill':self.skill, 'profile':self.profile, 'agent_mode':self.agent_mode}))
        tmp.replace(path)

    def background_busy(self):
        return self.agent.subagents.get_running_count()>0 or any(not t.done() for tasks in self.agent._active_tasks.values() for t in tasks)

    async def cancel_agent_tasks(self):
        # This console has one execution lane. Include children after their parent turn ends.
        current = asyncio.current_task()
        pending = {task for tasks in self.agent._active_tasks.values() for task in tasks}
        pending.update(self.agent.subagents._running_tasks.values())
        if self.running is not None:pending.add(self.running)
        pending = {task for task in pending if task is not current and not task.done()}
        # Cancel the complete snapshot before yielding, so one branch cannot keep another alive.
        cancelled = sum(task.cancel() for task in pending)
        if pending:await asyncio.gather(*pending, return_exceptions=True)
        return cancelled

    def reflection_busy(self):
        e = self.agent.experience
        return bool(e and (any(not t.done() for t in e._tasks.values()) or
                           any(not t.done() for t in e._cluster_tasks.values())))

    def configure_agent(self, config):
        sync_workspace_templates(self.workspace)
        provider = _make_provider(config)
        ep, em = _make_evolution_provider(config, provider)
        client, ids, coordinator, availability = _make_forge_components(config, provider)
        self.bus = MessageBus()
        original_publish = self.bus.publish_outbound
        async def publish(msg):
            if msg.channel in {'console', 'cli'}:
                key = msg.chat_id if msg.chat_id.startswith('console:') else 'console:'+msg.chat_id
                await self.emit('answer', session=key, content=msg.content, media=[url for p in (msg.media or []) if (url:=self.media_url(p))])
            else:
                await original_publish(msg)
        self.bus.publish_outbound = publish
        self.cron = CronService(get_cron_dir()/'jobs.json')
        self.agent = AgentLoop(
            bus=self.bus, provider=make_agent_provider(config, provider), workspace=self.workspace,
            model=config.agents.defaults.model,
            max_iterations=config.agents.defaults.max_tool_iterations,
            context_window_tokens=config.agents.defaults.context_window_tokens,
            brave_api_key=config.tools.web.search.api_key or None,
            web_proxy=config.tools.web.proxy or None, exec_config=config.tools.exec,
            cron_service=self.cron, restrict_to_workspace=True,
            mcp_servers=config.tools.mcp_servers, channels_config=config.channels,
            embodiment_registry=EmbodimentRegistry(config),
            forge_tool_client=client, forge_tool_invocation_ids=ids,
            forge_task_coordinator=coordinator, runtime_availability_provider=availability,
            evolution_config=config.agents.evolution, evolution_provider=ep, evolution_model=em)
        self._all_tools = {name:self.agent.tools.get(name) for name in self.agent.tools.tool_names}
        original_connect = self.agent._connect_mcp
        async def connect():
            await original_connect()
            self._all_tools.update({n:self.agent.tools.get(n) for n in self.agent.tools.tool_names})
            self.apply_tool_mode()
        self.agent._connect_mcp = connect
        self.apply_tool_mode()
        native_execute = self.agent.tools.execute
        async def execute(name, params):
            # The simulator has one Action lane; a model may batch multiple start calls.
            if name == 'forge_tool_start_action' and self.agent_mode == 'simulation':
                active = self.coordinator.store.active()
                pending = [e for e in active.execution_records
                           if e.semantics == 'action' and e.status not in TERMINAL] if active else []
                if pending:
                    result = encode({'ok': False, 'error': {
                        'type': 'pending_simulation_action',
                        'message': 'Wait for the existing Action terminal status and native result before starting another; do not replay an unknown Action.',
                        'pending': [{'record_id': e.record_id, 'invocation_id': e.invocation_id,
                                     'status': e.status} for e in pending]}})
                    job = self.job or {}
                    await self.emit('tool_result', session=job.get('session'),
                        request_id=job.get('request_id'), name=name, arguments=params, result=result)
                    return result
            # The selected simulation profile is the only execution asset in this console.
            if name == 'activate_skill' and self.agent_mode == 'simulation' and params.get('name') != self.skill:
                return 'Error: 当前仿真会话只能激活 '+self.skill
            if name == 'image':
                path = Path(params.get('image_path','')).expanduser()
                path = (path if path.is_absolute() else self.workspace/path).resolve()
                if not path.is_relative_to(self.workspace.resolve()):
                    return 'Error: image_path must be inside the configured workspace'
                params = {**params, 'image_path':str(path)}
                session = (self.job or {}).get('session','console:default')
                channel, chat_id = session.split(':',1)
                self.agent.tools.get('image').set_context(channel,chat_id)
            result = await native_execute(name, params)
            job = self.job or {}
            await self.emit('tool_result', session=job.get('session'),
                request_id=job.get('request_id'), name=name, arguments=params, result=result)
            return agent_forge_result(name, result)
        self.agent.tools.execute = execute
        self.coordinator = coordinator
        async def dispatch(msg):
            session = msg.session_key
            if msg.channel == 'system':
                session = msg.session_key_override or (msg.chat_id if ':' in msg.chat_id else 'cli:'+msg.chat_id)
            # Keep completed subagent results queued behind the same official processing lock.
            async with self.agent._processing_lock:
                if msg.channel != 'system' and self.coordinator.store.active():
                    from PhyAgentOS.bus.events import OutboundMessage
                    await self.bus.publish_outbound(OutboundMessage(channel=msg.channel,
                        chat_id=msg.chat_id,content='已有活动Forge任务，请等待终态或继续原任务。'))
                    return
                self.job = {'session':session,'request_id':'channel-'+uuid.uuid4().hex,'started_at':now()}
                try:
                    if msg.channel != 'system':msg.content = self.scoped_prompt(msg.content)
                    response = await self.agent._process_message(msg)
                    if response is not None:await self.bus.publish_outbound(response)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self.emit('error',session=session,content=str(exc))
                finally:
                    self.job = None
                    await self.refresh()
        self.agent._dispatch = dispatch
        self.channels = ChannelManager(config, self.bus)
        self.cron.on_job = self.on_cron_job
        hb = config.gateway.heartbeat
        self.heartbeat = HeartbeatService(workspace=self.workspace, provider=provider,
            model=config.agents.defaults.model, on_execute=self.on_heartbeat_execute,
            on_notify=self.on_heartbeat_notify, interval_s=hb.interval_s, enabled=hb.enabled)
        from console.real_commands import attach, REAL_EXECUTION_POLICY
        build_system_prompt=self.agent.context.build_system_prompt
        def command_system_prompt(skill_names=None):
            prompt=build_system_prompt(skill_names)
            return prompt+'\n\n'+REAL_EXECUTION_POLICY if self.agent_mode=='real' else prompt
        self.agent.context.build_system_prompt=command_system_prompt
        self.real_commands = attach(self)

    def apply_tool_mode(self):
        for name, tool in self._all_tools.items():
            if ((self.agent_mode == 'real' and name in {'real_observe','real_relative_action'}) or
                (self.agent_mode == 'general' and not name.startswith('real_')) or
                (self.agent_mode == 'simulation' and (name == 'activate_skill' or name.startswith('forge_')))):
                self.agent.tools.register(tool)
            else:
                self.agent.tools.unregister(name)

    def scoped_prompt(self, prompt):
        if prompt.strip().lower() in {'/new','/help'}:
            return prompt
        if self.agent_mode == 'real':
            return ('当前为PiPER真机命令模式，仅可使用real_observe与real_relative_action；后者在后端托管真实Skill激活、AgentTask绑定与动作终态记录，无须另调未暴露的生命周期工具。'
                '先根据用户目标观察真实三路相机及反馈，每段动作后重新观察；所有动作以当前反馈为起点。'
                '图像没有外参标定，不能从像素臆造绝对坐标；用小段相对动作和观察逐步靠近。'
                '原生Forge完成仅是程序完成，抓握/交接/放置必须看到物体实际状态。'
                '每次工具返回ok=false、动作失败、保护停止或结果unknown时立即停止本次后续动作并报告，不重试或换路径继续。'
                '只读任务不得下发动作；前端的真实用户请求决定本次是否允许动作，模型不能改变授权。'
                '不写文件、不调用shell、不加载ACT/DP、不修改控制底座；已有示教只参考任务顺序，不重放轨迹。'
                '这是新的在线闭环入口，尚未验证重复成功率。回复简洁，展示观察、动作和未完成事项。\n'+prompt)
        if self.agent_mode == 'simulation':
            return ('当前唯一执行资产为 '+self.skill+'/'+self.profile+
                '，仅仿真；先读取实时context与指导。不得连接真机、修改底座或绕过Forge执行。'
                '使用真实结果与证据，未知效果不重放。\n'+prompt)
        return ('当前为通用Agent工作区模式；文件操作限本工作区，不修改RPent/ROS/ACT/π0.5/PiPER底座，不连接真机。'
                '需要机器人执行时使用当前已安装仿真Skill和Forge；外部消息只按用户明确指定的收件人与内容发送。'
                'spawn为异步子Agent，收到启动回执后立即结束当前turn；完成结果会另行送达，不得exec sleep或轮询等待。\n'+prompt)

    def media_url(self, value):
        if isinstance(value,dict):
            data=value.get('data','')
            if not isinstance(data,str) or not data.startswith('data:image/') or ';base64,' not in data:return None
            header,payload=data.split(';base64,',1)
            suffix={'data:image/png':'.png','data:image/jpeg':'.jpg','data:image/webp':'.webp','data:image/gif':'.gif','data:image/bmp':'.bmp'}.get(header)
            if suffix is None:return None
            raw=base64.b64decode(payload,validate=True)
            directory=self.workspace/'.paos/console-media';directory.mkdir(parents=True,exist_ok=True)
            path=directory/(hashlib.sha256(raw).hexdigest()+suffix)
            if not path.exists():path.write_bytes(raw)
        else:
            path=Path(value).expanduser().resolve()
        if path.is_relative_to(self.workspace.resolve()) and path.is_file():
            return '/api/media?path='+quote(str(path.relative_to(self.workspace.resolve())))
        return None

    async def start_services(self):
        await self.agent._connect_mcp()
        await self.cron.start()
        await self.heartbeat.start()
        self.service_tasks = [asyncio.create_task(self.agent.run()),
                              asyncio.create_task(self.channels.start_all())]

    async def stop_services(self):
        self.heartbeat.stop()
        self.cron.stop()
        await self.channels.stop_all()
        pending = [t for tasks in self.agent._active_tasks.values() for t in tasks if not t.done()]
        pending += [t for t in self.agent.subagents._running_tasks.values() if not t.done()]
        for task in pending: task.cancel()
        if pending:await asyncio.gather(*pending,return_exceptions=True)
        self.agent.stop()
        for task in self.service_tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*self.service_tasks, return_exceptions=True)
        self.service_tasks = []
        await self.agent.close_mcp()

    def idle_execution_task(self):
        active = self.coordinator.store.active()
        if (not active or active.status.value != 'executing' or active.cancellation_requested
                or self.job or self.background_busy() or self.reflection_busy()):
            return None
        if any(e.semantics in {'action', 'session'} and e.status not in TERMINAL
               for e in active.execution_records):
            return None
        runtime = self.manager.state_store.load(self.skill)
        if runtime and (runtime.active_invocations or runtime.active_sessions):
            return None
        return active

    async def reload_credentials(self, config):
        if config.model_dump(exclude={'providers'}) != self.config.model_dump(exclude={'providers'}):
            raise ValueError('密钥更新不能修改模型或任务策略')
        await self.reload_configuration(config, idle_credentials=True)

    async def reload_configuration(self, config, *, idle_credentials=False):
        active = self.coordinator.store.active()
        if getattr(self, 'real_commands', None) and self.real_commands.configuration_busy():
            raise ValueError('真机任务仍在执行或结果待核对，不能重载配置')
        if (self.job or self.reflection_busy() or self.background_busy()
                or (active and not (idle_credentials and self.idle_execution_task()))):
            raise ValueError('任务或反思正在执行，不能重载配置')
        previous = self.config
        await self.stop_services()
        try:
            self.configure_agent(config)
            self.config = config
            self.configure_observation()
            await self.start_services()
        except (Exception, SystemExit):
            await self.stop_services()
            self.configure_agent(previous)
            self.config = previous
            self.configure_observation()
            await self.start_services()
            raise
        await self.emit('configuration_changed', content='配置已加载，当前任务记录与观察保持。')
        await self.refresh()

    async def set_agent_mode(self, mode):
        if mode not in {'simulation','general','real'}: raise ValueError('Invalid agent mode')
        if self.job or self.coordinator.store.active() or self.reflection_busy() or self.background_busy():
            raise ValueError('任务或反思正在执行，不能切换模式')
        self.agent_mode = mode
        self.apply_tool_mode()
        self.save_preferences()
        await self.emit('mode_changed',content='Agent模式：'+mode)
        await self.refresh()

    async def select_skill(self, name, profile):
        if (name,profile) not in {('piper-dual-sim','joint'),('piper-dual-sim','osc'),('rpent-piper-sim','rpent'),('rpent-piper-sim','joint'),('move-arm-by-ee','mujoco')}:
            raise ValueError('该profile未登记为可执行仿真')
        await self.collector.close()
        self.skill,self.profile = name,profile
        self.configure_observation()
        self.collector = self.new_collector()
        await self.collector.start()
        self.save_preferences()
        await self.refresh()

    async def on_cron_job(self, job):
        if self.job or self.control_busy or self.background_busy() or self.coordinator.store.active():
            raise RuntimeError('当前任务忙，未提交定时任务')
        key='console:cron-'+job.id
        rid='cron-'+uuid.uuid4().hex
        self.job={'session':key,'request_id':rid,'started_at':now()}
        await self.emit('accepted',session=key,request_id=rid,content='定时任务：'+job.payload.message)
        cron_tool=self.agent.tools.get('cron')
        token=cron_tool.set_cron_context(True) if cron_tool else None
        try:
            self.running=asyncio.create_task(self.process(job.payload.message,key,rid,propagate_errors=True))
            result=await self.running
        finally:
            if cron_tool and token is not None:cron_tool.reset_cron_context(token)
        if job.payload.deliver and job.payload.to and result:
            from PhyAgentOS.bus.events import OutboundMessage
            await self.bus.publish_outbound(OutboundMessage(channel=job.payload.channel or 'console',
                chat_id=job.payload.to,content=result))
        return result

    async def on_heartbeat_execute(self, content):
        if self.job or self.control_busy or self.background_busy() or self.coordinator.store.active():
            raise RuntimeError('当前任务忙，未提交Heartbeat')
        key='console:heartbeat';rid='heartbeat-'+uuid.uuid4().hex
        self.job={'session':key,'request_id':rid,'started_at':now()}
        await self.emit('accepted',session=key,request_id=rid,content=content)
        self.running=asyncio.create_task(self.process(content,key,rid,propagate_errors=True))
        return await self.running

    async def on_heartbeat_notify(self, content):
        # process() already writes the result into the local console journal.
        return None

    def new_collector(self):
        return ForgeObservationCollector(self.gateway, required_image_sources=self.image_sources,
            max_artifact_bytes=8*1024*1024, require_state=True)

    async def emit(self, kind, **values):
        self.seq += 1
        event = {'id': self.seq, 'type': kind, 'at': now(), **values}
        self.events.append(event)
        with (RUN / 'events.jsonl').open('a') as f:
            f.write(encode(event) + '\n')
        async with self.condition:
            self.condition.notify_all()
        return event

    def read_tasks(self):
        # List views need scalar metadata, not the cumulative action/evidence history.
        fields = ["task_id", "task_description", "status", "verification.mode",
                  "verdict.verdict", "origin_session_key", "primary_skill_binding.skill_name", "updated_at"]
        expressions = ["json_extract(record_json, '$."+field+"')" for field in fields]
        query = 'SELECT '+','.join(expressions)+",json_array_length(record_json, '$.revisions') FROM agent_tasks ORDER BY updated_at DESC LIMIT 60"
        keys = ['task_id','description','status','mode','verdict','session','skill','updated_at','revision']
        with sqlite3.connect(f'file:{self.database}?mode=ro', uri=True) as db:
            return [dict(zip(keys, row)) for row in db.execute(query)]

    def read_task(self, task_id):
        with sqlite3.connect(f'file:{self.database}?mode=ro', uri=True) as db:
            row = db.execute('SELECT record_json FROM agent_tasks WHERE task_id=?', (task_id,)).fetchone()
            if row is None:
                raise ValueError('Task not found')
            events = [{'type': r[0], 'at': r[1], 'payload': json.loads(r[2])} for r in db.execute(
                'SELECT event_type,created_at,payload_json FROM agent_task_events WHERE task_id=? ORDER BY event_id', (task_id,))]
        task = json.loads(row[0])
        files = []
        directory = (self.artifacts / task_id).resolve()
        if directory.is_relative_to(self.artifacts.resolve()) and directory.exists():
            files = [{'name': str(p.relative_to(directory)), 'bytes': p.stat().st_size} for p in sorted(directory.rglob('*')) if p.is_file()]
        return {'task': task, 'events': events, 'files': files}

    def session_index(self):
        sessions = {x['key']:{k:v for k,v in x.items() if k!='path'} for x in self.agent.sessions.list_sessions() if x['key'].startswith('console:')}
        for event in self.events:
            key = event.get('session')
            if key and key.startswith('console:'):
                record = sessions.setdefault(key, {'key':key,'created_at':event['at'],'updated_at':event['at']})
                record['updated_at'] = max(record['updated_at'],event['at'])
        return sorted(sessions.values(),key=lambda x:x['updated_at'],reverse=True)

    async def refresh(self):
        report = await asyncio.to_thread(self.manager.status, self.skill)
        runtime_id = report.state.runtime_instance_id if report.state else None
        previous_id = getattr(self, '_observation_runtime_id', None)
        if runtime_id:self._observation_runtime_id = runtime_id
        if previous_id and runtime_id and runtime_id != previous_id:
            # A normal CLI reset also restarts source sequences; discard the previous stream.
            await self.collector.close()
            self.collector = self.new_collector()
            if report.flow_running:await self.collector.start()
        tasks = await asyncio.to_thread(self.read_tasks)
        active = self.coordinator.store.active()
        manifest = self.manager.catalog.get(self.skill)
        observation = await self.collector.latest_snapshot()
        sim_status = ((observation.state.payload.get('runtime') or {}).get('sim_status') or {}) if observation.state else {}
        native_observation = sim_status.get('observation') or {}
        reset_required = bool(sim_status.get('reset_required') or native_observation.get('reset_required')
            or native_observation.get('failed') or native_observation.get('collision_stop'))
        native_cancel = self.skill=='move-arm-by-ee' or bool(sim_status.get('interruptible_actions'))
        nonterminal_actions = bool(active and any(e.semantics=='action' and not e.terminal for e in active.execution_records))
        cancel_supported = native_cancel or not nonterminal_actions
        self.snapshot = {
            'at': now(), 'core_version': __version__, 'model': self.config.agents.defaults.model,
            'skill': self.skill, 'skill_version': manifest.version if manifest else None, 'profile': self.profile,
            'asset_root': self.asset_root, 'task_variant':'air-handoff' if self.skill in {'piper-dual-sim','rpent-piper-sim'} else 'mujoco',
            'agent_mode':self.agent_mode, 'image_sources':self.image_sources,
            'interruptible_actions':native_cancel, 'cancel_supported':cancel_supported,
            'cancel_reason':'' if cancel_supported else '当前原生profile没有已确认的中途取消能力',
            'reset_required':reset_required,
            'reflection_busy':self.reflection_busy(), 'background_busy':self.background_busy(),
            'ready': report.ready and bool(report.state and report.state.profile == self.profile) and not reset_required, 'runtime': dataclasses.asdict(report),
            'job': self.job, 'control_busy': self.control_busy,
            'active_task': active.task_id if active else None,
            'idle_execution_task': active.task_id if report.ready and self.idle_execution_task() else None,
            'tasks': tasks,
            'sessions': self.session_index(),
            'last_event_id': self.seq,
        }
        return self.snapshot

    async def ticker(self):
        while True:
            try:
                active = self.coordinator.store.active()
                # Cancellation is followed by GET reconciliation, never by another motion POST.
                if active and active.cancellation_requested and not self.control_busy:
                    task = await self.coordinator.reconcile_nonterminal()
                    if task and not task.terminal and task.execution_records and all(
                        e.status in TERMINAL for e in task.execution_records):
                        final = await self.coordinator.finalize_task(task.task_id)
                        await self.emit('cancel_terminal', task_id=task.task_id,
                            session=final.origin_session_key,
                            content=f'Gateway 原生 Action 已到已知终态；task={final.task_id}，status={final.status.value}。')
                await self.refresh()
                async with self.condition:
                    self.condition.notify_all()
            except Exception as exc:
                self.snapshot['status_error'] = str(exc)
                logger.warning('Console status: {}', exc)
            await asyncio.sleep(2)

    async def process(self, prompt, session, request_id, propagate_errors=False):
        async def progress(content, *, tool_hint=False):
            await self.emit('tool' if tool_hint else 'progress', session=session,
                            request_id=request_id, content=content)
        try:
            async with self.agent._processing_lock:
                content = await self.agent.process_direct(self.scoped_prompt(prompt), session_key=session,
                    channel='console', chat_id=session.split(':',1)[1], on_progress=progress)
            if propagate_errors and content.startswith('Error:'):
                raise RuntimeError(content)
            await self.emit('answer', session=session, request_id=request_id, content=content)
            return content
        except asyncio.CancelledError:
            await self.emit('agent_cancelled', session=session, request_id=request_id,
                content='Agent 推理已取消；物理动作以 Gateway 原生终态为准。')
            raise
        except Exception as exc:
            logger.exception('Console turn failed')
            await self.emit('error', session=session, request_id=request_id, content=str(exc))
            if propagate_errors:raise
        finally:
            if self.job and self.job['request_id'] == request_id:
                self.job = None
            await self.refresh()

S = None
@asynccontextmanager
async def lifespan(app):
    global S
    S = Console()
    await S.collector.start()
    await S.start_services()
    await S.refresh()
    ticker = asyncio.create_task(S.ticker())
    await S.emit('server_started', content='Core / Forge console started')
    try:
        yield
    finally:
        if S.running and not S.running.done(): S.running.cancel()
        ticker.cancel()
        await asyncio.gather(*[t for t in [ticker,S.running] if t], return_exceptions=True)
        await S.stop_services()
        await S.collector.close()

def json_response(data, status=200):
    return JSONResponse(json.loads(encode(data)), status_code=status, headers={'Cache-Control':'no-store'})

def check_origin(request):
    origin = request.headers.get('origin')
    if origin and urlparse(origin).netloc != request.headers.get('host'):
        raise ValueError('Origin mismatch')

async def state(request):
    return json_response(S.snapshot)

async def events(request):
    cursor = int(request.headers.get('last-event-id') or request.query_params.get('after','0'))
    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            for event in list(S.events):
                if event['id'] > cursor:
                    cursor = event['id']
                    yield f'id: {cursor}\nevent: console\ndata: {encode(event)}\n\n'
            yield f'event: state\ndata: {encode(S.snapshot)}\n\n'
            try:
                async with S.condition:
                    await asyncio.wait_for(S.condition.wait(), 10)
            except TimeoutError:
                yield ': heartbeat\n\n'
    return StreamingResponse(stream(), media_type='text/event-stream', headers={
        'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})

async def message(request):
    reserved = False
    control_reserved = False
    try:
        check_origin(request)
        body = await request.json()
        prompt = str(body['content']).strip()
        session = str(body['session'])
        rid = str(body['request_id'])
        if not prompt or len(prompt)>20000 or not session.startswith('console:') or len(session)>100:
            raise ValueError('Invalid message or session')
        if rid in S.seen:
            return json_response({'request_id':rid, 'duplicate':True})
        command = prompt.lower()
        if command == '/stop':
            response = await cancel(request)
            if response.status_code >= 400:return response
            result = json.loads(response.body)
            S.seen.add(rid)
            await S.emit('accepted',session=session,request_id=rid,content=prompt)
            content = f"已取消 {result['local_tasks_cancelled']} 个本地 Agent 任务。"
            if result['task_id']:
                content += ' Forge 取消已请求，动作停止以 Gateway 原生终态为准。'
            await S.emit('answer',session=session,request_id=rid,content=content)
            return json_response({'request_id':rid,**result})
        if command == '/help':
            S.seen.add(rid)
            await S.emit('accepted',session=session,request_id=rid,content=prompt)
            await S.emit('answer',session=session,request_id=rid,content=(
                'PhyAgentOS 控制台命令：\n/new — 按官方流程归档记忆并新建对话\n'
                '/stop — 取消本地 Agent 任务并请求原生 Forge 取消\n'
                '/restart — 空闲时重载 Agent 配置与服务，保持仿真实例\n/help — 显示命令'))
            return json_response({'request_id':rid})
        if S.job or S.control_busy or S.management_run_busy or S.background_busy() or S.coordinator.store.active():
            return json_response({'error':'已有活动任务，请等待终态或请求取消。'},409)
        if command == '/restart':
            if S.reflection_busy():return json_response({'error':'经验反思正在执行，暂不能重载 Agent。'},409)
            S.control_busy = True
            control_reserved = True
            S.seen.add(rid)
            await S.emit('accepted',session=session,request_id=rid,content=prompt)
            await S.reload_configuration(load_config())
            await S.emit('answer',session=session,request_id=rid,content='Agent 配置与服务已重载，仿真实例保持运行。')
            return json_response({'request_id':rid,'reloaded':True})
        S.job = {'session':session,'request_id':rid,'started_at':now()}
        reserved = True
        report = await asyncio.to_thread(S.manager.status, S.skill)
        if command != '/new' and S.agent_mode == 'simulation' and (not report.ready or report.state.profile != S.profile):
            return json_response({'error':'PiPER 仿真 Runtime 尚未 ready。'},409)
        if command != '/new' and S.agent_mode == 'simulation':
            latest = await S.collector.latest_snapshot()
            sim_status = ((latest.state.payload.get('runtime') or {}).get('sim_status') or {}) if latest.state else {}
            native = sim_status.get('observation') or {}
            if sim_status.get('reset_required') or native.get('reset_required') or native.get('failed') or native.get('collision_stop'):
                return json_response({'error':'当前仿真 episode 已失败或要求复位，请正常复位后继续。'},409)
        S.seen.add(rid)
        await S.emit('accepted', session=session, request_id=rid, content=prompt)
        S.running = asyncio.create_task(S.process(prompt,session,rid))
        reserved = False
        await S.refresh()
        return json_response({'request_id':rid},202)
    except (ValueError,KeyError) as exc:
        return json_response({'error':str(exc)},400)
    except Exception as exc:
        return json_response({'error':str(exc)},500)
    finally:
        if reserved:
            S.job = None
        if control_reserved:
            S.control_busy = False
            await S.refresh()

async def cancel(request):
    acquired = False
    try:
        check_origin(request)
        job = S.job
        active = S.coordinator.store.active()
        nonterminal_actions = bool(active and any(e.semantics=='action' and not e.terminal for e in active.execution_records))
        if nonterminal_actions and not S.snapshot.get('interruptible_actions'):
            return json_response({'error':S.snapshot.get('cancel_reason') or '当前原生动作不支持取消'},409)
        if S.control_busy:
            return json_response({'error':'Skill 生命周期操作正在进行。'},409)
        S.control_busy = True
        acquired = True
        local_cancelled = await S.cancel_agent_tasks()
        active = S.coordinator.store.active()
        if active:
            result = await S.coordinator.cancel_task(active.task_id, reason='console_user_stop')
            await S.emit('cancel_requested', task_id=active.task_id,
                session=job['session'] if job else active.origin_session_key,
                content=f'原生取消请求已提交，task={active.task_id}，status={result.status.value}。')
        else:
            await S.emit('cancel_requested', session=job['session'] if job else None, content=f'已取消 {local_cancelled} 个本地 Agent 任务；当前无活动 Forge task。')
        return json_response({'requested':True, 'task_id':active.task_id if active else None,
            'local_tasks_cancelled':local_cancelled})
    except Exception as exc:
        return json_response({'error':str(exc)},500)
    finally:
        if acquired: S.control_busy = False
        await S.refresh()

async def runtime(request):
    acquired = False
    try:
        check_origin(request)
        action = (await request.json())['action']
        if action not in {'start','stop','reset'}: raise ValueError('Invalid runtime action')
        if S.job or S.control_busy or S.management_run_busy or S.background_busy() or S.coordinator.store.active():
            return json_response({'error':'活动 task／Agent 尚未终态，禁止生命周期操作。'},409)
        S.control_busy = True
        acquired = True
        report = await asyncio.to_thread(S.manager.status,S.skill)
        if report.state and (report.state.active_invocations or report.state.active_sessions):
            return json_response({'error':'仍有未确认终态 invocation／session。'},409)
        S.control_busy = True
        acquired = True
        await S.emit('runtime_started', content=f'正常 Skill {action}')
        commands = [['paos','skill',action,S.skill]] if action!='reset' else [
            ['paos','skill','stop',S.skill], ['paos','skill','start',S.skill,'--profile',S.profile]]
        if action=='start': commands[0] += ['--profile',S.profile]
        for cmd in commands:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=ROOT,
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            output,_ = await proc.communicate()
            await S.emit('runtime_log',content=output.decode(errors='replace'))
            if proc.returncode: raise RuntimeError(f'CLI exit {proc.returncode}')
        # Sequence counters restart with the simulator. Replace the official collector once.
        await S.collector.close()
        S.collector = S.new_collector()
        if action!='stop': await S.collector.start()
        await S.emit('runtime_completed',content=f'Skill {action} 完成')
        return json_response({'action':action,'completed':True})
    except Exception as exc:
        return json_response({'error':str(exc)},500)
    finally:
        if acquired: S.control_busy = False
        await S.refresh()

async def observation(request):
    snap = await S.collector.latest_snapshot()
    image = snap.images.get(S.image_sources[0])
    instant = datetime.now(timezone.utc)
    image_age = (instant-image.received_at).total_seconds() if image else None
    state_age = (instant-snap.state.received_at).total_seconds() if snap.state else None
    return json_response({'at':now(), 'image_sequence':image.sequence if image else None,
        'image_at':image.received_at.isoformat() if image else None,'image_age_s':image_age,
        'state_at':snap.state.received_at.isoformat() if snap.state else None,'state_age_s':state_age,
        'fresh':image_age is not None and state_age is not None and max(image_age,state_age)<3,
        'state':snap.state.payload if snap.state else None,
        'cameras':{k:{'sequence':v.sequence,'received_at':v.received_at.isoformat()} for k,v in snap.images.items()},
        'physics_paused_between_requests':True})

async def frame(request):
    snap = await S.collector.latest_snapshot()
    source = request.query_params.get('source',S.image_sources[0])
    if source not in S.image_sources:return Response(status_code=400)
    image = snap.images.get(source)
    if not image: return Response(status_code=503)
    return Response(image.data,media_type=image.media_type,headers={
        'Cache-Control':'no-store','X-Frame-Sequence':str(image.sequence),'X-Observed-At':image.received_at.isoformat()})

async def task(request):
    try: return json_response(await asyncio.to_thread(S.read_task,request.path_params['task_id']))
    except ValueError as exc: return json_response({'error':str(exc)},404)

async def session(request):
    key = request.query_params.get('key','')
    known = {s['key'] for s in S.agent.sessions.list_sessions()}
    messages = S.agent.sessions.get_or_create(key).messages if key in known else []
    journal = [e for line in (RUN/'events.jsonl').read_text().splitlines() if (e:=json.loads(line)).get('session')==key][-2000:]
    return json_response({'messages':messages,'events':journal,'last_event_id':S.seq})

async def evidence(request):
    try:
        task_id=request.query_params['task']
        S.read_task(task_id)
        base=(S.artifacts/task_id).resolve()
        path=(base/request.query_params['path']).resolve()
        if not base.is_relative_to(S.artifacts.resolve()) or not path.is_relative_to(base) or not path.is_file():
            raise ValueError('Artifact not found')
        return FileResponse(path,headers={'Cache-Control':'no-store'})
    except (KeyError,ValueError): return Response(status_code=404)

async def media(request):
    path=(S.workspace/request.query_params.get('path','')).resolve()
    if not path.is_relative_to(S.workspace.resolve()) or not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg','.gif','.webp','.bmp'}:
        return Response(status_code=404)
    return FileResponse(path,headers={'Cache-Control':'no-store'})

async def experience(request):
    from console.management import experience_get
    data=await asyncio.to_thread(experience_get,S,{'skill':request.query_params.get('skill',S.skill)})
    return json_response(data)

async def mode(request):
    acquired=False
    try:
        check_origin(request)
        if S.job or S.control_busy or S.management_run_busy or S.background_busy() or S.coordinator.store.active():
            return json_response({'error':'已有活动任务'},409)
        S.control_busy=True;acquired=True
        await S.set_agent_mode((await request.json())['mode'])
        return json_response({'agent_mode':S.agent_mode})
    except (ValueError,KeyError) as exc:return json_response({'error':str(exc)},400)
    finally:
        if acquired:S.control_busy=False
        await S.refresh()

async def stop_session(request):
    try:
        check_origin(request)
        body=await request.json()
        active=S.coordinator.store.active()
        if S.control_busy or not active or body.get('task_id')!=active.task_id:
            return json_response({'error':'当前Task或Session已改变'},409)
        invocation_id=request.path_params['invocation_id']
        response=await S.coordinator.stop_session(active.task_id,invocation_id)
        await S.emit('session_stop_requested',task_id=active.task_id,
            session=active.origin_session_key,invocation_id=invocation_id,content='已请求停止观测Session，等待原生终态。')
        return json_response(response,202)
    except (ValueError,KeyError) as exc:return json_response({'error':str(exc)},400)
    except Exception as exc:return json_response({'error':str(exc)},409)

async def continue_task(request):
    reserved=False
    try:
        check_origin(request)
        body=await request.json()
        tid=request.path_params['task_id']
        rid=str(body['request_id'])
        if rid in S.seen:return json_response({'request_id':rid,'duplicate':True})
        if S.job or S.control_busy or S.management_run_busy or S.background_busy():return json_response({'error':'已有活动任务'},409)
        active=S.coordinator.store.active()
        if not active or active.task_id != tid:
            return json_response({'error':'仅可继续当前活动任务'},409)
        replanning = active.status.value == 'awaiting_replan'
        if not replanning and not S.idle_execution_task():
            return json_response({'error':'任务仍忙，或存在尚未确认终态的 Action / Session'},409)
        if replanning:
            session=str(body.get('session') or active.origin_session_key)
            if session != active.origin_session_key:raise ValueError('必须使用原任务会话')
        else:
            session=str(body.get('session') or 'console:resume-'+uuid.uuid4().hex[:12])
            if not session.startswith('console:') or len(session)>100:raise ValueError('Invalid session')
        prompt=str(body.get('content','')).strip()
        if not prompt or len(prompt)>20000:raise ValueError('请提供续接指令')
        S.job={'session':session,'request_id':rid,'started_at':now()}
        reserved=True
        if not replanning:
            report=await asyncio.to_thread(S.manager.status,S.skill)
            if not report.ready or report.state.profile != S.profile:
                return json_response({'error':'当前仿真 Runtime 尚未 ready'},409)
        if replanning:
            # The official begin_revision tool enforces deadline/budget and preserves the same Task.
            prompt='继续已有任务 '+tid+'；先forge_task_get读取真实recovery_context，再forge_task_begin_revision追加revision，不新建Task。\n'+prompt
        else:
            prompt=('续接已有真实任务 '+tid+'；先forge_task_get读取当前合同、冻结binding及执行记录，'
                    '再用同Task Query读取实时观测。保留原revision、episode与验证条件；不新建Task、'
                    '不activate_skill、不begin_revision、不reset、不重放历史动作。'
                    '仅从已确认的当前物理状态完成剩余目标。\n'+prompt)
        S.seen.add(rid)
        await S.emit('accepted',session=session,request_id=rid,content=prompt)
        S.running=asyncio.create_task(S.process(prompt,session,rid))
        reserved=False
        return json_response({'request_id':rid,'task_id':tid,'session':session},202)
    except (ValueError,KeyError) as exc:return json_response({'error':str(exc)},400)
    finally:
        if reserved:S.job=None


REPORTS = {'m4-results.json','m4-learned-skill.md','evolved-guide-provenance.json','final-versions.json'}
async def report(request):
    name=request.query_params.get('name')
    if name not in REPORTS: return Response(status_code=404)
    return FileResponse(ROOT/'runs'/name,headers={'Cache-Control':'no-store'})

async def presets(request):
    piper = S.skill in {'piper-dual-sim', 'rpent-piper-sim'}
    namespace = 'piper' if S.skill == 'piper-dual-sim' else 'rpent'
    base = ROOT/'config/piper_sim' if S.skill == 'piper-dual-sim' else ROOT/'runs/rpent-forge' if piper else ROOT/'runs'
    manifest = S.manager.catalog.get(S.skill)
    has_session = piper and manifest and manifest.version not in {'0.2.0','0.2.1'}
    return json_response({'session':(f'仅当前PiPER仿真观测，不提交Action。activate当前Skill并读取{namespace}.observe_stream实时context；通过官方Forge Session start持续采集20秒、interval_ms=1000，轮询到终态并读取真实result，报告sample_count及manifest。遵循task绑定要求，若要求Task则创建verification off的只读观测Task，终态后finalize。' if has_session else None),
        'session_stop':('读取当前PiPER活动Session，对确切活动invocation调用官方forge_tool_stop_session，轮询确认stopped并读取result；不要创建新Session或Action。' if has_session else None),
        'motion':(base/'task-motion.txt').read_text(),
        'enforce':(base/'task-enforce.txt').read_text(),
        'readonly':'仅检查 '+S.skill+'/'+S.profile+' 仿真。先activate当前Skill，读取实时context与Query状态，报告支持的工具与限制；不创建task、不提交Action、不复位。'})



from console.management import routes as management_routes
from console.real_demo import routes as real_demo_routes
from console.real_commands import routes as real_command_routes

app = Starlette(lifespan=lifespan,routes=[
    Route('/api/state',state), Route('/api/events',events),Route('/api/observation',observation),
    Route('/api/frame',frame),Route('/api/presets',presets),Route('/api/message',message,methods=['POST']),
    Route('/api/cancel',cancel,methods=['POST']),Route('/api/runtime',runtime,methods=['POST']),
    Route('/api/tasks/{task_id}',task),Route('/api/session',session),Route('/api/evidence',evidence),
    Route('/api/report',report),Route('/api/experience',experience),Route('/api/media',media),
    Route('/api/mode',mode,methods=['POST']),Route('/api/sessions/{invocation_id}/stop',stop_session,methods=['POST']),Route('/api/tasks/{task_id}/continue',continue_task,methods=['POST']),
    *real_command_routes(),*real_demo_routes,*management_routes(lambda:S,check_origin),Mount('/',StaticFiles(directory=ROOT/'console/web/dist',html=True)),
])
