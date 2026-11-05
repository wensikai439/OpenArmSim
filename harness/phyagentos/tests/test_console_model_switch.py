"""Offline regression checks for the in-page model switch; no model or hardware calls."""
import asyncio
from types import SimpleNamespace
import pytest
from PhyAgentOS.config.schema import Config
from console import management, real_commands
from console.providers import active_model

def configured():
    return Config.model_validate({
        'agents': {'defaults': {'model': 'old', 'provider': 'custom'},
                   'modes': {'enabled': True, 'default_mode': 'main',
                             'models': {'main': {'model': 'old'}, 'multimodal': {'model': 'vision'}}}},
        'providers': {'custom': {'api_key': 'private-test-key', 'api_base': 'http://localhost:9/v1'}}})

def owner(config):
    return SimpleNamespace(config=config, agent_mode='real')

def test_switch_updates_actual_route_and_preserves_credentials(monkeypatch):
    console=owner(configured())
    async def save(c,data,**kwargs): c.config=Config.model_validate(data)
    monkeypatch.setattr(management,'save_configuration',save)
    result=asyncio.run(management.settings_post(console,{'action':'switch_model','model':'new','provider':'custom'}))
    assert result['active_model']=='new' and result['agent_mode']=='real'
    assert console.config.agents.defaults.model=='new'
    assert console.config.agents.modes.models['main'].model=='new'
    assert console.config.agents.modes.models['multimodal'].model=='vision'
    assert console.config.agents.modes.default_mode=='main'
    assert console.config.providers.custom.api_key=='private-test-key'
    assert result['providers']['custom']['api_key']==''
    assert 'private-test-key' not in str(result)

def test_real_mode_advanced_save_is_accepted(monkeypatch):
    console=owner(configured())
    monkeypatch.setattr(management,'load_config',lambda:console.config)
    async def save(c,data,**kwargs):c.config=Config.model_validate(data)
    monkeypatch.setattr(management,'save_configuration',save)
    result=asyncio.run(management.settings_post(console,{'action':'save','defaults':{'model':'advanced'}}))
    assert result['agent_mode']=='real' and result['active_model']=='advanced'

def test_unconfigured_provider_rejected():
    with pytest.raises(ValueError,match='尚未配置'):
        asyncio.run(management.settings_post(owner(configured()),{'action':'switch_model','model':'new','provider':'deepseek'}))

def test_unknown_execution_blocks_configuration(tmp_path,monkeypatch):
    monkeypatch.setattr(real_commands,'RUNS',tmp_path)
    c=real_commands.RealCommands(SimpleNamespace())
    real_commands.save(tmp_path/'active.json',{'phase':'unknown','do_not_resubmit':True})
    with pytest.raises(management.BusyError,match='结果待核对'):
        management.assert_idle(SimpleNamespace(real_commands=c))

def test_reload_reattaches_tools_without_losing_controller(tmp_path,monkeypatch):
    monkeypatch.setattr(real_commands,'RUNS',tmp_path)
    console=SimpleNamespace(_all_tools={},apply_tool_mode=lambda:None)
    controller=real_commands.attach(console)
    console.real_commands=controller
    controller.current=tmp_path/'command-one'
    controller.camera_unit='existing-camera.service'
    console._all_tools={}
    assert real_commands.attach(console) is controller
    assert controller.current.name=='command-one'
    assert controller.camera_unit=='existing-camera.service'
    assert set(console._all_tools)=={'real_observe','real_relative_action'}

def test_active_model_reports_configured_route():
    config=configured()
    config.agents.modes.models['main'].model='actual'
    assert active_model(config)=='actual'

def test_real_skill_selection_and_removal_protection():
    console=owner(configured())
    console.skill='piper-dual-sim'
    assert management.selected(console)['name']=='piper-real-runtime'
    for name in ('piper-real-runtime','piper-dual-sim'):
        with pytest.raises(ValueError,match='先切换'):
            asyncio.run(management.skills_post(console,{'action':'remove','name':name}))

def test_simulation_switch_cannot_change_real_command_binding():
    with pytest.raises(ValueError,match='真机命令页固定'):
        asyncio.run(management.skills_post(owner(configured()),{'action':'switch','name':'piper-dual-sim','profile':'joint'}))
