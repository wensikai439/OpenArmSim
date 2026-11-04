"""Official mode management using the configured provider factory, including MiMo."""
from __future__ import annotations

import os
from typing import Any

from PhyAgentOS.cli.commands import _make_provider
from PhyAgentOS.providers.base import LLMProvider
from PhyAgentOS.providers.providers_manager import ProvidersManager


class ConfiguredProvidersManager(ProvidersManager):
    """Preserve official mode/tool contracts without rebuilding providers via LiteLLM."""

    def __init__(self, config, default_provider):
        self._default_provider = default_provider
        self.generation = default_provider.generation
        settings = config.agents.modes
        if 'main' not in settings.models:
            raise ValueError('Enabled agent modes require a main model configuration')
        if settings.default_mode != 'auto' and settings.default_mode not in settings.models:
            raise ValueError('Agent default_mode must name a configured mode or auto')
        super().__init__(config, {}, settings.default_mode)
        for name, item in settings.models.items():
            self.add_mode(name, item.model, item.describe)

    def _create_provider(self, model_id: str, provider_name: str = 'auto'):
        if model_id == self._config.agents.defaults.model and provider_name == 'auto':
            return self._default_provider
        return _make_provider(self._config, model_override=model_id,
            provider_name_override=None if provider_name == 'auto' else provider_name)

    def add_mode(self, mode: str, model_id: str, describe: str):
        if not mode or mode == 'auto' or not model_id:
            raise ValueError('A named mode and model are required; auto is reserved for routing')
        super().add_mode(mode, model_id, describe or '')

    def set_default_mode(self, mode):
        if mode != 'auto' and mode not in self._modes:
            raise ValueError(f'Unknown model mode: {mode}')
        super().set_default_mode(mode)

    def remove_mode(self, mode):
        if mode not in self._modes:
            raise ValueError(f'Unknown model mode: {mode}')
        if mode in {'main', self._default_mode}:
            raise ValueError('Cannot remove main or the selected model mode; select another mode first')
        super().remove_mode(mode)

    async def chat_with_retry(self, messages: list[dict[str, Any]], tools=None, model=None,
                              max_tokens=LLMProvider._SENTINEL,
                              temperature=LLMProvider._SENTINEL,
                              reasoning_effort=LLMProvider._SENTINEL,
                              mode='', tool_choice=None):
        mode = mode or self._default_mode
        if mode == 'auto':
            # Auto routing is an explicit configuration choice; the recommended default is main.
            decision = await self.get_provider('main').chat_with_retry(messages=[
                {'role':'system', 'content':'Choose one configured model mode. Reply only with its name.\n'+self.summary_modes()},
                {'role':'user', 'content':str(messages[-1].get('content', '')) if messages else ''},
            ], max_tokens=50, temperature=0.1)
            candidate = (decision.content or '').strip()
            mode = candidate if candidate in self._modes else 'main'
        if mode not in self._modes:
            raise ValueError(f'Model mode {mode!r} is not configured')
        provider = self.get_provider(mode)
        # AgentLoop always passes defaults.model. It must not overwrite agent_mode selection.
        actual_model = model if model and model != self._config.agents.defaults.model else provider.get_default_model()
        options = {'messages': messages, 'tools': tools, 'model': actual_model, 'tool_choice': tool_choice}
        for name, value in [('max_tokens', max_tokens), ('temperature', temperature), ('reasoning_effort', reasoning_effort)]:
            if value is not LLMProvider._SENTINEL:
                options[name] = value
        # Keep the official provider's generation defaults and retry policy; add no retry layer.
        return await provider.chat_with_retry(**options)


def make_agent_provider(config, default_provider):
    """AgentLoop recognizes the official manager and registers agent_mode/image tools."""
    return ConfiguredProvidersManager(config, default_provider) if config.agents.modes.enabled else default_provider


def provider_capabilities(config):
    return {'model_modes_enabled': config.agents.modes.enabled,
        'configured_modes': list(config.agents.modes.models),
        'image_vision': 'multimodal' in config.agents.modes.models and config.agents.modes.enabled,
        'image_display': config.agents.modes.enabled,
        'image_generation': {'configured': bool(os.environ.get('DASHSCOPE_API_KEY')),
            'missing': [] if os.environ.get('DASHSCOPE_API_KEY') else ['DASHSCOPE_API_KEY'],
            'service': 'Official ImageTool / DashScope Qwen-Image', 'live_verified': False}}


def active_model(config):
    """Report the configured route used by AgentLoop, including named modes."""
    modes = config.agents.modes
    if modes.enabled:
        if modes.default_mode == 'auto':
            return 'auto (' + modes.models['main'].model + ')'
        return modes.models[modes.default_mode].model
    return config.agents.defaults.model
