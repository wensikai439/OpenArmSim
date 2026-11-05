"""Bounded API client. Secrets stay in the OS vault, never in run artifacts."""
import base64
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from harness.vault import SecretStore


def clean(value):
    if isinstance(value, str):
        return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}', '[REDACTED]', value)
    if isinstance(value, dict):
        if value.get('type') == 'image_url':
            url = value.get('image_url', {}).get('url', '')
            return {'type': 'image_url', 'payload_sha256': hashlib.sha256(url.encode()).hexdigest()}
        return {k: clean(v) for k, v in value.items()
                if k not in ('api_key', 'authorization', 'reasoning_content', 'thinking')}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


class ModelClient:
    def __init__(self, folder, *, request_limit=24, token_limit=60000):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.settings = SecretStore(namespace='openarm-aspire').load()
        if not self.settings or not self.settings.get('api_key'):
            raise RuntimeError('ASPIRE API is not configured in the system vault')
        self.requests = 0
        self.tokens = 0
        self.request_limit, self.token_limit = request_limit, token_limit
        self.usage = []
        previous = self.folder / 'usage.json'
        if previous.exists():
            saved = json.loads(previous.read_text())
            self.requests, self.tokens = saved['requests'], saved['total_tokens']
            self.usage = saved['usage']

    def snapshot(self):
        return {'model': self.settings['model'], 'base_url': self.settings['base_url'],
                'requests': self.requests, 'total_tokens': self.tokens,
                'limits': {'requests': self.request_limit, 'tokens': self.token_limit},
                'usage': self.usage}

    def ask(self, messages, *, tool_name, description, schema, max_tokens=3600):
        if self.requests >= self.request_limit or self.tokens >= self.token_limit:
            raise RuntimeError('Declared API budget exhausted')
        self.requests += 1
        prefix = self.folder / f'{self.requests:02d}'
        payload = {'model': self.settings['model'], 'messages': messages,
                   'tools': [{'type': 'function', 'function': {'name': tool_name,
                              'description': description, 'parameters': schema}}],
                   'tool_choice': {'type': 'function', 'function': {'name': tool_name}},
                   'max_tokens': max_tokens}
        prefix.with_suffix('.request.json').write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2)+'\n')
        request = urllib.request.Request(self.settings['base_url'].rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode(), headers={
                'Authorization': 'Bearer ' + self.settings['api_key'],
                'Content-Type': 'application/json', 'User-Agent': 'OpenArm-ASPIRE/1.0'})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read(4 * 1024 * 1024))
        except urllib.error.HTTPError as exc:
            # Headers and raw provider errors can contain credentials. Retain status only.
            prefix.with_suffix('.error.json').write_text(json.dumps({'http': exc.code})+'\n')
            raise RuntimeError(f'Model request failed: HTTP {exc.code}') from None
        prefix.with_suffix('.response.json').write_text(json.dumps(clean(data), ensure_ascii=False, indent=2)+'\n')
        usage = data.get('usage', {})
        self.tokens += int(usage.get('total_tokens', 0))
        self.usage.append({'request': self.requests, 'seconds': time.monotonic()-start,
                           'response_model': data.get('model'), 'tokens': usage})
        (self.folder / 'usage.json').write_text(json.dumps(self.snapshot(), indent=2)+'\n')
        choice = data['choices'][0]
        if choice.get('finish_reason') in ('length', 'content_filter'):
            raise RuntimeError('Incomplete model response; candidate was not executed')
        calls = choice['message'].get('tool_calls', [])
        matches = [c for c in calls if c.get('function', {}).get('name') == tool_name]
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one structured model submission')
        return json.loads(matches[0]['function']['arguments'])


def image_part(path):
    path = Path(path)
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    return {'type': 'image_url', 'image_url': {'url':
        'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()}}
