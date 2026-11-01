"""Explicit API editing with optional OS keyring persistence and redacted status."""
import os
import threading
import json
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit
from .provider import http_error_reason


class ModelSettings:
    def __init__(self, store=None):
        self.lock=threading.Lock()
        self.override=None
        self.store=store
        self.storage_error=None
        if store:
            try:self.override=store.load()
            except RuntimeError as exc:self.storage_error=str(exc)

    def snapshot(self):
        with self.lock:
            if self.override is not None:
                return dict(self.override)
        return {'base_url':os.environ.get('OPENARM_LLM_BASE','https://api.deepseek.com/v1'),
                'model':os.environ.get('OPENARM_LLM_MODEL','deepseek-chat'),
                'api_key':os.environ.get('OPENARM_LLM_KEY',''),
                'vision':os.environ.get('OPENARM_LLM_VISION')=='1'}

    def public(self):
        values=self.snapshot()
        return {k:v for k,v in values.items() if k!='api_key'} | {
            'configured':bool(values['api_key']),
            'storage':('system_keyring' if self.store else 'session') if self.override is not None else 'environment',
            'storage_error':self.storage_error}

    def configure(self, values):
        if set(values)-{'base_url','model','api_key','vision'} or not {'base_url','model','vision'}<=set(values):
            raise ValueError('base_url, model and vision are required')
        if not isinstance(values['base_url'],str):
            raise ValueError('API 地址必须是文本')
        current=self.snapshot()
        values=dict(values)
        replacement=values.get('api_key')
        if replacement is not None and not isinstance(replacement,str):
            raise ValueError('api_key must be a string')
        if not replacement or not replacement.strip():
            if values['base_url'].strip().rstrip('/')!=current['base_url'].rstrip('/'):
                raise ValueError('更换 API 地址时请填写对应密钥；现有配置未改变')
            values['api_key']=current['api_key']
        for name in ('base_url','model','api_key'):
            if not isinstance(values[name],str) or not 1<=len(values[name].strip())<=2000:
                raise ValueError('请完整填写 API 地址、模型名和密钥')
            if any(c in values[name] for c in '\r\n'):
                raise ValueError('配置不能含换行')
        parsed=urlsplit(values['base_url'])
        if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
            raise ValueError('API 地址不能包含凭证、查询或片段')
        if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost','::1')):
            raise ValueError('外部 API 使用 HTTPS；HTTP 仅用于本机模型服务')
        if type(values['vision']) is not bool:
            raise ValueError('vision must be boolean')
        clean={k:v.strip() if isinstance(v,str) else v for k,v in values.items()}
        clean['base_url']=clean['base_url'].rstrip('/')
        with self.lock:
            if self.store:self.store.save(clean)
            self.override=clean
            self.storage_error=None
        return self.public()

    def clear(self):
        with self.lock:
            cleared={'base_url':'https://api.deepseek.com/v1','model':'deepseek-chat','api_key':'','vision':False}
            # Persist an empty record so environment fallback cannot silently
            # reactivate a key after the user explicitly deletes it.
            if self.store:self.store.save(cleared)
            self.override=cleared
            self.storage_error=None
        return self.public()

    def test(self):
        values=self.snapshot()
        if not values['api_key']:
            raise ValueError('请先保存模型配置')
        request=urllib.request.Request(values['base_url']+'/chat/completions',
            data=json.dumps({'model':values['model'],'messages':[{'role':'user','content':'Reply only: connected'}],
                             'max_tokens':16}).encode(),
            headers={'Authorization':'Bearer '+values['api_key'],'Content-Type':'application/json'})
        started=time.monotonic()
        try:
            with urllib.request.urlopen(request,timeout=30) as response:
                data=json.loads(response.read(1024*1024))
            if not data.get('choices') or not data['choices'][0].get('message'):
                raise ValueError('invalid response')
        except urllib.error.HTTPError as exc:
            raise RuntimeError('模型连接测试失败（'+http_error_reason(exc,values['api_key'])+'）') from None
        except Exception:
            raise RuntimeError('模型连接测试失败，请检查 API 是否兼容 Chat Completions') from None
        return {'connected':True,'model':values['model'],'latency_seconds':round(time.monotonic()-started,2),
                'scope':'text connectivity only; tool use and vision require task verification'}
