"""Instance-scoped provider compatibility and bounded, redacted API errors."""
import json
import re
import urllib.request
import urllib.error
from urllib.parse import urlsplit

DOCUMENTATION='https://api-docs.deepseek.com/guides/thinking_mode/'


def adapt_request(request):
    url=urlsplit(request.full_url)
    if (url.scheme!='https' or url.hostname not in ('api.deepseek.com','zzq.si')
            or not url.path.rstrip('/').endswith('/chat/completions') or request.data is None):return request,None
    payload=json.loads(request.data)
    if url.hostname=='zzq.si':
        # Verified gateway rejection: this model only accepts its default
        # temperature. Keep images, tools, history and all budgets unchanged.
        if payload.get('model')!='claude-haiku-5-5' or 'temperature' not in payload:return request,None
        del payload['temperature']
        adapted=urllib.request.Request(request.full_url,data=json.dumps(payload).encode(),
            headers=dict(request.header_items()),method=request.get_method())
        return adapted,{'provider':'zzq.si','temperature':'provider_default',
            'reason':'claude-haiku-5-5 rejects non-default temperature',
            'source':'verified gateway HTTP 400 response'}
    if payload.get('model') not in ('deepseek-flash','deepseek-v4-pro'):return request,None
    payload['thinking']={'type':'disabled'}
    payload.pop('reasoning_effort',None)
    adapted=urllib.request.Request(request.full_url,data=json.dumps(payload).encode(),
        headers=dict(request.header_items()),method=request.get_method())
    return adapted,{'provider':'api.deepseek.com','thinking':'disabled','output_token_cap':payload.get('max_tokens'),
        'reason':'bounded tool loop does not persist private reasoning','source':DOCUMENTATION}


def http_error_reason(error, api_key=''):
    """Read only a gateway's JSON error message, never headers or raw bodies.

    Cache the safe text because both the request event and its caller report
    the same HTTPError, whose response stream can only be consumed once.
    """
    cached=getattr(error,'_openarm_reason',None)
    if cached is not None:return cached
    reason=f'HTTP {error.code}'
    try:
        body=json.loads(error.read(64*1024))
        detail=body.get('error') if isinstance(body,dict) else None
        message=detail.get('message') if isinstance(detail,dict) else None
        if isinstance(message,str) and message.strip():
            if api_key:message=message.replace(api_key,'[credential omitted]')
            message=re.sub(r'(?i)(?:sk-|bearer\s+)[a-z0-9_.-]+','[credential omitted]',message)
            message=re.sub(r'(?i)\b(api[_ -]?key|authorization|access[_ -]?token)\s*[:=]\s*[^\s,;]+',
                           r'\1=[credential omitted]',message)
            message=' '.join(message.split())
            reason+='：'+message[:240]
    except Exception:
        pass
    error._openarm_reason=reason
    return reason


def response_metadata(body):
    data=json.loads(body);choice=data['choices'][0];message=choice['message']
    reason=choice.get('finish_reason');tools=message.get('tool_calls') or []
    content=message.get('content')
    complete=reason not in ('length','content_filter','insufficient_system_resource','aborted')
    nonempty=bool(tools or isinstance(content,str) and content.strip())
    return {'finish_reason':reason,'has_tools':bool(tools),'has_content':bool(isinstance(content,str) and content.strip()),
        'response_accepted':bool(complete and nonempty),
        'scope':'response usability only; never establishes an action or task success'}


class CheckedResponse:
    def __init__(self,response,notify,adapter):self.response=response;self.notify=notify;self.adapter=adapter
    def __enter__(self):self.response.__enter__();return self
    def __exit__(self,*args):return self.response.__exit__(*args)
    def read(self,limit):
        body=self.response.read(limit);meta=response_metadata(body)
        self.notify({**(self.adapter or {}),**meta})
        if not meta['response_accepted']:raise ValueError('provider returned an empty or incomplete response; no tools dispatched')
        return body


class CheckedRequests:
    def __init__(self,base,notify,owner):self.base,self.notify,self.provider_owner=base,notify,owner
    def __getattr__(self,name):return getattr(self.base,name)
    def urlopen(self,request,*args,**kwargs):
        if not isinstance(request,urllib.request.Request) or request.data is None:
            return self.base.urlopen(request,*args,**kwargs)
        adapted,adapter=adapt_request(request)
        response=self.base.urlopen(adapted,*args,**kwargs)
        return CheckedResponse(response,self.notify,adapter)


def install_isolated(harness):
    def notify(detail):
        root=harness.active_session
        if root:
            harness.workspace.event(root,'protocol','API 响应检查',detail,
                'completed' if detail['response_accepted'] else 'error')
    # Scope interception to this Harness instance; the standard library and
    # other Harness instances retain their own unwrapped transport.
    if getattr(harness.request_transport,'provider_owner',None) is not harness:
        harness.request_transport=CheckedRequests(harness.request_transport,notify,harness)
