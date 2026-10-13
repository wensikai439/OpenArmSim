"""Process-local provider adapter for the experimental bounded tool loop.

DeepSeek's current default thinking mode needs reasoning replay. This harness
does not retain private reasoning, so use its documented non-thinking mode.
No new endpoint, key storage, token budget, retry or robot command is added.
"""
import json
import urllib.request
import urllib.error
from types import SimpleNamespace
from urllib.parse import urlsplit

DOCUMENTATION='https://api-docs.deepseek.com/guides/thinking_mode/'


def adapt_request(request):
    url=urlsplit(request.full_url)
    if (url.scheme!='https' or url.hostname!='api.deepseek.com'
            or not url.path.rstrip('/').endswith('/chat/completions') or request.data is None):return request,None
    payload=json.loads(request.data)
    if payload.get('model') not in ('deepseek-flash','deepseek-v4-pro'):return request,None
    payload['thinking']={'type':'disabled'}
    payload.pop('reasoning_effort',None)
    adapted=urllib.request.Request(request.full_url,data=json.dumps(payload).encode(),
        headers=dict(request.header_items()),method=request.get_method())
    return adapted,{'provider':'api.deepseek.com','thinking':'disabled','output_token_cap':payload.get('max_tokens'),
        'reason':'bounded tool loop does not persist private reasoning','source':DOCUMENTATION}


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
    def __init__(self,notify):self.notify=notify
    def __getattr__(self,name):return getattr(urllib.request,name)
    def urlopen(self,request,*args,**kwargs):
        if not isinstance(request,urllib.request.Request) or request.data is None:
            return urllib.request.urlopen(request,*args,**kwargs)
        adapted,adapter=adapt_request(request)
        response=urllib.request.urlopen(adapted,*args,**kwargs)
        return CheckedResponse(response,self.notify,adapter)


def install_isolated(harness):
    import harness.agent as agent
    def notify(detail):
        root=harness.active_session
        if root:
            harness.workspace.event(root,'protocol','API 响应检查',detail,
                'completed' if detail['response_accepted'] else 'error')
    # Replace this module's urllib binding only. The standard library, other
    # processes, the main server and ModelSettings' credential flow stay intact.
    agent.urllib=SimpleNamespace(request=CheckedRequests(notify),error=urllib.error)
