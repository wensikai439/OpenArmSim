"""Gateway packet preservation and error disclosure boundaries; no API calls."""
import io
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from .llm import ModelSettings
from .provider import adapt_request, http_error_reason


class ProviderChecks(unittest.TestCase):
    def request(self,url='https://zzq.si/v1/chat/completions',model='claude-haiku-5-5'):
        return urllib.request.Request(url,data=json.dumps({'model':model,'temperature':0,
            'messages':[{'role':'user','content':[{'type':'text','text':'pick'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,fixture'}}]}],
            'tools':[{'type':'function','function':{'name':'observe','parameters':{'type':'object'}}}],
            'parallel_tool_calls':False,'max_tokens':1600}).encode(),
            headers={'Authorization':'Bearer fixture-key','Content-Type':'application/json'})

    def test_haiku_only_removes_rejected_temperature(self):
        original=self.request();before=json.loads(original.data)
        adapted,meta=adapt_request(original);after=json.loads(adapted.data)
        self.assertEqual(after,{key:value for key,value in before.items() if key!='temperature'})
        self.assertEqual(json.loads(original.data),before)
        self.assertEqual(adapted.full_url,original.full_url)
        self.assertEqual(adapted.header_items(),original.header_items())
        self.assertEqual(adapted.get_method(),original.get_method())
        self.assertEqual(meta['temperature'],'provider_default')
        repeated,meta=adapt_request(adapted)
        self.assertIs(repeated,adapted);self.assertIsNone(meta)

    def test_other_hosts_models_and_endpoints_are_not_changed(self):
        for url,model in [('https://other.example/v1/chat/completions','claude-haiku-5-5'),
                          ('https://zzq.si/v1/chat/completions','other-model'),
                          ('https://zzq.si/v1/responses','claude-haiku-5-5')]:
            request=self.request(url,model)
            with self.subTest(url=url,model=model):
                adapted,meta=adapt_request(request)
                self.assertIs(adapted,request);self.assertIsNone(meta)
        request=self.request('https://api.deepseek.com/v1/chat/completions','deepseek-flash')
        adapted,_=adapt_request(request);payload=json.loads(adapted.data)
        self.assertEqual(payload['thinking'],{'type':'disabled'})
        self.assertEqual(payload['temperature'],0)

    def error(self,body,code=400):
        return urllib.error.HTTPError('https://zzq.si/v1/chat/completions',code,'error',{},io.BytesIO(body))

    def test_only_bounded_json_message_is_disclosed_and_cached(self):
        error=self.error(json.dumps({'error':{'message':'model does not support non-default temperature'},
            'request':{'secret':'must-not-escape'}}).encode())
        reason=http_error_reason(error)
        self.assertIn('non-default temperature',reason);self.assertNotIn('must-not-escape',reason)
        self.assertEqual(http_error_reason(error),reason)
        for body in (b'<html>secret traceback</html>',b'{"message":"private request body"}',b'[]'):
            self.assertEqual(http_error_reason(self.error(body)),'HTTP 400')
        self.assertLess(len(http_error_reason(self.error(json.dumps({'error':{'message':'x'*1000}}).encode()))),260)

    def test_key_and_bearer_values_are_redacted_in_test_errors(self):
        settings=ModelSettings()
        settings.configure({'base_url':'https://zzq.si/v1','model':'claude-haiku-5-5',
            'api_key':'arbitrary-private-key','vision':True})
        body={'error':{'message':'invalid arbitrary-private-key; Bearer another-token; sk-placeholder123; api_key=hidden-value'}}
        with patch('urllib.request.urlopen',side_effect=self.error(json.dumps(body).encode())):
            with self.assertRaises(RuntimeError) as caught:settings.test()
        message=str(caught.exception)
        for secret in ('arbitrary-private-key','another-token','sk-placeholder123','hidden-value'):
            self.assertNotIn(secret,message)
        self.assertIn('HTTP 400',message)


if __name__=='__main__':unittest.main()
