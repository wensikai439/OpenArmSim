"""Provider compatibility checks without keys, network or physics commands."""
import json
import unittest
import urllib.request
from harness.provider import adapt_request,response_metadata


class ProviderChecks(unittest.TestCase):
    def test_official_deepseek_adaptation_preserves_recipient_credentials_and_budget(self):
        payload={'model':'deepseek-flash','max_tokens':1600,'messages':[{'role':'user','content':'抓取'}],
            'thinking':{'type':'enabled'},'reasoning_effort':'high','tools':[{'type':'function'}]}
        request=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer test-placeholder','Content-Type':'application/json'})
        adapted,metadata=adapt_request(request);body=json.loads(adapted.data)
        self.assertEqual(body['thinking'],{'type':'disabled'});self.assertNotIn('reasoning_effort',body)
        self.assertEqual(body['max_tokens'],1600);self.assertEqual(body['messages'],payload['messages'])
        self.assertEqual(adapted.full_url,request.full_url);self.assertEqual(adapted.header_items(),request.header_items())
        self.assertNotIn('Authorization',json.dumps(metadata));self.assertEqual(json.loads(request.data),payload)

    def test_other_providers_and_unrecognized_models_are_not_reconfigured(self):
        for url,model in [('https://other.example/chat/completions','deepseek-flash'),
            ('https://api.deepseek.com/chat/completions','other'),('http://api.deepseek.com/chat/completions','deepseek-flash')]:
            request=urllib.request.Request(url,data=json.dumps({'model':model}).encode())
            adapted,metadata=adapt_request(request);self.assertIs(adapted,request);self.assertIsNone(metadata)

    def test_partial_tool_response_and_empty_private_reasoning_never_count_as_usable(self):
        for reason,message in [('length',{'tool_calls':[{'id':'partial'}]}),
            ('stop',{'content':'','reasoning_content':'private test content'}),('aborted',{'content':'success'})]:
            meta=response_metadata(json.dumps({'choices':[{'message':message,'finish_reason':reason}]}))
            self.assertFalse(meta['response_accepted']);self.assertNotIn('private test content',json.dumps(meta))
        meta=response_metadata(json.dumps({'choices':[{'message':{'tool_calls':[{'id':'real'}]},'finish_reason':'tool_calls'}]}))
        self.assertTrue(meta['response_accepted'])


if __name__=='__main__':unittest.main()
