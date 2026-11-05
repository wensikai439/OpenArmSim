"""Loopback transport for real decisions made in the active Codex conversation.
It never invents a reply or verdict. Original Forge and model schemas remain responsible for validation.
"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse,json,time,uuid,hashlib,os

def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
 os.replace(tmp,path)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--exchange',type=Path,required=True)
 ap.add_argument('--launch-file',type=Path,required=True)
 a=ap.parse_args();exchange=a.exchange.resolve();exchange.mkdir(exist_ok=True)
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send_json(self,status,value):
   raw=json.dumps(value,ensure_ascii=False).encode()
   self.send_response(status);self.send_header('Content-Type','application/json')
   self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  def do_GET(self):
   if self.path=='/healthz':self.send_json(200,{'reviewer':'Codex conversation','automatic_model_replies':False})
   else:self.send_json(404,{'error':'Unknown endpoint'})
  def do_POST(self):
   if self.path!='/v1/chat/completions':return self.send_json(404,{'error':'Unknown endpoint'})
   raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
   try:body=json.loads(raw)
   except Exception:return self.send_json(400,{'error':'Invalid JSON'})
   if body.get('model')!='custom/codex-session-reviewer' or body.get('tools'):
    return self.send_json(400,{'error':'Only current-conversation review without tool delegation is supported'})
   request_id=uuid.uuid4().hex;folder=exchange/request_id;folder.mkdir()
   (folder/'request.json').write_bytes(raw)
   write(folder/'pending.json',{'request_id':request_id,'request_sha256':digest(raw),'created_unix':time.time(),
    'reviewer':'Codex conversation','source':'actual official provider request','automatic_response':False,'wait_seconds':3600})
   deadline=time.monotonic()+3600
   while time.monotonic()<deadline:
    reply=folder/'response.json'
    if reply.exists():
     try:
      v=json.loads(reply.read_text())
      if v['request_id']!=request_id or v['request_sha256']!=digest(raw) or v['reviewer']!='Codex conversation':
       raise ValueError('Reply provenance does not match this actual request')
      if not isinstance(v['content'],dict):raise ValueError('Reply must contain the actual structured Codex decision')
      completion={'id':'chatcmpl-'+request_id,'object':'chat.completion','created':int(time.time()),
       'model':'custom/codex-session-reviewer','usage':None,
       'choices':[{'index':0,'message':{'role':'assistant','content':json.dumps(v['content'],ensure_ascii=False)},'finish_reason':'stop'}]}
      self.send_json(200,completion)
      write(folder/'delivered.json',{'at_unix':time.time(),'response_sha256':digest(reply.read_bytes()),
       'reviewer':'Codex conversation','provider_token_usage_available':False})
      return
     except (KeyError,ValueError,json.JSONDecodeError) as exc:
      write(folder/'rejected-response.json',{'error':str(exc),'not_an_official_verdict':True})
      return self.send_json(409,{'error':'Codex response provenance invalid; no substitute generated'})
    time.sleep(.25)
   write(folder/'expired.json',{'at_unix':time.time(),'actual_decision_received':False})
   self.send_json(409,{'error':'Codex decision missing; no automatic substitute or paid fallback'})
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 write(a.launch_file,{'pid':os.getpid(),'port':server.server_port,'api_base':f'http://127.0.0.1:{server.server_port}/v1',
  'exchange':str(exchange),'external_provider_requests':0,'reviewer':'Codex conversation'})
 server.serve_forever()
if __name__=='__main__':main()
