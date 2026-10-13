"""On-demand, explicitly uncalibrated contact workbench; main UI is untouched."""
import argparse
import copy
import json
import os
import sqlite3
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

os.environ.setdefault('MUJOCO_GL','egl')
from tools.contact_candidate import ContactCandidate,DERIVATION
from harness.composite import install_isolated
from harness.candidate import CandidateWorkspace,EXECUTION_CONTRACT,install_isolated as install_candidate
from harness.scene import SceneHarness,install_isolated as install_scene
from sim.scene import SceneDirectory
from sim.grasp import CONTRACT as GRASP_CONTRACT,install_isolated as install_grasp
from sim.environment import ROOT
from sim.learning import LearningJobs


class ExperimentArchive:
    """Readable archives; training still belongs to the original workbench."""
    busy=False
    def __init__(self):self.reader=LearningJobs()
    def status(self,job_id=None):return self.reader.status(job_id) if job_id else None
    def catalogue(self):return self.reader.catalogue()
    def evidence(self,run_id):return self.reader.evidence(run_id)
    def start(self,*args,**kwargs):raise ValueError('候选接触工作台仅验证执行；训练与参数辨识请在主平台进行')
    def stop(self,*args,**kwargs):raise ValueError('本工作台没有训练任务；请在启动训练的主平台处理')
    def close(self):self.reader.close()


class WorkbenchHarness(SceneHarness):
    def chat(self,prompt,*,role='execute',**kwargs):
        if role!='execute':raise ValueError('候选接触工作台仅验证执行；实验与训练请使用主平台')
        return super().chat(prompt,role=role,**kwargs)
    def _invoke(self,name,args,role,context=None):
        if role!='execute':raise ValueError('候选接触工作台仅验证执行；实验与训练请使用主平台')
        return super()._invoke(name,args,role,context)


def main():
    parser=argparse.ArgumentParser(description='独立接触候选 · 原生仿真 · 尚未物理标定')
    parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    if not 1024<=args.port<=65535 or args.port==8765:raise ValueError('use a separate local port, default 8766')
    os.environ.setdefault('OPENARM_POLICY_CONFIG',str(ROOT/'harness/policy.json'))
    install_isolated()
    install_grasp()
    install_candidate()
    install_scene()
    import sim.service as service
    service.OpenArmEnv=ContactCandidate
    import server as web
    web.PORT=args.port

    class CandidateService(service.SimulationService):
        def _publish(self):
            super()._publish()
            if not hasattr(self,'directory'):self.directory=SceneDirectory(self.env)
            with self.lock:
                self.cache['diagnostics']['scene_directory']=self.directory.snapshot(self.cache['observation_id'])
                self.cache['simulation_variant']=copy.deepcopy(DERIVATION)
                self.cache['grasp_contract']=copy.deepcopy(GRASP_CONTRACT)
                self.cache['execution_contract']=copy.deepcopy(EXECUTION_CONTRACT)
                self.cache['policy_configured']=False
                self.cache['policy_status']={'ready':False,'reason':'contact candidate; no matching validated learned-policy evaluation contract'}
                self.cache['calibration_gaps']=[*self.cache['calibration_gaps'],'候选接触：有限接触面与 NoSlip 尚未独立物理标定']
        def call(self,command,**arguments):
            if command=='policy':raise ValueError('候选接触模型尚无经过验证的学习策略评测契约；请在主平台使用 VLA')
            return super().call(command,**arguments)

    class CandidateHandler(web.Handler):
        def _file(self,path,root):
            if path.resolve()==(ROOT/'ui/index.html').resolve():
                content=path.read_text().replace('<title>OpenArm Lab</title>','<title>OpenArm Lab · 接触候选</title>')
                content=content.replace('具身智能工作台','接触候选 · 尚未标定')
                content=content.replace('仅仿真 · 真机未连接','独立候选 · 真机未连接')
                content=content.replace('<option value="research">实验与训练</option>','')
                return self._send(200,content.encode(),'text/html; charset=utf-8')
            return super()._file(path,root)
        def _json(self,status,value):
            if urlsplit(self.path).path=='/api/calibration' and status==200:
                value={**value,'contact_candidate':copy.deepcopy(DERIVATION)}
            return super()._json(status,value)

    server=ThreadingHTTPServer(('127.0.0.1',args.port),CandidateHandler)
    try:
        server.simulation=CandidateService()
        server.learning=ExperimentArchive()
        path=ROOT/'harness/state/contact.sqlite3';existed=path.exists()
        workspace=CandidateWorkspace(path)
        if not existed:
            # Import user preferences once, not current tasks or old physics
            # receipts. Later edits/forgetting in this journal stay persistent.
            with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as source:
                columns=[r[1] for r in source.execute('PRAGMA table_info(notes)')]
                rows=source.execute("SELECT * FROM notes WHERE scope='preference' AND invalidated=0").fetchall()
                with workspace.db:
                    if rows:workspace.db.executemany('INSERT INTO notes ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',rows)
        server.harness=WorkbenchHarness(server.simulation,server.learning,
            settings=web.ModelSettings(store=web.SecretStore()),workspace=workspace)
        from harness.provider import install_isolated as install_provider
        install_provider(server.harness)
        print(json.dumps({'url':f'http://127.0.0.1:{args.port}/','variant':'contact candidate',
            'physical_calibration':False,'main_server_changed':False,'automatic_start':False},ensure_ascii=False),flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if hasattr(server,'harness'):server.harness.close();server.harness.workspace.db.close()
        if hasattr(server,'learning'):server.learning.close()
        if hasattr(server,'simulation'):server.simulation.close()
        server.server_close()


if __name__=='__main__':main()
