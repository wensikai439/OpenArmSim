"""一键抓瓶：./run grab [任务]。

Preflight -> typed on-site confirmation -> device service -> Haiku agent with
--execute -> service shutdown (arms hold). Nothing starts if preflight fails.
"""
import json
import queue
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

try:
    from .service import preflight
except ImportError:
    from service import preflight

ROOT=Path(__file__).resolve().parents[1]
TASK='右手抓瓶子下半瓶身，交接给左手，然后放进篮子'


def wait_service(process,out,timeout=90.):
    """Bound startup wait even if a child stays alive without printing."""
    lines=queue.Queue();tail=[]
    def drain():
        for line in process.stdout:
            out.write(line);out.flush();lines.put(line)
        lines.put(None)
    threading.Thread(target=drain,daemon=True).start()
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:line=lines.get(timeout=min(.2,max(.001,deadline-time.monotonic())))
        except queue.Empty:continue
        if line is None:return None,tail
        tail=(tail+[line.rstrip()])[-15:]
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and row.get('service') and row.get('devices_started') is True:
            return row,tail
    return None,tail+['设备启动超过等待上限；没有启动 Haiku 动作任务']


def main(argv=None,*,ask=input):
    task=' '.join(sys.argv[1:] if argv is None else argv).strip() or TASK
    ready=preflight(ROOT)
    if ready['missing']:
        print('还不能在真机上运行，缺这些：')
        for group in ready['setup_groups'].values():
            print('  · %s：%d 项'%(group['name'],len(group['missing'])))
        print('运行 ./run calib 查看标定进度和下一步。');return 2
    print('任务：'+task,'⚠ 机械臂将会运动：有人在旁边、急停在手边、工作区没有人。按 Ctrl+C 随时停止（原地保持）。',sep='\n')
    if ask('确认开始请输入 y：').strip().lower()!='y':print('已取消，没有打开设备');return 1
    log=ROOT/'.runtime/logs'/('grab-'+time.strftime('%Y%m%d-%H%M%S')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    service=subprocess.Popen([sys.executable,'-B',str(ROOT/'app/service.py'),'service','--devices','--confirm-on-site'],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    out=log.open('w');tail=[]
    try:
        print('正在打开设备、初始化到 ACT 初始姿态…')
        started,tail=wait_service(service,out)
        if not started:
            print('设备服务没有启动成功，最后的输出：',*tail,'完整日志：'+str(log),sep='\n');return 2
        print('设备就绪，交给 Haiku 执行…')
        r=subprocess.run([sys.executable,'-B',str(ROOT/'app/robot.py'),'chat',task,'--execute'],cwd=ROOT,capture_output=True,text=True)
        out.write(r.stdout+r.stderr);out.flush()
        try:result=json.loads(r.stdout)
        except ValueError:print(r.stdout[-2000:],r.stderr[-2000:],sep='\n');return 2
        print('','Haiku：'+str(result.get('reply') or result.get('reason') or result.get('status')),sep='\n')
        for row in result.get('tools',[]):
            value=row['result']
            if row['tool']=='run_bottle_skill':
                print('技能结果：%s；任务成功判定：%s'%(value.get('status'),{True:'✓ 成功（有篮内支撑与松手证据）',None:'未确认'}.get(value.get('task_success'),'未确认')))
                if value.get('error'):print('原因：'+str(value['error']))
        print('完整记录在 records/ 最新的 *-chat-* 目录；服务日志：'+str(log))
        return 0 if r.returncode==0 else 2
    except KeyboardInterrupt:print('\n已停止');return 1
    finally:
        if service.poll() is None:
            service.send_signal(signal.SIGTERM)
            try:service.wait(30)
            except subprocess.TimeoutExpired:service.kill()
        out.close()


if __name__=='__main__':raise SystemExit(main())
