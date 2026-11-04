"""Exercise the existing native PREVIEW Worker; never an actual arm execution."""
import os
from pathlib import Path
import json
import sys
import time
import subprocess
ROOT=Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'bundles/piper-real-runtime/profiles/runtime'))
import native_replay_bridge as replay
out=Path(json.loads((ROOT/'runs/four-can-native-current.json').read_text())['run'])
native=replay.native_module()
trajectory=native.Trajectory(out/'right-j1-0p5deg-candidate.json')
worker=native.Worker(trajectory,out/'native-preview-supervisor',preview=True,managed=True,prepared_can=True)
result={'scope':'original native supervisor and PTY PREVIEW in private network namespace without CAN devices',
        'preview':True,'physical_execution_attempted':False,'states':[]}
def counters():
    rows=json.loads(subprocess.check_output(['/usr/sbin/ip','-j','-s','link','show','type','can'],text=True))
    return {row['ifname']:row.get('stats64',row.get('stats',{})).get('tx',{}).get('packets') for row in rows}
def wait_state(wanted):
    until=time.monotonic()+5
    while time.monotonic()<until:
        state=worker.status()
        if state['state']==wanted:
            result['states'].append({'state':wanted,'pid':state['pid'],'cursor_s':state['cursor_s']});return
        if state['error']:raise RuntimeError(state['error'])
        time.sleep(.05)
    raise RuntimeError('Native preview did not reach '+wanted+': '+str(worker.status()))
result['tx_before']=counters()
assert not result['tx_before'], 'This verification must run in a private network namespace without CAN devices'
try:
    real_shape=native.Worker(trajectory,out/'not-started-real-worker',preview=False,managed=True,prepared_can=True)
    result['real_entry_command_not_executed']=real_shape.build_command(False)
    assert '--prepared-can' in result['real_entry_command_not_executed']
    assert '--preview' not in result['real_entry_command_not_executed']
    worker.start(False);wait_state('AUTO')
    worker.command('pause',video=False);wait_state('HOLD')
    worker.command('takeover',video=False);wait_state('TELEOP')
    worker.command('play',video=False);wait_state('AUTO')
finally:
    worker.stop()
    result['final']=worker.status();result['tx_after']=counters()
    result['tx_unchanged']=result['tx_before']==result['tx_after']
    (out/'native-preview-supervisor.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
assert result['tx_unchanged']
assert result['final']['exit_code']==0 and result['final']['running'] is False
print(json.dumps({'evidence':str(out/'native-preview-supervisor.json'),'states':result['states'],
 'exit_code':result['final']['exit_code'],'tx_unchanged':True,'physical_execution_attempted':False},indent=2))
