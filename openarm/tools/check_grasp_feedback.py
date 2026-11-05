"""Native preparation followed by one real-API read-only quality review."""
import json
import time

from sim.environment import ROOT
from sim.recording import dump
from tools.check_api import http


def main():
    out=ROOT/'runs/20261008-hold';target=out/'api-review.json'
    if target.exists():raise ValueError('preserve prior review')
    s=http('/api/status');assert not s['job'] and not s['agent']['busy']
    prompt='检查左右手目前的夹持质量，区分接触抬升和稳定程度。只读，不调用任何运动、不重置、不修改计划；不能把两指接触视为长期稳定。'
    protocol={'prompt':prompt,'native_preparation':'one ordinary right-arm pick, source guarded; not an LLM motion task',
        'API':'one real model review of live feedback, no fake backend','hardware_access':False,'transfer_ready':False}
    dump(out/'api-review.protocol.json',protocol)
    http('/api/tools',{'name':'observe','arguments':{}})
    obs=s['observation_id']
    http('/api/tools',{'name':'inspect_manipulation','arguments':{'observation_id':obs}})
    response=http('/api/tools',{'name':'pick','arguments':{'observation_id':obs,'object_id':'bottle','arm':'right'}})
    dump(out/'native-preparation.json',response)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        s=http('/api/status')
        if not s['job']:break
        time.sleep(.5)
    else:raise TimeoutError('preparation still live; inspect same job, do not reset')
    dump(out/'before-review.json',s)
    response=http('/api/chat',{'message':prompt});root=response['turn_id'];dump(out/'review-start.json',response)
    print(json.dumps({'turn_id':root,'started':True}),flush=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        s=http('/api/status');a=http('/api/agent');turns=[t for t in a['turns'] if t['id']==root or t.get('parent')==root]
        if not s['job'] and not s['agent']['busy'] and turns and all(t['status'] not in ('running','waiting') for t in turns):break
        time.sleep(.5)
    else:raise TimeoutError('review still live; do not reset or start a second request')
    events=[e for t in turns for e in t['trace'] if e['kind']=='tool']
    inspections=[e['detail']['result'] for e in events if e['name']=='inspect_manipulation']
    forbidden=[e['name'] for e in events if e['name'] in ('pick','place','handover','relocate','move_relative','move_to_pose','set_gripper','run_policy','run_tamp','run_pi05_vla','prepare_simulation','set_plan')]
    quality=inspections[-1].get('grasp_quality') if inspections else None
    initial=json.loads((out/'before-review.json').read_text())
    unchanged=(s['observation_id']==initial['observation_id'] and s['observation']['state']==initial['observation']['state'])
    report={**protocol,'status':'passed' if quality and not forbidden and unchanged else 'failed',
        'turn_id':root,'turns':turns,'terminal':s,'actual_api_calls':sum(e['kind']=='api' for t in turns for e in t['trace']),
        'quality_delivered_to_model':quality,'forbidden_actions':forbidden,'physical_state_unchanged_during_review':unchanged,
        'scope':'tool-delivery and read-only behavior; not an independent model-judgment benchmark or stable-grasp proof'}
    dump(target,report);print(json.dumps({k:report[k] for k in ('status','actual_api_calls','forbidden_actions','physical_state_unchanged_during_review')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
