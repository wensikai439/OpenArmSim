"""One real-API forward-placement case under the new native timing contract."""
import json

from sim.environment import ROOT
from sim.recording import dump
from .check_relocation import api


def main():
    out=ROOT/'runs/20261008-planning/api.json'
    api('forward',out)
    result=json.loads(out.read_text())
    native=[e for e in result['feedbacks'] if e['execution']['skill']=='relocate']
    planning=[e.get('feedback',{}).get('planning_timing',{}) for e in native]
    wired=bool(planning and all(p.get('schema')=='planning-timing/1' and p.get('mode')=='measured_delay'
                              and p.get('hold_transitions',0)>0 for p in planning))
    result['planning_contract_verified']=wired
    result['description']='真实模型接收右手前放8厘米任务，首个计划锁定measured_delay／120秒；模型请求等待与底层规划等待均计入，目标和原生评分分别核对。'
    if not wired:result['status']='failed'
    dump(out,result)
    print(json.dumps({'status':result['status'],'actual_api_calls':result['actual_api_calls'],
        'planning_contract_verified':wired},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
