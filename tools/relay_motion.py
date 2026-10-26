"""Compare actual relay motion from timestamped encoders, not rendered video."""
import argparse
import json
from pathlib import Path

import numpy as np

ARMS=np.r_[0:7,8:15]


def stats(values):
    return {name:float(value) for name,value in {
        'p95':np.percentile(abs(values),95),'p99':np.percentile(abs(values),99),
        'max':np.max(abs(values)),'rms':np.sqrt(np.mean(values*values))}.items()}


def measure(folder):
    rows=json.loads((folder/'controller.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    time=np.array([r['time_s'] for r in rows])
    phases=np.array([r['phase'] for r in rows]);measured=np.array([r['measured'] for r in rows])[:,ARMS]
    out={'success':result['full_task_success'],'seconds':result['sim_seconds'],
         'samples':len(rows),'actual':{},'phases':{},'nominal':{},'applied':{}}
    for key,data in (('actual',measured),('nominal',np.array([r['requested'] for r in rows])[:,ARMS]),
                     ('applied',np.array([r['applied'] for r in rows])[:,ARMS])):
        t=time.copy()
        for order,name in enumerate(('velocity_rad_s','acceleration_rad_s2','jerk_rad_s3'),1):
            data=np.diff(data,axis=0)/np.diff(t)[:,None];t=(t[1:]+t[:-1])/2
            out[key][name]=stats(data)
            if key=='actual':
                for phase in dict.fromkeys(phases):
                    # Exclude cross-phase differences from per-phase summaries.
                    ids=np.flatnonzero(np.logical_and.reduce([phases[k:len(phases)-order+k]==phase for k in range(order+1)]))
                    if not len(ids):continue
                    out['phases'].setdefault(phase,{})[name]=stats(data[ids])
    # Last second of every held section: actual vertical motion, not reference.
    holds=[]
    for end in np.flatnonzero((phases=='hold') & np.r_[phases[1:]!='hold',True]):
        start=end
        while start>0 and phases[start-1]=='hold' and time[end]-time[start-1]<=1.:start-=1
        if time[end]-time[start]<.8:continue
        holds.append({'end_s':float(time[end]),'duration_s':float(time[end]-time[start]),
            'vertical_range_mm':{side:float(np.ptp([r['tcp'][side][2] for r in rows[start:end+1]])*1000) for side in ('left','right')}})
    out['steady_holds']=holds
    return out


def compare(baseline,candidate):
    before,after=measure(baseline),measure(candidate)
    return {'baseline':before,'candidate':after,'change_percent':{
        key:{metric:100*(after['actual'][key][metric]/before['actual'][key][metric]-1) for metric in ('p95','p99','max','rms')}
        for key in before['actual']},
        'method':'14 measured arm joints; finite differences with actual timestamps and midpoint time grids; gripper motors excluded; contact effects included; raw unfiltered samples',
        'scope':'simulation only; derivative estimates at native 30Hz, not certified hardware jerk bounds'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=compare(a.baseline,a.candidate)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'before_s':report['baseline']['seconds'],'after_s':report['candidate']['seconds'],
        'success':report['candidate']['success'],'change_percent':report['change_percent']},ensure_ascii=False))


if __name__=='__main__':main()
