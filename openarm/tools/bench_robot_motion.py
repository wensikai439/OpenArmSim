"""Compare r7/r8 reference timing on declared linear kinematic test doubles.

No physics, collision scene, CAN, camera or API. This is NOT measured robot speed.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import numpy as np

from deploy.evidence import finite
from deploy.motion import retime
from deploy.relative import bounded_target,goal_error
from deploy.relative_checks import Fixture,pose

ROOT=Path(__file__).resolve().parents[1]
BASELINE='3dc89bb10e1beb5645dd41feedeed3add4891ab1'


def run(output):
    old=subprocess.check_output(['git','show',BASELINE+':deploy/motion.py'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='retime')
    namespace={'np':np,'finite':finite}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-r7-retime','exec'),namespace)
    rows=[]
    for distance in [.03,.06,.08,.12]:
        for scale in [1.,2.,4.]:
            start=pose();target=pose(0,0,distance);prior=start.copy();old_targets=[]
            while goal_error(prior,target)[0]>.002:
                prior=bounded_target(prior,target);old_targets.append(prior.copy())
            f=Fixture();f.poses['right']=start.copy()
            for _ in range(32):
                if f._relative_step('right',target):break
            else:raise RuntimeError('reference fixture did not converge')
            new_targets=[r['right'] for r in f.moves]
            metrics=[]
            for points,timing in [(old_targets,namespace['retime']),(new_targets,retime)]:
                prior=0.;trajectory=[np.zeros(16)]*4
                for t in points:
                    q=np.zeros(16);q[8]=prior;goal=q.copy();goal[8]=t[2,3]*scale
                    trajectory.extend(timing(q,goal,velocity=.1,acceleration=.2))
                    # Compare nominal timed references without inventing API,
                    # planning or communication overhead between segments.
                    prior=goal[8]
                trajectory.extend([trajectory[-1]]*4)
                values=np.array(trajectory)
                metrics.append({'planned_moves':len(points),'reference_seconds':(len(values)-8)/30,
                    'sampled_reference_jerk_peak_rad_s3':float(abs(np.diff(values,n=3,axis=0)*30**3).max())})
            rows.append({'distance_m':distance,'joint_scale_rad_per_m':scale,'r7':metrics[0],'r8':metrics[1],
                'reference_time_change_percent':100*(metrics[1]['reference_seconds']/metrics[0]['reference_seconds']-1)})
    data={'schema':'robot-reference-motion-comparison/1','baseline_commit':BASELINE,
        'case_count':len(rows),'velocity_rad_s':.1,'acceleration_rad_s2':.2,'new_jerk_rad_s3':2.,
        'fixture':'linear map q_right_1 = scale * TCP_z; no added API/planning/communication overhead; no physics or collision checking',
        'median_reference_time_change_percent':float(np.median([r['reference_time_change_percent'] for r in rows])),
        'all_reference_moves':{'r7':sum(r['r7']['planned_moves'] for r in rows),'r8':sum(r['r8']['planned_moves'] for r in rows)},
        'slower_cases':[i for i,r in enumerate(rows) if r['reference_time_change_percent']>0],
        'rows':rows,'physical_validation_passed':False,'devices_started':False,'motion_commands':0}
    output=output.resolve()
    if not output.is_relative_to(ROOT/'runs') or output.exists():raise ValueError('new result must be under runs')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ['case_count','median_reference_time_change_percent','all_reference_moves','slower_cases','physical_validation_passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
