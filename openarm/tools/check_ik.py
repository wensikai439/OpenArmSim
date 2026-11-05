"""Fixed scratch-only old/bounded IK comparison. No main UI/hardware commands."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from sim.environment import ROOT
from sim.recording import dump
from sim.tamp import TampPlanner,PlanningFailure
from sim.ik import solve,seeds
from tools.contact_candidate import ContactCandidate

OUT=ROOT/'runs/20261008-ik'


def main(mode='pre'):
    global OUT
    if mode=='grasp':OUT=OUT/'grasp'
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise ValueError('preserve the frozen IK comparison')
    env=ContactCandidate(images=False);env.reset(seed=0)
    initial_q=env.data.qpos.copy();initial_v=env.data.qvel.copy();cases=[]
    for label,delta in [('original',[0,0,0]),('previous-failed',[.015,-.02,0])]:
        for theta in ('recorded',0,-np.pi/4,np.pi/4,np.pi/2):
            cases.append({'label':label,'delta_m':delta,'height_m':.195,'theta':theta})
    declaration={'schema':'scratch-ik-comparison/1','cases':cases,'seed':0,'episode':0,
        'seeds':'identical initial + three recorded close proposals + seeded near-initial random proposals,12 total',
        'limits':{'position_m':.0015,'rotation_rad':.018},'full_native_collision_guard':True,
        'live_state_edits':False,'hardware_access':False,'main_page_commands':0,
        'scope':'10 predefined pose IK probes, not full grasp/path or a statistical reachability test',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'solver_sha256':hashlib.sha256((ROOT/'sim/ik.py').read_bytes()).hexdigest(),
        'model':env.manifest()}
    dump(OUT/'protocol.json',declaration);rows=[]
    try:
        for index,case in enumerate(cases):
            for backend in ('original','bounded'):
                planner=TampPlanner(env,'pick_bottle',parameters={'arm':'left','object_id':'bottle'})
                planner._arm('left');planner.planning_qpos=env.data.qpos.copy()
                for finger in env.fingers['left']:planner.planning_qpos[finger.qposadr[0]]=max(finger.range)
                addr=env.model.joint('bottle_free').qposadr[0]
                planner.planning_qpos[addr:addr+3]+=case['delta_m']
                axis=env.data.xmat[env.bottle_id].reshape(3,3)[:,2]
                rotation=planner.receiver_rotation.copy() if case['theta']=='recorded' else planner._grasp_frame(axis,case['theta'])
                planner.rotation=rotation;planner._sync()
                bottle=planner.data.xpos[env.bottle_id].copy();target=bottle+np.array([0,0,case['height_m']])
                if mode=='pre':target-=.10*rotation[:,2]
                bounds=np.array([env.model.joint(f'openarm_left_joint{i}').range for i in range(1,8)])
                initial=planner.data.qpos[planner.qids].copy();seed_values=seeds(initial,planner.ik_seeds['left'],bounds,np.random.default_rng(42))
                started=time.monotonic();q=None;diagnostics=None;failure=None
                try:
                    if backend=='original':
                        q=planner._ik(target,mode=='grasp');diagnostics=copy.deepcopy(planner.result.get('planning_diagnostics'))
                    else:
                        q,diagnostics=solve(env.model,planner.data,planner.qids,planner.body,planner.local_point,
                            bounds,target,rotation,seed_values,check=planner._check,
                            valid=lambda value:not planner._collision(value,mode=='grasp'))
                        if q is None:failure='seed_search_exhausted'
                except PlanningFailure as exc:failure=exc.code;diagnostics=copy.deepcopy(planner.result.get('planning_diagnostics'))
                rows.append({'case':case,'backend':backend,'proposal_found':q is not None,'failure':failure,
                    'target_m':target.tolist(),'rotation':rotation.tolist(),'q':q.tolist() if q is not None else None,
                    'wall_s':time.monotonic()-started,'diagnostics':diagnostics,'seeds':np.asarray(seed_values).tolist()})
                np.testing.assert_array_equal(initial_q,env.data.qpos);np.testing.assert_array_equal(initial_v,env.data.qvel)
                dump(OUT/'results.json',{'complete':len(rows)==len(cases)*2,'cases':rows,'live_state_unchanged':True})
                print(json.dumps({'case':index,'backend':backend,'found':q is not None,'failure':failure,
                                  'seconds':rows[-1]['wall_s']},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('pre','grasp'),default='pre');args=parser.parse_args();main(args.mode)
