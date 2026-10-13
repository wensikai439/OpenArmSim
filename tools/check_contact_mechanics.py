"""Reproduce a preserved failed API task and audit its native contact wrenches."""
import argparse
import json
from pathlib import Path
import sqlite3

import mujoco
import numpy as np

from sim.environment import ROOT,OpenArmEnv
from sim.recording import dump


def contact_mechanics(env,side='right'):
    if side not in ('left','right'):raise ValueError('unknown diagnostic arm')
    m,d=env.model,env.data;com=d.xipos[env.bottle_id].copy()
    contacts=[];total=np.zeros(3);torque=np.zeros(3);columns=[]
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(c.geom1),int(c.geom2)
        bodies=[int(m.geom_bodyid[g]) for g in (a,b)]
        if env.bottle_id not in bodies:continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        if force[0]<=.05:continue
        frame=c.frame.reshape(3,3);sign=-1 if bodies[0]==env.bottle_id else 1
        world=sign*frame.T@force[:3];total+=world
        torque+=np.cross(c.pos-com,world)+sign*frame.T@force[3:]
        other=b if bodies[0]==env.bottle_id else a
        jp=[]
        for body in bodies:
            p,r=np.zeros((3,m.nv)),np.zeros((3,m.nv))
            mujoco.mj_jac(m,d,p,r,c.pos,body);jp.append(p@d.qvel)
        relative=frame@(jp[1]-jp[0])
        # A relaxed point-contact wrench span: no force/friction bounds.
        # Missing gravity in this span is stronger than low friction; membership
        # is only necessary, never sufficient for a stable grasp.
        radius=c.pos-com
        for axis in frame:
            columns.append(np.r_[axis,np.cross(radius,axis)/.104])
        if c.dim>=4:columns.append(np.r_[np.zeros(3),frame[0]/.104])
        if c.dim>=6:
            for axis in frame[1:]:columns.append(np.r_[np.zeros(3),axis/.104])
        normal=float(force[0]);mu=np.asarray(c.friction[:2])
        utilization=float(np.linalg.norm(force[1:3]/np.maximum(mu*normal,1e-12)))
        contacts.append({'geoms':[m.geom(g).name for g in (a,b)],'other_body':m.body(int(m.geom_bodyid[other])).name,
            'dimension':int(c.dim),'geom_dimensions':[int(m.geom_condim[g]) for g in (a,b)],
            'geom_priorities':[int(m.geom_priority[g]) for g in (a,b)],
            'friction':c.friction.tolist(),'normal_N':normal,'tangent_N':float(np.linalg.norm(force[1:3])),
            'force_on_bottle_world_N':world.tolist(),'position_world_m':c.pos.tolist(),
            'normal_world':frame[0].tolist(),'distance_m':float(c.dist),
            'tangential_speed_m_s':float(np.linalg.norm(relative[1:])),
            'tangential_friction_utilization':utilization})
    required=np.r_[-m.body_mass[env.bottle_id]*m.opt.gravity,np.zeros(3)]
    rank=0;unbalanced=float(np.linalg.norm(required));singular=[]
    if columns:
        u,s,_=np.linalg.svd(np.asarray(columns).T,full_matrices=True)
        rank=int(np.sum(s>1e-8));singular=s.tolist()
        unbalanced=float(np.linalg.norm(u[:,rank:].T@required))
    body=env.ee_ids[side];rot=d.xmat[body].reshape(3,3)
    aid=m.actuator(side+'_finger1_ctrl').id
    return {'tick':env.tick,'sim_seconds':float(d.time),'held':dict(env.evaluation['held']),
        'diagnostic_arm':side,
        'body_position_m':d.xpos[env.bottle_id].tolist(),
        'relative_position_m':(rot.T@(d.xpos[env.bottle_id]-d.xpos[body])).tolist(),
        'relative_rotation':(rot.T@d.xmat[env.bottle_id].reshape(3,3)).tolist(),
        'contact_force_world_N':total.tolist(),'weight_N':float(np.linalg.norm(required[:3])),
        'contact_torque_about_bottle_COM_world_Nm':torque.tolist(),
        'finger_actuator_force_N':float(d.actuator_force[aid]),
        'finger_q_m':[float(d.qpos[j.qposadr[0]]) for j in env.fingers[side]],
        'contacts':contacts,'relaxed_wrench_rank':rank,'singular_values':singular,
        'relaxed_missing_gravity_N':unbalanced,
        'wrench_scope':'instantaneous unbounded linear contact wrench span about bottle COM; rotational rows divided by 0.104 m. Necessary only; rank 6 is NOT proof of feasible friction or stable hold'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',default='runs/20261008-alignment/api-terminal.json')
    parser.add_argument('--output',default='runs/20261008-model/mechanics')
    args=parser.parse_args();source=(ROOT/args.report).resolve();out=(ROOT/args.output).resolve()
    if not source.is_relative_to(ROOT/'runs') or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('contact audit stays under local runs')
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads(source.read_text());root=report['turn_id'];paths=list((ROOT/'runs'/('20261008-agent-'+root[:8])/'waits').glob('*/manifest.json'))
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
        for execution in report['terminal']['agent']['plan']['execution_tools']:
            row=db.execute('SELECT run_id FROM episodes WHERE id=?',(execution['id'],)).fetchone()
            if row is None:raise ValueError('missing native execution')
            paths.append(ROOT/'runs'/row[0]/'manifest.json')
    paths.sort(key=lambda p:json.loads(p.read_text())['start_tick'])
    env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
    dump(out/'protocol.json',{'schema':'contact-mechanics/1','source_report':str(source.relative_to(ROOT)),
        'source_report_sha256':__import__('hashlib').sha256(source.read_bytes()).hexdigest(),
        'method':'replay actual archived applied targets; check every recorded robot state/tick, then read solved contacts and point velocities',
        'no_physics_or_pose_edits':True,'model':env.manifest(),'hardware_access':False,'transfer_ready':False})
    records=[];max_state_error=0.;samples=[]
    try:
        for path in paths:
            manifest=json.loads(path.read_text());folder=path.parent
            if env.tick!=manifest['start_tick']:raise ValueError('non-contiguous archived commands')
            transitions=[json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
            start=len(samples)
            for transition in transitions:
                env.step_tamp(np.asarray(transition['action']))
                reference=transition['next_observation']
                error=float(np.max(np.abs(env.state()-np.asarray(reference['state']))))
                if env.tick!=reference['tick'] or error>1e-9:raise ValueError('native state reproduction mismatch')
                max_state_error=max(max_state_error,error)
                samples.append({**contact_mechanics(env),'kind':manifest['kind'],'run':folder.name})
            records.append({'run':folder.name,'kind':manifest['kind'],'start_tick':manifest['start_tick'],
                            'end_tick':env.tick,'samples':len(samples)-start})
            print(json.dumps(records[-1]),flush=True)
        expected=report['terminal']['diagnostics']['manipulation_scene']['position_m']
        body_error=float(np.linalg.norm(env.data.xpos[env.bottle_id]-expected))
        if body_error>1e-9:raise ValueError('terminal bottle position mismatch')
        waits=[s for s in samples if s['kind']=='wait' and s['held']['right']]
        active=[c for s in waits for c in s['contacts'] if 'finger' in c['other_body']]
        summary={'status':'completed','records':records,'ticks':env.tick,'max_robot_state_error':max_state_error,
            'terminal_bottle_error_m':body_error,'held_wait_samples':len(waits),
            'finger_contact_dimensions':sorted({c['dimension'] for c in active}),
            'finger_normal_force_N_percentiles':np.percentile([c['normal_N'] for c in active],[5,50,95]).tolist(),
            'tangent_utilization_percentiles':np.percentile([c['tangential_friction_utilization'] for c in active],[5,50,95]).tolist(),
            'tangential_speed_mm_s_percentiles':(1000*np.percentile([c['tangential_speed_m_s'] for c in active],[5,50,95])).tolist(),
            'held_wait_wrench_rank_counts':{str(k):sum(s['relaxed_wrench_rank']==k for s in waits) for k in range(7)},
            'missing_gravity_N_percentiles':np.percentile([s['relaxed_missing_gravity_N'] for s in waits],[5,50,95]).tolist(),
            'physics_invalid':env.evaluation['invalid'],'samples':samples,'hardware_access':False,'transfer_ready':False,
            'limits':['30 Hz diagnostic samples, not all 1 ms substeps','rank is instantaneous relaxed geometry only; not a force-closure or disturbance test',
                      'old archive lacks frozen mesh identities; exact state reproduction is a local software check, not capture-time asset or physical calibration proof']}
        dump(out/'summary.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('samples','records')},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':main()
