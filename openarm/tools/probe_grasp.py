"""Read contact mechanics while reproducing one archived native execution."""
import json
import sqlite3

import mujoco
import numpy as np

from sim.environment import ROOT, OpenArmEnv
from sim.calibration import response_dependencies

OUT = ROOT / 'runs/20261008-grasp'


def sample(env):
    body = env.ee_ids['right']
    rotation = env.data.xmat[body].reshape(3, 3)
    tcp = env.data.xpos[body] + rotation @ np.array([0, .006, .075])
    p = env.data.xpos[env.bottle_id].copy()
    aid = env.model.actuator('right_finger1_ctrl').id
    contacts = []
    total = np.zeros(3)
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        ids = [int(env.model.geom_bodyid[c.geom1]), int(env.model.geom_bodyid[c.geom2])]
        if env.bottle_id not in ids:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, i, force)
        world = c.frame.reshape(3, 3).T @ force[:3]
        if ids[0] == env.bottle_id:
            world = -world
        total += world
        contacts.append({'geoms': [env.model.geom(c.geom1).name, env.model.geom(c.geom2).name],
                         'normal_N': float(force[0]), 'tangent_N': float(np.linalg.norm(force[1:3])),
                         'force_on_bottle_world_N': world.tolist(), 'friction': c.friction.tolist(),
                         'distance_m': float(c.dist)})
    return {'tick': env.tick, 'sim_seconds': float(env.data.time), 'body_m': p.tolist(),
            'tcp_m': tcp.tolist(), 'relative_m': (rotation.T @ (p-tcp)).tolist(),
            'finger_q_m': [float(env.data.qpos[j.qposadr[0]]) for j in env.fingers['right']],
            'finger_target_m': float(env.data.ctrl[aid]), 'finger_effort_N': float(env.data.actuator_force[aid]),
            'total_contact_world_N': total.tolist(), 'contacts': contacts,
            'held': dict(env.evaluation['held'])}


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / 'mechanics.json'
    if path.exists():
        raise ValueError('preserve previous experiment')
    report = json.loads((ROOT/'runs/20261008-decision/api-first-failure.json').read_text())
    paths = [ROOT/w['run_path'] for w in report['waits']]
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro', uri=True) as db:
        for execution in report['terminal']['agent']['plan']['execution_tools']:
            row = db.execute('SELECT run_id FROM episodes WHERE id=?', (execution['id'],)).fetchone()
            paths.append(ROOT/'runs'/row[0])
    paths.sort(key=lambda p: json.loads((p/'manifest.json').read_text())['start_tick'])
    env = OpenArmEnv(images=False)
    env.reset(seed=0)
    env.set_task_budget(120)
    records = []
    try:
        for record in paths:
            manifest = json.loads((record/'manifest.json').read_text())
            assert response_dependencies(env.manifest()) == response_dependencies(manifest)
            assert env.tick == manifest['start_tick']
            samples = [sample(env)]
            for line in (record/'transitions.jsonl').read_text().splitlines():
                transition = json.loads(line)
                env.step_tamp(np.asarray(transition['action']))
                assert env.tick == transition['next_observation']['tick']
                samples.append(sample(env))
            records.append({'path': str(record.relative_to(ROOT)), 'kind': manifest['kind'], 'samples': samples})
        aid = env.model.actuator('right_finger1_ctrl').id
        result = {'method': 'archived commands; native solved forces; no physics or pose edits',
                  'finger_gain': float(env.model.actuator_gainprm[aid, 0]),
                  'finger_limit_N': env.model.actuator_forcerange[aid].tolist(),
                  'bottle_mass_kg': float(env.model.body_mass[env.bottle_id]),
                  'cone': str(env.model.opt.cone), 'impratio': float(env.model.opt.impratio),
                  'records': records, 'hardware_access': False, 'transfer_ready': False}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        for record in records:
            if record['kind'] != 'wait' or not record['samples'][0]['held']['right']:
                continue
            start, end = record['samples'][0], record['samples'][-1]
            print(json.dumps({'seconds': end['sim_seconds']-start['sim_seconds'],
                              'relative_drift_mm': 1000*float(np.linalg.norm(np.subtract(end['relative_m'], start['relative_m']))),
                              'last': end}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
