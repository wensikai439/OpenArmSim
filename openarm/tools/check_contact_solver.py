"""Numerical sensitivity study; never apply its solver candidate to runtime."""
import json
import argparse
import sqlite3
import time

import numpy as np

from sim.environment import ROOT, OpenArmEnv
from tools.probe_grasp import sample

OUT = ROOT / 'runs/20261008-grasp/solver'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--after-grasp',action='store_true');args=parser.parse_args()
    output=ROOT/'runs/20261008-grasp/solver-hold' if args.after_grasp else OUT
    if output.exists():
        raise ValueError('preserve previous numerical study')
    output.mkdir()
    source = json.loads((ROOT/'runs/20261008-decision/api-first-failure.json').read_text())
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro', uri=True) as db:
        ident = source['terminal']['agent']['plan']['execution_tools'][0]['id']
        run = db.execute('SELECT run_id FROM episodes WHERE id=?', (ident,)).fetchone()[0]
    files = [ROOT/source['waits'][0]['run_path']/'transitions.jsonl', ROOT/'runs'/run/'transitions.jsonl']
    actions = [json.loads(line)['action'] for file in files for line in file.read_text().splitlines()]
    protocol = {'noslip_iterations': [0, 1, 10, 20], 'source_actions': [str(p.relative_to(ROOT)) for p in files],
                'hold_seconds': 10, 'budget_seconds': 120, 'all_other_parameters': 'unchanged',
                'scope': 'numerical contact sensitivity only; not calibration or production',
                'solver_change': 'after identical native grasp' if args.after_grasp else 'before grasp; resulting hold states can differ',
                'automatic_application': False, 'hardware_access': False, 'transfer_ready': False}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    summaries = []
    for count in protocol['noslip_iterations']:
        env = OpenArmEnv(images=False); env.reset(seed=0); env.set_task_budget(120)
        if not args.after_grasp:env.model.opt.noslip_iterations = count
        started = time.monotonic()
        try:
            for action in actions:
                env.step_tamp(np.asarray(action))
            if args.after_grasp:env.model.opt.noslip_iterations = count
            rows = [sample(env)]; target = env.last_action.copy()
            for _ in range(300):
                _, _, terminal, truncated, _ = env.step_tamp(target)
                rows.append(sample(env))
                if terminal or truncated:
                    break
            drift = float(np.linalg.norm(np.subtract(rows[-1]['relative_m'], rows[0]['relative_m'])))
            summary = {'noslip_iterations': count, 'hold_seconds': rows[-1]['sim_seconds']-rows[0]['sim_seconds'],
                       'relative_drift_m': drift, 'held': env.evaluation['held'],
                       'wall_seconds': time.monotonic()-started, 'invalid': env.evaluation['invalid']}
            (output/(str(count)+'.json')).write_text(json.dumps({**summary, 'samples': rows}, indent=2)+'\n')
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        finally:
            env.close()
    (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    main()
