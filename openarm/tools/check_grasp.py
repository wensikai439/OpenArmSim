"""Compare declared grasp locations without changing candidate physics."""
import json

import numpy as np

from sim.environment import ROOT, OpenArmEnv
from sim.tamp import TampPlanner, PlanningFailure
from tools.probe_grasp import sample

OUT = ROOT / 'runs/20261008-grasp/candidates'


class Candidate(TampPlanner):
    def __init__(self, *args, height, opening, **kwargs):
        super().__init__(*args, **kwargs)
        self.height, self.opening = height, opening

    def _pick(self):
        side = self.pick_arm
        self._arm(side)
        bottle = self.env.data.xpos[self.env.bottle_id].copy()
        grasp = bottle + [0, 0, self.height]
        pre = grasp - [ .13, 0, 0 ]
        self.base_action[15] = self.opening
        self._phase('approach'); yield from self._move(pre)
        self._phase('grasp'); yield from self._move(grasp)
        self._phase('close'); yield from self._grip(side, .05)
        if not self.env.evaluation['held'][side]:
            raise PlanningFailure('grasp_not_confirmed', 'no sustained bilateral contact')
        hand = self.env.data.xmat[self.body].reshape(3, 3)
        self.carried = (hand.T @ (self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
                        hand.T @ self.env.data.xmat[self.env.bottle_id].reshape(3, 3))
        self._phase('lift'); yield from self._move(grasp + [0, 0, .12], True)
        self._phase('hold')
        for _ in range(30):
            yield self.base_action.copy(), {'phase': 'hold'}
            if not self.env.evaluation['held'][side]:
                raise PlanningFailure('grasp_lost', 'bilateral contact lost during lift')
        lo, _ = self.env._bottle_bounds()
        if lo[2] < bottle[2]+.06:
            raise PlanningFailure('lift_not_confirmed', 'lift below original threshold')


def main():
    if OUT.exists():
        raise ValueError('preserve prior trial')
    OUT.mkdir()
    cases = [{'id': 'production', 'height_m': .195, 'opening_motor_rad': -.9}] + [
        {'id': f'height-{int(1000*h)}', 'height_m': h, 'opening_motor_rad': -1.26}
        for h in (.095, .145, .175, .185, .195)]
    protocol = {'cases': cases, 'episode': 0, 'seed': 0, 'budget_seconds': 120,
                'hold_seconds': 10, 'stability_window_seconds': 2, 'relative_limit_m': .001,
                'scope': 'grasp proposal comparison only; no friction, force, controller or solver tuning',
                'hardware_access': False, 'transfer_ready': False}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    results = []
    for case in cases:
        env = OpenArmEnv(images=False); env.reset(seed=0); env.set_task_budget(120)
        rows = []; failure = None
        try:
            if case['id'] == 'production':
                planner = TampPlanner(env, 'pick_bottle', parameters={'arm': 'right'})
            else:
                planner = Candidate(env, 'pick_bottle', parameters={'arm': 'right'},
                                    height=case['height_m'], opening=case['opening_motor_rad'])
            iterator = planner.actions()
            while True:
                q = env.data.qpos.copy()
                try:
                    action, diag = next(iterator)
                except StopIteration:
                    break
                assert np.array_equal(q, env.data.qpos)
                _, _, terminal, truncated, _ = env.step_tamp(action)
                if terminal or truncated:
                    raise RuntimeError('native terminal')
            target = env.last_action.copy()
            rows.append(sample(env))
            for _ in range(300):
                _, _, terminal, truncated, _ = env.step_tamp(target)
                rows.append(sample(env))
                if terminal or truncated:
                    raise RuntimeError('native terminal')
            drift = [float(np.linalg.norm(np.subtract(r['relative_m'], rows[0]['relative_m']))) for r in rows]
            windows = [float(np.linalg.norm(np.subtract(rows[i+60]['relative_m'], rows[i]['relative_m']))) for i in range(len(rows)-60)]
            result = {'case': case, 'status': 'stable' if max(windows) <= .001 else 'sliding',
                      'max_two_second_relative_drift_m': max(windows), 'total_relative_drift_m': drift[-1],
                      'end_held': env.evaluation['held'], 'samples': rows}
        except Exception as exc:
            failure = {'code': getattr(exc, 'code', 'execution_error'), 'message': str(exc)}
            result = {'case': case, 'status': 'failed', 'failure': failure, 'samples': rows}
        finally:
            env.close()
        results.append(result)
        (OUT/(case['id']+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, ensure_ascii=False), flush=True)
    (OUT/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in results], ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
