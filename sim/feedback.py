"""Private native diagnostics. Never returned as a policy observation or score."""
import json

import numpy as np


class FeedbackLog:
    def __init__(self, folder, env):
        self.folder = folder
        self.stream = (folder/'internal.jsonl').open('w')
        self.initial_z = float(env.data.xpos[env.bottle_id, 2])
        self.count = 0
        self.squared_error = self.max_error = self.max_lift = self.max_penetration = 0.
        self.min_distance = {'left': float('inf'), 'right': float('inf')}
        self.max_hold = {'left': 0., 'right': 0.}
        try:
            self.read(env)
        except Exception:
            self.stream.close()
            raise

    def read(self, env, action=None):
        state = env.state()
        target = np.asarray(env.last_action if action is None else action)
        indices = [*range(7), *range(8, 15)]
        error = target[indices] - state[indices]
        bottle = env.data.xpos[env.bottle_id].copy()
        tcp = {}
        for side, body in env.ee_ids.items():
            rotation = env.data.xmat[body].reshape(3, 3)
            tcp[side] = env.data.xpos[body] + rotation @ np.array([0., .006, .075])
            self.min_distance[side] = min(self.min_distance[side], float(np.linalg.norm(tcp[side]-bottle)))
            self.max_hold[side] = max(self.max_hold[side], env.contacts.hold_seconds[side])
        contact = env.contacts.read(env.data)
        self.max_error = max(self.max_error, float(np.max(np.abs(error))))
        self.squared_error += float(np.mean(error**2)); self.count += 1
        self.max_lift = max(self.max_lift, float(bottle[2]-self.initial_z))
        self.max_penetration = max(self.max_penetration, env.contacts.penetration)
        row = {'tick': env.tick, 'sim_seconds': float(env.data.time),
            'joint_endpoint_error_rad': error.tolist(),
            'gripper_motor_target_rad': target[[7, 15]].tolist(),
            'gripper_motor_feedback_rad': state[[7, 15]].tolist(),
            'tcp_m': {k:v.tolist() for k,v in tcp.items()}, 'bottle_m': bottle.tolist(),
            'bottle_upright_cos': float(env.data.xmat[env.bottle_id].reshape(3, 3)[2, 2]),
            'held': contact['held'], 'touching': contact['touching'],
            'hold_seconds': dict(env.contacts.hold_seconds),
            'peak_contact_force_N': float(env.contacts.peak_force),
            'penetration_m': float(env.contacts.penetration)}
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        return row

    def finish(self):
        self.stream.close()
        result = {'schema':'internal-native-feedback/1', 'samples':self.count,
            'source':'MuJoCo state/contact truth; evaluator diagnostics only',
            'policy_input':False, 'physical_calibration_validated':False,
            'joint_error_semantics':'applied target minus measured position at control-interval endpoint',
            'joint_endpoint_rmse_rad':float(np.sqrt(self.squared_error/self.count)),
            'max_joint_endpoint_error_rad':self.max_error,
            'min_tcp_bottle_distance_m':self.min_distance,
            'max_bottle_lift_m':self.max_lift, 'max_two_finger_hold_seconds':self.max_hold,
            'max_penetration_m':self.max_penetration}
        (self.folder/'internal-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return result
