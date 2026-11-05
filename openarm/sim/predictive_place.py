"""Independent loaded-servo candidate; never restored into a live state.

Geometric proposals retain the connected placement provider. Only a stopped
feedback correction can switch to bounded native actuator increments. Their
short-horizon collision proof uses full copied native dynamics, rather than
mistaking a biased actuator target for a measured configuration. Not deployed.
"""
import copy
import time

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .connected_place import ConnectedBottlePlacement
from .tamp import PlanningFailure, TampPlanner


class NativePreview:
    """Read-only preview for the current stateless native position-PD mode."""
    def __init__(self, env):
        if env.source_controller or mujoco.get_mjcb_control() is not None:
            raise PlanningFailure('preview_controller_unsupported',
                'Native preview only supports the stateless position-PD controller without callbacks')
        if env.model.nplugin:
            raise PlanningFailure('preview_plugin_unsupported', 'Stateful native plugins require separate preview validation')
        self.env = env
        self.data = mujoco.MjData(env.model)

    def collision(self, side):
        """Original TampPlanner deep-contact predicate on actual predicted bodies."""
        env = self.env; model = env.model
        for contact in self.data.contact:
            if contact.dist >= -.0005:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if min(a, b) < 0:
                continue
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            names = (model.body(ba).name or '', model.body(bb).name or '')
            robot = any('openarm_' in name for name in names)
            if not robot and env.bottle_id not in (ba, bb):
                continue
            if env.bottle_id in (ba, bb):
                other = names[1] if ba == env.bottle_id else names[0]
                allowed = {s for s, held in env.evaluation['held'].items() if held} | {side}
                if 'finger' in other and any(f'openarm_{s}_' in other for s in allowed):
                    continue
            return {'geoms': [model.geom(a).name, model.geom(b).name], 'depth_m': -float(contact.dist)}
        return None

    def run(self, action, side, *, seconds=.15, guard=True):
        env = self.env; model = env.model; started = time.monotonic()
        dt = float(model.opt.timestep)
        if not np.isfinite(seconds) or not dt <= seconds <= .2:
            raise ValueError('Preview horizon must be1physics step to200ms')
        action = env.validate_action(action, native_fingers=True)
        mujoco.mj_copyData(self.data, model, env.data)
        data = self.data
        for index, arm in enumerate(('left', 'right')):
            data.ctrl[env.aids[arm]] = action[index*8:index*8+7]
            data.ctrl[model.actuator(f'{arm}_finger1_ctrl').id] = env.grippers[arm].to_joint(action[index*8+7])
        finish = float(data.time) + round(seconds/dt)*dt
        tick = env.tick
        offset = getattr(env, '_periodic_offset', 0.)
        samples = 0; failure = None; minimum_upright = 1.; maximum_drift = 0.
        def check():
            nonlocal failure, minimum_upright, maximum_drift
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                failure = {'kind': 'non_finite_native_prediction'}; return
            upright = float(data.xmat[env.bottle_id].reshape(3,3)[2,2])
            minimum_upright = min(minimum_upright, upright)
            if guard:
                pair = self.collision(side)
                if pair:
                    failure = {'kind': 'native_collision', **pair}; return
                if upright < .98:
                    failure = {'kind': 'carried_axis_constraint', 'upright_cosine': upright}; return
                for arm in ('left', 'right'):
                    limits = np.asarray([model.joint(f'openarm_{arm}_joint{i}').range for i in range(1,8)])
                    q = data.qpos[env.qids[arm]]
                    if np.max(np.maximum(limits[:,0]-q,q-limits[:,1])) > .01:
                        failure = {'kind': 'native_joint_guard', 'arm': arm}; return
                for ident, origin in getattr(env, 'protected_objects', {}).items():
                    drift = float(np.linalg.norm(data.xpos[env.registry[ident]['body']] - origin))
                    maximum_drift = max(maximum_drift, drift)
                    if drift > .003:
                        failure = {'kind': 'non_target_object_displaced', 'object_id': ident, 'distance_m': drift}; return
        check()
        while failure is None and data.time + dt/2 < finish:
            tick += 1
            endpoint = min(finish, (tick/env.profile['control_hz']) + offset)
            if endpoint <= data.time + dt/2:
                raise ValueError('Preview requires the existing periodic control clock')
            while failure is None and data.time + dt/2 < endpoint:
                if env.profile['gravity_compensation'] == 'bias':
                    for arm in ('left', 'right'):
                        data.qfrc_applied[env.dids[arm]] = data.qfrc_bias[env.dids[arm]]
                mujoco.mj_step(model, data); samples += 1
                check()  # solved contact at the same pre-integration time as the native evaluator
            if failure is None:
                mujoco.mj_forward(model, data); check()
        return {'accepted': failure is None, 'failure': failure, 'requested_horizon_s': seconds,
            'integrated_seconds': float(data.time-env.data.time), 'physics_steps': samples,
            'minimum_upright_cosine': minimum_upright, 'maximum_non_target_drift_m': maximum_drift,
            'seconds': time.monotonic()-started, 'source': 'copied native1ms dynamics; prospective evidence only',
            'live_state_written': False, 'hardware_access': False, 'transfer_ready': False}


class PredictiveBottlePlacement(ConnectedBottlePlacement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview = NativePreview(self.env)
        self._preview_servo_used = False
        self.result['preview_servo_contract'] = {
            'source': 'native object/TCP/encoder feedback; privileged simulation only',
            'fallback': 'after completed native path and settling feedback; never initial search/path failure',
            'maximum_actuator_increment_rad': .008, 'prediction_horizon_s': .15,
            'prediction': 'original position-PD, limits, gravity compensation,1ms contacts and native free object',
            'changed_validation': 'small biased actuator increments checked by native predicted dynamics, not command-pose FK',
            'original_geometric_proposals_preserved': True,
            'original_physics_goal_release_support_stability_gates_preserved': True,
            'not_deployed': True, 'hardware_access': False, 'transfer_ready': False}

    def _servo(self, target, rotation, position_tolerance, *, release=False):
        env = self.env; history = []; stable = 0
        did = env.registry['bottle']['dofadr']
        jp = np.zeros((3,env.model.nv)); jr = jp.copy()
        limits = np.asarray([env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
        release_rotation = env.data.xmat[env.ee_ids[self.side]].reshape(3,3).copy() if release else None
        for index in range(361 if not release else 121):
            self._check()
            if env.evaluation.get('invalid') or not env.evaluation['held'][self.side]:
                raise PlanningFailure('grasp_lost', 'Predictive servo requires an actual valid native independent hold')
            self._arm(self.side,True); self._sync(); point, actual_rotation = self._pose()
            if release:
                actual = np.r_[env.data.xpos[env.bottle_id,:2],float(env._bottle_bounds()[0][2])]
                error = np.asarray(target)-actual
                quat = np.empty(4); angular = np.empty(3)
                mujoco.mju_mat2Quat(quat,(release_rotation@actual_rotation.T).ravel())
                mujoco.mju_quat2Vel(angular,quat,1.)
            else:
                error = np.asarray(target)-point
                quat = np.empty(4); angular = np.empty(3)
                mujoco.mju_mat2Quat(quat,(rotation@actual_rotation.T).ravel())
                mujoco.mju_quat2Vel(angular,quat,1.)
            velocity = env.data.qvel[did:did+6].copy()
            pose_ok = (np.linalg.norm(error[:2])<=position_tolerance and abs(error[2])<=.006
                if release else np.linalg.norm(error)<=position_tolerance and np.linalg.norm(angular)<=.06)
            upright = float(env.data.xmat[env.bottle_id].reshape(3,3)[2,2])
            speed_ok = np.linalg.norm(velocity[:3])<=.01 and np.linalg.norm(velocity[3:])<=.15
            stable = stable+1 if pose_ok and speed_ok and upright>=.98 else 0
            row = {'iteration':index,'sim_time':float(env.data.time),'position_error_m':float(np.linalg.norm(error)),
                'error_m':error.tolist(), 'object_origin_m':env.data.xpos[env.bottle_id].tolist(),
                'object_bottom_z_m':float(env._bottle_bounds()[0][2]),
                'angular_error_rad':float(np.linalg.norm(angular)), 'upright_cosine':upright,
                'linear_speed_m_s':float(np.linalg.norm(velocity[:3])),
                'angular_speed_rad_s':float(np.linalg.norm(velocity[3:])), 'stable_endpoints':stable}
            history.append(row)
            self.result.setdefault('private_native_servos', {})['release' if release else str(target)] = history
            if stable>=6:
                return
            if index == (120 if release else 360):
                raise PlanningFailure('native_servo_not_stable','Original target and low-speed gate did not settle; no release')
            action = self.base_action.copy()
            if not pose_ok:
                jac_point = env.data.xpos[env.bottle_id].copy() if release else point
                mujoco.mj_jac(env.model,self.data,jp,jr,jac_point,self.body)
                jac = np.vstack([jp[:,self.dids],.15*jr[:,self.dids]])
                cartesian = np.r_[np.clip(.6*error-.06*velocity[:3],-.004,.004),.15*.4*angular]
                current = self.base_action[self.start:self.start+7]
                if release:
                    # Independent coordinate clipping distorts the desired
                    # Cartesian direction. Solve the bounded native actuator
                    # increment jointly; zero-width coordinates stay fixed.
                    low = np.maximum(-.008,limits[:,0]-current)
                    high = np.minimum(.008,limits[:,1]-current)
                    active = high-low>1e-12
                    delta = np.zeros(7)
                    if active.any():
                        matrix = np.vstack([jac[:,active],.01*np.eye(int(active.sum()))])
                        fit = lsq_linear(matrix,np.r_[cartesian,np.zeros(int(active.sum()))],
                            bounds=(low[active],high[active]),method='bvls',tol=1e-9,max_iter=30)
                        if not fit.success or not np.isfinite(fit.x).all():
                            raise PlanningFailure('servo_increment_unreachable','Bounded native actuator increment did not solve')
                        delta[active] = fit.x
                    action[self.start:self.start+7] = current+delta
                    row['bounded_increment_rad'] = delta.tolist()
                    row['linearized_increment_m_rad'] = (jac@delta).tolist()
                else:
                    delta = jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(6),cartesian)
                    action[self.start:self.start+7] = np.clip(current+np.clip(delta,-.008,.008),limits[:,0],limits[:,1])
            preview = self.preview.run(action,self.side)
            row['preview'] = preview
            if not preview['accepted']:
                raise PlanningFailure('native_preview_rejected','Original native collision/upright/limits rejected the next actuator increment')
            self.base_action = action.copy()
            yield action,{'phase':self.phase,'native_preview_servo':index,'scene_source':'simulator_ground_truth'}
            yield from self._check_hold(action)

    def _move(self,target,allow_fingers=False,*,position_tolerance=.012):
        active = self._active() and allow_fingers
        rotation = self.rotation.copy() if self.rotation is not None else None
        self.result.pop('tracking_feedback',None)
        try:
            yield from super()._move(target,allow_fingers,position_tolerance=position_tolerance)
        except PlanningFailure as exc:
            tracking = self.result.get('tracking_feedback')
            if not active or rotation is None or tracking is None or exc.code not in ('goal_collision','ik_unreachable','tracking_error'):
                raise
            # Current actual body still must pass the exact original predicate.
            self._arm(self.side,True); self._sync()
            if TampPlanner._collision(self,env_q(self),True):
                raise
            self.result.setdefault('stopped_geometric_corrections',[]).append({
                'failure_code':exc.code,'tracking':copy.deepcopy(tracking),
                'planning_diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
            self._preview_servo_used = True
            self.rotation = rotation
            yield from self._servo(np.asarray(target),rotation,position_tolerance)
            self.result.pop('planning_diagnostics',None)

    def _grip(self,side,value,*,samples=45,ramp_samples=24):
        if not (self._preview_servo_used and self.phase=='release_left' and self.placement_surface=='table'):
            yield from super()._grip(side,value,samples=samples,ramp_samples=ramp_samples)
            return
        precision = min(.003,self.env.placement_target['tolerance_m']*.6)
        target = np.r_[self.placement_xy,self.env.placement_surface('table')['top_z']+.002]
        yield from self._servo(target,None,precision,release=True)
        # Identical native finger ramp, after the same extra low-speed release
        # alignment. Support, no-touch, upright and1s stability still evaluated
        # by the inherited complete place implementation.
        yield from TampPlanner._grip(self,side,value,samples=samples,ramp_samples=ramp_samples)


def env_q(planner):
    return planner.env.data.qpos[planner.qids].copy()
