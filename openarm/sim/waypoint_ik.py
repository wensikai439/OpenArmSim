"""Scratch IK proposals reserve headroom for SourceMIT motor targets.

The zero-velocity endpoint calculation uses CURRENT controller history and
model gravity bias. It is a proposal heuristic, not a future-history guarantee,
payload/contact identification, or permission to skip native execution gates.
"""
import copy
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from sim.controller import SourceMIT
from sim.tamp import TampPlanner, PlanningFailure
from sim.waypoint_control import WaypointAdapter

NUMERICAL_COMMAND_MARGIN = .005
MINIMUM_JOINT_MARGIN = .02001
MAX_REFINEMENTS = 3


def endpoint(planner, q):
    env = planner.env; side = planner.side
    previous = getattr(planner, '_kinematics_only', False)
    planner._kinematics_only = False
    try:
        planner._at(q)
        ids = env.aids[side]; dofs = env.dids[side]
        kp = env.model.actuator_gainprm[ids, 0]; kd = -env.model.actuator_biasprm[ids, 2]
        desired = planner.data.qfrc_bias[dofs].copy()
        limits = env.model.actuator_forcerange[ids]
        if np.any(desired < limits[:, 0]) or np.any(desired > limits[:, 1]):
            raise PlanningFailure('controller_endpoint_effort', 'Model gravity at proposal exceeds original source actuator force')
        history = env.source_controller
        low = np.asarray(q)-1.; high = np.asarray(q)+1.
        velocity = np.zeros(7)
        # Extended analytic search is scratch arithmetic only. Its result is
        # explicitly compared to ORIGINAL motor command limits; never sent.
        for _ in range(40):
            middle = (low+high)*.5
            force = WaypointAdapter.effort(middle, q, velocity, kp, kd,
                history.previous[side], history.accumulated[side], limits)
            below = force < desired
            low = np.where(below, middle, low); high = np.where(below, high, middle)
        command = (low+high)*.5
        force = WaypointAdapter.effort(command, q, velocity, kp, kd,
            history.previous[side], history.accumulated[side], limits)
        if np.max(abs(force-desired)) > 3e-5:
            raise PlanningFailure('controller_endpoint_inversion', 'Endpoint force inversion is not independently representable')
        start = 0 if side == 'left' else 8
        bounds = np.column_stack((env.action_space.low[start:start+7], env.action_space.high[start:start+7]))
        slack = np.minimum(command-bounds[:, 0], bounds[:, 1]-command)
        return {'native_command_rad': command, 'required_offset_rad': command-q,
            'minimum_command_slack_rad': float(slack.min()), 'command_slack_rad': slack,
            'model_bias_Nm': desired, 'force_inversion_error_Nm': float(np.max(abs(force-desired))),
            'current_source_calls': history.calls[side],
            'assumption': 'zero endpoint velocity, current previous-error/integral history; not predicted payload contact dynamics'}
    finally:
        planner._kinematics_only = previous


def refine(planner, target, rotation, seed, low, high, allow_fingers, tolerance):
    previous = getattr(planner, '_kinematics_only', False)
    planner._kinematics_only = True
    cache = None; queries = 0
    jp = np.zeros((3, planner.env.model.nv)); jr = jp.copy()
    def evaluate(q):
        nonlocal cache, queries
        planner._check()
        if cache is not None and np.array_equal(cache[0], q):
            return cache[1:]
        planner._at(q); point, actual = planner._pose()
        quat = np.empty(4); angular = np.empty(3)
        mujoco.mju_mat2Quat(quat, (rotation@actual.T).ravel()); mujoco.mju_quat2Vel(angular, quat, 1.)
        mujoco.mj_jac(planner.env.model, planner.data, jp, jr, point, planner.body)
        cache = (q.copy(), point-target, jp[:, planner.dids].copy(), angular.copy())
        queries += 1
        return cache[1:]
    initial = np.clip(seed, low, high)
    try:
        solve = minimize(lambda q: .5*float((q-seed)@(q-seed)), initial,
            jac=lambda q: q-seed, method='SLSQP', bounds=list(zip(low, high)),
            constraints=[{'type': 'eq', 'fun': lambda q: evaluate(q)[0], 'jac': lambda q: evaluate(q)[1]},
                {'type': 'ineq', 'fun': lambda q: 1.-float(np.sum((evaluate(q)[2]/(.018-1e-6))**2))}],
            options={'maxiter': 100, 'ftol': 1e-10})
        q = np.asarray(solve.x); error, _, angle = evaluate(q)
        posed = np.linalg.norm(error) < tolerance and np.linalg.norm(angle) < .018
        bounded = np.isfinite(q).all() and np.all(q >= low) and np.all(q <= high)
        collision = planner._collision(q, allow_fingers) if posed and bounded else None
        accepted = posed and bounded and collision is False
        return (q.copy() if accepted else None), {
            'accepted': bool(accepted), 'optimizer_success': bool(solve.success), 'optimizer_message': str(solve.message),
            'position_error_m': float(np.linalg.norm(error)), 'rotation_error_rad': float(np.linalg.norm(angle)),
            'original_collision_checked': collision is not None, 'collision': collision,
            'kinematic_queries': queries, 'proposed_joint_bounds_rad': np.column_stack((low, high))}
    finally:
        planner._kinematics_only = previous


def install():
    original = TampPlanner._ik
    def inverse(self, target, allow_fingers=False, *, position_tolerance=.0015):
        q = original(self, target, allow_fingers, position_tolerance=position_tolerance)
        env = self.env
        if env.source_controller is None or self.rotation is None:
            return q
        if type(env.source_controller) is not SourceMIT:
            raise PlanningFailure('controller_pose_unsupported', 'Only audited SourceMIT candidate is supported')
        probe = endpoint(self, q)
        if probe['minimum_command_slack_rad'] >= NUMERICAL_COMMAND_MARGIN:
            return q
        record = {'schema': 'controller-pose-proposal/1', 'arm': self.side, 'phase': self.phase,
            'target_position_m': np.asarray(target).copy(), 'original_q_rad': q.copy(), 'initial_endpoint': probe,
            'numerical_motor_margin_rad': NUMERICAL_COMMAND_MARGIN, 'refinements': [],
            'source': 'scratch geometry and current source history; heuristic only', 'accepted': False,
            'native_task_goals_scores_and_plant_unchanged': True, 'hardware_access': False, 'transfer_ready': False}
        self.result.setdefault('private_controller_poses', []).append(record)
        bounds = np.asarray([env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1, 8)])
        low = bounds[:, 0]+MINIMUM_JOINT_MARGIN; high = bounds[:, 1]-MINIMUM_JOINT_MARGIN
        for index in range(MAX_REFINEMENTS):
            bias = probe['required_offset_rad']
            low = np.maximum(low, bounds[:, 0]-np.minimum(bias, 0)+NUMERICAL_COMMAND_MARGIN)
            high = np.minimum(high, bounds[:, 1]-np.maximum(bias, 0)-NUMERICAL_COMMAND_MARGIN)
            if np.any(low >= high):
                break
            candidate, detail = refine(self, np.asarray(target), self.rotation.copy(), q, low, high,
                allow_fingers, position_tolerance)
            record['refinements'].append(detail)
            if candidate is None:
                break
            q = candidate; probe = endpoint(self, q); detail['endpoint'] = probe
            if probe['minimum_command_slack_rad'] >= NUMERICAL_COMMAND_MARGIN-1e-6:
                record.update(accepted=True, accepted_q_rad=q.copy())
                return q
        raise PlanningFailure('controller_pose_unreachable', 'Original pose lacks a checked motor-headroom proposal; native target unchanged')
    TampPlanner._ik = inverse


# Explicit opt-in only; importing this module does not patch the default planner.
install_isolated = install
