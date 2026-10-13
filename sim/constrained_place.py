"""Optional bounded task-space descent; existing demo/provider is retained.

The task requires position and an upright bottle, not a prescribed wrist yaw or
an exactly vertical axis. Only scratch IK changes here. Native contact, loaded
edges, object feedback, release/support and the original success gates remain.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize
from types import SimpleNamespace

from .cartesian_place import CartesianBottlePlacement
from .tamp import PlanningFailure


class ConstrainedBottlePlacement(CartesianBottlePlacement):
    def _ik(self, target, allow_fingers=False, *, position_tolerance=.0015):
        if not (getattr(self, '_pre_release_servo', False) and self.carried is not None
                and self.phase in ('place', 'release_left') and position_tolerance <= .0002):
            return super()._ik(target, allow_fingers, position_tolerance=position_tolerance)
        self._sync()
        initial = self.data.qpos[self.qids].copy()
        target = np.asarray(target, dtype=float)
        bounds = np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range
                             for i in range(1, 8)])
        # A stricter planning cone leaves margin for native tracking. The actual
        # task's .98 upright gate is unchanged, and every increment is checked.
        upright_min = .99
        jp = np.zeros((3, self.env.model.nv)); jr = jp.copy()
        cached = None; evaluations = 0
        self.stats['ik_attempts'] += 1

        def evaluate(q):
            nonlocal cached, evaluations
            self._check()
            if cached is not None and np.array_equal(q, cached[0]):
                return cached[1:]
            self._at(q)
            point, _ = self._pose()
            axis = self.data.xmat[self.env.bottle_id].reshape(3, 3)[:, 2].copy()
            mujoco.mj_jac(self.env.model, self.data, jp, jr, point, self.body)
            # Rotation of the carried-body axis follows the hand's angular
            # Jacobian. In world coordinates d(axis_z)/dq = (w x axis)_z.
            axis_z_jac = np.array([axis[1], -axis[0], 0.]) @ jr[:, self.dids]
            cached = (q.copy(), point - target, jp[:, self.dids].copy(),
                      float(axis[2]), axis_z_jac.copy())
            evaluations += 1
            return cached[1:]

        def objective(q):
            delta = q - initial
            return .5 * float(delta @ delta)

        def feasible(q):
            error, _, upright, _ = evaluate(q)
            return (np.isfinite(q).all() and np.linalg.norm(error) < position_tolerance
                and upright >= upright_min - 1e-5
                and np.all(q >= bounds[:, 0] - 1e-8)
                and np.all(q <= bounds[:, 1] + 1e-8))

        class FeasiblePose(Exception):
            pass

        iterations = 0
        def callback(q):
            nonlocal iterations
            iterations += 1
            # Feedback descent needs a feasible small update, not convergence
            # of a secondary minimum-motion objective. Stop as soon as the
            # native constraints pass, before a long bound-active line search
            # makes the proposal stale while the real simulated object moves.
            if feasible(q):
                raise FeasiblePose(q.copy())

        try:
            result = minimize(objective, np.clip(initial, bounds[:, 0], bounds[:, 1]),
                jac=lambda q: q - initial, bounds=bounds, method='SLSQP',
                constraints=[
                    {'type': 'eq', 'fun': lambda q: evaluate(q)[0],
                     'jac': lambda q: evaluate(q)[1]},
                    {'type': 'ineq', 'fun': lambda q: evaluate(q)[2] - upright_min,
                     'jac': lambda q: evaluate(q)[3]}],
                callback=callback, options={'maxiter': 130, 'ftol': 1e-9})
        except FeasiblePose as found:
            result = SimpleNamespace(x=found.args[0], success=False, nit=iterations,
                message='Native pose constraints passed; stopped before secondary-objective convergence')
        error, _, upright, _ = evaluate(result.x)
        diagnostic = {'source': 'native position/upright Jacobians with bounded task constraints',
            'optimizer': 'SLSQP', 'optimizer_success': bool(result.success),
            'optimizer_message': str(result.message), 'iterations': int(result.nit),
            'native_evaluations': evaluations, 'position_error_m': float(np.linalg.norm(error)),
            'position_tolerance_m': position_tolerance, 'upright_cosine': upright,
            'planning_upright_minimum': upright_min, 'task_upright_minimum': .98,
            'yaw_free': True, 'secondary_objective': 'small joint motion; optimality is not required or claimed',
            'joint_limits_and_original_collision_checks': True}
        # Optimizer status describes convergence of the minimum-motion
        # objective, not feasibility. Bound-active line searches can stop with
        # a valid pose. Accept only independently recomputed finite constraints
        # and native collision/edge checks; never accept from status alone.
        constraints_ok = feasible(result.x)
        diagnostic['independently_recomputed_constraints_passed'] = bool(constraints_ok)
        diagnostic['planning_cone_numerical_tolerance'] = 1e-5
        if not constraints_ok:
            self.result['planning_diagnostics'] = {'arm': self.side, 'phase': self.phase,
                'best_ik': diagnostic, 'blocked_pairs': []}
            raise PlanningFailure('ik_unreachable', 'No bounded position/upright-cone solution')
        if self._collision(result.x, allow_fingers):
            self.result['planning_diagnostics'] = {'arm': self.side, 'phase': self.phase,
                'best_ik': diagnostic, 'blocked_pairs': [self.last_collision]}
            raise PlanningFailure('goal_collision', 'Bounded task-space solution collides under original geometry')
        self.result['object_constraint_IK'] = diagnostic
        self.result.pop('planning_diagnostics', None)
        return result.x.copy()
