"""Opt-in receipt-bound height/yaw hints for the existing transport tool."""
import copy
import numpy as np
from .goals import same_xy
from sim.transit_hint import clearance


def install_isolated(base):
    if getattr(base, 'RECEIPT_TRANSIT_HINTS', False):
        return base
    class HintedHarness(base):
        RECEIPT_TRANSIT_HINTS = True
        def _pick_hint(self, snapshot, goal, arm):
            for execution in reversed((self.workspace.active_plan or {}).get('execution_tools', [])):
                if execution['skill'] in ('handover', 'run_policy', 'set_gripper'): break
                if execution['skill'] != 'pick': continue
                try:
                    receipt = self.workspace.execution_receipt('execute', execution['id'],
                        context=snapshot['memory_context'], include_oracle=True)
                except ValueError: break
                f = receipt['feedback']; future = f.get('future_placement') or {}
                if not isinstance(future, dict): break
                after = receipt.get('evidence', {}).get('after', {}).get('observation_id', '')
                if (receipt['evidence_kind'] != 'simulator_oracle' or not f.get('primitive_success')
                    or f.get('verification_source') != 'native_two_finger_contact_and_lift'
                    or f.get('held_by') != arm or f.get('object_id') != goal['object_id']
                    or after.split(':')[0] != snapshot['observation_id'].split(':')[0]
                    or not same_xy(future.get('target_xy_m', []), goal['target_xy_m'])): break
                return future, execution['id']
            return {}, None

        def _lookahead(self, snapshot, goal, arm):
            angle, ident = super()._lookahead(snapshot, goal, arm)
            if ident is not None: return angle, ident
            future, ident = self._pick_hint(snapshot, goal, arm)
            yaw = future.get('yaw_rad')
            if ident is None or type(yaw) not in (int, float) or not np.isfinite(yaw): return angle, None
            obj = next(row for row in snapshot['diagnostics']['scene_directory']['objects'] if row['object_id'] == goal['object_id'])
            rotation = np.asarray(obj['rotation_matrix']); current = np.arctan2(rotation[1,0], rotation[0,0])
            return float(np.arctan2(np.sin(yaw-current), np.cos(yaw-current))), ident

        def _invoke(self, name, args, role, context=None):
            hint = None
            if name == 'resolve_placement_transit' and 'clearance_m' not in args:
                snapshot = self.service.snapshot(); goal = (self.workspace.active_plan or {}).get('placement_goal')
                if goal and goal['surface'] == 'table' and args.get('object_id') == goal['object_id']:
                    future, ident = self._pick_hint(snapshot, goal, args.get('arm'))
                    height, hint = clearance(future, goal['target_xy_m'])
                    if hint['used']:
                        args = copy.deepcopy(args); args['clearance_m'] = height
                        hint['execution_id'] = ident
            # Original freshness, root, hold, goal and perception-boundary gates
            # run before any proposal is returned. Explicit model height stays.
            result = super()._invoke(name, args, role, context)
            if hint is not None: result['intermediate_height_hint'] = hint
            return result
    return HintedHarness
