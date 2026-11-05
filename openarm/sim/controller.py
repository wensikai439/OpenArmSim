"""Candidate of the audited collector's MIT command state, not hardware code.

Only numerical state is reproduced. No CAN imports, devices or remote writes.
Capture-time source/firmware/torque saturation and initial accumulator values
are still unknown. This mode must be explicitly selected for offline studies.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class SourceMIT:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous={side:np.zeros(7) for side in ('left','right')}
        self.accumulated={side:np.zeros(7) for side in ('left','right')}
        self.calls={side:0 for side in ('left','right')}

    def update(self,side,target,measured):
        if side not in self.previous:raise ValueError('unknown arm')
        target=np.asarray(target,dtype=float);measured=np.asarray(measured,dtype=float)
        if target.shape!=(7,) or measured.shape!=(7,) or not np.isfinite(target).all() or not np.isfinite(measured).all():
            raise ValueError('MIT state requires finite 7-D target and feedback')
        error=np.clip(target-measured,-.3,.3)
        difference=error-self.previous[side]
        self.previous[side]=error.copy()
        self.accumulated[side]=np.clip(self.accumulated[side]+error,-10.,10.)
        self.calls[side]+=1
        return {'position_rad':target.copy(),'velocity_input':difference.copy(),
                'feedforward_input':self.accumulated[side].copy(),'call':self.calls[side]}

    def contract(self,initialization='zero'):
        if initialization not in ('zero','static_gravity_proxy'):raise ValueError('unknown controller initialization')
        path=ROOT/'reference/source/inputs/controller.py'
        manifest_path=ROOT/'reference/source/manifest.json'
        source=None
        if path.is_file() and manifest_path.is_file():
            source=hashlib.sha256(path.read_bytes()).hexdigest()
            expected=json.loads(manifest_path.read_text())['files']['controller.py']['sha256']
            if source!=expected:raise ValueError('audited controller source identity changed')
        else:raise ValueError('source-MIT candidate requires audited source snapshot')
        return {'mode':'source_mit_candidate','source_sha256':source,'capture_time_version_bound':False,
            'update_clock':'once per arm command invocation; never per physics substep',
            'error_clip_rad':.3,'velocity_input':'clipped-error difference without division by dt',
            'feedforward_input':'cumulative clipped error, limited to +/-10 per joint',
            'force_law':'kp*(target-q) + kd*(velocity_input-qvel) + feedforward_input',
            'native_force_limit':'whole force goes through simulated actuator forcerange; real driver limits unknown',
            'extra_gravity_compensation':False,'initial_state':initialization,
            'initial_state_scope':('zero accumulator hypothesis' if initialization=='zero' else
                'initial simulated gravity torque clipped to +/-10 as an accumulator proxy; NOT measured controller history'),
            'first_update':'first advance, not reset',
            'hardware_access':False,'physically_calibrated':False,'transfer_ready':False}
