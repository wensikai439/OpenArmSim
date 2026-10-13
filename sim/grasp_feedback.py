"""Bounded, read-only object-in-hand feedback for the physics owner.

Contact/lift and recent relative-pose stability are different observations.
This monitor never changes actions, native success predicates or parameters.
"""
import copy

import numpy as np


class GraspFeedback:
    window_s=2.
    position_limit_m=.001
    rotation_limit_rad=.02

    def __init__(self):
        self.rows={side:[] for side in ('left','right')}
        self.latest={side:{'status':'not_held','samples':0} for side in self.rows}
        self.last_time={side:None for side in self.rows}
        self.episodes={side:0 for side in self.rows}
        self.active={side:False for side in self.rows}

    def sample(self,side,timestamp,position,rotation,*,held,continuous_contact_s):
        if side not in self.rows:raise ValueError('unknown arm')
        t=float(timestamp);contact=float(continuous_contact_s)
        p=np.asarray(position,dtype=float);r=np.asarray(rotation,dtype=float)
        if p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite([t,contact]).all() or contact<0:
            raise ValueError('invalid grasp feedback sample')
        previous=self.last_time[side]
        if previous is not None and t<previous-1e-9:raise ValueError('grasp feedback clock went backwards')
        if previous is not None and abs(t-previous)<1e-9:return
        self.last_time[side]=t
        rows=self.rows[side]
        # A contact interruption inside a long control interval must invalidate
        # the history even if the latest endpoint again reports two fingers.
        gap=0. if previous is None else t-previous
        interrupted=bool(rows and contact<gap-1e-6)
        if not held or interrupted:
            rows.clear();self.active[side]=False
        if held:
            if not self.active[side]:self.episodes[side]+=1;self.active[side]=True
            rows.append((t,p.copy(),r.copy()))
            while len(rows)>1 and t-rows[1][0]>=self.window_s-1e-6:rows.pop(0)
        span=rows[-1][0]-rows[0][0] if rows else 0.
        drift=angle=0.
        if rows:
            positions=np.asarray([row[1] for row in rows])
            rotations=np.asarray([row[2] for row in rows]).reshape(len(rows),9)
            drift=float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)))
            angle=float(np.arccos(np.clip((np.min(rotations@rotations.T)-1)/2,-1,1)))
        status=('not_held' if not held else 'insufficient_duration' if span<self.window_s-1e-6 or contact<self.window_s-1e-6 else
                'sliding' if drift>self.position_limit_m or angle>self.rotation_limit_rad else 'stable_recent_window')
        self.latest[side]={'status':status,'timestamp_s':t,'samples':len(rows),'contact_episode':self.episodes[side],
            'window_span_s':span,'continuous_two_finger_s':contact,'relative_position_range_m':drift,
            'relative_rotation_range_rad':angle}

    def read(self,env):
        for side in self.rows:
            body=env.ee_ids[side];hand=env.data.xmat[body].reshape(3,3)
            self.sample(side,float(env.data.time),hand.T@(env.data.xpos[env.bottle_id]-env.data.xpos[body]),
                hand.T@env.data.xmat[env.bottle_id].reshape(3,3),
                held=env.evaluation['held'][side],continuous_contact_s=env.contacts.hold_seconds[side])

    def summary(self):
        return {'schema':'grasp-feedback/1','arms':copy.deepcopy(self.latest),
            'window_s':self.window_s,'limits':{'position_m':self.position_limit_m,'rotation_rad':self.rotation_limit_rad},
            'source':'simulator_ground_truth','sampling':'each control endpoint; continuous contact from solved 1ms substeps',
            'scope':'recent relative-pose window only; contact/lift, long-duration hold and real transfer are separate',
            'changes_task_score':False,'hardware_access':False,'transfer_ready':False}
