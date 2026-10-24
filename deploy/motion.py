"""Continuous dual-arm hold kernel, separated from slow API/tool decisions.

Needs a physically admitted planner and a real initialized LegacyArm. Nothing
starts at import/construction, and this module supplies no unchecked IK fallback.
"""
import threading
import time

import numpy as np

try:
    from .evidence import finite
except ImportError:
    from evidence import finite


ARM=np.r_[0:7,8:15]


class Servo:
    def __init__(self,driver,limits,record,*,rate=30,step_rad=.01,tracking_rad=np.deg2rad(3)):
        self.driver=driver;self.limits=limits;self.record=record;self.rate=rate
        self.step=step_rad;self.tracking=tracking_rad;self.target=None;self.state=None
        self.lock=threading.Lock();self.stop_event=threading.Event();self.thread=None;self.failure=None;self.sends=0
        self.condition=threading.Condition(self.lock);self.generation=0;self.applied_generation=-1

    def validate(self,q):
        q=finite(q,(16,))
        for side,i in [('left',0),('right',8)]:
            low,high=self.limits[side]
            if np.any(q[i:i+7]<low) or np.any(q[i:i+7]>high):raise ValueError('command exceeds URDF limits')
            lo,hi=self.limits[side+'_gripper']
            if not lo<=q[i+7]<=hi:raise ValueError('command exceeds accepted jaw motor limits')
        return q

    def start(self):
        if self.thread is not None:raise RuntimeError('servo already started')
        if self.driver.controller is None or set(self.driver.controller._activated_sides)!={'left','right'}:
            raise RuntimeError('explicit accepted initialization required')
        self.state=self.driver.read();self.target=self.validate(self.state['q']).copy()
        self.thread=threading.Thread(target=self._loop,name='openarm-hold',daemon=True);self.thread.start()

    def _loop(self):
        deadline=time.monotonic()
        try:
            while not self.stop_event.is_set():
                state=self.driver.read();measured=finite(state['q'],(16,))
                with self.lock:target=self.target.copy();generation=self.generation
                if not -.01<=time.time()-state['timestamp_unix_s']<=.1:raise RuntimeError('feedback watchdog expired')
                if np.max(abs(target[ARM]-measured[ARM]))>self.tracking:raise RuntimeError('joint tracking limit exceeded')
                self.driver.send(target,state);self.state=state;self.sends+=1
                with self.condition:
                    self.applied_generation=generation;self.condition.notify_all()
                deadline+=1/self.rate
                if time.monotonic()-deadline>.1:raise RuntimeError('servo overrun; no command catch-up burst')
                self.stop_event.wait(max(0.,deadline-time.monotonic()))
        except Exception as exc:
            self.failure=str(exc);self.stop_event.set();self.record('stop',{'reason':self.failure,'motion_outcome':'requires measured/operator review'})
            # Best effort holding cannot substitute for firmware watchdog or a
            # hardware emergency stop; its acceptance remains a separate gate.
            try:self.driver.controller.hold_last_validated_position()
            except Exception:self.record('error',{'reason':'hold failed; operator intervention required'})

    def follow(self,path,planner,scene,*,guard=lambda:None):
        if not self.thread or self.failure or self.stop_event.is_set():raise RuntimeError('servo unavailable')
        points=[self.validate(q).copy() for q in path]
        if not points:raise ValueError('empty path')
        with self.lock:start=self.target.copy()
        prior=start
        for q in points:
            if np.max(abs(q[ARM]-prior[ARM]))>self.step:raise ValueError('trajectory requires retiming; joint step too large')
            if np.max(abs(q[[7,15]]-prior[[7,15]]))>.03:raise ValueError('jaw command step too large')
            prior=q
        # Validation must run on the exact final full dual-arm path and current
        # scene, including carried-object geometry. IK feasibility alone is not
        # a certificate. Planner is a deterministic admitted provider, never LLM.
        planner.validate_path([start,*points],scene)
        self.record('path_validated',{'points':len(points),'planner':planner.identity,'scene_revision':scene['revision']})
        for q in points:
            guard()
            if self.failure or self.stop_event.is_set():raise RuntimeError(self.failure or 'servo stopped')
            with self.condition:
                self.target=q.copy();self.generation+=1;generation=self.generation
                deadline=time.monotonic()+.2
                while self.applied_generation<generation:
                    if self.failure or self.stop_event.is_set():raise RuntimeError(self.failure or 'servo stopped')
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise RuntimeError('no servo receipt; stop rather than skip waypoints')
                    self.condition.wait(remaining)
        self.record('path_dispatched',{'points':len(points),'qualification':'not an arrival or grasp-success verdict'})

    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(2)
        if self.thread and self.thread.is_alive():raise RuntimeError('servo failed to stop; operator intervention required')
