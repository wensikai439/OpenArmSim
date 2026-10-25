"""Continuous dual-arm hold kernel, separated from slow API/tool decisions.

Needs a physically admitted planner and a real initialized LegacyArm. Nothing
starts at import/construction, and this module supplies no unchecked IK fallback.
"""
import threading
import time
from collections import deque

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
        self.watchdog=None;self.progress=time.monotonic();self._halt_lock=threading.Lock()
        self.feedback=deque(maxlen=180);self.guard=lambda:None

    def halt(self,reason):
        """Latch BEFORE logging/holding. A failed disk must never skip the hold."""
        with self._halt_lock:
            if self.failure is not None:return
            self.failure=str(reason);self.stop_event.set()
            if hasattr(self.driver,'stopped'):self.driver.stopped=True
        with self.condition:self.condition.notify_all()
        try:
            self.record('stop',{'reason':self.failure,'motion_outcome':'requires measured/operator review'})
        except Exception:pass
        finally:
            try:self.driver.controller.hold_last_validated_position()
            except Exception:
                try:self.record('error',{'reason':'hold failed; operator intervention required'})
                except Exception:pass

    def _watch(self):
        # Independent of a blocked read/send/logger. This latches future
        # commands; firmware must still handle a blocked CAN call or host death.
        while not self.stop_event.wait(.02):
            if time.monotonic()-self.progress>.15:
                self.halt('independent servo progress watchdog expired');return

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
        self.progress=time.monotonic()
        self.thread=threading.Thread(target=self._loop,name='openarm-hold',daemon=True);self.thread.start()
        self.watchdog=threading.Thread(target=self._watch,name='openarm-watchdog',daemon=True);self.watchdog.start()

    def _loop(self):
        deadline=time.monotonic()
        try:
            while not self.stop_event.is_set():
                self.guard()
                state=self.driver.read();measured=finite(state['q'],(16,))
                self.feedback.append(state)
                with self.lock:target=self.target.copy();generation=self.generation
                if not -.01<=time.time()-state['timestamp_unix_s']<=.1:raise RuntimeError('feedback watchdog expired')
                if np.max(abs(target[ARM]-measured[ARM]))>self.tracking:raise RuntimeError('joint tracking limit exceeded')
                if self.stop_event.is_set():break
                self.driver.send(target,state);self.state=state;self.sends+=1;self.progress=time.monotonic()
                with self.condition:
                    self.applied_generation=generation;self.condition.notify_all()
                deadline+=1/self.rate
                if time.monotonic()-deadline>.1:raise RuntimeError('servo overrun; no command catch-up burst')
                # A delayed tick never produces a burst of accumulated ticks.
                if deadline<time.monotonic():deadline=time.monotonic()+1/self.rate
                self.stop_event.wait(max(0.,deadline-time.monotonic()))
        except Exception as exc:
            self.halt(exc)

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
        if planner.validate_path([start,*points],scene) is not True:
            raise ValueError('planner did not explicitly accept the exact full path')
        self.record('path_validated',{'points':len(points),'planner':planner.identity,'scene_revision':scene['revision']})
        try:
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
        except Exception as exc:
            self.halt(exc);raise
        self.record('path_dispatched',{'points':len(points),'qualification':'not an arrival or grasp-success verdict'})

    def close(self):
        self.halt('session closed')
        if self.thread:self.thread.join(2)
        if self.watchdog:self.watchdog.join(2)
        if self.thread and self.thread.is_alive():raise RuntimeError('servo failed to stop; operator intervention required')


def retime(start,goal,*,rate=30,velocity=.15,acceleration=.3,minimum_seconds=.3):
    """Quintic interpolation with zero endpoint velocity/acceleration, radians.

    This smooths an already collision-planned segment. It is never an obstacle
    avoidance planner and the resulting full path must be checked again.
    """
    start=finite(start,(16,));goal=finite(goal,(16,))
    if not all(np.isfinite(v) and v>0 for v in [rate,velocity,acceleration,minimum_seconds]):
        raise ValueError('invalid trajectory timing')
    distance=float(np.max(abs(goal-start)))
    duration=max(minimum_seconds,1.875*distance/velocity,np.sqrt(5.774*distance/acceleration))
    ticks=max(2,int(np.ceil(duration*rate)))
    t=np.arange(1,ticks+1)/ticks;s=10*t**3-15*t**4+6*t**5
    return [start+(goal-start)*v for v in s]
