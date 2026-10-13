"""Native preview parity, immutability and intentional rejection checks."""
import json
from types import SimpleNamespace

import mujoco
import numpy as np

from harness.object_handover import ProtectedObjectEnv
from .predictive_place import NativePreview
from .tamp import PlanningFailure


def main():
    rows=[]; env=ProtectedObjectEnv(images=False)
    try:
        env.reset(seed=0); env.task_termination='skill'; env.set_task_budget(120)
        predictor=NativePreview(env)
        model_before=env.model.actuator_biasprm.copy()
        q0=env.data.qpos.copy(); v0=env.data.qvel.copy(); ctrl0=env.data.ctrl.copy(); t0=float(env.data.time)
        action=env.last_action.copy(); action[1]+=.002; action[9]-=.002
        result=predictor.run(action,'right',seconds=.15,guard=False)
        assert result['physics_steps']==150 and abs(result['integrated_seconds']-.15)<1e-8
        for a,b in [(q0,env.data.qpos),(v0,env.data.qvel),(ctrl0,env.data.ctrl),
                    (model_before,env.model.actuator_biasprm)]:np.testing.assert_array_equal(a,b)
        assert float(env.data.time)==t0 and env.tick==0 and env._clock_mode is None
        rows.append({'check':'full150ms preview leaves live state, clock and model unchanged','passed':True})
        # Verify each native endpoint after a copied prediction. All actuation,
        # warm start and gravity compensation must reproduce exact execution.
        errors=[]
        for index in range(12):
            action=env.last_action.copy(); action[1]+=.001 if index%2 else -.001
            target=(env.tick+1)/env.profile['control_hz']+getattr(env,'_periodic_offset',0.)
            seconds=round((target-env.data.time)/env.model.opt.timestep)*env.model.opt.timestep
            predictor.run(action,'right',seconds=seconds,guard=False)
            predicted={name:getattr(predictor.data,name).copy() for name in ['qpos','qvel','ctrl','qacc_warmstart','qfrc_applied']}
            env.step_tamp(action)
            error=max(float(np.max(abs(value-getattr(env.data,name)))) if value.size else 0 for name,value in predicted.items())
            errors.append(error)
            assert error==0.,(index,error)
        rows.append({'check':'12original30Hz endpoints exact native dynamic parity','passed':True,'maximum_absolute_error':max(errors)})
        try:
            invalid=env.last_action.copy(); invalid[0]=100
            predictor.run(invalid,'right')
            raise AssertionError('native bounds were bypassed')
        except ValueError:pass
        rows.append({'check':'original native action-limit rejection','passed':True})
        # Collision classification on a deliberately intersecting scratch-only
        # free body; no live geometry or task ever modified.
        mujoco.mj_copyData(predictor.data,env.model,env.data)
        address=env.registry['bottle']['qposadr']
        predictor.data.qpos[address:address+3]=env.data.xpos[env.registry['block']['body']]
        mujoco.mj_forward(env.model,predictor.data)
        pair=predictor.collision('left')
        assert pair and any('block' in name for name in pair['geoms']),pair
        rows.append({'check':'actual scratch bottle/block penetration rejected','passed':True,'pair':pair})
        try:
            NativePreview(SimpleNamespace(source_controller=object()))
            raise AssertionError('stateful controller accepted')
        except PlanningFailure as exc:assert exc.code=='preview_controller_unsupported'
        rows.append({'check':'unsupported stateful source controller fails closed','passed':True})
        print(json.dumps({'passed':all(row['passed'] for row in rows),'checks':rows},ensure_ascii=False,indent=2))
    finally:env.close()


if __name__=='__main__':main()
