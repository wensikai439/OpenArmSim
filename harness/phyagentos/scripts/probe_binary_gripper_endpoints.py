"""Isolated empty-gripper endpoint physics probe; does not produce an official task verdict."""
import argparse,json,os,hashlib,sys,traceback
from pathlib import Path
import numpy as np
def save(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+"\n")
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--closed-clearance-m",type=float,default=0.0)
    ap.add_argument("--via-task-action",action="store_true")
    a=ap.parse_args();assert 0.0<=a.closed_clearance_m<=.001
    if a.via_task_action:
        assert a.closed_clearance_m==.0001, "Binary task mode uses the fixed 0.1mm numerical clearance"
        os.environ["PAOS_GRIPPER_MODE"]="binary_endpoints"
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ["PAOS_CANCEL_REQUEST"]=str(a.out/"cancel-request.json")
    from run_joint_expert_native import install,PiperAirHandoffTask
    install();task=None;error=None;checks=[]
    try:
        task=PiperAirHandoffTask(a.out/"sim",render=False,run_id=a.out.name)
        original_safety=task.ws.safety
        def record_safety_failure():
            try:original_safety()
            except Exception:
                import mujoco
                contacts=[]
                for c in task.data.contact[:task.data.ncon]:
                    if c.dist<-.001:
                        contacts.append({"dist_m":float(c.dist),"geoms":[mujoco.mj_id2name(task.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)],"bodies":[mujoco.mj_id2name(task.model,mujoco.mjtObj.mjOBJ_BODY,int(task.model.geom_bodyid[g])) for g in (c.geom1,c.geom2)]})
                save(a.out/"safety-contact-failure.json",{"contacts":contacts,"control_step":int(task.ws.step_count),"qpos":task.data.qpos.tolist(),"qvel":task.data.qvel.tolist(),"guard_bypassed":False})
                raise
        task.ws.safety=record_safety_failure
        initial=task.observe();save(a.out/"initial-observation.json",initial)
        assert not any(initial["assist_active"].values()) and not initial["failed"]
        for side in ("left","right"):
            for mode,command,target in (("OPEN",-1.,.1),("CLOSE",1.,0.)):
                start=int(task.ws.step_count)
                consumed_command=command if mode=="OPEN" else 1.0-a.closed_clearance_m/.05
                if a.via_task_action:
                    result=task.step({"id":"set_gripper","payload":{"side":side,"open_or_close":mode.lower()}})
                    save(a.out/f"{side}-{mode.lower()}-native-action.json",result)
                    assert result.get("primitive_completed") is True and result.get("verified") is True, result.get("error")
                else:
                    task.gripper(side,consumed_command,40)  # native physics/controller stepping, no qpos assignment
                o=task.observe();index=task.ws.robot._action_split_indexes[side+"_gripper"][0]
                actual_command=float(task.ws.action[index])
                width=float(o["gripper_feedback"][side]["joint_opening_m"])
                warning_counts=[int(w.number) for w in task.model.warning] if hasattr(task.model,"warning") else [int(w.number) for w in task.data.warning]
                item={"side":side,"mode":mode,"native_requested_command":command,"native_consumed_command":actual_command,"task_endpoint_goal_m":target,
                      "servo_width_target_m":target if mode=="OPEN" else a.closed_clearance_m,
                      "measured_named_slide_opening_m":width,"absolute_error_m":abs(width-target),
                      "control_step_start":start,"control_step_end":int(task.ws.step_count),
                      "warnings":warning_counts,"assist_active":o["assist_active"],"failed":o["failed"],
                      "qpos_finite":bool(np.isfinite(task.data.qpos).all()),"qvel_finite":bool(np.isfinite(task.data.qvel).all())}
                checks.append(item);save(a.out/"checks.json",checks)
                assert actual_command==consumed_command
                assert abs(width-target)<=.002, "Endpoint opening did not reach the predeclared 2mm tolerance"
                assert not any(warning_counts) and item["qpos_finite"] and item["qvel_finite"]
                assert not any(o["assist_active"].values()) and not o["failed"]
        save(a.out/"final-observation.json",task.observe())
    except Exception as e:error={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
    finally:
        if task is not None:
            control_steps=int(task.ws.step_count);dt=float(task.model.opt.timestep)
            task.close()
        else:control_steps=0;dt=None
    report={"probe_success":error is None,"checks":checks,"error":error,
            "scope":"Empty-finger full endpoints only; not a bottle grip/handoff rollout or official verdict.",
            "threshold_fixed_before_execution_m":.002,"closed_numerical_clearance_m":a.closed_clearance_m,"new_owned_simulator_instance":True,
            "via_task_action":a.via_task_action,"gripper_mode":"binary_endpoints" if a.via_task_action else "direct endpoint probe",
            "original_live_runtime_touched":False,"hardware_connected":False,"model_requests":0,
            "task_verdict_created":False,"qpos_overrides_during_probe":0,"assist_enabled":False,
            "control_steps_total":control_steps,"physics_dt_s":dt,
            "physics_substeps_expected":control_steps*int(round(.05/dt)) if dt else 0,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(a.out/"acceptance.json",report)
    print(json.dumps({"probe_success":report["probe_success"],"checks":checks,"report":str(a.out/"acceptance.json"),"error":error}),flush=True)
    return 0 if error is None else 1
if __name__=="__main__":raise SystemExit(main())
