"""Bounded native diagnostics for executed joint targets; no official verdict or dataset admission."""
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(str(Path(__file__).resolve().parents[1]))
ASSET = Path(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa")))))
sys.path.insert(0, str(ASSET/"src"))
sys.path.insert(0, str(ROOT/"bundles/piper-dual-sim/profiles/piper"))
# Optional exact asset lock for NEW project-owned timing diagnostics.
SOURCE_LOCK_PATH = os.environ.get("PAOS_FROZEN_SIM_SOURCE_LOCK")
if SOURCE_LOCK_PATH:
    import hashlib
    lock = json.loads(Path(SOURCE_LOCK_PATH).read_text())
    assert lock["asset_root"] == str(ASSET)
    for name, entry in lock["modules"].items():
        assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    import piper_robocasa
    piper_robocasa.__path__.insert(0, str(Path(SOURCE_LOCK_PATH).parent/"piper_robocasa"))
from joint_expert_adapter import install
from run_sim_expert import load_program, validate_proposal, geometry, physical_success
from piper_robocasa.pose_preflight import check_pose_path

def save(out, name, value):
    (out/name).write_text(json.dumps(value, indent=2, default=str)+"\n")

def smoke(task):
    ws=task.ws
    initial={s:ws.data.qpos[ws.qindexes[s]].copy() for s in ("left","right")}
    results=[]
    for side,delta in (("left",.02),("right",-.02)):
        scratch=mujoco.MjData(ws.model)
        scratch.qpos[:]=ws.data.qpos
        q=initial[side].copy();q[4]+=delta
        scratch.qpos[ws.qindexes[side]]=q
        mujoco.mj_kinematics(ws.model,scratch)
        a,b=ws.robot._action_split_indexes[side]
        ws.action[a:a+3]=scratch.site_xpos[ws.eef_ids[side]]
        ws.action[a+3:b]=Rotation.from_matrix(scratch.site_xmat[ws.eef_ids[side]].reshape(3,3)).as_rotvec()
        for _ in range(int(2 * ws.control_hz)):task.tick()
        actual=ws.data.qpos[ws.qindexes[side]].copy()
        command=ws.joint_command[a:b].copy()
        tracking=float(np.max(np.abs(command-actual)))
        intended=float(np.max(np.abs(command-q)))
        if tracking>.01 or intended>.005:raise RuntimeError("Small joint target failed physical tracking")
        results.append({"side":side,"target_rad":command.tolist(),"actual_rad":actual.tolist(),
                        "max_tracking_error_rad":tracking,"expected_fk_joint_error_rad":intended})
    before={s:ws.data.qpos[ws.qindexes[s]].copy() for s in ("left","right")}
    for _ in range(int(5 * ws.control_hz)):task.tick()
    drift=max(float(np.max(np.abs(ws.data.qpos[ws.qindexes[s]]-before[s]))) for s in before)
    if drift>.01:raise RuntimeError("Five second hold drift exceeded original bound")
    return {"small_moves":results,"hold_5s_max_joint_drift_rad":drift,
            "physics_steps":ws.step_count*int(round((1/ws.control_hz)/ws.model.opt.timestep)),
            "control_steps":ws.step_count,"warnings":[int(w.number) for w in ws.data.warning]}

def expert(task, program, out, max_proposals):
    propose,admission=load_program(program)
    save(out,"program-admission.json",admission)
    o=task.observe();initial_transfer_count=len(o["composable_transfers"]);rows=[];memory={}
    for _ in range(max_proposals):
        o["expert_geometry"]=geometry()
        p=propose(o,memory);validate_proposal(p)
        if p.get("abort"):raise RuntimeError("Expert abort: "+p["abort"])
        if p.get("done"):
            if not physical_success(o,initial_transfer_count):raise RuntimeError("Premature done")
            return {"physical_goal_met":True,"proposals":len(rows),
                    "actions_dispatched":sum("result" in r for r in rows),"final_observation":o}
        action=p["action"];a=action["arguments"];r={"sequence":len(rows)+1,"proposal":p}
        rows.append(r)
        if action["tool_id"] in {"piper.move_arm","rpent.move_arm"}:
            verification=check_pose_path(task.ws,a["side"],a["target_world_m"],a["target_rotation_rad"])
            r["same_parameter_preflight"]=verification
            if not verification["accepted"]:
                if o.get("last_pose_check",{}).get("arguments")==a:raise RuntimeError("Repeated unchanged rejected pose")
                o["last_pose_check"]={"arguments":a,"verification":verification}
                memory=p["memory"]
                r["planning_status"]="rejected_before_action"
                save(out,"actions.json",rows)
                print(json.dumps({"sequence":r["sequence"],"status":"rejected_before_action"}),flush=True)
                continue
        result=task.step({"id":action["tool_id"].split(".",1)[1],"payload":a})
        r["result"]=result;save(out,"actions.json",rows)
        o=task.observe();save(out,"latest-observation.json",o)
        print(json.dumps({"sequence":r["sequence"],"tool":action["tool_id"],"control_step":o["control_step"],
                          "completed":result.get("primitive_completed"),"error":result.get("error")}),flush=True)
        if not result.get("primitive_completed") or not result.get("verified"):raise RuntimeError(str(result))
        memory=p["memory"]
    raise RuntimeError("Native proposal budget exhausted")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--smoke",action="store_true");ap.add_argument("--program",type=Path)
    ap.add_argument("--max-proposals",type=int,default=180);ap.add_argument("--render",action="store_true")
    ap.add_argument("--binary-gripper",action="store_true")
    ap.add_argument("--scene-config",type=Path)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    # Standalone diagnostics own their cancellation path, separate from any running Skill.
    os.environ["PAOS_CANCEL_REQUEST"] = str(out/"cancel-request.json")
    if args.scene_config:
        cfg = json.loads(args.scene_config.read_text())
        os.environ["PAOS_CONTROL_HZ"] = str(cfg["world"]["control_hz"])
    from native_shim import CooperativeTask as PiperAirHandoffTask
    if args.binary_gripper:os.environ["PAOS_GRIPPER_MODE"]="binary_endpoints"
    install()
    timing = os.environ.get("PAOS_MOTION_TIMING_PROFILE", "legacy")
    if timing != "legacy":
        if timing != "feedback_joint_v1":raise ValueError("Unknown project timing profile")
        from joint_collection_timing import install as install_timing, metadata
        install_timing()
        save(out, "timing-profile.json", metadata())
    task=None;error=None;result=None
    try:
        task=PiperAirHandoffTask(out/"sim",render=args.render,run_id=out.name,config_path=args.scene_config)
        if timing != "legacy":
            actual_gains = {side:task.ws.robot.part_controllers[side].kp.tolist() for side in ("left","right")}
            save(out,"timing-profile.json",{**metadata(),"controller_kp":actual_gains})
            contract = json.loads((out/"sim/joint_command_contract.json").read_text())
            contract["kp"] = actual_gains
            contract["gain_selection"] = "Existing shared scene collection_fast, applied after workstation initialization"
            save(out/"sim","joint_command_contract.json",contract)
        save(out,"initial-observation.json",task.observe())
        result=smoke(task) if args.smoke else expert(task,args.program,out,args.max_proposals)
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc)};traceback.print_exc()
    finally:
        if task:
            save(out,"final-observation.json",task.observe());task.save_artifacts();task.close()
        save(out,"summary.json",{"mode":"native diagnostic; no official Task success or dataset admission",
                                  "passed":error is None,"result":result,"error":error,"model_requests":0})
    return 0 if error is None else 1
if __name__=="__main__":sys.exit(main())
