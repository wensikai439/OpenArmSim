"""Prepare one formally successful joint-command episode; render actual recorded pre-states."""
from __future__ import annotations
import argparse, hashlib, json, sys, os
from pathlib import Path
import cv2, mujoco, numpy as np
from PIL import Image

ROOT=Path(str(Path(__file__).resolve().parents[1]))
ASSET=Path(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa")))))
sys.path.insert(0,str(ASSET/"src"))
# Select the exact recorded asset before importing simulator modules.
source_lock = os.environ.get("PAOS_FROZEN_SIM_SOURCE_LOCK")
if source_lock:
    lock = json.loads(Path(source_lock).read_text())
    for entry in lock["modules"].values():
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Renderer asset source hash mismatch")
    import piper_robocasa
    piper_robocasa.__path__.insert(0, str(Path(source_lock).parent/"piper_robocasa"))
from piper_robocasa.air_handoff_task import PiperAirHandoffTask
from run_sim_expert import physical_success, observation
SIDES=("left","right")
CAMERAS=("cam_left","cam_middle","cam_right")
NAMES=[name for side in SIDES for name in ([f"{side}_joint{i}" for i in range(1,7)]+[f"{side}_gripper_width"])]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,default=str)+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--task-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    task_dir=args.task_dir.resolve();out=args.out.resolve()
    task=read(task_dir/"task-final.json");sources=read(task_dir/"dataset-sources.json")
    final=observation(read(task_dir/"final-query.json"))
    if task["status"]!="succeeded" or task.get("verdict",{}).get("verdict")!="success":
        raise ValueError("Only original official succeeded/success Task is admitted")
    if not physical_success(final) or final.get("controller_mode")!="JOINT_POSITION:absolute":
        raise ValueError("Strict physical goal and actual joint profile are required")
    source=sources[0];sim=(ROOT/source["sim_dir"]).resolve()
    if final["episode_id"]!=sim.parent.name or task["task_id"]!=source["label"]:
        raise ValueError("Task/episode source association mismatch")
    contract=read(sim/"joint_command_contract.json")
    fps=int(contract["control_hz"])
    if fps not in (20,30):raise ValueError("Unsupported source sampling rate")
    if contract["input_type"]!="absolute" or contract["joint_unit"]!="rad":
        raise ValueError("Wrong source command semantics")
    rows=[json.loads(l) for l in (sim/"joint_commands.jsonl").read_text().splitlines()]
    if len(rows)<32 or any(not r.get("completed") for r in rows):
        raise ValueError("Incomplete or too short command trace")
    binary_present=["gripper_task_modes" in r for r in rows]
    if any(binary_present) and not all(binary_present):
        raise ValueError("Partial explicit binary intent trace")
    binary=all(binary_present)
    modes=np.asarray([[r["gripper_task_modes"][s] for s in SIDES] for r in rows]) if binary else None
    if binary:
        binary_contract=read(sim/"binary_gripper_contract.json")
        if binary_contract["task_mode_values"]!={"CLOSE":0,"OPEN":1} or not np.isin(modes,[0,1]).all():
            raise ValueError("Invalid explicit binary intent contract")
    model=mujoco.MjModel.from_xml_path(str(sim/"model.xml"))
    splits=contract["native_action_splits"]
    indexes={s:[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)])
                for n in contract["joint_names"][s]] for s in SIDES}
    fingers={s:[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"gripper0_{s}_{f}_finger_joint")])
                for f in ("l","r")] for s in SIDES}
    state=[];action=[];assist=[];assist_post=[];times=[];post_qpos=[];tracking=[];max_delta=0.
    for n,r in enumerate(rows):
        cmd=np.asarray(r["command_native"],dtype=np.float64)
        if r["transition_index"]!=n or r["physics_substeps_done"]!=r["physics_substeps_expected"]:
            raise ValueError("Trace step association mismatch")
        if not r.get("command_created_before_step"):raise ValueError("No prior command evidence")
        q=np.asarray(r["pre"]["qpos"]);post=np.asarray(r["post"]["qpos"])
        if not all(np.isfinite(np.asarray(x)).all() for x in (cmd,q,post,r["pre"]["qvel"],r["post"]["qvel"],r["ctrl_first"])):
            raise ValueError("Nonfinite source")
        if any(r["post"]["warnings"]):raise ValueError("Numerical warning")
        if abs(r["post"]["time_s"]-r["pre"]["time_s"]-1/fps)>1e-7:raise ValueError("Source record period differs from actual contract")
        if n:
            np.testing.assert_array_equal(q,rows[n-1]["post"]["qpos"])
            if abs(r["pre"]["time_s"]-rows[n-1]["post"]["time_s"])>1e-9:raise ValueError("Time discontinuity")
        sr=[];ar=[];tr=[]
        for s in SIDES:
            a,b=splits[s];goal=cmd[a:b]
            np.testing.assert_array_equal(goal,r["controller_consumed_joint_targets_rad"][s])
            limits=np.asarray(contract["joint_limits_rad"][s])
            if np.any(goal<limits[:,0]-1e-9) or np.any(goal>limits[:,1]+1e-9):raise ValueError("Target outside joint bounds")
            if n:max_delta=max(max_delta,float(np.max(np.abs(goal-np.asarray(rows[n-1]["command_native"])[a:b]))))
            gr=float(cmd[splits[s+"_gripper"][0]])
            if not -1<=gr<=1:raise ValueError("Invalid native gripper command")
            sr.extend(q[indexes[s]].tolist()+[float(np.abs(q[fingers[s]]).sum())])
            ar.extend(goal.tolist()+[.05*(1-gr)])
            tr.extend((goal-post[indexes[s]]).tolist())
        state.append(sr);action.append(ar);tracking.append(tr)
        times.append(r["pre"]["time_s"]);post_qpos.append(post)
        assist.append(r["pre"]["eq_active"])
        assist_post.append(r["post"]["eq_active"])
    if max_delta>.5+1e-9:raise ValueError("Actual joint command discontinuity exceeds original 0.5rad bound")
    out.mkdir(parents=True,exist_ok=False);folder=out/source["label"];folder.mkdir()
    numeric=dict(state=np.asarray(state,dtype=np.float32),action=np.asarray(action,dtype=np.float32),
                 assist=np.asarray(assist,dtype=np.float32),assist_post=np.asarray(assist_post,dtype=np.float32),
                 source_transition_index=np.arange(len(rows),dtype=np.int64),source_time_s=np.asarray(times),
                 post_qpos=np.asarray(post_qpos),tracking_error_rad=np.asarray(tracking))
    if binary:numeric["gripper_task_modes"]=modes.astype(np.int8)
    np.savez_compressed(folder/"transitions.npz",**numeric)
    replay=PiperAirHandoffTask(out/"state-renderer",render=True,run_id="offline-state-replay-only",config_path=sim/"scene_spec.json")
    image_index=[]
    try:
        if sha(sim/"model.xml")!=sha(out/"state-renderer/model.xml"):
            raise ValueError("Independent renderer model differs from source")
        for n,r in enumerate(rows):
            replay.ws.restore_state(r["pre"]["full_state"])
            np.testing.assert_array_equal(replay.data.qpos,r["pre"]["qpos"])
            if replay.data.time!=r["pre"]["time_s"]:raise ValueError("Renderer state time mismatch")
            refs={}
            for camera in CAMERAS:
                rgb=cv2.resize(replay.ws.frame(camera),(320,240),interpolation=cv2.INTER_AREA)
                dest=folder/"images"/camera/f"{n:06d}.jpg";dest.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb).save(dest,quality=90)
                refs[camera]={"path":str(dest.relative_to(folder)),"sha256":sha(dest)}
            image_index.append({"transition_index":n,"source_time_s":r["pre"]["time_s"],"images":refs})
            if n%400==0:print(json.dumps({"rendered_pre_states":n+1,"total":len(rows)}),flush=True)
    finally:
        replay.close()
    save(folder/"image-index.json",image_index)
    manifest={
        "task_id":task["task_id"],"episode_id":final["episode_id"],"label":source["label"],
        "selected_for_export":True,
        "selected_for_training":final.get("scene_initialization",{}).get("split") == "train",
        "training_scope":"native asset joint-command dataset; frozen whole-episode split and cross-domain limits retained",
        "program_origin":source["program_origin"],
        "scene_initialization":read(sim/"scene_initialization.json") if (sim/"scene_initialization.json").exists() else None,
        "scenario_record":read(task_dir/"scenario-record.json") if (task_dir/"scenario-record.json").exists() else None,
        "length":len(rows),"state_names":NAMES,
        "transitions_sha256":sha(folder/"transitions.npz"),"image_index_sha256":sha(folder/"image-index.json"),
        "sources":{str(p):sha(p) for p in (task_dir/"task-final.json",sim/"joint_commands.jsonl",
                                         sim/"joint_command_contract.json",sim/"model.xml",sim/"physics_trace.jsonl")},
        "physical_goal_met":True,"official_verdict":"original Task succeeded/success",
        "semantic_reviewer":read(task_dir/"summary.json").get("reviewer","deepseek-flash"),
        "reviewer_provenance":read(task_dir/"reviewer.json") if (task_dir/"reviewer.json").exists() else None,
        "command_consumption_verified_rows":len(rows),"pre_post_qpos_continuity":True,
        "max_joint_command_delta_rad":max_delta,
        "tracking_error_abs_max_rad":float(np.max(np.abs(tracking))),
        "tracking_error_abs_p99_rad":float(np.percentile(np.abs(tracking),99)),
        "assist_order":["right","left"],"assist_pre_and_post_preserved":True,
        "binary_gripper_modes":{"available":binary,"order":list(SIDES),
            "source":"joint_commands.jsonl.gripper_task_modes" if binary else None,
            "inferred_from_servo_width":False,
            "contract":binary_contract if binary else None},
        "binary_contract_sha256":sha(sim/"binary_gripper_contract.json") if binary else None,
        "control_hz":fps,"state_unit":"12 joints rad + two measured software finger openings m",
        "action_unit":"12 actually consumed absolute joint targets rad + two servo software opening targets m",
        "image_source":"actual recorded pre-state replay in independent renderer; no physics stepping",
        "cross_domain_zero_sign_calibrated":False,"physical_gripper_gap_calibrated":False}
    if binary:manifest["sources"][str(sim/"binary_gripper_contract.json")]=sha(sim/"binary_gripper_contract.json")
    save(folder/"manifest.json",manifest)
    top={"fps":fps,"action_schema":"piper_dual_executed_joint_absolute_software_width_v1","action_names":NAMES,
         "state_names":NAMES,"cameras":list(CAMERAS),"robot_type":"piper_dual_joint_contact_assisted_sim",
         "image_source":manifest["image_source"],"episodes":[manifest],"cross_domain_mixing_ready":False,
         "remaining_cross_domain_contracts":["Explicit camera/interpolation/assistance data admission",
                                            "Real binary task-label transition coverage",
                                            "Physical calibration for hardware deployment remains separate"],
         "training_started":False,"preparer_sha256":sha(__file__)}
    save(out/"manifest.json",top)
    print(json.dumps({"selected_episode":task["task_id"],"transitions":len(rows),"out":str(out)}),flush=True)
    return 0
if __name__=="__main__":sys.exit(main())
