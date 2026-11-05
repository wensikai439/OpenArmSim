"""Align recorded MuJoCo transitions and render derived RGB in an isolated MjData."""
from __future__ import annotations
import os
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import mujoco
import numpy as np

ROOT = Path(str(Path(__file__).resolve().parents[1]))
CAMERAS = ("cam_right", "overview")
ACTION_NAMES = [f"{side}.tcp.{axis}" for side in ("right", "left") for axis in ("x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad")] + ["right.gripper_command", "left.gripper_command"]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""): h.update(block)
    return h.hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

def find_obs(obj):
    if isinstance(obj, dict):
        if all(k in obj for k in ("episode_id", "control_step", "physics_step", "bottle_world_m")): return obj
        for v in obj.values():
            found = find_obs(v)
            if found is not None: return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_obs(v)
            if found is not None: return found
    return None

def outputs(record):
    d = (record.get("response") or {}).get("data", {})
    return d.get("result", {}).get("outputs", {})

def select(task, initial, final, actions, allow_assisted):
    verdict = task.get("verdict") or {}
    transfers = final.get("composable_transfers", [])[len(initial.get("composable_transfers", [])):]
    handoff = any(x.get("donor") == "right" and x.get("receiver") == "left" and len(x.get("contact_geoms", [])) >= 2
                  and x.get("qpos_change") == 0 and x.get("table_contact") is False and x.get("bottle_world_m", [0,0,0])[2] > .78 for x in transfers)
    checks = {
        "official_task_succeeded": task["status"] == "succeeded",
        "official_verdict_success": verdict.get("verdict") == "success" and verdict.get("verifier_status") == "completed",
        "all_official_criteria_satisfied": bool(verdict.get("criteria")) and all(x["status"] == "satisfied" for x in verdict.get("criteria", [])),
        "all_actions_verified": bool(actions) and all(x["status"] == "succeeded" and outputs(x).get("native_result", {}).get("primitive_completed") is True
                                                     and outputs(x).get("native_result", {}).get("verified") is True for x in actions),
        "same_task_airborne_transfer": handoff,
        "bucket_settled": final.get("bucket_bottom_contact") is True and final["bucket_xy_error_m"] < .055 and final["bottle_speed_norm"] < .05,
        "native_healthy_and_released": final.get("failed") is False and final.get("collision_stop") is None
            and not any(final["numerical_warnings"]) and not any(final["assist_active"].values()),
        "assisted_data_explicitly_allowed": allow_assisted,
    }
    return all(checks.values()), checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-assisted", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    specs = json.loads(args.sources.read_text())
    index = {"version": "piper_sim_transitions_v1", "fps": 20, "action_schema": "piper_dual_tcp_absolute_world_rotvec_v1",
        "robot_type": "piper_dual_tcp", "action_names": ACTION_NAMES, "action_units": ["m"]*3+["rad"]*3+["m"]*3+["rad"]*3+["normalized"]*2,
        "action_frame": "world", "action_mode": "absolute", "gripper_range": [-1,1],
        "image_shape": [240,320,3], "cameras": list(CAMERAS), "episodes": [],
        "image_source": "state_replay_from_recorded_simulation",
        "image_processing": "native MuJoCo 640x480 and native cam_right intrinsics remap, then INTER_AREA resize to 320x240; no text overlay",
        "asset": str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa")))), "asset_ownership": "shared independent simulation asset; RPent and PhyAgentOS are consumers",
        "pure_contact_grasp": False, "geometry_calibrated": False,
        "training_compatibility": "TCP actions; not directly compatible with existing joint-position ACT actions",
        "recording_limitations": "qpos/time/action recorded; qvel, ctrl and controller internal state absent. Rendering only; no action replay claim.",
        "preparer_sha256": sha(__file__), "native_camera_implementation_sha256": sha(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa"))) / 'src/piper_robocasa/workstation.py'))}
    for spec in specs:
        task_path = ROOT / spec["task_file"]
        task = json.loads(task_path.read_text()); task = task.get("task", task)
        records = [x for revision in task["revisions"] for x in revision["execution_records"]]
        queries = [find_obs(x["response"]) for x in records if x["tool_id"] == "rpent.observe" and x["status"] == "succeeded"]
        queries = [x for x in queries if x is not None]
        initial, final = queries[0], queries[-1]
        actions = [x for x in records if x["semantics"] == "action"]
        sim = ROOT / spec["sim_dir"]; trace = sim/"physics_trace.jsonl"; model_path = sim/"model.xml"
        before_hashes = {str(x): sha(x) for x in (task_path, trace, model_path)}
        rows = [json.loads(line) for line in trace.open()]
        first, last = initial["control_step"], final["control_step"]
        rows = [x for x in rows if first <= x["step"] <= last]
        if len(rows) < 2 or rows[-1]["step"] != last: raise ValueError("Task trace range is missing")
        if initial["episode_id"] != final["episode_id"] or sim.parent.name != initial["episode_id"]: raise ValueError("Episode mismatch")
        pairs = list(zip(rows[:-1], rows[1:]))
        if any(b["step"] != a["step"]+1 or abs(b["time"]-a["time"]-.05) > 1e-7 for a,b in pairs): raise ValueError("Trace gap")
        model = mujoco.MjModel.from_xml_path(str(model_path))
        if abs(model.opt.timestep-.0005) > 1e-12: raise ValueError("Unexpected actual physics profile")
        joints = [(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),int(model.jnt_qposadr[i]),int(model.jnt_type[i])) for i in range(model.njnt)]
        robot_joints = [(name,adr,kind) for name,adr,kind in joints if name.startswith(("robot0_", "gripper0_"))]
        if len(robot_joints) != 16: raise ValueError("Unexpected robot joint layout")
        state_names = [x[0] for x in robot_joints]; adr = [x[1] for x in robot_joints]
        selected, checks = select(task,initial,final,actions,args.allow_assisted)
        folder = args.out/spec["label"]; folder.mkdir()
        qpos = np.asarray([a["qpos"] for a,b in pairs],dtype=np.float64)
        next_qpos = np.asarray([b["qpos"] for a,b in pairs],dtype=np.float64)
        action = np.asarray([b["action"] for a,b in pairs],dtype=np.float32)
        if qpos.shape[1] != model.nq or action.shape[1] != 14 or not all(np.isfinite(x).all() for x in (qpos,next_qpos,action)): raise ValueError("Invalid numerical trace")
        if np.max(np.abs(action[:,12:])) > 1: raise ValueError("Invalid gripper command")
        np.savez_compressed(folder/"transitions.npz",state=qpos[:,adr].astype(np.float32),next_state=next_qpos[:,adr].astype(np.float32),
            action=action,state_qpos=qpos,next_qpos=next_qpos,sim_time=np.asarray([a["time"] for a,b in pairs]),
            control_step=np.asarray([a["step"] for a,b in pairs]),next_control_step=np.asarray([b["step"] for a,b in pairs]),
            assist=np.asarray([[a["assist"][side] for side in ("right","left")] for a,b in pairs],dtype=np.bool_),
            next_assist=np.asarray([[b["assist"][side] for side in ("right","left")] for a,b in pairs],dtype=np.bool_),
            phase=np.asarray([b["phase"] for a,b in pairs]))
        entry = {"label":spec["label"],"task_id":task["task_id"],"episode_id":initial["episode_id"],
            "status":task["status"],"formal_verdict":(task.get("verdict") or {}).get("verdict"),
            "selected_for_training":selected,"selection_checks":checks,"transitions":len(pairs),
            "task_start_control_step":first,"task_end_control_step":last,"retained_start_control_step":rows[0]["step"],
            "dropped_initial_action_count":max(0,rows[0]["step"]-first), "transition_alignment":"row[i].qpos -- row[i+1].action --> row[i+1].qpos",
            "state_names":state_names,"state_units":["rad" if x[2] == 3 else "m" for x in robot_joints],
            "source":spec,"source_sha256":before_hashes,"transitions_sha256":sha(folder/"transitions.npz"),
            "assisted_grasp":True,"program_origin":spec["program_origin"],
            "final_native_metrics":{k:final.get(k) for k in ("failed","reset_required","bucket_bottom_contact","bucket_xy_error_m","bottle_speed_norm","assist_active","collision_stop","numerical_warnings","task_success","air_handoff_verified")},
            "image_source":index["image_source"] if selected else None}
        if selected:
            data = mujoco.MjData(model)
            # Separate model/data, forward kinematics only. No simulator RPC, mj_step or live qpos writes.
            renderer = mujoco.Renderer(model,480,640)
            y,x = np.mgrid[:480,:640].astype(np.float32); f = 240/np.tan(np.pi/6)
            u = ((x-321.5523986816406)*f/610.22998046875+319.5).astype(np.float32)
            v = ((y-253.65721130371094)*f/609.4248657226562+239.5).astype(np.float32)
            def render(q,t,camera):
                data.qpos[:] = q; data.time = t
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                rgb = renderer.render().copy()
                return cv2.remap(rgb,u,v,cv2.INTER_LINEAR) if camera == "cam_right" else rgb
            for camera in CAMERAS: (folder/"images"/camera).mkdir(parents=True)
            for n,(a,b) in enumerate(pairs):
                for camera in CAMERAS:
                    rgb = cv2.resize(render(a["qpos"],a["time"],camera),(320,240),interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(folder/"images"/camera/f"{n:06d}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if n % 200 == 0: print(json.dumps({"label":spec["label"],"rendered":n+1,"total":len(pairs)}),flush=True)
            # Compare against actual terminal camera captures from this same Task, not against synthetic expectations.
            native_frames = records[-1]["response"]["data"]["response"]["result"]["outputs"]["frames"]
            comparisons = {}
            for camera in CAMERAS:
                reference = next(x for x in native_frames if x["camera_id"] == camera)
                native_path = sim/reference["path"]
                rgb = render(rows[-1]["qpos"],rows[-1]["time"],camera)
                actual = cv2.cvtColor(cv2.imread(str(native_path)),cv2.COLOR_BGR2RGB)
                mae = float(np.abs(rgb.astype(float)-actual.astype(float)).mean())
                comparisons[camera] = {"source_path":str(native_path),"source_sha256":sha(native_path),"mean_absolute_pixel_error":mae,"accepted":mae<=1.0}
                cv2.imwrite(str(folder/f"terminal-replay-{camera}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            renderer.close()
            entry["replay_comparison"] = comparisons
            if not all(x["accepted"] for x in comparisons.values()): raise ValueError("Offline camera replay differs from native capture")
            image_records = [{"frame_index":n,"control_step":a["step"],"sim_time":a["time"],
                "images":{camera:{"path":str(Path("images")/camera/f"{n:06d}.png"),"sha256":sha(folder/"images"/camera/f"{n:06d}.png")} for camera in CAMERAS}} for n,(a,b) in enumerate(pairs)]
            save(folder/"image-index.json",image_records)
            entry["image_index_sha256"] = sha(folder/"image-index.json")
        after_hashes = {x:sha(x) for x in before_hashes}
        if before_hashes != after_hashes: raise RuntimeError("Source changed during derivation")
        entry["raw_sources_unchanged"] = True
        save(folder/"manifest.json",entry)
        index["episodes"].append(entry)
        save(args.out/"manifest.json",index)
        print(json.dumps({"task_id":task["task_id"],"selected":selected,"transitions":len(pairs),"checks":checks}),flush=True)
    save(args.out/"manifest.json",index)
    print(json.dumps({"completed":str(args.out),"training_episodes":sum(x["selected_for_training"] for x in index["episodes"])}),flush=True)

if __name__ == "__main__": main()
