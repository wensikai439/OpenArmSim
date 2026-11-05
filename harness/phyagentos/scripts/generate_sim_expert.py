"""Use one DS Flash request per expert version, preserving prompts, responses and failure provenance."""
from __future__ import annotations
import os
import argparse, ast, asyncio, hashlib, json, time
from pathlib import Path
from PhyAgentOS.config.loader import load_config
from PhyAgentOS.cli.commands import _make_provider
ROOT=Path(str(Path(__file__).resolve().parents[1]))

def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def obs(x):
    if isinstance(x,dict):
        if "bottle_world_m" in x and "tcp_world_m" in x:return x
        for v in x.values():
            r=obs(v)
            if r is not None:return r
    elif isinstance(x,list):
        for v in x:
            r=obs(v)
            if r is not None:return r
def compact(o):
    keys=["episode_id","physics_step","control_step","bottle_world_m","bucket_world_m","tcp_world_m","tcp_rotation_rad","bottle_rotation_matrix",
        "assist_active","failed","reset_required","collision_stop","numerical_warnings","finger_contact_now","receiver_ready",
        "measured_grasp_candidates","tabletop_approach","primitive_geometry","bucket_bottom_contact","bucket_xy_error_m","bottle_speed_norm","composable_transfers"]
    return {k:o[k] for k in keys if k in o}

CONTRACT="""Write an observation-feedback expert program for actual contact-assisted right-to-left airborne bottle handoff and release/settling in the left bucket on our shared independent PiPER MuJoCo asset.
Return Python source only, at most 180 lines. Keep helpers compact. Use required observed priors, never fixed absolute coordinate fallbacks. Abort if required priors are absent. Only import math. Define propose(observation, memory), with empty dict memory at start. No top-level execution except import/constants/function definitions.
Return exactly one proposal per call:
{"action":{"tool_id":"rpent.move_arm","arguments":{"side":"right|left","target_world_m":[x,y,z],"target_rotation_rad":[rx,ry,rz]}},"memory":{...},"reason":"..."}
or same structure with rpent.set_gripper {side,open_or_close:"open|close"}, rpent.grasp_bottle {side}, rpent.release_bottle {side}.
Return {"done":True,"memory":{...},"reason":"..."} only when observed new airborne composable_transfer right->left, bucket_bottom_contact=true, bucket_xy_error_m<0.055, bottle_speed_norm<0.05, no assistance active, failed=false, collision_stop=null and warnings all zero.
Return {"abort":"specific reason","memory":{...}} on native failed/reset_required/collision_stop/warnings or missing mandatory measured observations. Never turn a native failure into done, never continue a failed episode.
Trusted runner provides fresh native observations after each successful action; it performs same-argument rpent.check_pose before every move, serial admission/terminal waits, and strict official Task verification. A rejected read-only pose returns observation.last_pose_check={arguments,verification} for a new proposal in the same bounded planning loop, with no Action or physics advance. Handle this feedback; never repeat identical rejected arguments. Your program ONLY proposes actions; do not call APIs, files, network, subprocess, eval/exec, importlib, hardware, time, random, simulator or database.
No fixed episode action list, no trajectory copied from history, no unconditional success counter. Derive targets from current bottle/bucket/TCP observations, pose priors and bottle_rotation_matrix. Maintain phase/progress in memory. Segment movement, observe achieved TCP, maintain hold orientation when carrying, guard measured assistance/contact before lifting/transfer and release. Do not blindly close +1: grasp_bottle uses the native tested contact-triggered assistance profile.
Positions are absolute WORLD metres, rotations absolute WORLD exponential-map radians. Both arms are six joints with joint-limited IK. Bottle_world_m is bottle BASE, not centre. Physics dt=0.0005, control=20Hz, native move interpolates at <=4mm control tick plus only 40 convergence ticks, failure if actual TCP error>20mm. Forge timeout does not lengthen convergence. IK check is kinematic only, not proof of dynamics.
Tabletop_approach contains right horizontal-side grasp priors, including transition rotation, staging, pregrasp and grasp targets. Stage orientation/clearance before descent. Primitive_geometry.handoff.shared_bottle_base_region_world_m is a prior usable for staging; it is not an executed route.
A geometry calculation: target pinch midpoint = bottle_base + bottle_R @ [0,0,grasp_height]. Tool TCP = pinch midpoint - tool_R @ expert_geometry.finger_pair_mesh_midpoint_tcp_local_m[side]. Use Rodrigues rotation from rotvec. Metadata center approx [0,0,-0.02025]m is measured from exact current asset finger mesh AABBs; x/y are tiny and finger symmetry can change slightly. Recheck actual bilateral contacts through native grasp.
Native grasp_bottle(left) while right assists only establishes receiver_ready=left after two measured receiving finger bodies contact; it does NOT transfer yet. Require actual receiver_ready=left, two left finger contacts, right hold/contact, no table contact and bottle base z>0.78, then set_gripper(right,open) triggers measured constraint transfer. Require a NEW composable_transfers entry and left hold/contact before any movement. Report assisted grasp, never pure friction.
Do not modify task_success or air_handoff_verified: composable actions may legitimately leave macro-only flags false.
A high-level goal is still a hypothesis until a fresh physical rollout. Prefer aborting unsupported geometry to fabricating progress.
"""

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--previous",type=Path)
    ap.add_argument("--feedback-task",type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)  # Existing attempt means no duplicate paid request.
    cfg=load_config()
    if cfg.agents.defaults.model not in {"custom/deepseek-flash","deepseek-flash"}:raise RuntimeError("Expected explicitly selected DS Flash configuration")
    provider=_make_provider(cfg)
    if (provider.api_base or "").rstrip("/") not in {"https://api.deepseek.com","https://api.deepseek.com/v1"}:raise RuntimeError("Unexpected Flash endpoint")
    initial_path=ROOT/"runs/ik-handoff-20261004/forge-handoff-02/initial-query.json"
    initial=compact(obs(json.loads(initial_path.read_text())))
    geometry_path=ROOT/"runs/deepseek-flash-handoff-20261010-04/finger-geometry-diagnosis.json"
    geometry=json.loads(geometry_path.read_text())
    measured={"finger_pair_mesh_midpoint_tcp_local_m":{s:a["finger_pair_mesh_midpoint_tcp_local_m"] for s,a in geometry["arms"].items()},
        "source_path":str(geometry_path),"source_sha256":sha(geometry_path),"geometry_calibrated":False}
    context={"real_initial_observation":initial,"expert_geometry":measured,"goal":"actual assisted air handoff and bucket settling"}
    context["real_initial_observation"]["expert_geometry"]=measured
    if args.previous:
        if not args.feedback_task:raise ValueError("Revision requires actual failure source")
        t=json.loads(args.feedback_task.read_text());t=t.get("task",t)
        records=[x for r in t["revisions"] for x in r["execution_records"]]
        failed_actions=[x for x in records if x["semantics"]=="action" and x["status"]=="failed"]
        if failed_actions:
            failed=failed_actions[-1];i=records.index(failed)
            check=next((x for x in reversed(records[:i]) if x["tool_id"]=="rpent.check_pose" and x["arguments"]==failed["arguments"]),None) if failed["tool_id"]=="rpent.move_arm" else None
            native=failed["response"]["data"]["result"]["outputs"]["native_result"]
            rejection=None
            failure={"kind":"executed_action_failure","record_id":failed["record_id"],"arguments":failed["arguments"],
                "error":native.get("error"),"control_steps":native.get("control_steps")}
            after_observation=compact(native["relay_state"])
        else:
            rejected=[x for x in records if x["tool_id"]=="rpent.check_pose" and x.get("response",{}).get("data",{}).get("response",{}).get("result",{}).get("outputs",{}).get("verification",{}).get("accepted") is False]
            if not rejected:raise ValueError("No actual failed Action or rejected kinematic candidate in feedback Task")
            check=rejected[-1];i=records.index(check)
            rejection=check["response"]["data"]["response"]["result"]["outputs"]
            failure={"kind":"read_only_preflight_rejection_no_action","record_id":check["record_id"],"arguments":check["arguments"],
                "verification":rejection["verification"],"action_dispatched":False}
            final_query=next(x for x in reversed(records) if x["tool_id"]=="rpent.observe")
            after_observation=compact(obs(final_query["response"]))
        before=next(x for x in reversed(records[:i]) if x["tool_id"]=="rpent.observe")
        prior_source=args.feedback_task.parent/"program-source.json"
        prior=json.loads(prior_source.read_text()) if prior_source.exists() else {}
        same_program=prior.get("sha256")==sha(args.previous) and prior.get("task_id")==t["task_id"]
        review_path=args.previous.parent/"static-review.json"
        if not review_path.exists(): review_path=args.previous.parent/"admission.json"
        feedback={"source_task_id":t["task_id"],"source_task_file":str(args.feedback_task),"source_task_sha256":sha(args.feedback_task),
            "origin":"generated expert rollout" if same_program else "historical external/planner rollout; not a rollout of the previous expert",
            "generated_program_caused_this_failure":same_program,
            "failure":failure,"formal_verdict":t.get("verdict"),
            "same_parameter_kinematic_check":{"record_id":check["record_id"],"arguments":check["arguments"],"receipt":check["response"]} if check else None,
            "before_failure_observation":compact(obs(before["response"])),"after_failure_observation":after_observation,
            "known_limits":{"native_position_failure_threshold_m":.02,"max_extra_control_ticks":40},
            "hypotheses_to_test":["Use real read-only check feedback to revise an unadmitted candidate position/orientation; recheck every new proposal.",
                "Retain short measured hold moves and donor separation; no retry after an executed native failure."] }
        save(args.out/"feedback.json",feedback)
        context={"previous_program":args.previous.read_text(),"previous_sha256":sha(args.previous),"actual_failure_feedback":feedback,
            "previous_static_review":json.loads(review_path.read_text()),"expert_geometry":measured,"revision_request":"Revise the feedback expert around the measured failure: retreat empty donor after confirmed transfer/opening; <=10mm held-left movement with unchanged measured rotation; save transfer-time measured bottle-to-TCP offset. Native failure must abort immediately. Fix every previous static-review finding. Transfer fields are donor/receiver/contact_geoms/qpos_change/table_contact/bottle_world_m; from/to do NOT exist. Save initial transfer count and require a new matching airborne bilateral-contact entry. Fully approach the right observed grasp target before grasp_bottle; lift using measured TCP-to-bottle offset, not base.z as TCP. For left receiving pose use current bottle pose, horizontal right measured tool rotation and mesh pinch offset with a separate bottle-axis grasp height to avoid donor fingers. Carry bottle over rim using held offset; release from a small clearance above rim if receiving grasp is low, rather than lowering gripper into bucket walls. Compute all targets from observed asset geometry. Abort if held contact/assistance is lost. Done must satisfy physical contract. Do not claim these hypotheses solved the error."}
    if args.previous:
        context["revision_request"] += (" Address this actual feedback first. A rejected check_pose is a read-only planning rejection, not an executed failed Action. The runner now passes observation.last_pose_check={arguments,verification} on rejected candidates without physics advancement or Action dispatch; proposes within one bounded loop, and aborts unchanged rejected arguments. For an empty arm use the returned suggested_sample_pose or revise orientation/position from measured priors, then let the runner recheck. For a held bottle maintain its upright pose and use smaller measured movement instead of blindly changing its orientation. Do not confuse rotation-vector sign/wrap with matrix orientation. Avoid following measured orientation drift into joint-limit failure during empty-arm clearance. All revised targets must be derived from current observations/check feedback, not a historical trajectory.")
    if args.previous:
        context["revision_request"] += (" Handle actual rejected poses in held carry phases, not only empty-arm phases. When current native hold/contact is valid, consider the actual suggested_sample_pose as a new candidate, and predict held bottle orientation R_new_tcp @ transpose(R_current_tcp) @ R_current_bottle from the measured rigid assistance transform. Only choose it if predicted bottle upright z remains >=0.98; actual observations must still confirm hold/upright after execution. Derive each desired bottle goal to TCP using the CURRENT measured TCP minus bottle-base offset, updated after any accepted orientation change. Remember an accepted revised hold rotation for later proposals so you do not revert to the rejected old orientation. Apply the same measured transform reasoning to receiving-arm candidates if needed. If a supported candidate cannot be found, abort honestly; no hidden motion or changes to physical thresholds.")
    if args.previous and args.feedback_task.parent.name == "airborne-receiver-native-rollout":
        action_path=args.feedback_task.parent/"actions.json"
        action_rows=json.loads(action_path.read_text())
        actual=[]
        for row in action_rows:
            proposal=row.get("proposal",{});action=proposal.get("action",{})
            if action.get("tool_id")!="rpent.move_arm" or action.get("arguments",{}).get("side")!="left":continue
            native=row.get("terminal_status",{}).get("data",{}).get("result",{}).get("outputs",{}).get("native_result",{})
            actual.append({"sequence":row["sequence"],"phase":proposal.get("memory",{}).get("phase"),
                "arguments":action["arguments"],"actual_tcp":native.get("relay_state",{}).get("tcp_world_m",{}).get("left"),
                "verification":native.get("verification"),"error":native.get("error")})
        context["actual_receiver_motion_window"]={"source_path":str(action_path),"source_sha256":sha(action_path),"rows":actual[-9:]}
        context["revision_request"] += (" This actual executed failure occurred BEFORE transfer, in low receiving approach with nearly downward measured wrist rotation. Same-parameter native preflight accepted, but actual residual grew from 6.7 to16.3 to19.5 to20.2mm while y motion stagnated and z drifted down. Smaller steps or unchanged downward wrist alone are not established solutions. Fix only this actual obstacle; preserve working source pickup, carry and donor yaw, all native thresholds and180-proposal budget. Consider raising/staging the EMPTY receiver at clearance derived from observed bottle top/diameter and its own initial pose before changing to a geometry-derived reachable grasp orientation; do not force a rotation at the previous low anchor or repeat a stagnant approach. Alternatively propose an observation-derived mutually reachable contact band, with exact same-argument check_pose on every candidate. Bottle base is not center. Native grasp(left) ONLY sets receiver_ready, then donor OPEN triggers actual transfer; require both measured receiving contacts/readiness before opening and a NEW transfer/left assistance afterward. Return complete program; fewer than300 lines, keep unchanged successful phases compact. No base/controller tuning, simulator changes, old trajectory copying or unknown Action retry.")
    messages=[{"role":"system","content":CONTRACT},{"role":"user","content":json.dumps(context,ensure_ascii=False)}]
    save(args.out/"request.json",{"model":"deepseek-flash","provider":"custom","endpoint":provider.api_base,
        "max_tokens":6000,"temperature":.2,"reasoning_effort":"none","messages":messages,"generator_sha256":sha(Path(__file__))})
    save(args.out/"request-state.json",{"status":"submitted_once","submitted_at":time.time(),"automatic_retries":0})
    print(json.dumps({"request":"revision" if args.previous else "generate","model":"deepseek-flash","out":str(args.out)}),flush=True)
    try:
        response=await provider.chat(messages,model="deepseek-flash",max_tokens=6000,temperature=.2,reasoning_effort="none")
    finally:
        await provider._client.close()
    save(args.out/"response.json",{"content":response.content,"finish_reason":response.finish_reason,"usage":response.usage,
        "reasoning_content":response.reasoning_content,"tool_calls":[str(x) for x in response.tool_calls]})
    save(args.out/"request-state.json",{"status":"response_received_once","received_at":time.time(),"automatic_retries":0,"response_sha256":sha(args.out/"response.json")})
    if response.finish_reason!="stop" or response.tool_calls:raise RuntimeError("Generation did not complete; raw response retained, no retry")
    code=(response.content or "").strip()
    if code.startswith("```"):
        lines=code.splitlines()
        if lines[-1].strip()!="```":raise ValueError("Incomplete code fence")
        code="\n".join(lines[1:-1])
    ast.parse(code)
    (args.out/"expert.py").write_text(code+"\n")
    save(args.out/"program.json",{"model":"deepseek-flash","sha256":sha(args.out/"expert.py"),
        "parent_sha256":sha(args.previous) if args.previous else None,"feedback_source_task_id":t["task_id"] if args.previous else None,
        "usage":response.usage,"syntax":"passed","fresh_physical_rollout":False,"admission":"pending static review"})
    print(json.dumps({"saved":str(args.out/"expert.py"),"sha256":sha(args.out/"expert.py"),"usage":response.usage}),flush=True)

if __name__=="__main__":asyncio.run(main())
