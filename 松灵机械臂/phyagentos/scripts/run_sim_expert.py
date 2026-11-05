"""Run a reviewed feedback expert through official Task-bound Forge actions, with no implicit reset or retries."""
from __future__ import annotations
import os
import argparse, ast, asyncio, hashlib, inspect, json, math, socket, sys, time, urllib.request
from pathlib import Path
ROOT=Path(str(Path(__file__).resolve().parents[1]))
ALLOWED={namespace+"."+operation for namespace in ("piper","rpent") for operation in ("move_arm","set_gripper","grasp_bottle","release_bottle")}
TERMINAL={"completed","failed","rejected","cancelled","expired"}
SAFE_METHODS={"get","copy","values","items","append","pop"}
SAFE_MATH={"sqrt","sin","cos","acos","atan2","pi","isfinite"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(out,name,obj):(out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+"\n")
def console(path):
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open("http://127.0.0.1:19081"+path,timeout=15) as r:return json.load(r)
def data(response):
    if response.get("ok") is False:raise RuntimeError(str(response.get("error")))
    return response.get("data",response)
def observation(response):return data(response)["response"]["result"]["outputs"]["observation"]
def physical_success(o,initial_count=0):
    transfer=any(t.get("donor")=="right" and t.get("receiver")=="left" and len(t.get("contact_geoms",[]))>=2
        and t.get("qpos_change")==0 and t.get("table_contact") is False and t.get("bottle_world_m",[0,0,0])[2]>.78
        for t in o.get("composable_transfers",[])[initial_count:])
    return transfer and o.get("bucket_bottom_contact") is True and o["bucket_xy_error_m"]<.055 and o["bottle_speed_norm"]<.05 \
        and o.get("failed") is False and not o.get("reset_required") and o.get("collision_stop") is None \
        and not any(o["numerical_warnings"]) and not any(o["assist_active"].values())

def load_program(path):
    source=path.read_text();tree=ast.parse(source)
    admission=json.loads(path.with_name("admission.json").read_text())
    if admission.get("approved_for_simulation_attempt") is not True or admission["sha256"]!=sha(path):raise ValueError("Program must match reviewed admission")
    for n in ast.walk(tree):
        if isinstance(n,(ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal)):raise ValueError("Unsupported expert construct")
        if isinstance(n,ast.Import) and any(a.name!="math" for a in n.names):raise ValueError("Only math import allowed")
        if isinstance(n,ast.Name) and (n.id.startswith("__") or n.id in {"open","eval","exec","compile","globals","locals","getattr","setattr","input","breakpoint"}):raise ValueError("Forbidden expert capability")
        if isinstance(n,ast.Attribute) and (n.attr.startswith("_") or (n.attr not in SAFE_METHODS and not (isinstance(n.value,ast.Name) and n.value.id=="math" and n.attr in SAFE_MATH))):raise ValueError("Unsupported expert attribute: "+n.attr)
    import builtins
    safe={k:getattr(builtins,k) for k in ("dict","list","tuple","float","int","bool","str","len","min","max","abs","range","enumerate","zip","all","any","sum","round","isinstance","ValueError")}
    def only_math(name,*args,**kwargs):
        if name!="math":raise ImportError("Only math")
        return math
    safe["__import__"]=only_math
    namespace={"__builtins__":safe}
    exec(compile(tree,str(path),"exec"),namespace)
    if not callable(namespace.get("propose")):raise ValueError("Missing propose function")
    return namespace["propose"],admission

def validate_proposal(p):
    if not isinstance(p,dict) or not isinstance(p.get("memory"),dict):raise ValueError("Proposal must carry memory")
    choices=sum(bool(p.get(k)) for k in ("action","done","abort"))
    if choices!=1:raise ValueError("Expected exactly one action/done/abort")
    if "action" not in p:return
    action=p["action"];tool=action["tool_id"];a=action["arguments"]
    if tool not in ALLOWED or a.get("side") not in {"right","left"}:raise ValueError("Action not allowed")
    required={"side"}
    if tool in {"piper.move_arm","rpent.move_arm"}:
        required|={"target_world_m","target_rotation_rad"}
        for key in ("target_world_m","target_rotation_rad"):
            if len(a[key])!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in a[key]):raise ValueError("Invalid pose")
    elif tool in {"piper.set_gripper","rpent.set_gripper"}:
        required.add("open_or_close")
        if a["open_or_close"] not in {"open","close"}:raise ValueError("Invalid gripper operation")
    if set(a)!=required:raise ValueError("Unexpected action arguments")

def geometry():
    p=ROOT/"runs/deepseek-flash-handoff-20261010-04/finger-geometry-diagnosis.json"
    g=json.loads(p.read_text())
    return {"finger_pair_mesh_midpoint_tcp_local_m":{s:a["finger_pair_mesh_midpoint_tcp_local_m"] for s,a in g["arms"].items()},
        "source_path":str(p),"source_sha256":sha(p),"geometry_calibrated":False}

def probe(args,out,propose):
    cases=json.loads(args.probe_file.read_text());rows=[]
    for case in cases:
        o=case["observation"];o["expert_geometry"]=geometry()
        p=propose(o,case.get("memory",{}));validate_proposal(p)
        expected=case["expect"]
        mode="action" if p.get("action") else "done" if p.get("done") else "abort"
        if mode!=expected["mode"]:raise AssertionError(case["label"]+": unexpected "+mode)
        if p.get("done") and not physical_success(o,case.get("memory",{}).get("initial_transfer_count",0)):raise AssertionError("Unsupported done")
        if expected.get("step_bound_m") is not None:
            a=p["action"]["arguments"];side=a["side"]
            distance=math.dist(a["target_world_m"],o["tcp_world_m"][side])
            if distance>expected["step_bound_m"]:raise AssertionError("Carry proposal exceeds requested feedback step")
            if a["target_rotation_rad"]!=o["tcp_rotation_rad"][side]:raise AssertionError("Carry rotation changed")
        rows.append({"label":case["label"],"source":case["source"],"proposal":p,"passed":True})
    save(out,"probes.json",rows)
    save(out,"summary.json",{"mode":"offline decision checks on real recorded observations","passed":len(rows),
        "program_sha256":sha(args.program),"model_requests":0,"actions_dispatched":0,"new_rollouts":0,
        "limitation":"Explicit phase memory probes are program checks; they are not executed physical trajectories."})
    print(json.dumps({"offline_probes_passed":len(rows),"out":str(out)}),flush=True)

async def run(args,out,propose,admission):
    state=console("/api/state");save(out,"before-console-state.json",state)
    if state.get("skill") != args.skill:raise RuntimeError("Console selection differs from requested Skill; no motion")
    namespace="piper" if args.skill=="piper-dual-sim" else "rpent"
    rs=state.get("runtime",{}).get("state",{}) or {}
    busy=any(state.get(k) for k in ("job","active_task","control_busy","background_busy","reflection_busy")) \
        or any(rs.get(k) for k in ("active_invocations","active_sessions","active_task_bindings"))
    if busy:
        save(out,"summary.json",{"status":"blocked","reason":"Existing official runtime ownership is not idle; no Task, reset or motion dispatched.",
            "active_task_bindings":rs.get("active_task_bindings"),"model_requests":0,"tasks_created":0,"actions_dispatched":0,"new_rollouts":0})
        print("BLOCKED: existing runtime ownership; no Task/reset/motion dispatched",flush=True)
        return 3
    from PhyAgentOS.config.loader import load_config
    from PhyAgentOS.cli.commands import _make_provider,_make_forge_components
    from PhyAgentOS.agent.experience.store import ExperienceStore
    from PhyAgentOS.agent.experience.activation import SkillActivationManager
    from PhyAgentOS.verification.contracts import TaskVerificationContract
    cfg=load_config()
    if args.reviewer == "codex":
        from urllib.parse import urlparse
        from PhyAgentOS.providers.custom_provider import CustomProvider
        endpoint=urlparse(args.codex_review_url or "")
        if endpoint.scheme!="http" or endpoint.hostname!="127.0.0.1" or endpoint.path!="/v1":
            raise ValueError("Codex review requires the explicitly launched loopback endpoint")
        cfg.agents.defaults.model="custom/codex-session-reviewer"
        cfg.agents.verification.model=cfg.agents.defaults.model
        cfg.agents.verification.provider="custom"
        cfg.agents.verification.timeout_s=3660
        cfg.providers.custom.api_base=args.codex_review_url
        cfg.providers.custom.api_key="local-reviewer-placeholder"
        cfg.agents.evolution.model=cfg.agents.defaults.model
        cfg.agents.evolution.provider="custom"
        save(out,"reviewer.json",{"reviewer":"Codex conversation","model":cfg.agents.defaults.model,
            "api_base":args.codex_review_url,"paid_ds_requests":0,"user_requested_replacement":True})
    elif args.reviewer == "deepseek-flash" and (cfg.agents.defaults.model not in {"deepseek-flash","custom/deepseek-flash"} or cfg.agents.verification.model not in {None,"","deepseek-flash","custom/deepseek-flash"}):
        raise RuntimeError("DS Flash required for the deepseek-flash reviewer")
    with socket.socket() as s:
        s.bind(("127.0.0.1",0));cfg.agents.verification.service_port=s.getsockname()[1]
    if args.reviewer == "configured":
        save(out,"reviewer.json",{"reviewer":"configured PAO provider","model":cfg.agents.defaults.model,
            "provider":cfg.get_provider_name(cfg.agents.defaults.model),
            "api_base":cfg.get_api_base(cfg.agents.defaults.model),
            "reasoning_effort":cfg.agents.defaults.reasoning_effort,
            "source":"PAO default configuration"})
    provider=(CustomProvider(api_key=cfg.providers.custom.api_key,api_base=cfg.providers.custom.api_base,
        default_model=cfg.agents.defaults.model,timeout_s=3660) if args.reviewer=="codex" else _make_provider(cfg))
    client,ids,coord,availability=_make_forge_components(cfg,provider)
    experience = None
    if args.verification_mode == "enforce" and cfg.agents.evolution.enabled:
        from PhyAgentOS.agent.experience.analyzer import ModelExperienceAnalyzer
        from PhyAgentOS.agent.experience.coordinator import ExperienceCoordinator
        evolution = cfg.agents.evolution
        if args.reviewer=="deepseek-flash" and (evolution.model not in {None, "", "deepseek-flash", "custom/deepseek-flash"} or evolution.provider not in {None, "", "custom"}):
            raise RuntimeError("DS Flash required for experience analysis")
        experience = ExperienceCoordinator(
            workspace=cfg.workspace_path,
            analyzer=ModelExperienceAnalyzer(provider=provider, model=cfg.agents.defaults.model,timeout_s=3660 if args.reviewer=="codex" else 180),
            task_coordinator=coord, runtime_availability_provider=availability,
            binding_resolver=coord.binding_resolver,
            min_successful_episodes=evolution.min_successful_episodes,
            min_lesson_episodes=evolution.min_lesson_episodes,
            max_lessons_per_skill=evolution.max_lessons_per_skill,
            max_calls=evolution.max_evolution_calls_per_run)
        activation_manager = experience.activation
        coord.set_experience(experience)
    else:
        activation_manager=SkillActivationManager(workspace=cfg.workspace_path,store=ExperienceStore(cfg.workspace_path),binding_resolver=coord.binding_resolver,runtime_availability_provider=availability)
    coord.set_activation_manager(activation_manager)
    session="expert:"+out.name
    description="Use reviewed observation-feedback expert "+sha(args.program)+" on independent shared PiPER simulation: actual assisted right-to-left airborne bottle handoff and release/settling in left bucket."
    if args.require_joint_targets:
        description += " Use the joint profile: actual absolute radian joint targets must be consumed by the official JointPositionController and logged with pre/post states."
    if args.require_binary_gripper:
        description += " Use binary OPEN/CLOSE task commands with contact-limited closing toward the 0.1mm empty-gripper endpoint; retain actual per-step servo targets separately from task intent."
    activation_manager.begin_turn(session,description)
    # Freeze the actually installed runtime Skill for formal execution. Promoted
    # workspace workflow text remains pending its official install lifecycle; it
    # must not shadow or silently replace this published Forge binding.
    from PhyAgentOS.skill_runtime.catalog import SkillCatalog
    registered = SkillCatalog().get(args.skill)
    skill_path = registered.resolve_bundle_path(Path("SKILL.md"))
    activated=activation_manager._activate_bound(session_key=session,name=args.skill,role="primary",
        skill={"name":args.skill,"path":str(skill_path),"source":"installed"},
        content=skill_path.read_text(encoding="utf-8"),skill_version=registered.version)
    if inspect.isawaitable(activated):activated=await activated
    activation,content,lessons=activated
    save(out,"activation.json",activation.model_dump(mode="json"))
    if args.verification_mode == "off":
        description += " Native-only diagnostic; semantic verification deferred. No M4 root or success-dataset admission."
        coord.verifier.max_calls = 0
    contract=TaskVerificationContract.model_validate({"mode":args.verification_mode,"goal":description,
        "success_criteria":[
            "All same-task native Actions complete with primitive_completed=true and verified=true; no failed, unknown or unresolved Action.",
            "A new same-episode composable_transfers entry has donor=right, receiver=left, at least two receiver contact geoms, qpos_change=0, table_contact=false, and bottle base z>0.78m.",
            "Final same-task observe reports bucket_bottom_contact=true, bucket_xy_error_m<0.055, bottle_speed_norm<0.05, both assist_active=false, failed=false, collision_stop=null, and numerical_warnings all zero.",
            "Final physics_step and sim_time_s advance from initial same-task observe with actual before/after robot state and both camera evidence."],
        "constraints":["Simulation only, independent shared asset; contact-triggered assistance explicitly reported.",
            "Use frozen Task-bound registered Forge Query/Action, same-argument IK/FK preflight and serial terminal waits. No qpos writes, native RPC motion, hardware, database edits, implicit reset, relaxed thresholds or retries of unknown effects.",
            "Legacy macro fields task_success, handoff_verified and air_handoff_verified are retained as returned, but are not additional success criteria for this composable-primitives task. Assess the exact four criteria from measured transfer, release and settling evidence; identify any genuine conflicting physical evidence explicitly. Program proposals and offline state-rendered images are not proof of new rollout success."],
        "evidence_policy":{"required_kinds":["rgb_image","robot_state"],"required_sources":["image/cam_right","image/overview"],"minimum_association":"best_effort"}})
    save(out,"contract.json",contract.model_dump(mode="json"))
    task=await coord.create_task(task_description=description,verification=contract,activation_id=activation.activation_id,origin_session_key=session)
    task_id=task.task_id;save(out,"task-created.json",task.model_dump(mode="json"))
    save(out,"program-source.json",{"task_id":task_id,"path":str(args.program),"sha256":sha(args.program),"admission":admission,"geometry":geometry(),"runtime_skill":args.skill,"proposal_namespace_translation":namespace+".* (operation and arguments unchanged; frozen program file not edited)"})
    actions=[];memory={};error=None
    try:
        activation_manager.record_tool(session,"forge_tool_invoke_query",{"task_id":task_id,"tool_id":namespace+".observe","arguments":{}})
        receipt=await coord.invoke_query(task_id,namespace+".observe",{},timeout_ms=30000);save(out,"initial-query.json",receipt);o=observation(receipt)
        if o["control_step"]!=0 or o["failed"] or o.get("reset_required") or o["composable_transfers"] or any(o["assist_active"].values()):raise RuntimeError("Fresh healthy factory episode required; no motion")
        if args.require_joint_targets:
            if o.get("controller_mode") != "JOINT_POSITION:absolute" or not o.get("joint_command_trace_path"):
                raise RuntimeError("The joint expert profile is required; no motion dispatched")
        if args.require_binary_gripper and o.get("gripper_command_mode") != "binary_endpoints":
            raise RuntimeError("The binary gripper profile is required; no motion dispatched")
        if args.scene_config:
            scene_path=args.scene_config.resolve();cfg_scene=json.loads(scene_path.read_text())
            expected_sha=sha(scene_path);observed_scene=o.get("scene_initialization",{})
            if observed_scene.get("source_sha256")!=expected_sha:
                raise RuntimeError("Fresh runtime does not match requested initial scene; no motion")
            for obj,key in (("bottle","bottle_world_m"),("bucket","bucket_world_m")):
                if math.dist(o[key],cfg_scene["objects"][obj]["base_world_m"])>1e-6:
                    raise RuntimeError("Actual initial "+obj+" position differs from requested scene; no motion")
            save(out,"scenario-record.json",{"source_path":str(scene_path),"sha256":expected_sha,
                "scenario":cfg_scene.get("paos_scenario",{}),"native":observed_scene,
                "initial_bottle_world_m":o["bottle_world_m"],"initial_bucket_world_m":o["bucket_world_m"]})
        initial=o.copy();episode_id=o["episode_id"]
        for _ in range(args.max_actions):
            if o["episode_id"]!=episode_id or o["failed"] or o.get("reset_required"):raise RuntimeError("Episode changed or failed; stop")
            o["expert_geometry"]=geometry()
            p=propose(o,memory);validate_proposal(p)
            if p.get("action"):
                p={**p,"action":{**p["action"],"tool_id":namespace+"."+p["action"]["tool_id"].split(".",1)[1]}}
            if p.get("abort"):raise RuntimeError("Expert abort: "+p["abort"])
            if p.get("done"):
                if not physical_success(o,len(initial["composable_transfers"])):raise RuntimeError("Premature expert done rejected")
                break
            a=p["action"];row={"sequence":len(actions)+1,"observation_control_step":o["control_step"],"proposal":p}
            actions.append(row);save(out,"actions.json",actions)
            if a["tool_id"]==namespace+".move_arm":
                activation_manager.record_tool(session,"forge_tool_invoke_query",{"task_id":task_id,"tool_id":namespace+".check_pose","arguments":a["arguments"]})
                preflight=await coord.invoke_query(task_id,namespace+".check_pose",a["arguments"],timeout_ms=30000)
                row["same_parameter_preflight"]=preflight;save(out,"actions.json",actions)
                native=data(preflight)["response"]["result"]["outputs"]
                if native.get("read_only_confirmed") is not True:raise RuntimeError("IK/FK preflight did not confirm read-only behavior; no motion")
                if native["verification"].get("accepted") is not True:
                    prior=o.get("last_pose_check",{})
                    if prior.get("arguments")==a["arguments"]:raise RuntimeError("Expert repeated an unchanged rejected pose; no motion")
                    # A rejected read-only candidate has no Action effect to retry.
                    # Feed its measured check back to the expert within this budget.
                    o["last_pose_check"]={"arguments":a["arguments"],"verification":native["verification"]}
                    memory=p["memory"];row["planning_status"]="rejected_before_action";save(out,"actions.json",actions)
                    print(json.dumps({"sequence":row["sequence"],"planning_status":"rejected_before_action"}),flush=True)
                    continue
            activation_manager.record_tool(session,"forge_tool_start_action",{"task_id":task_id,"tool_id":a["tool_id"],"arguments":a["arguments"]})
            accepted=await coord.start_action(task_id,a["tool_id"],a["arguments"],timeout_ms=30000)
            row["accepted"]=accepted;save(out,"actions.json",actions)
            iid=data(accepted)["invocation_id"];deadline=time.monotonic()+45
            while True:
                status=await client.invocation_status(iid);coord.observe_action(task_id,iid,status);d=data(status)
                row["terminal_status"]=status;save(out,"actions.json",actions)
                if d.get("phase") in TERMINAL:break
                if time.monotonic()>deadline:raise TimeoutError("Unknown terminal effect; invocation preserved, no retry: "+iid)
                await asyncio.sleep(.5)
            native=d.get("result",{}).get("outputs",{}).get("native_result",{})
            if d.get("result",{}).get("status")!="succeeded" or native.get("primitive_completed") is not True or native.get("verified") is not True:raise RuntimeError("Actual Action failed; no replay: "+iid)
            memory=p["memory"]
            activation_manager.record_tool(session,"forge_tool_invoke_query",{"task_id":task_id,"tool_id":namespace+".observe","arguments":{}})
            receipt=await coord.invoke_query(task_id,namespace+".observe",{},timeout_ms=30000)
            save(out,"latest-query.json",receipt);o=observation(receipt)
            print(json.dumps({"sequence":row["sequence"],"tool":a["tool_id"],"control_step":o["control_step"]}),flush=True)
        else:raise RuntimeError("Expert action budget exhausted")
        save(out,"final-query.json",receipt)
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc)};save(out,"execution-error.json",error)
        if all(x.terminal for x in coord.get_task(task_id).execution_records):
            try:
                final_receipt=await coord.invoke_query(task_id,namespace+".observe",{},timeout_ms=30000)
                save(out,"final-query.json",final_receipt)
            except Exception as capture_error:
                save(out,"final-observation-error.json",{"type":type(capture_error).__name__,"message":str(capture_error)})
    finally:
        current=coord.get_task(task_id)
        if all(x.terminal for x in current.execution_records):
            activation_manager.record_tool(session,"forge_task_finalize",{"task_id":task_id})
            if experience is not None:
                experience.bind_forge_task(task_id,session_key=session,forge_binding=current.primary_skill_binding)
            finalized=await coord.finalize_task(task_id);save(out,"task-final.json",finalized.model_dump(mode="json"))
        save(out,"task-latest.json",coord.get_task(task_id).model_dump(mode="json"))
        save(out,"task-events.json",coord.store.events(task_id,limit=500))
        if experience is not None:
            # Await only the current Task's officially scheduled reflection and clusters.
            # Do not start/import/replay legacy pending jobs.
            while True:
                pending=[t for t in list(experience._tasks.values())+list(experience._cluster_tasks.values()) if not t.done()]
                if not pending:break
                await asyncio.gather(*pending,return_exceptions=True)
                await asyncio.sleep(0)
            try:episode=experience.store.get_episode_by_root(task_id)
            except KeyError:episode=None
            save(out,"experience-result.json",{"calls":experience.calls,
                "episode":episode.model_dump(mode="json") if episode else None,
                "min_successful_episodes":experience.evolution.min_successful_episodes,
                "min_lesson_episodes":experience.evolution.min_lesson_episodes})
            experience.stop()
        coord.verifier.stop();await client.close();await provider._client.close()
    task=coord.get_task(task_id)
    if (out/"task-final.json").exists() and "episode_id" in locals():
        save(out,"dataset-sources.json",[{"label":task_id,"task_file":str((out/"task-final.json").relative_to(ROOT)),
            "sim_dir":"runs/"+("piper-runtime" if args.skill=="piper-dual-sim" else "rpent-runtime")+"/"+episode_id+"/sim",
            "program_origin":{"type":"reviewed_llm_expert","model":admission.get("model"),"path":str(args.program),"sha256":sha(args.program),"admission":admission}}])
    final_native = observation(json.loads((out/"final-query.json").read_text())) if (out/"final-query.json").exists() else None
    physical_goal_met = bool(final_native and "initial" in locals()
        and physical_success(final_native, len(initial["composable_transfers"])))
    rollout_succeeded = error is None and physical_goal_met and task.status.value == "succeeded"
    save(out,"summary.json",{"task_id":task_id,"status":"succeeded" if rollout_succeeded else "failed",
        "official_task_status":task.status.value,"verification_mode":args.verification_mode,"physical_goal_met":physical_goal_met,
        "semantic_verification_deferred":args.verification_mode=="off","model_requests":0 if args.verification_mode=="off" else None,
        "reviewer":args.reviewer,"reviewer_provenance_file":str(out/"reviewer.json") if (out/"reviewer.json").exists() else None,
        "proposals":len(actions),"actions_dispatched":sum("accepted" in x for x in actions),"error":error,
        "formal_verdict":task.verdict.model_dump(mode="json") if task.verdict else None,"program_sha256":sha(args.program)})
    return 0 if rollout_succeeded else 1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skill",choices=("piper-dual-sim","rpent-piper-sim"),default="piper-dual-sim")
    ap.add_argument("--program",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument("--run",action="store_true");mode.add_argument("--probe-file",type=Path)
    ap.add_argument("--max-actions",type=int,default=180)
    ap.add_argument("--verification-mode",choices=("enforce","off"),default="enforce")
    ap.add_argument("--reviewer",choices=("configured","deepseek-flash","codex"),default="configured",help="Use the PAO default model/provider unless an explicit reviewer is selected.")
    ap.add_argument("--codex-review-url",help="Explicit loopback endpoint for actual replies by the active Codex conversation.")
    ap.add_argument("--require-joint-targets",action="store_true")
    ap.add_argument("--require-binary-gripper",action="store_true")
    ap.add_argument("--scene-config",type=Path,help="Require the already-reset runtime to match this immutable initial scene; does not reset.")
    args=ap.parse_args();args.program=args.program.resolve();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    propose,admission=load_program(args.program)
    if args.probe_file:probe(args,args.out,propose);return 0
    return asyncio.run(run(args,args.out,propose,admission))

if __name__=="__main__":sys.exit(main())
