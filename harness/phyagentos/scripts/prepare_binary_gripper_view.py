"""Build source-backed binary ACT views; retain unknown labels and assistance provenance."""
from pathlib import Path
import argparse,hashlib,json,os
for k in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):os.environ[k]="1"
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from probe_act_data import forward_probe
P=Path(str(Path(os.environ.get("PAOS_PROJECTS_ROOT", str(Path(__file__).resolve().parents[1] / "external/projects")))))
CAMERAS=[f"observation.images.cam_{s}" for s in ("left","middle","right")]
ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
GRIP=np.array([6,13])
CHUNK=32
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,j):Path(p).write_text(json.dumps(j,indent=2,ensure_ascii=False)+"\n")
def candidate(g,domain):
    widths=g["action"][:,GRIP]
    if domain=="sim":
        mode=np.asarray(g["gripper_task_modes"],dtype=np.int8).copy()
        assert mode.shape==widths.shape and np.isin(mode,[0,1]).all()
    else:
        mode=np.full(widths.shape,-1,dtype=np.int8)
        mode[widths==np.float32(.1)]=1
        mode[widths==np.float32(0)]=0
    known=mode>=0
    eligible=np.zeros(len(mode),dtype=bool)
    change=np.zeros((len(mode),2),dtype=bool)
    horizon_assist=np.full((len(mode),2),-1,dtype=np.int8)
    horizon_assist_known=np.zeros((len(mode),2),dtype=bool)
    for n in range(len(mode)):
        end=int(g["episode_end_exclusive"][n])
        ix=np.minimum(n+np.arange(CHUNK),end-1)
        eligible[n]=bool(known[ix].all())
        change[n]=((mode[ix[1:]]!=mode[ix[:-1]]) & known[ix[1:]] & known[ix[:-1]]).any(0)
        hk=(g["assist_known"][ix] & g["assist_post_known"][ix]).all(0)
        active=((g["assist"][ix]>0)|(g["assist_post"][ix]>0)).any(0)
        horizon_assist_known[n]=hk
        horizon_assist[n,hk]=active[hk]
    intent=g["action"].copy();intent[:,GRIP]=mode
    state=g["state"][:,ARM].copy()
    np.testing.assert_array_equal(intent[:,ARM],g["action"][:,ARM])
    np.testing.assert_array_equal(state,g["state"][:,ARM])
    return dict(state_joint12=state,task_intent_action14=intent,gripper_mode=mode,gripper_mode_known=known,
        anchor_eligible=eligible,chunk_has_known_mode_change=change,
        source_action14=g["action"],source_feedback_gripper_m=g["state"][:,GRIP],
        source_row=g["source_row"],source_transition_index=g["source_transition_index"],
        source_time_s=g["source_time_s"],episode_index=g["episode_index"],
        split=np.asarray(g["split"],dtype="U32") if "split" in g else np.full(len(mode),"candidate_sim_pool",dtype="U32"),
        view_frame_index=g["view_frame_index"],target_timestamp_s=g["target_timestamp_s"],
        episode_end_exclusive=g["episode_end_exclusive"],assist=g["assist"],assist_known=g["assist_known"],
        assist_post=g["assist_post"],assist_post_known=g["assist_post_known"],
        horizon_assist=horizon_assist,horizon_assist_known=horizon_assist_known)
def sample(loaded,g,index,domain):
    raw=loaded[int(g["source_row"][index])]
    end=int(g["episode_end_exclusive"][index]);future=index+np.arange(CHUNK);ix=np.minimum(future,end-1)
    assert g["gripper_mode_known"][ix].all() and g["anchor_eligible"][index]
    np.testing.assert_array_equal(raw["observation.state"].numpy()[ARM],g["state_joint12"][index])
    np.testing.assert_array_equal(raw["action"].numpy(),g["source_action14"][index])
    if domain=="sim":
        assert g["horizon_assist_known"][index].all()
        np.testing.assert_array_equal(raw["meta.gripper_task_modes"].numpy(),g["gripper_mode"][index])
        np.testing.assert_array_equal(raw["meta.assist_post_active"].numpy()[[1,0]],g["assist_post"][index])
    pool="real_assist_unknown" if domain=="real" else ("sim_assisted" if g["horizon_assist"][index].any() else "sim_unassisted")
    out={"observation.state":torch.from_numpy(g["state_joint12"][index].copy()),
         "action":torch.from_numpy(g["task_intent_action14"][ix].copy()),
         "action_is_pad":torch.from_numpy(future>=end),
         "meta.domain":domain,"meta.assistance_pool":pool,"task":raw["task"],
         "meta.gripper_label_origin":"exact_recorded_software_endpoints" if domain=="real" else "joint_commands.jsonl.gripper_task_modes",
         "meta.source_row_index":int(g["source_row"][index]),
         "meta.source_gripper_width_targets_m":torch.from_numpy(g["source_action14"][ix][:,GRIP].copy()),
         "meta.source_signed_gripper_feedback_m":torch.from_numpy(g["source_feedback_gripper_m"][index].copy()),
         "meta.gripper_mode_known":torch.from_numpy(g["gripper_mode_known"][ix].copy()),
         "meta.assist_active":torch.from_numpy(g["assist"][index].copy()),
         "meta.assist_known":torch.from_numpy(g["assist_known"][index].copy()),
         "meta.horizon_assist_active":torch.from_numpy(g["horizon_assist"][index].copy()),
         "meta.horizon_assist_known":torch.from_numpy(g["horizon_assist_known"][index].copy())}
    for c in CAMERAS:
        assert tuple(raw[c].shape) in ((3,480,640),(3,240,320)) and torch.isfinite(raw[c]).all()
        out[c]=torch.nn.functional.interpolate(raw[c][None],size=(240,320),mode="bilinear",align_corners=False)[0]
    return out
class BinaryACTView(Dataset):
    """Expose only source anchors whose complete same-episode chunk has known labels."""
    def __init__(self,loaded,grid,domain,indices):
        self.loaded,self.grid,self.domain,self.indices=loaded,grid,domain,np.asarray(indices,dtype=np.int64)
    def __len__(self):return len(self.indices)
    def __getitem__(self,n):return sample(self.loaded,self.grid,int(self.indices[n]),self.domain)
def future_rows(g,anchors):
    selected=np.concatenate([np.minimum(n+np.arange(CHUNK),int(g["episode_end_exclusive"][n])-1) for n in anchors])
    return np.unique(selected)
def coverage(g,split):
    result={}
    same=g["episode_index"][1:]==g["episode_index"][:-1]
    mode=g["gripper_mode"];known=g["gripper_mode_known"]
    for tag,mask in split.items():
        anchors=np.flatnonzero(mask & g["anchor_eligible"])
        targets=future_rows(g,anchors) if len(anchors) else np.array([],dtype=np.int64)
        transition=[]
        target_mask=np.zeros(len(mode),dtype=bool);target_mask[targets]=True
        for side in (0,1):
            pairs=same & target_mask[1:] & target_mask[:-1] & mask[1:] & mask[:-1] & known[1:,side] & known[:-1,side]
            transition.append({
                "close_to_open":int((pairs & (mode[:-1,side]==0) & (mode[1:,side]==1)).sum()),
                "open_to_close":int((pairs & (mode[:-1,side]==1) & (mode[1:,side]==0)).sum())})
        result[tag]={"eligible_anchors":len(anchors),"target_rows":len(targets),
            "mode_counts_left_right":{"close":[int((mode[targets,i]==0).sum()) for i in (0,1)],
                                     "open":[int((mode[targets,i]==1).sum()) for i in (0,1)]},
            "anchors_containing_transition_left_right":g["chunk_has_known_mode_change"][anchors].sum(0).tolist(),
            "adjacent_known_transitions_left_right":transition,
            "episodes_with_eligible_anchors":np.unique(g["episode_index"][anchors]).tolist()}
    return result
def unknown_spans(g):
    spans=[]
    for ep in np.unique(g["episode_index"]):
        rows=np.flatnonzero(g["episode_index"]==ep)
        for side in (0,1):
            bad=~g["gripper_mode_known"][rows,side]
            bounds=np.flatnonzero(np.diff(np.r_[False,bad,False]))
            for a,b in bounds.reshape(-1,2):
                ix=rows[a:b];width=g["source_action14"][ix,GRIP[side]]
                spans.append({"episode":int(ep),"side":["left","right"][side],
                    "view_start":int(ix[0]),"view_end_exclusive":int(ix[-1]+1),
                    "source_row_start":int(g["source_row"][ix[0]]),"source_row_end":int(g["source_row"][ix[-1]]),
                    "start_timestamp_s":float(g["target_timestamp_s"][ix[0]]),
                    "end_timestamp_s":float(g["target_timestamp_s"][ix[-1]]),
                    "software_width_min_max_m":[float(width.min()),float(width.max())],"label":"UNKNOWN"})
    return spans
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view-root",type=Path,required=True)
    ap.add_argument("--sim-root",type=Path,required=True)
    ap.add_argument("--real-registry",type=Path,default=P/"piper_training_project/datasets/pick_place_gripper_v1")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    cfg=read(args.real_registry/"dataset.json");split=read(args.real_registry/"split.json")
    real_root=P/"piper_training_project"/cfg["root"]
    align=read(args.view_root/"acceptance.json")
    assert Path(align["sim"]["root"]).resolve()==args.sim_root.resolve()
    assert Path(align["real"]["root"]).resolve()==real_root.resolve()
    provenance=read(args.sim_root/"export-provenance.json")
    assert provenance["binary_gripper_modes"]["available"] and not provenance["binary_gripper_modes"]["inferred_from_servo_width"]
    prepared=Path(provenance["prepared_root"]);manifest=read(prepared/"manifest.json")
    files=[Path(__file__),Path(__file__).with_name("probe_act_data.py"),args.real_registry/"split.json",
        args.real_registry/"dataset.json",args.real_registry/"dataset.lock.json",
        args.view_root/"real-20hz-view.npz",args.view_root/"sim-20hz-view.npz",args.view_root/"acceptance.json",
        real_root/"meta/piper_gripper_mapping.json",real_root/"meta/piper_sources.json",
        args.sim_root/"export-provenance.json",args.sim_root/"data/chunk-000/file-000.parquet"]
    files.extend(Path(p) for episode in manifest["episodes"] for p in episode["sources"])
    files.extend((args.sim_root/"data").glob("chunk-*/*.parquet"))
    before={str(p):sha(p) for p in dict.fromkeys(files)}
    actual={"real":LeRobotDataset(cfg["repo_id"],root=real_root,video_backend="pyav",download_videos=False),
        "sim":LeRobotDataset(provenance["repo_id"],root=args.sim_root,download_videos=False)}
    grids={};counts={};masks={};views={};fit_state=[];fit_action=[];rows=[];selected={}
    for domain in ("real","sim"):
        original=np.load(args.view_root/f"{domain}-20hz-view.npz",allow_pickle=False)
        g=candidate(original,domain);grids[domain]=g
        train=np.isin(g["episode_index"],split["train"]) if domain=="real" else np.isin(g["split"],["train","candidate_sim_pool"])
        validation=~train if domain=="real" else g["split"]=="validation"
        test=np.zeros_like(train) if domain=="real" else g["split"]=="test"
        masks[domain]={"train":train,"validation":validation,"test":test}
        g["train_anchor_indices"]=np.flatnonzero(train & g["anchor_eligible"])
        g["validation_anchor_indices"]=np.flatnonzero(validation & g["anchor_eligible"])
        g["test_anchor_indices"]=np.flatnonzero(test & g["anchor_eligible"])
        np.savez_compressed(args.out/f"{domain}-binary-task-view.npz",**g)
        views[domain]=BinaryACTView(actual[domain],g,domain,g["train_anchor_indices"])
        assert len(views[domain])>0
        eligible=g["anchor_eligible"]
        assisted=(g["horizon_assist"]>0).any(1) & g["horizon_assist_known"].all(1)
        free=(g["horizon_assist"]==0).all(1) & g["horizon_assist_known"].all(1)
        unknown=~g["horizon_assist_known"].all(1)
        counts[domain]={"frames":len(eligible),"known_both_grippers":int(g["gripper_mode_known"].all(1).sum()),
            "unknown_labels_left_right":(~g["gripper_mode_known"]).sum(0).tolist(),
            "eligible_chunk32_anchors":int(eligible.sum()),
            "assist_pools":{"assisted":int((eligible&assisted).sum()),"unassisted":int((eligible&free).sum()),"unknown":int((eligible&unknown).sum())},
            "coverage":coverage(g,masks[domain]),"raw_feedback_preserved":True,"actual_servo_action_preserved":True}
        anchors=g["train_anchor_indices"];targets=future_rows(g,anchors)
        if domain=="real":assert not np.isin(g["episode_index"][targets],split["validation"]).any()
        else:assert np.isin(g["split"][targets],["train","candidate_sim_pool"]).all()
        assert g["gripper_mode_known"][targets].all()
        fit_state.append(g["state_joint12"][anchors]);fit_action.append(g["task_intent_action14"][targets])
        changing=g["chunk_has_known_mode_change"][anchors].any(1)
        choice=int(np.flatnonzero(changing)[0]) if changing.any() else 0
        row=views[domain][choice];rows.append(row);n=int(anchors[choice])
        # Decode first and last admitted anchors through the installed loader too.
        for idx in (0,len(views[domain])-1):views[domain][idx]
        if domain=="real" and len(g["validation_anchor_indices"]):
            sample(actual[domain],g,int(g["validation_anchor_indices"][0]),domain)
        selected[domain]={"view_row":n,"source_row":int(g["source_row"][n]),"episode":int(g["episode_index"][n]),
            "gripper_modes_left_right":g["gripper_mode"][n].tolist(),"source_width_targets_m":g["source_action14"][n,GRIP].tolist(),
            "chunk_has_mode_change_left_right":g["chunk_has_known_mode_change"][n].tolist(),
            "assistance_pool":row["meta.assistance_pool"],"source_task":row["task"],"actual_loader_source_match":True}
        print(json.dumps({"phase":domain+"_filtered_loader_passed","eligible_train_anchors":len(views[domain])}),flush=True)
    save(args.out/"real-unknown-label-spans.json",unknown_spans(grids["real"]))
    batch=next(iter(DataLoader(rows,batch_size=2,num_workers=0)))
    forward=forward_probe(batch,CAMERAS,np.concatenate(fit_state),np.concatenate(fit_action),args.out,"binary-task")
    assert forward["batch_shapes"]["observation.state"]==[2,12] and forward["prediction_shape"]==[2,32,14]
    after={p:sha(p) for p in before};assert after==before
    cov=counts["real"]["coverage"]
    coverage_ready=all(cov[tag]["adjacent_known_transitions_left_right"][side][direction]>0
        for tag in ("train","validation") for side in (0,1) for direction in ("open_to_close","close_to_open"))
    contract={
        "purpose":"Offline binary ACT data ingestion; no hardware or policy-performance admission.",
        "state":"12 unmodified arm joints rad, left6/right6; signed/width gripper feedback retained outside model state.",
        "action":"left6 absolute joint targets rad / left mode bit / right6 absolute targets rad / right mode bit",
        "modes":{"CLOSE":0,"OPEN":1,"UNKNOWN":-1},
        "label_filter":"All 32 same-episode future labels must be known; end padding repeats last valid target and is marked.",
        "sim_label_source":"joint_commands.jsonl.gripper_task_modes; never inferred from contact width",
        "real_label_source":"Exact float32 0/.1m targets in the preserved derivative; intermediate widths remain unknown.",
        "cameras":{"keys":CAMERAS,"resize":[320,240],"normalization":"fixed ImageNet for interface probe",
            "treatment":"Named domain-specific RGB views. No geometry equality, rectification or calibrated wrist claim.",
            "sim_geometry_calibrated":align["cameras"]["sim_geometry_calibrated"],
            "sim_wrist_mounts_calibrated":align["cameras"]["sim_wrist_mounts_calibrated"]},
        "real_interpolation":{"source_state_alignment":align["real_original_state_alignment"],
            "treatment":"Inherited linear interpolation explicitly retained. 20Hz ZOH selection does not recover raw sensor causality."},
        "assistance":{"sim":"Pre and post per control step, and OR across the target chunk; assisted and unassisted pools separate.",
            "real":"-1/known=false, retained as unknown pool. Never classed as unassisted.",
            "experimental_mixed_probe":"May load flagged assisted sim with unknown-assistance real; does not validate unassisted transfer."},
        "normalization":"Real and sim train anchor states and union of their valid future targets; all real/sim validation and sim test excluded.",
        "task_scope":"Preserve original task text. The real registry describes right black bottle pickup, handoff to left and release into the left red bucket; sim uses the matching bottle-handoff/bucket goal with explicitly measured contact assistance.",
        "full_task_admission_requirement":"Observed known OPEN->CLOSE and CLOSE->OPEN supervision for each side in real train and validation.",
        "binary_endpoint_execution":"Already physically accepted in binary-gripper-unblock-20261011T124607Z; no new execution required."
    }
    report={"counts":counts,"selected_actual_loader_rows":selected,"data_contract":contract,"forward":forward,
        "source_sha256_before":before,"source_sha256_after":after,"source_files_unchanged":True,
        "sim_source_tasks":align["sim"]["source_tasks"],"state_input_shape":[12],"action_output_shape":[32,14],
        "filtered_dataset_class":"BinaryACTView","actual_loader":"lerobot.datasets.lerobot_dataset.LeRobotDataset",
        "mixed_labelled_window_interface_ready":True,"real_binary_transition_coverage_ready":coverage_ready,
        "cross_domain_mixing_ready":coverage_ready,"existing_state14_ACT_checkpoint_compatible":False,
        "remaining_requirements":[] if coverage_ready else ["Real binary task-intent labels spanning OPEN/CLOSE transitions. Existing continuous leader widths cannot establish those labels."],
        "continuous_gripper_calibration_required_for_state12":False,
        "physical_camera_or_joint_calibration_claimed":False,"unassisted_policy_transfer_validated":False,
        "validation_excluded_from_normalization":True,"sim_validation_and_test_excluded_from_normalization":True,"normalization_state_rows":sum(len(x) for x in fit_state),
        "normalization_action_rows":sum(len(x) for x in fit_action),
        "new_simulation_rollouts":0,"new_model_requests":0,"training_started":False,"optimizer_updates":0,"hardware_connected":False}
    save(args.out/"data-contract.json",contract);save(args.out/"acceptance.json",report)
    print(json.dumps({"completed":str(args.out),"counts":counts,"selected":selected,
        "state_shape":forward["batch_shapes"]["observation.state"],"prediction_shape":forward["prediction_shape"],
        "cross_domain_mixing_ready":coverage_ready}),flush=True)
if __name__=="__main__":main()
