"""Build read-only 20 Hz sim/real ACT candidate views and verify real loaders; no training."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os
from pathlib import Path
for key in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):os.environ[key]="1"
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader,Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from probe_act_data import forward_probe

P=Path(str(Path(os.environ.get("PAOS_PROJECTS_ROOT", str(Path(__file__).resolve().parents[1] / "external/projects")))))
CAMERAS=[f"observation.images.cam_{x}" for x in ("left","middle","right")]
NAMES=[f"{s}_{j}" for s in ("left","right") for j in [*(f"joint{i}" for i in range(1,7)),"gripper"]]
TARGET_FPS=20
CHUNK=32

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def table_arrays(root,sim):
    columns=["observation.state","action","episode_index","frame_index","timestamp","index"]
    optional=("meta.assist_post_active","meta.gripper_task_modes","meta.source_transition_index","meta.source_time_s")
    present=set(pq.read_schema(root/"data/chunk-000/file-000.parquet").names)
    if sim:columns.extend(["observation.assist_active"]+[k for k in optional if k in present])
    table=pq.read_table(sorted((root/"data").glob("chunk-*/*.parquet")),columns=columns)
    result={k:np.asarray(table[k].to_pylist(),dtype=np.float32) for k in ("observation.state","action")}
    result.update({k:table[k].to_numpy() for k in ("episode_index","frame_index","timestamp","index")})
    if sim:result["assist"]=np.asarray(table["observation.assist_active"].to_pylist(),dtype=np.float32)[:,[1,0]]
    else:result["assist"]=np.full((len(table),2),-1,dtype=np.float32)
    post_known=sim and "meta.assist_post_active" in present
    result["assist_post"]=np.asarray(table["meta.assist_post_active"].to_pylist(),dtype=np.float32)[:,[1,0]] if post_known else np.full((len(table),2),-1,dtype=np.float32)
    result["assist_post_known"]=np.full((len(table),2),post_known,dtype=bool)
    if sim and "meta.gripper_task_modes" in present:
        result["gripper_task_modes"]=np.asarray(table["meta.gripper_task_modes"].to_pylist(),dtype=np.int8)
    result["source_transition_index"]=np.asarray(table["meta.source_transition_index"].to_pylist(),dtype=np.int64).ravel() if sim and "meta.source_transition_index" in present else result["frame_index"].copy()
    result["source_time_s"]=np.asarray(table["meta.source_time_s"].to_pylist(),dtype=np.float64).ravel() if sim and "meta.source_time_s" in present else result["timestamp"].copy()
    return result

def make_grid(source,fps,split,domain):
    blocks=[];ranges=[];cursor=0
    np.testing.assert_array_equal(source["index"],np.arange(len(source["index"])))
    for ep in np.unique(source["episode_index"]):
        rows=np.flatnonzero(source["episode_index"]==ep)
        np.testing.assert_array_equal(rows,np.arange(rows[0],rows[-1]+1))
        n=len(rows);np.testing.assert_array_equal(source["frame_index"][rows],np.arange(n))
        drift=float(np.max(np.abs(source["timestamp"][rows]-np.arange(n)/fps)))
        assert drift<=1e-4,(domain,ep,drift)
        count=(n-1)*TARGET_FPS//fps+1
        local=np.arange(count,dtype=np.int64)*fps//TARGET_FPS
        target_t=np.arange(count,dtype=np.float64)/TARGET_FPS
        lag=target_t-local/fps
        assert local[-1]<n and lag.min()>=-1e-12 and lag.max()<1/fps+1e-12
        if domain=="real":
            tag="train" if int(ep) in split["train"] else "validation"
            assert int(ep) in split[tag]
        else:tag=source.get("episode_split",{}).get(int(ep),"candidate_sim_pool")
        blocks.append({"source_row":rows[local],"episode_index":np.full(count,ep,dtype=np.int64),
                       "split":np.full(count,tag,dtype="U32"),
                       "view_frame_index":np.arange(count,dtype=np.int64),
                       "target_timestamp_s":target_t,"nominal_sample_lag_s":lag})
        ranges.append({"episode":int(ep),"source_start":int(rows[0]),"source_end_exclusive":int(rows[-1]+1),
                       "view_start":cursor,"view_end_exclusive":cursor+count,"split":tag,
                       "source_frames":n,"view_frames":count,"max_nominal_sample_lag_s":float(lag.max()),
                       "source_timestamp_rounding_max_s":drift,
                       "unrepresented_tail_duration_s":float((n-1)/fps-target_t[-1])})
        cursor+=count
    grid={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    ix=grid["source_row"]
    grid.update(state=source["observation.state"][ix],action=source["action"][ix],
                assist=source["assist"][ix],assist_known=np.full((len(ix),2),domain=="sim",dtype=bool),
                source_frame_index=source["frame_index"][ix],source_dataset_timestamp_s=source["timestamp"][ix])
    ends=np.empty(len(ix),dtype=np.int64)
    for r in ranges:ends[r["view_start"]:r["view_end_exclusive"]]=r["view_end_exclusive"]
    grid["episode_end_exclusive"]=ends
    for k in ("assist_post","assist_post_known","source_transition_index","source_time_s","gripper_task_modes"):
        if k in source:grid[k]=source[k][ix]
    assert grid["state"].shape==grid["action"].shape==(len(ix),14)
    assert np.isfinite(grid["state"]).all() and np.isfinite(grid["action"]).all()
    # Verify every selected value against the source, without applying sign/zero/width transforms.
    np.testing.assert_array_equal(grid["action"],source["action"][ix])
    np.testing.assert_array_equal(grid["state"],source["observation.state"][ix])
    return grid,ranges

class ACT20HzView(Dataset):
    def __init__(self,loaded,grid,domain,split):
        self.loaded,self.grid,self.domain,self.split=loaded,grid,domain,split
        self.fps=TARGET_FPS
    def __len__(self):return len(self.grid["source_row"])
    def __getitem__(self,index):
        g=self.grid;raw=self.loaded[int(g["source_row"][index])]
        row={k:raw[k] for k in ("observation.state","action","timestamp","frame_index","episode_index","index","task_index","task",*CAMERAS) if k in raw}
        end=int(g["episode_end_exclusive"][index]);future=index+np.arange(CHUNK)
        selected=np.minimum(future,end-1)
        row["observation.state"]=torch.from_numpy(g["state"][index].copy())
        row["action"]=torch.from_numpy(g["action"][selected].copy())
        row["action_is_pad"]=torch.from_numpy(future>=end)
        row["observation.assist_active"]=torch.from_numpy(g["assist"][index].copy())
        row["meta.assist_known"]=torch.from_numpy(g["assist_known"][index].copy())
        row["meta.assist_post_active"]=torch.from_numpy(g["assist_post"][index].copy())
        row["meta.assist_post_known"]=torch.from_numpy(g["assist_post_known"][index].copy())
        if "gripper_task_modes" in g:row["meta.gripper_task_modes"]=torch.from_numpy(g["gripper_task_modes"][index].copy())
        row["meta.domain"]=self.domain
        row["meta.split"]=str(g["split"][index])
        row["meta.source_row_index"]=int(g["source_row"][index])
        row["meta.source_frame_index"]=int(g["source_frame_index"][index])
        row["meta.view_frame_index"]=int(g["view_frame_index"][index])
        row["meta.source_dataset_timestamp_s"]=float(g["source_dataset_timestamp_s"][index])
        row["meta.nominal_sample_lag_s"]=float(g["nominal_sample_lag_s"][index])
        row["meta.source_fps"]=self.loaded.fps
        row["meta.view_fps"]=TARGET_FPS
        row["meta.original_state_alignment"]="linear_interpolation" if self.domain=="real" else "recorded_pre_action_sim_state"
        row["meta.image_source"]="captured_video" if self.domain=="real" else "rendered_from_recorded_pre_state_without_step"
        row["meta.state_gripper_semantics"]="signed_follower_feedback_unmodified" if self.domain=="real" else "sum_of_absolute_finger_slide_qpos"
        row["timestamp"]=torch.tensor(g["target_timestamp_s"][index],dtype=torch.float64)
        row["frame_index"]=torch.tensor(g["view_frame_index"][index])
        for c in CAMERAS:
            row[c]=torch.nn.functional.interpolate(row[c][None],size=(240,320),mode="bilinear",align_corners=False)[0]
        return row

def check_sample(view,source,index):
    g=view.grid;n=int(g["source_row"][index]);end=int(g["episode_end_exclusive"][index])
    row=view[index];raw=view.loaded[n]
    np.testing.assert_array_equal(raw["observation.state"].numpy(),source["observation.state"][n])
    np.testing.assert_array_equal(raw["action"].numpy(),source["action"][n])
    np.testing.assert_array_equal(row["observation.state"].numpy(),source["observation.state"][n])
    future=np.minimum(index+np.arange(CHUNK),end-1)
    np.testing.assert_array_equal(row["action"].numpy(),source["action"][g["source_row"][future]])
    np.testing.assert_array_equal(row["action_is_pad"].numpy(),index+np.arange(CHUNK)>=end)
    assert int(raw["episode_index"])==int(g["episode_index"][index])
    image_checks={}
    for c in CAMERAS:
        assert tuple(row[c].shape)==(3,240,320) and torch.isfinite(row[c]).all()
        assert float(row[c].std())>0
        image_checks[c]={"shape":list(row[c].shape),"std":float(row[c].std())}
    return {"view_row":index,"source_row":n,"episode":int(g["episode_index"][index]),
            "split":row["meta.split"],"state_and_action_source_match":True,
            "padded_actions":int(row["action_is_pad"].sum()),"images":image_checks}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-root",type=Path,required=True)
    ap.add_argument("--real-registry",type=Path,default=P/"piper_training_project/datasets/pick_place_gripper_v1")
    ap.add_argument("--kinematics-audit",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cfg=read(args.real_registry/"dataset.json")
    real_root=P/"piper_training_project"/cfg["root"]
    split=read(args.real_registry/"split.json");lock=read(args.real_registry/"dataset.lock.json")
    assert not set(split["train"])&set(split["validation"])
    assert len(split["train"])==88 and len(split["validation"])==10
    kin=read(args.kinematics_audit);assert kin["software_qpos_equals_urdf"]
    sim_info=read(args.sim_root/"meta/info.json");real_info=read(real_root/"meta/info.json")
    provenance=read(args.sim_root/"export-provenance.json")
    sim_repo_id=provenance.get("repo_id","local/phyagentos-piper-assisted-joint-v1")
    assert real_info["fps"]==30 and sim_info["fps"]==20
    assert real_info["features"]["action"]["names"]==NAMES
    for feature in ("action","observation.state"):
        sim_names=[n.replace("gripper_width","gripper") for n in sim_info["features"][feature]["names"]]
        assert sim_names==NAMES and real_info["features"][feature]["names"]==NAMES
    assert [k for k in sim_info["features"] if k.startswith("observation.images.")]==CAMERAS
    assert cfg["cameras"]==CAMERAS
    files=[Path(__file__),Path(__file__).with_name("probe_act_data.py"),args.kinematics_audit,
           args.real_registry/"dataset.json",args.real_registry/"split.json",args.real_registry/"dataset.lock.json",
           args.sim_root/"meta/info.json",args.sim_root/"export-provenance.json",args.sim_root/"data/chunk-000/file-000.parquet",
           P/"piper_robocasa/configs/workstation_camera.json"]
    files.extend(real_root/p for p in lock["files"])
    files.extend((args.sim_root/"data").glob("chunk-*/*.parquet"))
    before={str(p):sha(p) for p in files}
    for p,expected in lock["files"].items():assert before[str(real_root/p)]==expected,p
    sim_source=table_arrays(args.sim_root,True);real_source=table_arrays(real_root,False)
    sim_source["episode_split"]={int(e["episode_index"]):
        (e.get("scenario_record") or {}).get("scenario",{}).get("split","candidate_sim_pool") for e in provenance["episodes"]}
    if any(v not in {"train","validation","test","candidate_sim_pool"} for v in sim_source["episode_split"].values()):
        raise ValueError("Invalid admitted simulation split")
    assert len(real_source["index"])==real_info["total_frames"]
    assert len(sim_source["index"])==provenance["total_frames"]==sum(x["length"] for x in provenance["episodes"])
    grids={};ranges={}
    for domain,source,fps in (("sim",sim_source,20),("real",real_source,30)):
        grids[domain],ranges[domain]=make_grid(source,fps,split,domain)
        np.savez_compressed(args.out/f"{domain}-20hz-view.npz",**grids[domain])
        save(args.out/f"{domain}-episode-ranges.json",ranges[domain])
    sim=LeRobotDataset(sim_repo_id,root=args.sim_root,download_videos=False)
    real=LeRobotDataset(cfg["repo_id"],root=real_root,video_backend="pyav",download_videos=False)
    views={"sim":ACT20HzView(sim,grids["sim"],"sim",split),"real":ACT20HzView(real,grids["real"],"real",split)}
    # Check numerical padding at all 99 episode boundaries, without decoding unnecessary frames.
    boundary_checks=[]
    for domain,source in (("sim",sim_source),("real",real_source)):
        g=grids[domain]
        for r in ranges[domain]:
            n=r["view_end_exclusive"]-1
            future=np.minimum(n+np.arange(CHUNK),n)
            assert np.all(g["episode_index"][future]==r["episode"])
            np.testing.assert_array_equal(g["action"][future],np.repeat(source["action"][g["source_row"][n]][None],CHUNK,axis=0))
            boundary_checks.append({"domain":domain,"episode":r["episode"],"last_row_valid":True,"padded_actions":31})
    checks={};sample_rows={}
    for domain in ("sim","real"):
        chosen=ranges[domain] if domain=="sim" else [ranges["real"][split["train"][0]],ranges["real"][split["validation"][0]],ranges["real"][split["train"][-1]]]
        sample_rows[domain]=sorted(set(i for r in chosen for i in (r["view_start"],(r["view_start"]+r["view_end_exclusive"]-1)//2,r["view_end_exclusive"]-1)))
        checks[domain]=[check_sample(views[domain],sim_source if domain=="sim" else real_source,i) for i in sample_rows[domain]]
        save(args.out/f"{domain}-loader-checks.json",checks[domain])
        print(json.dumps({"phase":f"{domain}_actual_loader_passed","samples":len(checks[domain]),"frames":len(views[domain])}),flush=True)
    real_g=grids["real"];fit=np.isin(real_g["episode_index"],split["train"])
    assert not np.isin(real_g["episode_index"][fit],split["validation"]).any()
    sim_g=grids["sim"]
    sim_fit=np.isin(sim_g["split"],["train","candidate_sim_pool"])
    fit_state=np.concatenate((real_g["state"][fit],sim_g["state"][sim_fit]))
    fit_action=np.concatenate((real_g["action"][fit],sim_g["action"][sim_fit]))
    rows=[views["real"][ranges["real"][split["train"][0]]["view_start"]],views["sim"][0]]
    batch=next(iter(DataLoader(rows,batch_size=2,num_workers=0)))
    forward=forward_probe(batch,CAMERAS,fit_state,fit_action,args.out,"candidate-dual-domain")
    after={p:sha(p) for p in before};assert before==after
    camera=read(P/"piper_robocasa/configs/workstation_camera.json")
    report={"versions":{p:importlib.metadata.version(p) for p in ("lerobot","torch","torchvision","numpy","pyarrow")},
      "actual_loader":"lerobot.datasets.lerobot_dataset.LeRobotDataset","view_class":"ACT20HzView",
      "common_fps":20,"action_chunk_size":32,"future_action_offsets_s":[i/20 for i in range(CHUNK)],
      "horizon_last_action_s":31/20,"target_frame_shape":[14],"order":NAMES,
      "time_mapping":"source_local_frame=floor(view_local_frame*source_fps/20); exact nominal integer frame grid",
      "real_original_state_alignment":read(real_root/"meta/piper_sources.json")["state_alignment"],
      "no_raw_sensor_causality_claim":"Real source state was already linearly interpolated; this view neither reverses nor conceals that interpolation. Tiny stored float32 timestamp rounding is reported separately.",
      "sim":{"root":str(args.sim_root.resolve()),"repo_id":sim_repo_id,"source_tasks":[x["source_task_id"] for x in provenance["episodes"]],
             "explicit_binary_gripper_modes":"gripper_task_modes" in sim_g,
             "assist_post_preserved":bool(sim_g["assist_post_known"].all()),
             "source_frames":len(sim_source["index"]),"view_frames":len(sim_g["source_row"]),"episodes":len(ranges["sim"]),
             "assist_flags_preserved":True,"assist_side_order":["left","right"],"assist_active_count":sim_g["assist"].sum(0).tolist()},
      "real":{"root":str(real_root),"source_frames":len(real_source["index"]),"view_frames":len(real_g["source_row"]),"episodes":len(ranges["real"]),
              "train_episodes":88,"validation_episodes":10,"train_view_frames":int(fit.sum()),"validation_view_frames":int((~fit).sum()),
              "original_split_preserved":True,"signed_gripper_state_unchanged":True,
              "state_gripper_min_m":real_g["state"][:,[6,13]].min(0).tolist(),"state_gripper_max_m":real_g["state"][:,[6,13]].max(0).tolist(),
              "assist_active":-1,"assist_known":False,"max_nominal_sample_lag_s":float(real_g["nominal_sample_lag_s"].max())},
      "boundary_checks":boundary_checks,"loader_checks":checks,"candidate_interface_forward":forward,
      "normalization_scope":{"real_train_frames":int(fit.sum()),"sim_candidate_frames":int(sim_fit.sum()),"real_validation_excluded":True,"sim_validation_and_test_excluded":True,
                             "status":"Interface probe only; no approved mixed-training normalization contract"},
      "software_joint_mapping":{"qpos_equals_urdf":True,"zero_offset_applied":0,"sign_multiplier":1,"evidence":str(args.kinematics_audit)},
      "cameras":{"keys":CAMERAS,"shape":[3,240,320],"preprocessing":"named views; bilinear resize only; fixed ImageNet normalization for interface probe",
                 "no_overview_alias":True,"sim_source_status":camera.get("source_status"),
                 "sim_geometry_calibrated":camera.get("acceptance",{}).get("geometry_calibrated"),"sim_wrist_mounts_calibrated":camera.get("wrist_mounts_calibrated"),
                 "sim_original_projection_threshold_passed":camera.get("acceptance",{}).get("original_pass"),
                 "sim_world_mapping_status":camera.get("world_mapping_status")},
      "cross_domain_mixing_ready":False,
      "remaining_requirements":["Signed hardware gripper feedback to physical two-finger opening mapping with reference measurements.",
                                "Camera geometry/domain contract: wrist mounts uncalibrated, middle geometry remains candidate.",
                                "Explicit acceptance of inherited real-state interpolation and unknown real assistance treatment before mixed training."],
      "source_sha256_before":before,"source_sha256_after":after,"source_files_unchanged":True,
      "view_sha256":{domain:sha(args.out/f"{domain}-20hz-view.npz") for domain in grids},
      "training_started":False,"optimizer_updates":0,"actions_dispatched":0,"new_model_requests":0,
      "shared_asset_or_ACT_sources_modified":False}
    save(args.out/"acceptance.json",report)
    save(args.out/"view-contract.json",{k:report[k] for k in ("common_fps","action_chunk_size","future_action_offsets_s","time_mapping","real_original_state_alignment","no_raw_sensor_causality_claim","order","software_joint_mapping","cameras","cross_domain_mixing_ready","remaining_requirements")})
    print(json.dumps({"completed":str(args.out),"real_view_frames":len(real_g["source_row"]),"sim_view_frames":len(sim_g["source_row"]),
                      "prediction_shape":forward["prediction_shape"],"cross_domain_mixing_ready":False}),flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
