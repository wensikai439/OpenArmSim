"""Export actual consumed joint-command episodes; verify LeRobot chunks and ACT CPU interface."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()

def save(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepared",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--repo-id",default="local/phyagentos-piper-assisted-joint-v1")
    args=ap.parse_args()
    api_version=importlib.metadata.version("lerobot")
    if api_version != "0.4.4": raise ValueError("Use the existing LeRobot 0.4.4 ACT CPU profile")
    legacy=api_version=="0.3.3"
    source=json.loads((args.prepared/"manifest.json").read_text())
    if source["action_schema"] != "piper_dual_executed_joint_absolute_software_width_v1":
        raise ValueError("Expected actually consumed absolute joint targets")
    episodes=[x for x in source["episodes"] if x.get("selected_for_export", x["selected_for_training"])]
    if not episodes: raise ValueError("No approved actual-success episodes")
    state_names=episodes[0]["state_names"]
    if any(x["state_names"]!=state_names for x in episodes): raise ValueError("State layout mismatch")
    binary=all(x.get("binary_gripper_modes",{}).get("available",False) for x in episodes)
    features={"observation.state":{"dtype":"float32","shape":(14,),"names":state_names},
        "action":{"dtype":"float32","shape":(14,),"names":source["action_names"]},
        "observation.assist_active":{"dtype":"float32","shape":(2,),"names":["right","left"]}}
    features["meta.assist_post_active"]={"dtype":"float32","shape":(2,),"names":["right","left"]}
    features["meta.source_transition_index"]={"dtype":"int64","shape":(1,),"names":["transition_index"]}
    features["meta.source_time_s"]={"dtype":"float64","shape":(1,),"names":["recorded_pre_time_s"]}
    if binary:features["meta.gripper_task_modes"]={"dtype":"int64","shape":(2,),"names":["left","right"]}
    for camera in source["cameras"]:
        features["observation.images."+camera]={"dtype":"image","shape":(240,320,3),"names":["height","width","channel"]}
    dataset=LeRobotDataset.create(repo_id=args.repo_id,root=args.out,fps=int(source["fps"]),robot_type=source["robot_type"],
        features=features,use_videos=False,image_writer_threads=2)
    exported=[]
    for episode in episodes:
        folder=args.prepared/episode["label"]
        numeric=np.load(folder/"transitions.npz",allow_pickle=False)
        if sha(folder/"transitions.npz")!=episode["transitions_sha256"]: raise ValueError("Prepared trace modified")
        indices=json.loads((folder/"image-index.json").read_text())
        if sha(folder/"image-index.json")!=episode["image_index_sha256"]: raise ValueError("Image index modified")
        for n,row in enumerate(indices):
            frame={"observation.state":numeric["state"][n].copy(),"action":numeric["action"][n].copy(),
                "observation.assist_active":numeric["assist"][n].astype(np.float32),
                "meta.assist_post_active":numeric["assist_post"][n].astype(np.float32),
                "meta.source_transition_index":numeric["source_transition_index"][n:n+1].astype(np.int64),
                "meta.source_time_s":numeric["source_time_s"][n:n+1].astype(np.float64)}
            if binary:frame["meta.gripper_task_modes"]=numeric["gripper_task_modes"][n].astype(np.int64)
            for camera in source["cameras"]:
                ref=row["images"][camera];path=folder/ref["path"]
                if sha(path)!=ref["sha256"]: raise ValueError("Derived image modified")
                with Image.open(path) as im: frame["observation.images."+camera]=np.asarray(im.convert("RGB")).copy()
            language="Contact-assisted right-to-left airborne bottle handoff, followed by release and settling in the left bucket."
            if legacy: dataset.add_frame(frame,task=language)
            else:
                frame["task"]=language
                dataset.add_frame(frame)
            if n % 400 == 0:print(json.dumps({"task_id":episode["task_id"],"added_frames":n+1}),flush=True)
        dataset.save_episode()
        exported.append({"episode_index":len(exported),"source_task_id":episode["task_id"],"source_episode_id":episode["episode_id"],
            "program_origin":episode["program_origin"],"length":len(indices),
            "scenario_record":episode.get("scenario_record"),"scene_initialization":episode.get("scene_initialization"),
            "manifest_sha256":sha(folder/"manifest.json")})
    dataset.stop_image_writer()
    if not legacy: dataset.finalize()
    del dataset
    # This is the actual installed LeRobot loader, with ACT-style future action chunk and terminal padding.
    chunk_size=32
    loaded=LeRobotDataset(args.repo_id,root=args.out,delta_timestamps={"action":[i/int(source["fps"]) for i in range(chunk_size)]})
    # Check episode boundaries through the actual loader, including terminal padding.
    cursor=0;all_numeric=[]
    for episode in episodes:
        numeric=np.load(args.prepared/episode["label"]/"transitions.npz",allow_pickle=False)
        length=len(numeric["state"])
        all_numeric.append({k:numeric[k] for k in numeric.files})
        for local in sorted(set((0,length//2,length-1))):
            n=cursor+local;row=loaded[n]
            np.testing.assert_allclose(row["observation.state"].numpy(),numeric["state"][local],atol=1e-6)
            selected=np.minimum(local+np.arange(chunk_size),length-1)
            np.testing.assert_allclose(row["action"].numpy(),numeric["action"][selected],atol=1e-6)
            np.testing.assert_array_equal(row["action_is_pad"].numpy(),local+np.arange(chunk_size)>=length)
            np.testing.assert_array_equal(row["meta.assist_post_active"].numpy(),numeric["assist_post"][local])
            if binary:np.testing.assert_array_equal(row["meta.gripper_task_modes"].numpy(),numeric["gripper_task_modes"][local])
            for camera in source["cameras"]:
                im=row["observation.images."+camera]
                assert tuple(im.shape)==(3,240,320) and torch.isfinite(im).all()
        cursor+=length
    numeric={k:np.concatenate([v[k] for v in all_numeric]) for k in all_numeric[0]}
    assert cursor==len(loaded)
    import pyarrow.parquet as pq
    table=pq.read_table(sorted((args.out/"data").glob("chunk-*/*.parquet")))
    for key,column in (("state","observation.state"),("action","action"),("assist","observation.assist_active"),("assist_post","meta.assist_post_active")):
        np.testing.assert_array_equal(np.asarray(table[column].to_pylist(),dtype=np.float32),numeric[key])
    np.testing.assert_array_equal(np.asarray(table["meta.source_transition_index"].to_pylist()).ravel(),numeric["source_transition_index"])
    np.testing.assert_array_equal(np.asarray(table["meta.source_time_s"].to_pylist()).ravel(),numeric["source_time_s"])
    if binary:np.testing.assert_array_equal(np.asarray(table["meta.gripper_task_modes"].to_pylist()),numeric["gripper_task_modes"])
    batch=next(iter(DataLoader(loaded,batch_size=2,num_workers=0,shuffle=False)))
    from probe_act_data import forward_probe
    cameras=["observation.images."+c for c in source["cameras"]]
    interface=forward_probe(batch,cameras,numeric["state"],numeric["action"],args.out,"joint")
    report={"repo_id":args.repo_id,"prepared_root":str(args.prepared.resolve()),
        "binary_gripper_modes":{"available":binary,"source":"joint_commands.jsonl.gripper_task_modes" if binary else None,
            "feature":"meta.gripper_task_modes","order":["left","right"],"inferred_from_servo_width":False},
        "all_exported_numeric_rows_source_match":True,"assist_pre_and_post_preserved":True,
        "actual_dataset_loader":"lerobot.datasets.lerobot_dataset.LeRobotDataset","versions":{x:importlib.metadata.version(x) for x in ["lerobot","torch","torchvision","numpy","datasets"]},
        "total_episodes":loaded.num_episodes,"total_frames":len(loaded),"fps":loaded.fps,
        "action_chunk_size":chunk_size,"batch_shapes":{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},
        "state_action_source_alignment":"passed for first/middle/last rows","episode_end_action_padding":"passed",
        "training_started":False,"device":"cpu","source_manifest_sha256":sha(args.prepared/"manifest.json"),
        "exporter_sha256":sha(__file__),"episodes":exported,"image_source":source["image_source"],
        "action_schema":source["action_schema"],"ACT_interface":interface,"directly_mixable_with_current_joint_ACT":False,
        "mixed_training_requirement":"Match robot state, action semantics, units, frames, camera preprocessing and assistance labels before combining with real/joint datasets."}
    save(args.out/"export-provenance.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
