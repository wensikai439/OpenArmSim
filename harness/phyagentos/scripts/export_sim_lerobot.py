"""Stream approved TCP episodes into local LeRobot and verify real action chunks on CPU."""
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
    ap.add_argument("--repo-id",default="local/phyagentos-piper-assisted-tcp-v1")
    args=ap.parse_args()
    api_version=importlib.metadata.version("lerobot")
    if api_version not in {"0.3.3","0.4.4"}: raise ValueError("Only the two inspected installed LeRobot APIs are supported")
    legacy=api_version=="0.3.3"
    source=json.loads((args.prepared/"manifest.json").read_text())
    episodes=[x for x in source["episodes"] if x["selected_for_training"]]
    if not episodes: raise ValueError("No approved actual-success episodes")
    state_names=episodes[0]["state_names"]
    if any(x["state_names"]!=state_names for x in episodes): raise ValueError("State layout mismatch")
    features={"observation.state":{"dtype":"float32","shape":(16,),"names":state_names},
        "action":{"dtype":"float32","shape":(14,),"names":source["action_names"]},
        "observation.assist_active":{"dtype":"float32","shape":(2,),"names":["right","left"]}}
    for camera in source["cameras"]:
        features["observation.images."+camera]={"dtype":"image","shape":(240,320,3),"names":["height","width","channel"]}
    dataset=LeRobotDataset.create(repo_id=args.repo_id,root=args.out,fps=20,robot_type=source["robot_type"],
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
                "observation.assist_active":numeric["assist"][n].astype(np.float32)}
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
            "program_origin":episode["program_origin"],"length":len(indices),"manifest_sha256":sha(folder/"manifest.json")})
    dataset.stop_image_writer()
    if not legacy: dataset.finalize()
    del dataset
    # This is the actual installed LeRobot loader, with ACT-style future action chunk and terminal padding.
    chunk_size=32
    loaded=LeRobotDataset(args.repo_id,root=args.out,delta_timestamps={"action":[i/20 for i in range(chunk_size)]})
    numeric=np.load(args.prepared/episodes[0]["label"]/"transitions.npz",allow_pickle=False)
    for n in (0,len(loaded)//2,len(loaded)-1):
        row=loaded[n]
        np.testing.assert_allclose(row["observation.state"].numpy(),numeric["state"][n],atol=1e-6)
        expected=np.stack([numeric["action"][min(n+i,len(numeric["action"])-1)] for i in range(chunk_size)])
        np.testing.assert_allclose(row["action"].numpy(),expected,atol=1e-6)
        expected_pad=np.asarray([n+i>=len(loaded) for i in range(chunk_size)])
        np.testing.assert_array_equal(row["action_is_pad"].numpy(),expected_pad)
        for camera in source["cameras"]:
            image=row["observation.images."+camera]
            assert tuple(image.shape)==(3,240,320) and torch.isfinite(image).all()
    batch=next(iter(DataLoader(loaded,batch_size=2,num_workers=0,shuffle=False)))
    report={"actual_dataset_loader":"lerobot.datasets.lerobot_dataset.LeRobotDataset","versions":{x:importlib.metadata.version(x) for x in ["lerobot","torch","torchvision","numpy","datasets"]},
        "total_episodes":loaded.num_episodes,"total_frames":len(loaded),"fps":loaded.fps,
        "action_chunk_size":chunk_size,"batch_shapes":{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},
        "state_action_source_alignment":"passed for first/middle/last rows","episode_end_action_padding":"passed",
        "training_started":False,"device":"cpu","source_manifest_sha256":sha(args.prepared/"manifest.json"),
        "exporter_sha256":sha(__file__),"episodes":exported,"image_source":source["image_source"],
        "action_schema":source["action_schema"],"directly_mixable_with_current_joint_ACT":False,
        "mixed_training_requirement":"Match robot state, action semantics, units, frames, camera preprocessing and assistance labels before combining with real/joint datasets."}
    save(args.out/"export-provenance.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
