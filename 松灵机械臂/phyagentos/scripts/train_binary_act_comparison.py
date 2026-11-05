"""Run a fixed real-only or mixed upstream ACT comparison on frozen whole-episode views."""
from pathlib import Path
import argparse,copy,datetime,fcntl,hashlib,importlib.metadata,inspect,json,os,sys,time
from types import SimpleNamespace
P=Path(str(Path(os.environ.get("PAOS_PROJECTS_ROOT", str(Path(__file__).resolve().parents[1] / "external/projects")))))
TRAIN=Path(os.environ.get("PIPER_TRAINING_ROOT", str(P/"piper_training_project")))
os.environ.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",
    HF_HOME=str(TRAIN/".cache/huggingface/act/cu128"),TORCH_HOME=str(TRAIN/".cache/torch/act/cu128"),
    WANDB_MODE="disabled",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
    TQDM_DISABLE="1")
os.environ.pop("PYTHONPATH",None);os.environ.pop("PYTHONHOME",None)
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.default import DatasetConfig,WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts import lerobot_train as upstream
from safetensors.torch import load_file
from prepare_binary_gripper_view import BinaryACTView,CAMERAS,ARM,GRIP,future_rows,sha,read,save
EXPECTED_BACKBONE="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
class MixedView(Dataset):
    def __init__(self,real,sim,meta):
        self.real,self.sim,self.meta=real,sim,meta
        self.num_frames=len(real)+len(sim)
        self.num_episodes=len(np.unique(real.grid["episode_index"][real.indices]))+len(np.unique(sim.grid["episode_index"][sim.indices]))
    def __len__(self):return self.num_frames
    def __getitem__(self,n):
        return self.real[n] if n<len(self.real) else self.sim[n-len(self.real)]
class BalancedBatches(Sampler):
    """Exactly half each domain per batch, shuffled cycles within each domain."""
    def __init__(self,n_real,n_sim,batch_size,steps,seed):
        self.nr,self.ns,self.half,self.steps,self.seed=n_real,n_sim,batch_size//2,steps,seed
        self.batch_size=batch_size
        if n_real<=0: raise ValueError("No real training anchors")
    def __len__(self):return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed)
        def domain_stream(n,offset):
            while True:
                for i in rng.permutation(n):yield int(i)+offset
        real=domain_stream(self.nr,0);sim=domain_stream(self.ns,self.nr)
        for _ in range(self.steps):
            batch=([next(real) for _ in range(self.half)]+[next(sim) for _ in range(self.half)]) if self.ns else [next(real) for _ in range(self.batch_size)]
            yield [batch[int(i)] for i in rng.permutation(len(batch))]
def evaluate(policy,preprocessor,views,stats,output):
    policy.eval()
    reports={}
    std=torch.tensor(stats["action"]["std"])
    mean=torch.tensor(stats["action"]["mean"])
    for name,view in views.items():
        counts=np.zeros((2,2),dtype=np.int64);correct=np.zeros_like(counts)
        total_abs=0.;valid_items=0;arm_abs=np.zeros(12);valid_actions=0;anchors=0
        by_pool={}
        for raw in DataLoader(view,batch_size=16,shuffle=False,num_workers=0):
            batch=preprocessor(raw)
            with torch.inference_mode():pred=policy.predict_action_chunk(batch).cpu()
            target=raw["action"];valid=~raw["action_is_pad"]
            assert torch.isfinite(pred).all()
            normalized_target=(target-mean)/std
            diff=(pred-normalized_target).abs()
            total_abs+=float(diff[valid].sum());valid_items+=int(valid.sum())*14
            physical=pred*std+mean
            arm_abs+=(physical[...,ARM]-target[...,ARM]).abs()[valid].sum(0).numpy()
            valid_actions+=int(valid.sum());anchors+=len(valid)
            actual=target[...,GRIP].long();guess=(physical[...,GRIP]>=.5).long()
            for side in range(2):
                for mode in (0,1):
                    mask=valid & (actual[...,side]==mode)
                    counts[side,mode]+=int(mask.sum())
                    correct[side,mode]+=int(((guess[...,side]==mode)&mask).sum())
            for b,pool in enumerate(raw["meta.assistance_pool"]):
                row=by_pool.setdefault(pool,{"anchors":0,"valid_actions":0,"normalized_abs_sum":0.})
                row["anchors"]+=1;row["valid_actions"]+=int(valid[b].sum())
                row["normalized_abs_sum"]+=float(diff[b][valid[b]].sum())
        for row in by_pool.values():
            row["normalized_masked_action_l1"]=row.pop("normalized_abs_sum")/(14*row["valid_actions"])
        reports[name]={"anchors":anchors,"batches":(anchors+15)//16,"valid_future_actions":valid_actions,
            "normalized_masked_action_l1_zero_latent":total_abs/valid_items,
            "joint_mae_rad_mean":float((arm_abs/valid_actions).mean()),
            "joint_mae_rad_per_joint":(arm_abs/valid_actions).tolist(),
            "gripper_class_counts_left_right_close_open":counts.tolist(),
            "gripper_correct_left_right_close_open":correct.tolist(),
            "gripper_accuracy_left_right":(correct.sum(1)/counts.sum(1)).tolist(),
            "assistance_groups":by_pool,
            "split":name,"whole_episode_held_out":True,
            "complete_eligible_evaluation_view":anchors==len(view),
            "actual_task_success_rate_measured":False}
    save(output,{"utc":utc(),"policy_evaluation_mode":"eval/zero latent; no gradients",
        "metrics":reports,"notes":["Real validation has no gripper transitions and right OPEN class is absent.",
            "Simulation evaluation uses frozen independent validation/test episodes excluded from all training and normalization.",
            "Window-weighted offline errors are not physical task success or proof of benefit from mixing."]})
    return reports
def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; no CPU long-training fallback")
    if importlib.metadata.version("lerobot")!="0.4.4":raise RuntimeError("Unexpected LeRobot version")
    cfg_real=read(TRAIN/"datasets/pick_place_gripper_v1/dataset.json")
    real_root=TRAIN/cfg_real["root"]
    split=read(TRAIN/"datasets/pick_place_gripper_v1/split.json")
    import urllib.request
    with urllib.request.urlopen(cfg_real["capture_status_url"],timeout=5) as r:cap=json.load(r)
    capture={k:cap[k] for k in ("recording","conversion","queued","stack","demo")}
    save(args.out/"capture-idle.json",capture)
    if any(capture[k] for k in ("recording","conversion","queued")):raise RuntimeError("Capture is busy")
    backbone=Path(torch.hub.get_dir())/"checkpoints/resnet18-f37072fd.pth"
    if not backbone.is_file() or sha(backbone)!=EXPECTED_BACKBONE:raise RuntimeError("Approved ImageNet weights missing/changed")
    env_root=TRAIN/".venvs/act/cu128"
    if sha(env_root/"requirements.lock.txt")!=read(env_root/"environment.json")["lock_sha256"]:
        raise RuntimeError("ACT CUDA environment lock changed")
    contract=read(args.data_run/"binary-view/acceptance.json")
    assert contract["mixed_labelled_window_interface_ready"]
    assert contract["validation_excluded_from_normalization"]
    assert len(split["train"])==88 and len(split["validation"])==10 and not set(split["train"])&set(split["validation"])
    lock=read(TRAIN/"datasets/pick_place_gripper_v1/dataset.lock.json")
    actual_files={p.relative_to(real_root).as_posix() for d in ("data","meta","videos") for p in (real_root/d).rglob("*") if p.is_file()}
    if actual_files!=set(lock["files"]):raise RuntimeError("Real dataset file set changed")
    for rel,digest in lock["files"].items():
        if sha(real_root/rel)!=digest:raise RuntimeError("Real dataset changed: "+rel)
    data_preservation=read(args.data_run/"preservation.json")
    if data_preservation.get("source_files_unchanged") is False:
        assert data_preservation["training_data_sources_unchanged"]
        assert data_preservation["evaluation_sim_modules_match_original_baseline"]
    preservation_scope=data_preservation.get("source_preservation_scope","Original protected sources")
    protected=dict(data_preservation["source_sha256_after"])
    more=[args.data_run/"binary-view"/n for n in ("real-binary-task-view.npz","sim-binary-task-view.npz",
        "binary-task-normalization.json","data-contract.json","acceptance.json")]
    more+=[Path(__file__),Path(__file__).with_name("prepare_binary_gripper_view.py"),
        env_root/"requirements.lock.txt",env_root/"environment.json",TRAIN/"models/act/model.json",backbone,
        Path(inspect.getsourcefile(ACTPolicy)),Path(inspect.getsourcefile(upstream))]
    for p in more:protected[str(p)]={"sha256":sha(p),"bytes":p.stat().st_size}
    for p,e in protected.items():
        if sha(p)!=e["sha256"]:raise RuntimeError("Protected source changed: "+p)
    save(args.out/"source-fingerprints-before.json",protected)
    provenance=read(args.data_run/"lerobot-0.4.4/export-provenance.json")
    loaded={"real":LeRobotDataset(cfg_real["repo_id"],root=real_root,video_backend="pyav",download_videos=False),
        "sim":LeRobotDataset(provenance["repo_id"],root=args.data_run/"lerobot-0.4.4",download_videos=False)}
    grids={d:dict(np.load(args.data_run/f"binary-view/{d}-binary-task-view.npz",allow_pickle=False)) for d in ("real","sim")}
    rg,sg=grids["real"],grids["sim"]
    assert np.isin(rg["episode_index"][rg["train_anchor_indices"]],split["train"]).all()
    assert np.isin(rg["episode_index"][rg["validation_anchor_indices"]],split["validation"]).all()
    train_views={d:BinaryACTView(loaded[d],grids[d],d,
            grids[d]["train_anchor_indices"] if d=="real" or args.domain_mode=="mixed" else np.empty(0,dtype=np.int64))
            for d in ("real","sim")}
    if args.domain_mode=="mixed" and not len(train_views["sim"]):raise RuntimeError("No eligible simulation training anchors")
    stats=read(args.data_run/"binary-view/binary-task-normalization.json")
    active_domains=("real","sim") if args.domain_mode=="mixed" else ("real",)
    if args.domain_mode=="real-only":
        stats=copy.deepcopy(stats)
        for key,field,selector in [("observation.state","state_joint12",lambda g:g["train_anchor_indices"]),
                ("action","task_intent_action14",lambda g:future_rows(g,g["train_anchor_indices"]))]:
            values=rg[field][selector(rg)]
            stats[key]={"mean":values.mean(0).tolist(),"std":np.maximum(values.std(0),1e-6).tolist(),"fit_frames":len(values)}
    assert contract["sim_validation_and_test_excluded_from_normalization"]
    for d in active_domains:
        if "split" in grids[d]:
            assert np.isin(grids[d]["split"][grids[d]["train_anchor_indices"]],["train","candidate_sim_pool"]).all()
    for key,field,selector in [("observation.state","state_joint12",lambda g:g["train_anchor_indices"]),
        ("action","task_intent_action14",lambda g:future_rows(g,g["train_anchor_indices"]))]:
        values=np.concatenate([grids[d][field][selector(grids[d])] for d in active_domains])
        np.testing.assert_allclose(stats[key]["mean"],values.mean(0),rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(stats[key]["std"],np.maximum(values.std(0),1e-6),rtol=1e-5,atol=1e-6)
        assert len(values)==stats[key]["fit_frames"]
    save(args.out/"normalization.json",{"fit":"Unique anchors/targets of selected training domains; fixed ImageNet RGB","training_domains":list(active_domains),"sim_validation_and_test_excluded":True,
        "real_validation_excluded":True,"statistics":stats,"sampling_weights_do_not_change_statistics":True})
    norm={k:{s:torch.tensor(v,dtype=torch.float32) for s,v in row.items() if s in ("mean","std")} for k,row in stats.items()}
    for c in CAMERAS:norm[c]={"mean":torch.tensor([.485,.456,.406])[:,None,None],"std":torch.tensor([.229,.224,.225])[:,None,None]}
    features={k:copy.deepcopy(loaded["real"].meta.features[k]) for k in ("observation.state","action",*CAMERAS)}
    features["observation.state"]["shape"]=[12]
    if features["observation.state"].get("names") is not None:
        features["observation.state"]["names"]=[features["observation.state"]["names"][i] for i in ARM]
    for c in CAMERAS:features[c]["shape"]=[240,320,3]
    meta=SimpleNamespace(features=features,stats=norm)
    mixed=MixedView(train_views["real"],train_views["sim"],meta)
    eval_indices={"real_validation":rg["validation_anchor_indices"],
        "sim_validation":sg["validation_anchor_indices"],"sim_test":sg["test_anchor_indices"]}
    if not len(eval_indices["sim_validation"]) or not len(eval_indices["sim_test"]):
        raise RuntimeError("Independent successful simulation validation/test episodes required")
    for name in ("sim_validation","sim_test"):
        expected="validation" if name=="sim_validation" else "test"
        assert (sg["split"][eval_indices[name]]==expected).all()
        assert not np.intersect1d(sg["train_anchor_indices"],eval_indices[name]).size
    eval_views={name:BinaryACTView(loaded["real" if name=="real_validation" else "sim"],
        rg if name=="real_validation" else sg,"real" if name=="real_validation" else "sim",indices)
        for name,indices in eval_indices.items()}
    initial_eval_views={k:v for k,v in eval_views.items() if k!="sim_test"}
    np.savez_compressed(args.out/"evaluation-indices.npz",**eval_indices)
    inputs={"observation.state":PolicyFeature(type=FeatureType.STATE,shape=(12,))}
    inputs.update({c:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320)) for c in CAMERAS})
    settings=read(TRAIN/"models/act/model.json")["policy"]
    policy_cfg=ACTConfig(**settings,input_features=inputs,
        output_features={"action":PolicyFeature(type=FeatureType.ACTION,shape=(14,))},device="cuda",push_to_hub=False)
    cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/phyagentos-binary-mixed-v1",root=args.data_run/"binary-view"),
        policy=policy_cfg,output_dir=args.out/"trainer",job_name=args.out.name,seed=args.seed,batch_size=args.batch_size,
        num_workers=4,steps=args.steps,eval_freq=0,save_freq=min(500,args.steps),log_freq=100,
        wandb=WandBConfig(enable=False))
    save(args.out/"request.json",{"utc":utc(),"authorization":"User: 推进混合训练",
        "stage":"Bounded mixed labelled-window ACT experiment; full binary task coverage not admitted",
        "data_run":str(args.data_run),"steps":args.steps,"seed":args.seed,"batch_size":args.batch_size,
        "domain_mode":args.domain_mode,"real_sim_batch_counts":[args.batch_size if args.domain_mode=="real-only" else args.batch_size//2,0 if args.domain_mode=="real-only" else args.batch_size//2],"sampling":"Shuffled cycles within training domains; equal optimizer steps and total batch size",
        "source_train_anchors":{d:len(v) for d,v in train_views.items()},
        "original_real_train_episodes":split["train"],"original_real_validation_episodes":split["validation"],
        "sim_source_episode_ids":[e["source_episode_id"] for e in provenance["episodes"]] if "episodes" in provenance else ["paos-rpent-40a953e74d51"],
        "sim_training_episodes":[int(x) for x in np.unique(sg["episode_index"][sg["train_anchor_indices"]])],
        "sim_validation_episodes":[int(x) for x in np.unique(sg["episode_index"][sg["validation_anchor_indices"]])],
        "sim_test_episodes":[int(x) for x in np.unique(sg["episode_index"][sg["test_anchor_indices"]])],
        "real_assistance":"unknown",
        "domain_task_assist_are_policy_inputs":False,"real_camera_state_alignment_limits":contract["data_contract"],
        "cross_domain_mixing_ready":contract["cross_domain_mixing_ready"],"full_task_ready":False,
        "model_architecture":"Full upstream ACT; new state12 policy head; local ImageNet ResNet18 visual initialization only",
        "old_state14_checkpoint_loaded":False,"new_sim_rollouts":0,"llm_requests":0,"hardware_connected":False})
    save(args.out/"versions-and-artifacts.json",{"interpreter":sys.executable,"packages":{n:importlib.metadata.version(n) for n in
        ("torch","torchvision","lerobot","numpy","datasets","pyarrow","accelerate")},
        "gpu":torch.cuda.get_device_name(0),"backbone":{"path":str(backbone),"sha256":EXPECTED_BACKBONE},
        "policy_source":str(inspect.getsourcefile(ACTPolicy)),"upstream_train_source":str(inspect.getsourcefile(upstream)),
        "upstream_sources_modified":False})
    original_make=upstream.make_policy
    initial={}
    def create_policy(*a,**k):
        policy=original_make(*a,**k)
        conv=next(m for m in policy.modules() if isinstance(m,torch.nn.Conv2d))
        assert torch.equal(conv.weight.detach().cpu(),torch.load(backbone,map_location="cpu",weights_only=True)["conv1.weight"])
        save(args.out/"backbone-initialization.json",{"source_sha256":EXPECTED_BACKBONE,"first_convolution_exact_match":True,
            "total_parameters":sum(p.numel() for p in policy.parameters()),"state_shape":[12],"action_shape":[32,14]})
        pre,_=make_pre_post_processors(policy_cfg=policy.config,dataset_stats=norm)
        initial.update(evaluate(policy,pre,initial_eval_views,stats,args.out/"evaluation-initial.json"))
        return policy
    counts={"real":0,"sim":0};pools={};unique={"real":set(),"sim":set()}
    def batches(_):
        loader=DataLoader(mixed,batch_sampler=BalancedBatches(len(train_views["real"]),len(train_views["sim"]),
            args.batch_size,args.steps,args.seed),num_workers=4,pin_memory=True,prefetch_factor=2)
        for raw in loader:
            for d,pool,src in zip(raw["meta.domain"],raw["meta.assistance_pool"],raw["meta.source_row_index"].tolist()):
                counts[d]+=1;pools[pool]=pools.get(pool,0)+1;unique[d].add(int(src))
            assert raw["meta.domain"].count("real")==(args.batch_size if args.domain_mode=="real-only" else args.batch_size//2)
            yield raw
    updates=0;records=[]
    original_update=upstream.update_policy
    def checked_update(*a,**k):
        nonlocal updates
        tracker,detail=original_update(*a,**k);updates+=1
        row={"step":updates,"loss":float(tracker.loss.val),"grad_norm":float(tracker.grad_norm.val),
            "lr":float(tracker.lr.val),"update_seconds":float(tracker.update_s.val),
            "l1_loss":detail.get("l1_loss"),"kld_loss":detail.get("kld_loss")}
        if not np.isfinite([row["loss"],row["grad_norm"]]).all():raise RuntimeError("Nonfinite training")
        with (args.out/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        records.append(row)
        if updates==1 or updates%100==0:
            status={"utc":utc(),"status":"training","optimizer_updates":updates,"target_steps":args.steps,
                "last_metrics":row,"sampled_domain_counts":counts,"sampled_assistance_pools":pools,
                "elapsed_seconds":time.perf_counter()-start}
            save(args.out/"status.json",status);print(json.dumps(status),flush=True)
        return tracker,detail
    upstream.make_dataset=lambda _:mixed
    upstream.make_policy=create_policy
    upstream.cycle=batches
    upstream.update_policy=checked_update
    sys.argv=[sys.argv[0]]
    print(json.dumps({"phase":"starting_upstream_train","anchors":mixed.num_frames,"steps":args.steps}),flush=True)
    upstream.train(cfg)
    checkpoint=(args.out/"trainer/checkpoints/last").resolve()
    saved_step=read(checkpoint/"training_state/training_step.json")["step"]
    assert saved_step==args.steps==updates
    for f in ("pretrained_model/model.safetensors","training_state/optimizer_state.safetensors"):
        tensors=load_file(str(checkpoint/f))
        assert tensors and all(torch.isfinite(t).all() for t in tensors.values())
    policy=ACTPolicy.from_pretrained(str(checkpoint/"pretrained_model")).to("cuda")
    pre,_=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(checkpoint/"pretrained_model"),
        preprocessor_overrides={"device_processor":{"device":"cuda"}})
    final=evaluate(policy,pre,eval_views,stats,args.out/"evaluation-final.json")
    save(args.out/"evaluation-comparison.json",{"initial_and_final_same_indices":True,
        "normalized_l1_initial_final":{n:[initial[n]["normalized_masked_action_l1_zero_latent"],
            final[n]["normalized_masked_action_l1_zero_latent"]] for n in initial},
        "initial_baseline":"Same ImageNet-initialized policy before updates; domain_mode identifies the actual trained control","domain_mode":args.domain_mode,
        "causal_benefit_of_mixing_established":False})
    after={p:{"sha256":sha(p),"bytes":Path(p).stat().st_size} for p in protected}
    if after!=protected:raise RuntimeError("Protected source changed during training")
    save(args.out/"preservation.json",{"utc":utc(),"source_files_unchanged":True,"protected_files":len(after),
        "protected_bytes":sum(v["bytes"] for v in after.values()),"source_sha256_after":after,
        "source_preservation_scope":preservation_scope,"input_preservation_report":str(args.data_run/"preservation.json")})
    summary={"utc":utc(),"status":"completed","optimizer_updates":updates,"target_steps":args.steps,"exit_code":0,
        "elapsed_seconds":time.perf_counter()-start,"checkpoint":str(checkpoint),
        "checkpoint_step":saved_step,"checkpoint_model_and_optimizer_finite":True,
        "sampled_domain_counts":counts,"sampled_assistance_pools":pools,"unique_source_rows":{d:len(v) for d,v in unique.items()},
        "batch_size":args.batch_size,"initial_loss":records[0]["loss"],"final_loss":records[-1]["loss"],
        "mean_loss_first_100":float(np.mean([r["loss"] for r in records[:100]])),
        "mean_loss_last_100":float(np.mean([r["loss"] for r in records[-100:]])),
        "real_validation":final["real_validation"],"sim_validation":final["sim_validation"],"sim_test":final["sim_test"],"domain_mode":args.domain_mode,"source_files_unchanged":True,"upstream_training_completed":True,
        "new_simulation_rollouts":0,"llm_requests":0,"hardware_connected":False,"full_binary_task_ready":False,
        "task_success_measured":False,"unassisted_policy_transfer_validated":False,
        "source_preservation_scope":preservation_scope,"shared_source_drift_report":str(args.data_run/"preservation.json")}
    save(args.out/"result.json",summary);save(args.out/"status.json",summary)
    print(json.dumps(summary),flush=True)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-run",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--domain-mode",choices=("real-only","mixed"),required=True)
    ap.add_argument("--steps",type=int,default=2000);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--batch-size",type=int,default=16);args=ap.parse_args()
    if args.steps<=0 or args.batch_size<2 or args.batch_size%2:raise ValueError("Positive steps and even batch size required")
    with (TRAIN/"runs/.training.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
if __name__=="__main__":main()
