"""Offline ACT checkpoint inference in the existing ACT CUDA environment."""
from pathlib import Path
import argparse,json,os,sys
TRAIN=Path(str(Path(os.environ.get("PIPER_TRAINING_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_training_project")))))
os.environ.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",
 HF_HOME=str(TRAIN/".cache/huggingface/act/cu128"),TORCH_HOME=str(TRAIN/".cache/torch/act/cu128"),
 TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
os.environ.pop("PYTHONPATH",None);os.environ.pop("PYTHONHOME",None)
import numpy as np,torch
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);a=ap.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42)
 policy=ACTPolicy.from_pretrained(str(a.checkpoint)).to("cuda").eval();policy.reset()
 pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(a.checkpoint),
  preprocessor_overrides={"device_processor":{"device":"cuda"}})
 print(json.dumps({"type":"ready","state":12,"action":14,"n_action_steps":policy.config.n_action_steps,
  "chunk_size":policy.config.chunk_size,"parameters":sum(p.numel() for p in policy.parameters())}),flush=True)
 for line in sys.stdin:
  request=json.loads(line)
  if request.get("type")=="stop":return
  state=np.asarray(request["state12"],dtype=np.float32);assert state.shape==(12,) and np.isfinite(state).all()
  batch={"observation.state":torch.from_numpy(state)[None]}
  for name,path in request["rgb_jpeg_paths"].items():
   with Image.open(path) as f:rgb=np.asarray(f.convert("RGB"),dtype=np.float32)/255.
   assert rgb.shape==(240,320,3)
   batch["observation.images."+name]=torch.from_numpy(rgb.copy()).permute(2,0,1)[None]
  with torch.inference_mode():action=post(policy.select_action(pre(batch))).detach().cpu().reshape(-1)
  assert action.shape==(14,) and torch.isfinite(action).all()
  print(json.dumps({"type":"action","control_step":request["control_step"],"action14":action.tolist()}),flush=True)
if __name__=="__main__":main()
