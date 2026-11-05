"""Strict local ACT inference from an authorized, hash-verified checkpoint copy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from http.server import HTTPServer

import numpy as np

from .pi05_server import digest, read_json, Handler
from .policy_wire import PROTOCOL, TASK, IMAGE_SHAPES

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'models/act/checkpoint'


def audit():
    source = read_json(CHECKPOINT.parent/'source.json')
    if not source.get('remote_hash_verified') or not source.get('local_copy_complete'):
        raise ValueError('ACT copy has not been verified')
    hashes = {}
    for name, entry in source['files'].items():
        p = (CHECKPOINT/name).resolve()
        if not p.is_relative_to(CHECKPOINT.resolve()) or p.stat().st_size != entry['bytes'] or digest(p)!=entry['sha256']:
            raise ValueError('ACT file changed: '+name)
        hashes[name] = entry['sha256']
    config = read_json(CHECKPOINT/'config.json')
    inputs = {'observation.state':{'type':'STATE','shape':[16]},
              **{'observation.images.'+k:{'type':'VISUAL','shape':[3,240,320]} for k in IMAGE_SHAPES}}
    if config['type']!='act' or config['input_features']!=inputs or config['output_features']!={'action':{'type':'ACTION','shape':[16]}}:
        raise ValueError('ACT dimensions/camera keys do not match reviewed OpenArm checkpoint')
    if config['chunk_size']!=32 or config['n_action_steps']!=16 or config.get('temporal_ensemble_coeff') is not None:
        raise ValueError('ACT differs from the reviewed deployment chunk contract')
    norm={'VISUAL':'MEAN_STD','STATE':'MEAN_STD','ACTION':'MEAN_STD'}
    if config['normalization_mapping']!=norm:
        raise ValueError('ACT normalization mismatch')
    for label,kind in [('pre','normalizer_processor'),('post','unnormalizer_processor')]:
        steps=read_json(CHECKPOINT/f'policy_{label}processor.json')['steps']
        if any('relative' in s['registry_name'] or 'absolute' in s['registry_name'] for s in steps):
            raise ValueError('unexpected action conversion')
        selected=[s for s in steps if s['registry_name']==kind]
        if len(selected)!=1 or selected[0]['config']['norm_map']!=norm:
            raise ValueError('stored ACT normalization mismatch')
    import hashlib
    return {'protocol':PROTOCOL,'policy_type':'act','checkpoint':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'files_sha256':hashes,'chunk_size':32,'execute_horizon':16,'num_inference_steps':0,
            'action_semantics':'openarm_absolute_joint16','state_semantics':'openarm_encoder16',
            'camera_shapes':{k:list(v) for k,v in IMAGE_SHAPES.items()},
            'model_image_shapes':{k:[3,240,320] for k in IMAGE_SHAPES},
            'image_adapter':'RGB uint8 /255 → CHW float32 → bilinear 240x320, align_corners=False; matches reviewed deployment',
            'task':TASK,'physical_transfer_validated':False}


class Policy:
    def __init__(self, device='cuda'):
        import torch
        import lerobot
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_file
        self.metadata=audit()
        runtime=read_json(ROOT/'sim/learning.json')
        actual=Path(lerobot.__file__).resolve().parent.parent
        revision=subprocess.check_output(['git','-C',str(actual),'rev-parse','HEAD'],text=True).strip()
        changed=subprocess.check_output(['git','-C',str(actual),'status','--porcelain','--untracked-files=no'],text=True)
        if actual!=Path(runtime['source']).resolve() or revision!=runtime['source_revision'] or changed.strip():
            raise ValueError('ACT must use the pinned clean official LeRobot runtime')
        if device=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; no silent CPU fallback')
        torch.set_num_threads(4)
        self.torch=torch
        config=PreTrainedConfig.from_pretrained(str(CHECKPOINT),local_files_only=True)
        config.device=device
        config.pretrained_backbone_weights=None  # every backbone tensor is loaded from checkpoint
        self.model=ACTPolicy(config)
        weights=load_file(str(CHECKPOINT/'model.safetensors'),device='cpu')
        self.model.load_state_dict(weights,strict=True)
        actual_weights=self.model.state_dict()
        if any(not torch.equal(actual_weights[k].cpu(),v.to(dtype=actual_weights[k].dtype)) for k,v in weights.items()):
            raise ValueError('ACT tensor verification failed')
        self.model.to(device).eval()
        self.pre,self.post=make_pre_post_processors(config,pretrained_path=str(CHECKPOINT),
            preprocessor_overrides={'device_processor':{'device':device,'float_dtype':None}},
            postprocessor_overrides={'device_processor':{'device':'cpu','float_dtype':None}})
        self.metadata.update(device=device,strict_weights_verified=True,runtime_revision=revision,adapter_sha256=digest(Path(__file__)))
        self.session=None;self.sequence=0;self.deadline=0

    def begin(self, seed):
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('invalid seed')
        if self.session and time.monotonic()<self.deadline:raise ValueError('policy session already active')
        self.torch.manual_seed(seed);self.model.reset();self.pre.reset();self.post.reset()
        self.session=uuid.uuid4().hex;self.sequence=0;self.deadline=time.monotonic()+180
        return {'session':self.session,'seed':seed,'metadata':self.metadata}

    def infer(self, data):
        if not self.session or data['session'].item()!=self.session or time.monotonic()>=self.deadline:
            raise ValueError('stale or mismatched policy session')
        if data['sequence'].item()!=self.sequence or self.sequence>=75:raise ValueError('invalid observation sequence or exhausted action budget')
        torch=self.torch
        batch={'observation.state':torch.from_numpy(data['state'].copy()),'task':TASK}
        for name in IMAGE_SHAPES:
            image=torch.from_numpy(data[name].copy()).permute(2,0,1).float().div(255)
            batch['observation.images.'+name]=torch.nn.functional.interpolate(image.unsqueeze(0),size=(240,320),mode='bilinear',align_corners=False).squeeze(0).contiguous()
        self.sequence+=1;started=time.monotonic()
        with torch.inference_mode():
            actions=self.post(self.model.predict_action_chunk(self.pre(batch)))[0].cpu().float().numpy()
        if actions.shape!=(32,16) or not np.isfinite(actions).all():raise ValueError('invalid ACT action chunk')
        self.deadline=time.monotonic()+180
        return {'actions':actions.tolist(),'sequence':self.sequence-1,'inference_seconds':time.monotonic()-started}

    def end(self, session):
        if session!=self.session:raise ValueError('invalid policy session')
        self.model.reset();self.session=None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.add_argument('--port',type=int,default=8003)
    args=parser.parse_args()
    if args.check:print(json.dumps(audit(),indent=2));return
    cache=ROOT/'.runtime/cache/act';cache.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(cache),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_HOME=str(cache))
    policy=Policy();server=HTTPServer(('127.0.0.1',args.port),Handler);server.policy=policy
    print(json.dumps({'status':'ready','metadata':policy.metadata}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
