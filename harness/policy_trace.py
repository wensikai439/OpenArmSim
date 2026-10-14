"""Local policy receipts for diagnosis, without changing actions or guards.

Successful predictions retain all rows; rejected requests additionally retain
their exact input NPZ. Pixels and raw chunks stay in the experiment archive.
"""
from pathlib import Path
import hashlib

import numpy as np

from sim.environment import ROOT
from sim.policy import LeRobotClient, make_client
from sim.policy_wire import decode, IMAGE_SHAPES
from sim.timing import PolicyTiming


class TracingClient(LeRobotClient):
    def request(self, path, body=None, content_type='application/json'):
        if path != '/infer':
            return super().request(path, body, content_type)
        self.response_capture = {'input':body,'response':None}
        # Only the bounded local policy request; never a provider/API packet.
        result = super().request(path, body, content_type)
        self.response_capture['response'] = result
        return result


def archive_response(owner, inference, capture, *, horizon, gripper_bounds):
    folder = Path(owner.path).resolve()
    if not folder.is_relative_to((ROOT/'runs').resolve()) or folder == (ROOT/'runs').resolve():
        raise ValueError('policy receipts require the active native recorder directory')
    body = capture.get('input')
    if not isinstance(body,bytes) or len(body)>8*1024*1024:
        raise ValueError('bounded encoded local policy input required')
    observation=decode(body)
    response=capture.get('response')
    path=folder/'policy';path.mkdir(exist_ok=True)
    index=inference['request_id']
    if type(index) is not int or not 0<=index<1200:raise ValueError('invalid recorder request index')
    prefix=f'{index:04d}'
    diagnostic={'schema':'policy-response-trace/1','request_id':index,
        'sequence':int(observation['sequence']), 'state':observation['state'].tolist(),
        'input_sha256':hashlib.sha256(body).hexdigest(),
        'image_sha256':{k:hashlib.sha256(observation[k].tobytes()).hexdigest() for k in IMAGE_SHAPES},
        'source':'exact local policy input and response; no physical calibration claim',
        'validation_changed':False,'hardware_access':False,'transfer_ready':False}
    if isinstance(response,dict) and 'actions' in response:
        try:
            actions=np.asarray(response['actions'],np.float32)
            diagnostic['shape']=list(actions.shape)
            if actions.ndim>2 or actions.size>3200:raise ValueError('response outside trace array budget')
            diagnostic['finite']=bool(np.isfinite(actions).all())
            filename=prefix+'-actions.npy';destination=path/filename
            if destination.exists():raise ValueError('do not overwrite a prediction receipt')
            np.save(destination,actions,allow_pickle=False)
            diagnostic['actions_file']='policy/'+filename
            diagnostic['actions_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
            if actions.ndim==2 and actions.shape[1]==16:
                grip=actions[:,[7,15]]
                low,high=np.asarray(gripper_bounds,np.float32)
                bad=np.isfinite(grip)&((grip<low-1e-5)|(grip>high+1e-5))
                diagnostic.update(execute_horizon=horizon,
                    out_of_range_selected=int(bad[:horizon].sum()),out_of_range_unused=int(bad[horizon:].sum()),
                    gripper_bounds_rad=[low.tolist(),high.tolist()],
                    out_of_range=[{'row':int(r),'side':('left','right')[int(c)],'value_rad':float(grip[r,c])}
                        for r,c in np.argwhere(bad)[:32]])
        except (TypeError,ValueError) as exc:
            diagnostic['response_trace_error']=str(exc)
    if not inference['response_valid']:
        filename=prefix+'-input.npz';destination=path/filename
        if destination.exists():raise ValueError('do not overwrite a rejected input receipt')
        destination.write_bytes(body)
        diagnostic['rejected_input_file']='policy/'+filename
    return diagnostic


class TracingTiming(PolicyTiming):
    def actions(self, env, client, config, event, cancelled=lambda:False, **kwargs):
        owner=getattr(event,'__self__',None)
        def record(value):
            capture=getattr(client,'response_capture',None)
            if value.get('type')=='policy_inference' and capture is not None:
                try:
                    # Recorder/env ownership is explicit; it is not an LLM path.
                    if owner is None:raise ValueError('native recorder owner unavailable')
                    trace=archive_response(owner,value,capture,horizon=config['execute_horizon'],
                        gripper_bounds=[env.action_space.low[[7,15]],env.action_space.high[[7,15]]])
                except Exception as exc:
                    # A diagnostic failure cannot mask the actual policy outcome.
                    trace={'schema':'policy-response-trace/1','trace_error':str(exc),'validation_changed':False}
                finally:
                    client.response_capture=None
                value={**value,'response_diagnostic':trace}
            event(value)
        yield from super().actions(env,client,config,record,cancelled,**kwargs)


def traced_client(config,seed=0):
    """Wrap only the supported local HTTP policy; preserve other backends."""
    return TracingClient(config,seed) if config.get('backend')=='lerobot_http' else make_client(config,seed)


def install_isolated(base):
    """Inject local policy diagnostics into one service class, without module patches."""
    if getattr(base,'_openarm_policy_trace',False):return base
    class TracedService(base):
        _openarm_policy_trace=True
        def __init__(self,*args,**kwargs):
            super().__init__(*args,policy_client_factory=traced_client,
                policy_timing_class=TracingTiming,**kwargs)
    return TracedService
