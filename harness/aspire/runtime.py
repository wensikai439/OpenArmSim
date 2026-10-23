"""Official ASPIRE ProgramExecutor over isolated native OpenArm primitives.

This adapter is explicitly geometry-oracle based. It does not claim visual
localization, the official benchmark protocol, or hardware transfer.
"""
import argparse
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_BUNDLED = ROOT/'third_party/aspire'
SOURCE = Path(os.environ.get('OPENARM_ASPIRE_SOURCE', str(_BUNDLED if _BUNDLED.exists() else Path('/home/wsk/ASPIRE/libero-pro/source'))))
sys.path.insert(0, str(SOURCE/'src'))
from cap_harness.runtime import ProgramExecutor
from cap_harness.registry import ToolRegistry
from cap_harness.contracts import ExecutionResult
from cap_harness.errors import ApiError, ErrorCode


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if hasattr(value, 'items'): return dict(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial, allow_nan=False)+'\n')


def validate_case(case):
    if set(case) != {'seed','bottle_xy_m'} or type(case['seed']) is not int:
        raise ValueError('case must contain seed and bottle_xy_m')
    xy=case['bottle_xy_m']
    if (not isinstance(xy,list) or len(xy)!=2 or any(type(x) not in (float,int) for x in xy)
        or not np.isfinite(xy).all() or not .40<=xy[0]<=.65 or not -.32<=xy[1]<=-.07):
        raise ValueError('fixture outside declared right-side table region')
    return case


def fixture_environment(base, case):
    validate_case(case)
    class FixtureEnv(base):
        def reset(self, **kwargs):
            kwargs['seed']=case['seed']
            outcome=super().reset(**kwargs)
            if self.tick or float(self.data.time)!=0:
                raise RuntimeError('fixture is allowed only before the first physics step')
            address=self.registry['bottle']['qposadr']
            self.data.qpos[address:address+2]=case['bottle_xy_m']
            mujoco.mj_forward(self.model,self.data)
            self.evaluation=self._evaluate(0.)
            self._protect()
            return outcome
    return FixtureEnv


def evaluate_route(stages, evaluation, *, program_ok):
    """Task score requires the directed physical chain, not a model assertion."""
    order=[]
    for row in stages:
        if not row.get('ok'): continue
        args=row['arguments']; name=row['tool']
        if name=='pick' and args.get('arm')=='right': order.append('right-pick')
        if name=='handover' and args.get('from_arm')=='right' and args.get('to_arm')=='left':
            order.append('right-left')
        if name=='place' and args.get('arm')=='left' and args.get('receptacle_id')=='basket':
            order.append('left-basket')
    expected=['right-pick','right-left','left-basket']
    cursor=0
    for name in order:
        if cursor<len(expected) and name==expected[cursor]: cursor+=1
    route=cursor==3
    placement=bool(evaluation.get('inside_basket') and evaluation.get('supported_by_basket')
        and evaluation.get('placement',{}).get('confirmed')
        and not any(evaluation.get('held',{}).values()) and not evaluation.get('invalid'))
    return {'directed_route':route,'stable_released_basket':placement,
            'program_completed':bool(program_ok),'task_success':bool(program_ok and route and placement)}


class PublicTrace:
    """Primitive spans compatible with the official executor recorder hook."""
    def __init__(self, path):
        self.path=path

    @contextmanager
    def span(self, name, *, category, inputs):
        row={'tool':name,'category':category,'inputs':inputs,'started':time.monotonic()}
        class Span:
            def output(self, value, *, ok=True): row['ok']=ok
        try:
            yield Span()
        finally:
            row['wall_seconds']=time.monotonic()-row.pop('started')
            with self.path.open('a') as stream:
                stream.write(json.dumps(row, default=serial, allow_nan=False)+'\n')

    def keyframe(self, label):
        # All custom motion tools capture synchronized images themselves.
        pass


class OpenArmTrial:
    validate_case = staticmethod(validate_case)
    fixture_environment = staticmethod(fixture_environment)

    def __init__(self, case, output):
        self.case=self.validate_case(case); self.output=output
        self.stages=[]; self.native_paths={}; self.motion_calls=0
        self.service=None
        os.environ['OPENARM_SCENE_OBJECTS']='[]'
        from tools.object_workbench import build_backend
        from sim.recording import Recorder
        import sim.scene_library as library
        factory,_=build_backend()
        env_type=self.fixture_environment(library.LibraryEnv,case)
        owner=self
        class TrialRecorder(Recorder):
            def __init__(self, env, kind, **kwargs):
                kwargs['root']=output/'native'
                super().__init__(env,kind,**kwargs)
                owner.native_paths[self.path.name]=self.path
        # Process-local adapter selection. No other process or production file changes.
        with patch.object(library,'LibraryEnv',env_type):
            self.service=factory(recorder_class=TrialRecorder)
        self.service.call('task_budget',seconds=120)
        self.capture('initial')

    def capture(self, label):
        folder=self.output/'media';folder.mkdir(exist_ok=True)
        snapshot=self.service.snapshot()
        for camera in ('overview','head','left','right'):
            (folder/f'{label}-{camera}.jpg').write_bytes(self.service.frame(camera,snapshot['observation_id']))

    def get_observation(self):
        snapshot=self.service.snapshot(); diagnostics=snapshot['diagnostics']
        scene=diagnostics['manipulation_scene']
        return {'source':'simulation_geometry_oracle',
                'held_by':list(scene['held_by']), 'bottle_center_m':list(scene['position_m']),
                'upright':bool(scene['upright']), 'tick':snapshot['observation']['tick'],
                'sim_seconds':snapshot['observation']['sim_time']}

    def execute(self, tool, task, parameters):
        self.motion_calls+=1
        if self.motion_calls>12:
            raise ValueError('declared12motion-call budget exhausted')
        before=self.get_observation()
        index=len(self.stages)+1
        row={'tool':tool,'arguments':parameters,'before':before,'ok':False}
        self.capture(f'{index:02d}-before-{tool}')
        start=time.monotonic()
        try:
            self.service.call('tamp',task=task,parameters=parameters,
                observation_id=self.service.snapshot()['observation_id'],realtime=False,
                timing_mode='paused',initiator={'type':'aspire-generated-program'})
            while self.service.snapshot().get('job'):
                if time.monotonic()-start>175:
                    self.service.call('stop'); raise TimeoutError('primitive wall-time budget exhausted')
                time.sleep(.03)
            snapshot=self.service.snapshot(); native=snapshot.get('last_run') or {}
            run_id=native.get('run_id'); path=self.native_paths.get(run_id)
            if path is None: raise RuntimeError('accepted primitive has no native recording')
            summary=json.loads((path/'summary.json').read_text())
            physical=summary.get('metrics',{}).get('tamp',{})
            row.update(ok=physical.get('primitive_success') is True,
                steps=summary['transitions'],native_run=str(path.relative_to(self.output)),
                failure_code=physical.get('failure_code'),phase=physical.get('phase'),
                error=summary.get('error'),after=self.get_observation(),wall_seconds=time.monotonic()-start)
        except Exception as exc:
            row.update(error=str(exc),failure_code=getattr(exc,'code','adapter_error'),
                       after=self.get_observation(),wall_seconds=time.monotonic()-start)
        self.stages.append(row)
        self.capture(f'{index:02d}-after-{tool}')
        dump(self.output/'stages.json',self.stages)
        error=None if row['ok'] else ApiError(ErrorCode.EXECUTION_FAILED,
            str(row.get('error') or row.get('failure_code') or 'native primitive failed'),
            details={'failure_code':str(row.get('failure_code'))})
        return ExecutionResult(ok=row['ok'],steps_executed=int(row.get('steps',0)),error=error,
            diagnostics={'current_holder':row['after']['held_by'],'failure_code':row.get('failure_code')})

    def pick(self, arm='right', hold_seconds=2, next_handover=None):
        if arm not in ('left','right') or type(hold_seconds) not in (int,float) or not 2<=hold_seconds<=10:
            raise ValueError('pick requires valid arm and2..10hold_seconds')
        params={'object_id':'bottle','arm':arm,'hold_seconds':hold_seconds}
        if next_handover is not None:
            if next_handover!={'to_arm':'left'} or arm!='right':
                raise ValueError('only right-to-left next_handover is declared for this task')
            params['next_handover']=dict(next_handover)
        return self.execute('pick','pick_bottle',params)

    def handover(self, from_arm='right', to_arm='left', hold_seconds=2, placement_aware=False):
        if (from_arm not in ('left','right') or to_arm not in ('left','right') or from_arm==to_arm
            or type(hold_seconds) not in (int,float) or not 2<=hold_seconds<=10):
            raise ValueError('handover requires distinct arms and2..10hold_seconds')
        if type(placement_aware) is not bool: raise ValueError('placement_aware must be boolean')
        params={'object_id':'bottle','from_arm':from_arm,'to_arm':to_arm,'hold_seconds':hold_seconds}
        if placement_aware: params['placement_aware']=True
        return self.execute('handover','handover_bottle',params)

    def place(self, arm='left', receptacle_id='basket', shape_clearance=False):
        if arm not in ('left','right') or receptacle_id!='basket':
            raise ValueError('this task requires placement into the original basket')
        if type(shape_clearance) is not bool:
            raise ValueError('shape_clearance must be boolean')
        params={'object_id':'bottle','arm':arm,'receptacle_id':receptacle_id}
        if shape_clearance: params['shape_clearance']=True
        return self.execute('place','place_bottle',params)

    def move_relative(self, arm, x_m, y_m, z_m):
        xyz=[x_m,y_m,z_m]
        if (arm not in ('left','right') or any(type(v) not in (int,float) for v in xyz)
            or not np.isfinite(xyz).all() or np.linalg.norm(xyz)>.15):
            raise ValueError('relative motion requires finite vector of at most15cm')
        return self.execute('move_relative','move_relative',{'arm':arm,'delta_m':xyz,'frame':'world'})

    def registry(self):
        names=('get_observation','pick','handover','place','move_relative')
        registry=ToolRegistry(shared_public_names=names,public_extension_allowlist=())
        for name in names:
            registry.register(getattr(self,name),name=name,public=True,
                layer='shared_high_level',capability='custom OpenArm simulator-geometry adapter')
        return registry

    def close(self):
        if self.service:self.service.close()


def run(program, case, output, *, host_type=OpenArmTrial):
    output.mkdir(parents=True,exist_ok=False)
    (output/'program.py').write_bytes(program.read_bytes())
    dump(output/'case.json',case)
    host=None
    report={'program_ok':False,'task_success':False,'protocol_success':False,
            'source':'NVIDIA ASPIRE ProgramExecutor with custom OpenArm oracle adapter',
            'oracle_geometry':True,'vision_only':False,'hardware_access':False,'transfer_ready':False,
            'program_sha256':hashlib.sha256(program.read_bytes()).hexdigest()}
    try:
        host=host_type(case,output)
        execution=ProgramExecutor(host.registry(),recorder=PublicTrace(output/'trace.jsonl')).execute_program(program.read_text())
        evaluation=host.service.snapshot()['diagnostics']['evaluation']
        grade=evaluate_route(host.stages,evaluation,program_ok=execution.ok)
        report.update(program_ok=execution.ok,task_success=grade['task_success'],
            protocol_success=grade['directed_route'] and grade['stable_released_basket'],
            grade=grade,end_evaluation=evaluation,stages=host.stages,
            calls=[{f.name:getattr(c,f.name) for f in fields(c)} for c in execution.calls],
            error=str(execution.error) if execution.error else None,
            native_runs=[str(p.relative_to(output)) for p in host.native_paths.values()])
        failures=[s for s in host.stages if not s['ok']]
        report['failure_stage']=failures[-1]['tool'] if failures else None
        report['failure_code']=failures[-1].get('failure_code') if failures else None
        host.capture('final')
    except Exception as exc:
        report['error']=str(exc);report['failure_stage']='runtime'
        if host:report['stages']=host.stages
    finally:
        if host:host.close()
        dump(output/'result.json',report)
    print(json.dumps({'task_success':report['task_success'],'program_ok':report['program_ok'],
                      'failure_stage':report.get('failure_stage'),'error':report.get('error')}),flush=True)
    return 0 if report['task_success'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'runs'):
        parser.error('output must be a new directory inside runs/')
    raise SystemExit(run(args.program,json.loads(args.case.read_text()),output))
