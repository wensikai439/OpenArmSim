"""Single-owner native recovery lab. No reset, snapshot restore or hardware API.

Short grasp actions and pausable native handover/place phases expose meaningful
checkpoints. Faults are explicit diagnostic actuator-command loss or a one-shot
perceived-pose error, never a live object/joint pose edit or model change.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

from .runtime import ROOT, ProgramExecutor, ToolRegistry, PublicTrace, dump, fixture_environment, validate_case
from sim.adaptive_grasp import KinematicIK
from sim.grasp import candidates, cylinder_bands
from sim.recording import Recorder
from sim.tamp import TampPlanner, PlanningFailure


class SegmentPlanner(KinematicIK, TampPlanner):
    TASKS=(*TampPlanner.TASKS,'segment')

    def __init__(self, env, *, mode, arm='right', grasp=None, fraction=0.):
        super().__init__(env,'segment',parameters={'arm':arm,'object_id':'bottle'})
        self.mode,self.grasp,self.fraction=mode,grasp,fraction
        self._arm(arm,bool(env.evaluation['held'][arm]))

    def actions(self):
        side=self.side;env=self.env
        if self.mode in ('pregrasp','approach'):
            if any(env.evaluation['held'].values()):raise PlanningFailure('already_holding','Empty gripper required')
            self.rotation=np.array(self.grasp['rotation'])
            self.ik_seeds[side]=[np.array(q) for q in self.grasp['seeds']]+self.ik_seeds[side]
            if self.mode=='pregrasp':
                self._phase('approach');yield from self._grip(side,env.grippers[side].open_motor)
                yield from self._move(np.array(self.grasp['pregrasp_m']))
            else:
                self._phase('grasp');yield from self._move(np.array(self.grasp['grasp_m']),True)
        elif self.mode=='gripper':
            if self.fraction>0 and env.evaluation['held'][side]:
                raise PlanningFailure('release_forbidden','Use independently verified handover/place for held-object release')
            self._phase('close' if self.fraction==0 else 'gripper')
            motor=env.grippers[side].open_motor*self.fraction+.05*(1-self.fraction)
            yield from self._grip(side,motor)
            if self.fraction==0 and not env.evaluation['held'][side]:
                raise PlanningFailure('grasp_not_confirmed','No sustained two-finger hold after closure')
        elif self.mode=='lift':
            if not env.evaluation['held'][side]:raise PlanningFailure('hold_required','Verify actual hold before lift')
            initial=float(env._bottle_bounds()[0][2]);self._sync();point,rot=self._pose();self.rotation=rot.copy()
            self._phase('lift');yield from self._move(point+[0,0,.12],True)
            self._phase('hold')
            from sim.decision_timing import GraspStability
            audit=GraspStability(env)
            for _ in range(60):
                yield self.base_action.copy(),{'phase':'hold','scene_source':'simulator_ground_truth'}
                audit.read()
                if not env.evaluation['held'][side]:raise PlanningFailure('grasp_lost','Native hold lost during lift audit')
            if env._bottle_bounds()[0][2]<initial+.06:raise PlanningFailure('lift_not_confirmed','Actual bottle did not lift60mm')
            self.result['hold_audit']=audit.summary()
            if audit.summary()['arms'][side]['status']!='stable_within_audit_window':
                raise PlanningFailure('grasp_unstable','Object-in-hand audit failed')
        elif self.mode=='finish_transfer':
            if not all(env.evaluation['held'].values()):
                raise PlanningFailure('dual_hold_required','Both actual holds required before donor release')
            self._phase('release_right');yield from self._grip('right',env.grippers['right'].open_motor)
            if not env.evaluation['held']['left']:raise PlanningFailure('receiver_grasp_lost','Receiver lost hold')
            from sim.fast_handover import ForwardBottleHandover
            self._phase('retreat');self._arm('right');self._sync();point,_=self._pose()
            yield from ForwardBottleHandover._retreat_after_forward_handover(self,point)
            self._phase('hold')
            for _ in range(60):
                yield self.base_action.copy(),{'phase':'hold','scene_source':'simulator_ground_truth'}
                if not env.evaluation['held']['left'] or env.evaluation['touching']['right']:
                    raise PlanningFailure('handover_not_confirmed','Independent left hold not maintained')
        else:raise ValueError('unknown segment')
        self.result.update(primitive_success=True,status='completed')


class RecoveryWorld:
    PUBLIC=('observe','propose_grasps','pregrasp','approach','close_gripper','open_gripper',
        'verify_hold','lift','retreat','start_transfer','advance','finish_transfer','start_place')

    def __init__(self, case, output, *, environment_factory=fixture_environment):
        allowed={'seed','bottle_xy_m','fault','pose_bias_m'}
        if set(case)-allowed or not {'seed','bottle_xy_m','fault'}<=set(case):
            raise ValueError('expected seed, bottle_xy_m, fault and optional pose_bias_m')
        validate_case({k:case[k] for k in ('seed','bottle_xy_m')})
        bias=case.get('pose_bias_m',.05)
        if type(bias) not in (int,float) or not np.isfinite(bias) or not .04<=bias<=.07:
            raise ValueError('diagnostic pose bias must be .04 to .07m')
        if case.get('fault') not in ('none','right_close_loss','receiver_close_loss','stale_grasp'):
            raise ValueError('unknown diagnostic fault')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.case=copy.deepcopy(case);dump(self.output/'case.json',case)
        from tools.object_workbench import build_backend
        factory,_=build_backend();self.planner_factory=factory._bound_tamp_planner_factory
        from sim.scene_library import LibraryEnv
        base={k:case[k] for k in ('seed','bottle_xy_m')}
        self.env=environment_factory(LibraryEnv,base)(images=False,selection=[])
        self.env.reset(seed=case['seed']);self.env.set_task_budget(120)
        self.rows=[];self.route=[];self.catalogue={};self.checkpoint=None
        self.active=None;self.last_grasp=None;self.last_failure=None;self.calls=0
        self.used_faults=set();self.pose_bias_active=case['fault']=='stale_grasp'
        self.pose_revision=0;self.recovery_actions=0
        self.source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        dump(self.output/'environment.json',self.env.manifest())
        self.capture('initial')

    def capture(self,label):
        folder=self.output/'media';folder.mkdir(exist_ok=True)
        for camera in ('head','right','overview'):
            Image.fromarray(self.env.render(camera)).save(folder/f'{label}-{camera}.jpg',quality=83)

    def _guard(self,readonly=False):
        self.calls+=1
        if self.calls>64:raise ValueError('64tool-call budget exhausted')
        self.env.require_current_sources()
        if not readonly and (self.env._done or self.env.evaluation['invalid']):raise ValueError('terminal physical state; recovery forbidden')

    def observe(self):
        self._guard(readonly=True)
        return self._observation()

    def _observation(self):
        """Unmetered final audit; not an extra public tool or policy backdoor."""
        env=self.env
        pos=env.data.xpos[env.bottle_id].copy()
        if self.pose_bias_active:pos[2]+=self.case.get('pose_bias_m',.05)
        self.observation_tick=env.tick
        return {'source':'simulation_geometry_with_diagnostic_sensor_or_command_faults',
            'tick':env.tick,'sim_seconds':float(env.data.time),'pose_revision':self.pose_revision,
            'terminal':bool(env._done),'invalid':bool(env.evaluation['invalid']),
            'held_by':[s for s,v in env.evaluation['held'].items() if v],
            'touching':dict(env.evaluation['touching']),
            'upright':bool(env.data.xmat[env.bottle_id].reshape(3,3)[2,2]>.98),
            'bottle_center_m':pos.tolist(),'gripper_motor_feedback_rad':{
                'left':float(env.state()[7]),'right':float(env.state()[15])},
            'gripper_open_rad':{s:env.grippers[s].open_motor for s in ('left','right')},
            'checkpoint_id':self.checkpoint['id'] if self.checkpoint else None,
            'current_grasp_id':self.last_grasp,'last_failure':self.last_failure,
            'completed_steps':list(self.route),
            'operation':None if not self.active else {'kind':self.active['kind'],'phase':self.active.get('phase')},
            'remaining_seconds':self.env.task_window()['remaining_seconds']}

    def propose_grasps(self, exclude_ids=()):
        self._guard()
        if self.active:raise ValueError('finish or recover current operation first')
        if any(self.env.evaluation['held'].values()):raise ValueError('cannot plan empty-hand grasp while holding')
        if not self.env.data.xmat[self.env.bottle_id].reshape(3,3)[2,2]>.98:
            raise ValueError('this recovery provider supports upright bottles only')
        blocked={self.catalogue[x]['signature'] for x in exclude_ids if x in self.catalogue
            and self.catalogue[x]['pose_revision']==self.pose_revision}
        planner=SegmentPlanner(self.env,mode='pregrasp');planner._sync()
        bands=cylinder_bands(self.env.model,planner.data,self.env.bottle_id)
        before=self.env.data.qpos.copy();accepted=[];rejected=[]
        for i,c in enumerate(candidates(bands)):
            signature=(c['geom'],round(c['yaw_rad'],6),round(c['pitch_rad'],6))
            if signature in blocked:continue
            value=copy.deepcopy(c)
            if self.pose_bias_active:
                value['grasp_m'][2]+=self.case.get('pose_bias_m',.05)
                value['pregrasp_m'][2]+=self.case.get('pose_bias_m',.05)
            planner.rotation=value['rotation'];saved=copy.deepcopy(planner.ik_seeds['right'])
            try:
                planner.planning_qpos=self.env.data.qpos.copy()
                for finger in self.env.fingers['right']:planner.planning_qpos[finger.qposadr[0]]=max(finger.range)
                q=planner._ik(value['grasp_m'],True);planner.ik_seeds['right'].insert(0,q)
                pre=planner._ik(value['pregrasp_m']);planner._path(pre)
                if not planner._edge(pre,q,True):raise PlanningFailure('approach_collision','Connected approach rejected')
                ident=f'g{self.pose_revision}-{len(self.catalogue):03d}'
                row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in value.items()}
                row.update(id=ident,seeds=[q.tolist(),pre.tolist()],signature=signature,
                    pose_revision=self.pose_revision)
                self.catalogue[ident]=row
                accepted.append({k:row[k] for k in ('id','geom','yaw_rad','pitch_rad','grasp_m','pregrasp_m')})
                if len(accepted)>=3:break
            except PlanningFailure as exc:
                rejected.append({'index':i,'code':exc.code})
            finally:planner.ik_seeds['right']=saved
        np.testing.assert_array_equal(before,self.env.data.qpos)
        dump(self.output/f'proposals-{self.calls:02d}.json',{'accepted':accepted,'rejected':rejected})
        return {'candidates':accepted,'pose_revision':self.pose_revision}

    def _grasp(self, ident):
        if ident not in self.catalogue:raise ValueError('unknown grasp id')
        row=self.catalogue[ident]
        if row['pose_revision']!=self.pose_revision:raise ValueError('stale grasp; re-observe and propose again')
        return row

    def _begin(self,planner,kind):
        if self.active:raise ValueError('an operation is already active')
        recorder=Recorder(self.env,'recovery',record_images=False,seed=self.case['seed'],
            details={'kind':kind,'research_source_sha256':self.source_hash},root=self.output/'native')
        self.active={'planner':planner,'iterator':iter(planner.actions()),'kind':kind,
            'recorder':recorder,'pending':None,'phase':None,'before':recorder.observation(self.env),
            'started_tick':self.env.tick,'injected':set()}

    def _step(self,action,diagnostic):
        op=self.active;env=self.env;requested=np.array(action,copy=True);applied=requested.copy()
        fault=self.case['fault'];phase=diagnostic.get('phase')
        key=('right' if op['kind']=='close_right' else 'left' if op['kind']=='handover' and phase=='dual_hold' else None)
        target_fault='right_close_loss' if key=='right' else 'receiver_close_loss'
        if key and fault==target_fault and (key not in self.used_faults or key in op['injected']):
            op['injected'].add(key);self.used_faults.add(key)
            applied[15 if key=='right' else 7]=env.grippers[key].open_motor
            diagnostic={**diagnostic,'diagnostic_actuator_command_loss':key,'requested_action':requested.tolist()}
        _,_,terminal,truncated,info=env.step_tamp(applied)
        after=op['recorder'].observation(env)
        op['recorder'].append(op['before'],applied,after,info,terminated=terminal,truncated=truncated,diagnostics=diagnostic)
        op['before']=after
        return terminal or truncated

    def _finish(self,ok,code=None,message=None):
        op=self.active;planner=op['planner'];kind=op['kind']
        row={'tool':kind,'ok':ok,'failure_code':code,'message':message,
            'start_tick':op['started_tick'],'end_tick':self.env.tick,'phase':planner.phase,
            'native':str(op['recorder'].path.relative_to(self.output))}
        op['recorder'].finish('completed' if ok else 'error',self.env.info(),error=message,
            metrics={'tamp':{**planner.result,'primitive_success':ok,'failure_code':code}})
        try:op['iterator'].close()
        except Exception:pass
        self.active=None;self.rows.append(row)
        if not ok:
            self.last_failure={'code':code,'tool':kind,'tick':self.env.tick}
            if self.pose_bias_active:
                self.pose_bias_active=False;self.pose_revision+=1
            self.capture(f'failure-{len(self.rows):02d}')
        else:
            if kind in ('lift','handover','finish_transfer','place'):self.route.append(kind)
        dump(self.output/'stages.json',self.rows)
        return {'ok':ok,'done':True,'error_code':code,'kind':kind,'phase':planner.phase}

    def advance(self):
        self._guard()
        if not self.active:raise ValueError('no active operation')
        op=self.active;executed_phase=None
        try:
            while True:
                if op['pending'] is None:action,diagnostic=next(op['iterator'])
                else:action,diagnostic=op.pop('pending');op['pending']=None
                phase=diagnostic.get('phase',op['planner'].phase)
                if executed_phase is not None and phase!=executed_phase:
                    op['pending']=(action,diagnostic);op['phase']=phase
                    return {'ok':True,'done':False,'phase_completed':executed_phase,'next_phase':phase}
                executed_phase=phase;op['phase']=phase
                if self._step(action,diagnostic):
                    if op['kind']=='place' and self.env.evaluation.get('placement',{}).get('confirmed'):
                        return self._finish(True)
                    return self._finish(False,'terminal_physics','Native task ended; no reset/recovery')
        except StopIteration:
            return self._finish(op['planner'].result.get('primitive_success') is True,
                None if op['planner'].result.get('primitive_success') is True else 'missing_receipt')
        except PlanningFailure as exc:return self._finish(False,exc.code,str(exc))

    def _once(self,planner,kind):
        self._begin(planner,kind)
        for _ in range(24):
            row=self.advance()
            if row['done']:return row
        raise ValueError('phase budget exhausted')

    def pregrasp(self,grasp_id):
        self._guard();row=self._grasp(grasp_id);self.last_grasp=grasp_id
        result=self._once(SegmentPlanner(self.env,mode='pregrasp',grasp=row),'pregrasp')
        if result['ok']:
            probe=SegmentPlanner(self.env,mode='pregrasp');probe._sync();p,rot=probe._pose()
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,rot.ravel())
            self.checkpoint={'id':f'pregrasp-{self.env.tick}','position_m':p.tolist(),
                'quaternion_xyzw':np.r_[quat[1:],quat[0]].tolist(),'held_by':[]}
        return result

    def approach(self,grasp_id):
        self._guard();return self._once(SegmentPlanner(self.env,mode='approach',grasp=self._grasp(grasp_id)),'approach')

    def close_gripper(self,arm='right'):
        self._guard()
        if arm not in ('right','left'):raise ValueError('invalid arm')
        if arm=='left' and not self.env.evaluation['held']['right']:
            raise ValueError('receiver closure requires donor hold')
        return self._once(SegmentPlanner(self.env,mode='gripper',arm=arm,fraction=0.),'close_'+arm)

    def open_gripper(self,arm='right'):
        self._guard()
        if arm not in ('right','left'):raise ValueError('invalid arm')
        return self._once(SegmentPlanner(self.env,mode='gripper',arm=arm,fraction=1.),'open_'+arm)

    def verify_hold(self,arm='right'):
        self._guard()
        if arm not in ('right','left'):raise ValueError('invalid arm')
        return {'ok':bool(self.env.evaluation['held'][arm]),'arm':arm,
            'contact_seconds':float(self.env.contacts.hold_seconds[arm])}

    def lift(self):
        self._guard();return self._once(SegmentPlanner(self.env,mode='lift'),'lift')

    def retreat(self,checkpoint_id):
        self._guard()
        if not self.checkpoint or checkpoint_id!=self.checkpoint['id']:raise ValueError('unknown checkpoint')
        if any(self.env.evaluation['held'].values()):raise ValueError('empty checkpoint cannot roll back a held object')
        if self.env.evaluation['touching']['right']:raise ValueError('open empty gripper before retreat')
        self.recovery_actions+=1
        planner=TampPlanner(self.env,'move_to_pose',parameters={'arm':'right','frame':'world',
            'position_m':self.checkpoint['position_m'],'quaternion_xyzw':self.checkpoint['quaternion_xyzw']})
        return self._once(planner,'retreat')

    def start_transfer(self,placement_aware=True):
        self._guard()
        planner=self.planner_factory(self.env,'handover_bottle',parameters={'object_id':'bottle',
            'from_arm':'right','to_arm':'left','hold_seconds':2,'placement_aware':placement_aware})
        self._begin(planner,'handover');return {'ok':True,'done':False}

    def finish_transfer(self):
        self._guard();return self._once(SegmentPlanner(self.env,mode='finish_transfer'),'finish_transfer')

    def start_place(self):
        self._guard()
        planner=self.planner_factory(self.env,'place_bottle',parameters={'object_id':'bottle',
            'arm':'left','receptacle_id':'basket','shape_clearance':True})
        self._begin(planner,'place');return {'ok':True,'done':False}

    def registry(self):
        registry=ToolRegistry(shared_public_names=self.PUBLIC,public_extension_allowlist=())
        for name in self.PUBLIC:registry.register(getattr(self,name),name=name,public=True,
            layer='shared_high_level',capability='native recovery lab; oracle sensors; no reset')
        return registry

    def execute(self,program,label):
        (self.output/f'{label}.py').write_text(program)
        result=ProgramExecutor(self.registry(),recorder=PublicTrace(self.output/'trace.jsonl')).execute_program(program)
        return {'program_ok':result.ok,'error':str(result.error) if result.error else None,
                'state':self._observation()}

    def result(self):
        e=self.env.evaluation
        cursor=0
        expected=({'lift'},{'handover','finish_transfer'},{'place'})
        for step in self.route:
            if cursor<3 and step in expected[cursor]:cursor+=1
        chain=cursor==3
        score=bool(chain and e.get('placement',{}).get('confirmed') and e['inside_basket']
            and e['supported_by_basket'] and not any(e['held'].values()) and not e['invalid'])
        return {'task_success':score,'route':self.route,'native_failures':self.last_failure,
            'stages':self.rows,'fault_used':sorted(self.used_faults),'pose_error_cleared':not self.pose_bias_active,
            'sim_seconds':float(self.env.data.time),'calls':self.calls,
            'evaluation':e,'hardware_access':False,'transfer_ready':False,'reset_count':1,
            'post_start_pose_restores':0,'source_sha256':self.source_hash}

    def close(self):
        if self.active:self._finish(False,'interrupted','Lab closed while operation active')
        self.env.close()
