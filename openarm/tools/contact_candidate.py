"""Explicit cold-compiled contact candidate, separate from the main simulator.

MjSpec is changed before compilation/evidence capture. Recording guards and
original success predicates remain intact. This is not physical calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from sim import environment as environment_module
from sim.environment import ROOT,OpenArmEnv
from sim.recording import Recorder,dump

SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DERIVATION={'schema':'contact-patch-candidate/1','contact_dimension':4,'noslip_iterations':3,
    'applies_to':'bottle and both finger bodies, including their other contact pairs',
    'friction_coefficients':'unchanged original declared values; torsional coefficient becomes active',
    'source_sha256':SOURCE_HASH,'physically_calibrated':False,'production_default_changed':False,
    'scope':'finite-area torsional contact plus numerical slip suppression; research candidate only',
    'source':'https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip',
    'hardware_access':False,'transfer_ready':False}


class CandidateSpec:
    @staticmethod
    def from_string(*args,**kwargs):
        spec=mujoco.MjSpec.from_string(*args,**kwargs)
        spec.option.noslip_iterations=3
        for geom in spec.geoms:
            name=geom.parent.name
            if name=='task_bottle' or name.startswith('openarm_') and 'finger' in name:geom.condim=4
        return spec


class Compiler:
    MjSpec=CandidateSpec
    def __getattr__(self,name):return getattr(mujoco,name)


class ContactCandidate(OpenArmEnv):
    def __init__(self,*args,**kwargs):
        profile=dict(kwargs.pop('profile',{}) or {})
        profile.update(contact_model='finite_patch_candidate',contact_derivation=DERIVATION)
        # Only this process's construction is redirected through public MjSpec.
        # No model/audit reset occurs after the base environment captures it.
        with patch.object(environment_module,'mujoco',Compiler()):
            super().__init__(*args,profile=profile,**kwargs)
        assert self.model.opt.noslip_iterations==3
        for g in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[g]);name=self.model.body(body).name
            if body==self.bottle_id or name.startswith('openarm_') and 'finger' in name:
                if self.model.geom_condim[g]!=4:
                    raise ValueError('cold candidate contact inventory incomplete: '+str((self.model.geom(g).name,name,int(self.model.geom_condim[g]))))


def check():
    out=ROOT/'runs/20261008-grip/cold-composite';out.mkdir(exist_ok=False)
    original=OpenArmEnv(images=False);env=None;rec=None
    try:
        original.reset(seed=0);baseline=original.manifest()
        env=ContactCandidate(images=False);env.reset(seed=0);candidate=env.manifest()
        assert baseline['model_evidence']['inputs']==candidate['model_evidence']['inputs']
        assert baseline['model_evidence']['compiled']['sha256']!=candidate['model_evidence']['compiled']['sha256']
        assert not candidate['model_evidence']['runtime_parameters']['changed_fields']
        for field in ('body_mass','body_inertia','geom_friction','actuator_gainprm','actuator_biasprm','actuator_forcerange','jnt_range','mesh_vert'):
            np.testing.assert_array_equal(getattr(original.model,field),getattr(env.model,field),err_msg=field)
        dump(out/'manifest.json',{'original':baseline,'candidate':candidate,'derivation':DERIVATION})
        rec=Recorder(env,'diagnostic',root=out/'episodes',record_images=False,seed=0,details=DERIVATION)
        before=rec.observation(env);_,_,term,trunc,info=env.step_tamp(env.last_action.copy())
        rec.append(before,env.last_action.copy(),rec.observation(env),info,terminated=term,truncated=trunc)
        rec.finish('diagnostic_complete',env.info(),metrics={'candidate_cold_compiled':True})
        # The normal parameter guard must still reject a later mutation.
        env.model.opt.noslip_iterations=20
        try:Recorder(env,'diagnostic',root=out/'rejected',record_images=False,seed=0)
        except ValueError:guard_rejected=True
        else:raise ValueError('ordinary recorder accepted a changed candidate model')
        result={'passed':True,'source_inputs_identical':True,'compiled_models_distinct':True,
            'material_mass_inertia_gains_limits_meshes_identical':True,'later_mutation_rejected':guard_rejected,
            'candidate_recording_run':str(rec.path.relative_to(ROOT)),'derivation':DERIVATION,
            'candidate_parameters_frozen_before_recording':True,'hardware_access':False,'transfer_ready':False}
        dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if rec and not rec._closed:rec.finish('error',env.info(),error='cold candidate check incomplete')
        original.close()
        if env:env.close()


def api(case,output=None):
    import sim.service as service_module
    from tools import check_cache
    from harness.composite import install_isolated
    install_isolated()
    service_module.OpenArmEnv=ContactCandidate  # isolated process only
    check_cache.OUT=ROOT/'runs/20261008-grip'
    check_cache.isolated_api(output or 'api-'+case+'-composite.json',arm='right' if case=='right' else 'left',
        via_arm=None if case=='right' else 'right',distance_m=.08 if case=='right' else .06,
        experiment_details={'contact_candidate':DERIVATION,'same_normal_recording_guard':True,
                            'main_server_model_changed':False,'child_stage_binding':'verified_native_receipts',
                            'child_binding_sha256':hashlib.sha256((ROOT/'harness/composite.py').read_bytes()).hexdigest()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['check','api'],required=True)
    parser.add_argument('--case',choices=['right','transfer'],default='right')
    parser.add_argument('--output');args=parser.parse_args()
    if args.output and ('/' in args.output or not args.output.endswith('.json')):raise ValueError('simple JSON filename required')
    if args.mode=='check':check()
    else:api(args.case,args.output)
