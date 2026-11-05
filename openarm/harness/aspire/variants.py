"""Versioned initial-condition stress fixtures; never change a running task.

The parameter ranges are engineering stress ranges, not fitted real-world
distributions. The original v1 position-only fixtures remain unchanged.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .runtime import ROOT, OpenArmTrial, fixture_environment, run, validate_case


def validate_variant(case):
    expected = {'schema','seed','bottle_xy_m','bottle_yaw_rad','basket_offset_m',
                'bottle_mass_scale','bottle_friction_scale'}
    if set(case) != expected or case['schema'] != 'relay-fixture/2':
        raise ValueError('unsupported variant fixture')
    validate_case({k:case[k] for k in ('seed','bottle_xy_m')})
    for name, low, high in (('bottle_yaw_rad',-np.pi,np.pi),
                            ('bottle_mass_scale',.8,1.2),('bottle_friction_scale',.8,1.2)):
        value=case[name]
        if type(value) not in (float,int) or not np.isfinite(value) or not low<=value<=high:
            raise ValueError('out-of-range variant: '+name)
    offset=case['basket_offset_m']
    if (not isinstance(offset,list) or len(offset)!=2
            or any(type(x) not in (float,int) for x in offset)
            or not np.isfinite(offset).all() or max(abs(x) for x in offset)>.025):
        raise ValueError('basket shift exceeds25mm')
    return case


def variant_environment(base, case):
    validate_variant(case)
    original=fixture_environment(base,{k:case[k] for k in ('seed','bottle_xy_m')})
    class VariantEnv(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Select a model variant ONCE during construction. The ordinary
            # immutable-parameter audit remains enabled throughout execution.
            self.model.body_pos[self.box_id,:2] += case['basket_offset_m']
            self.model.body_mass[self.bottle_id] *= case['bottle_mass_scale']
            self.model.body_inertia[self.bottle_id] *= case['bottle_mass_scale']
            self.model.geom_friction[self.bottle_geoms] *= case['bottle_friction_scale']
            qpos=self.data.qpos.copy();qvel=self.data.qvel.copy()
            mujoco.mj_setConst(self.model,self.data)
            self.data.qpos[:]=qpos;self.data.qvel[:]=qvel
            mujoco.mj_forward(self.model,self.data)
            from sim.model_evidence import ParameterAudit, compiled_evidence
            self._parameter_audit=ParameterAudit(self.model)
            self._model_evidence['compiled']=compiled_evidence(self.model)

        def reset(self, **kwargs):
            result=super().reset(**kwargs)
            if self.tick or self.data.time:
                raise RuntimeError('variant is allowed only before first physics step')
            address=self.registry['bottle']['qposadr']
            half=case['bottle_yaw_rad']/2
            self.data.qpos[address+3:address+7]=[np.cos(half),0,0,np.sin(half)]
            mujoco.mj_forward(self.model,self.data)
            self.evaluation=self._evaluate(0.)
            self._protect()
            return result
    return VariantEnv


class VariantTrial(OpenArmTrial):
    validate_case=staticmethod(validate_variant)
    fixture_environment=staticmethod(variant_environment)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--program',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('output must stay inside runs/')
    raise SystemExit(run(args.program,validate_variant(json.loads(args.case.read_text())),out,host_type=VariantTrial))
