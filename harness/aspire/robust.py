"""Reject upper-half donor grasps before lift, using native contact evidence.

Local simulation only. Same 14 tools, program, budgets and physics as v3.
The stricter lower-body gate invokes the existing supported-release recovery.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .body import BodyWorld
from .runtime import ROOT,dump


def lower_half(contact):
    points=contact.get('contacts',[])
    return bool(points) and all(p.get('geom')=='bottle_body' and
        type(p.get('body_fraction')) in (int,float) and math.isfinite(p['body_fraction'])
        and 0.<=p['body_fraction']<=.5 for p in points)


class RobustWorld(BodyWorld):
    def __init__(self,case,output):
        super().__init__(case,output)
        self.region_checks=[]
        self.source_hash=hashlib.sha256(Path(__file__).read_bytes()+
            Path(__file__).with_name('body.py').read_bytes()+
            Path(__file__).with_name('precision.py').read_bytes()+
            Path(__file__).with_name('recovery.py').read_bytes()).hexdigest()

    def close_gripper(self,arm='right'):
        result=super().close_gripper(arm)
        if arm!='right' or not result['ok']:return result
        contact=self.grasp_contacts.get('right',{})
        accepted=lower_half(contact)
        self.region_checks.append({'tick':self.env.tick,'accepted':accepted,
            'max_fraction':max([p['body_fraction'] for p in contact.get('contacts',[])],default=None),
            'required_fraction_max':.5,'source':'actual force-bearing bottle-body contact, simulator geometry'})
        if accepted:return result
        self.rows[-1].update(ok=False,native_contact_ok=True,failure_code='grasp_region_mismatch',
            message='Every force-bearing donor contact must be in the lower half of the bottle body')
        self.last_failure={'code':'grasp_region_mismatch','tool':'close_right','tick':self.env.tick}
        self.capture('upper-half-rejected')
        if self.pose_bias_active:self.pose_bias_active=False;self.pose_revision+=1
        dump(self.output/'stages.json',self.rows)
        return {**result,'ok':False,'error_code':'grasp_region_mismatch'}

    def result(self):
        result=super().result()
        accepted=lower_half(self.grasp_contacts.get('right',{}))
        result['task_success']=bool(result['task_success'] and accepted)
        result['body_relay'].update(right_contact_fraction_max=.5,
            right_lower_half_verified=accepted,
            body_contact_verified=bool(result['body_relay']['body_contact_verified'] and accepted))
        result['region_checks']=getattr(self,'region_checks',[])
        return result

    def close(self):
        dump(self.output/'grasp-region-checks.json',self.region_checks)
        super().close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('local output belongs in runs/')
    from .recovery_campaign import run_trial
    run_trial(json.loads(a.case.read_text()),out,a.program,'frozen',world_type=RobustWorld)
