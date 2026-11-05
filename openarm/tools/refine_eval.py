"""Second round: disclose reused failures as development; hold out new points."""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.robust_eval import save


def main():
    out=ROOT/'runs/20261024-robust/refine'
    if out.exists():raise ValueError('round already exists')
    out.mkdir();origin=ROOT/'runs/20261024-robust'
    (out/'baseline.py').write_bytes((origin/'baseline.py').read_bytes())
    cases=[]
    for partition,seed in [('heldout',2207),('heldout',2215),('development',2003),('development',2005)]:
        source=origin/'cases'/partition/(str(seed)+'.json')
        p=out/'cases/development'/source.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes())
        cases.append({'seed':seed,'partition':'development','group':'combined','region':'reused-counterexample',
            'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    rng=random.Random(2026102412)
    for i in range(8):
        xy=[round(rng.uniform(.435,.585),6),round(rng.uniform(-.285,-.115),6)]
        for stress in (False,True):
            seed=2400+2*i+int(stress)
            case={'seed':seed,'bottle_xy_m':xy,'fault':'none','motion_profile':'precision','grasp_region':'body'}
            if stress:case.update(bottle_yaw_rad=round(rng.uniform(-3.14,3.14),6),
                basket_offset_m=[round(rng.uniform(-.023,.023),6) for _ in range(2)],
                bottle_mass_scale=round(rng.uniform(.8,1.2),6),bottle_friction_scale=round(rng.uniform(.8,1.2),6),
                fault=('none','right_close_loss','stale_grasp','receiver_close_loss')[i%4],
                pose_bias_m=round(rng.uniform(.04,.07),6))
            p=out/'cases/heldout'/(str(seed)+'.json');save(p,case)
            cases.append({'seed':seed,'partition':'heldout','group':'combined' if stress else 'nominal','region':'fresh',
                'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save(out/'protocol.json',{'schema':'bottle-robustness/2','random_seed':2026102412,'cases':cases,
        'development':'two round1 heldout failures explicitly reused as development, plus two older regressions',
        'heldout':'8 independent new positions crossed with nominal/combined; 16 conditions per method',
        'original_round1_report_unchanged':'30/32 for both methods',
        'baseline_module':'harness.aspire.robust','candidate_module':'harness.aspire.placement',
        'acceptance':'same native gates and each right contact in lower half, <=5mm stationary left drop',
        'timing':'paused simulation; no hardware or visual localization validation',
        'no_neural_training':True,'hardware_access':False,'transfer_ready':False})
    save(out/'acceptance.json',json.loads((origin/'acceptance.json').read_text()))
    print('Second round frozen; previous heldout failures are now development only')


if __name__=='__main__':main()
