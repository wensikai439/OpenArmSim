"""Export a small real-only harness, not the simulation platform."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import argparse

from .package_skill import ROOT,UPSTREAM_REV,RUNTIME_FILES


def main():
    existing=json.loads((ROOT/'skills/bottle_recovery/manifest.json').read_text())
    version=existing['version']
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'handoff/OpenArmRobot-20261025-r6.zip')
    p.add_argument('--collision',type=Path,default=ROOT/'runs/20261025-real/collision.json')
    a=p.parse_args();destination=a.output.resolve()
    if destination.exists():raise ValueError('refuse overwriting an existing robot package')
    upstream=Path('/home/wsk/ASPIRE/libero-pro/source')
    files={
        'run':(ROOT/'deploy/run_robot').read_bytes(),
        'README.md':(ROOT/'deploy/ROBOT.md').read_bytes(),
        'AGENTS.md':(ROOT/'deploy/robot-AGENTS.md').read_bytes(),
        'app/robot.py':(ROOT/'deploy/robot.py').read_bytes(),
        'app/skill/program.py':(ROOT/'skills/bottle_recovery/program.py').read_bytes(),
        'app/skill/SKILL.md':(ROOT/'deploy/ROBOTSKILL.md').read_bytes(),
        'app/试运行.md':(ROOT/'deploy/试运行.md').read_bytes(),
        'config/hardware.json':(ROOT/'deploy/hardware.json').read_bytes(),
        'config/calibration.json':(ROOT/'deploy/calibration.json').read_bytes(),
    }
    for name in ['service','evidence','devices','motion','planning','calibration','runtime','prepare']:
        files['app/'+name+'.py']=(ROOT/('deploy/'+name+'.py')).read_bytes()
    for source,target in [('robot_checks','robot_checks'),('service_checks','service_checks'),('runtime_checks','checks')]:
        files['app/'+target+'.py']=(ROOT/('deploy/'+source+'.py')).read_bytes()
    if a.collision.is_file():files['config/collision.json']=a.collision.read_bytes()
    program_hash=hashlib.sha256(files['app/skill/program.py']).hexdigest()
    if program_hash!=existing['program_sha256']:raise ValueError('admitted skill hash mismatch')
    config={'schema':'openarm-real-harness/2','robot_service':'http://127.0.0.1:8772','program_sha256':program_hash,
            'skill_version':version,'robot_contract':'openarm-body-v4/1',
            'right_contact_fraction_max':.5,'real_validation_passed':False,'api_storage':'.private/api.json'}
    files['config/robot.json']=(json.dumps(config,indent=2)+'\n').encode()
    files['app/skill/manifest.json']=(json.dumps({
        'version':version,'program_sha256':program_hash,'admitted_in':'candidate simulation only',
        'real_validation_passed':False,'real_tool_implementations_included':True,
        'runtime_release':6,'physical_acceptance_required':True,
        'robot_contract':'openarm-body-v4/1','right_contact_fraction_max':.5,
        'placement_contract':'object center and upright axis preserved; checked axial-yaw alternatives',
        'simulation_backend_included':False},indent=2)+'\n').encode()
    hashes={}
    for name in RUNTIME_FILES:
        data=subprocess.check_output(['git','-C',str(upstream),'show',UPSTREAM_REV+':'+name])
        files['app/aspire/'+name]=data;hashes[name]=hashlib.sha256(data).hexdigest()
    files['app/aspire/ORIGIN.json']=(json.dumps({'url':'https://github.com/NVlabs/ASPIRE',
        'revision':UPSTREAM_REV,'files':hashes,'scope':'executor and registry only; no simulator or robot backend'},indent=2)+'\n').encode()
    mutable=['config/calibration.json','config/collision.json']
    manifest={'schema':'openarm-real-only/3','runtime_release':6,'source_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'simulator_included':False,'hardware_driver_included':False,'hardware_adapter_included':True,'skills_real_validated':False,
        'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items()) if name not in mutable},
        'mutable_configuration':mutable,
        'configuration_templates':{name:hashlib.sha256(files[name]).hexdigest() for name in mutable if name in files}}
    files['records/manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as out:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name=='run' else 0o100644)<<16;out.writestr(info,data)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.sha256').write_text(digest+'  '+destination.name+'\n')
    print(json.dumps({'archive':str(destination),'sha256':digest,'bytes':destination.stat().st_size,'simulator_included':False}))


if __name__=='__main__':main()
