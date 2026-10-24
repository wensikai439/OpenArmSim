"""Export a small real-only harness, not the simulation platform."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from .package_skill import ROOT,UPSTREAM_REV,RUNTIME_FILES


def main():
    existing=json.loads((ROOT/'skills/bottle_recovery/manifest.json').read_text())
    version=existing['version']
    destination=ROOT/f'handoff/OpenArmRobot-20261024-v{version}.zip'
    if destination.exists():raise ValueError('refuse overwriting an existing robot package')
    upstream=Path('/home/wsk/ASPIRE/libero-pro/source')
    files={
        'run':(ROOT/'deploy/run_robot').read_bytes(),
        'README.md':(ROOT/'deploy/ROBOT.md').read_bytes(),
        'harness/robot.py':(ROOT/'deploy/robot.py').read_bytes(),
        'skills/bottle_recovery/program.py':(ROOT/'skills/bottle_recovery/program.py').read_bytes(),
        'skills/bottle_recovery/SKILL.md':(ROOT/'deploy/ROBOTSKILL.md').read_bytes(),
        'docs/试运行.md':(ROOT/'deploy/试运行.md').read_bytes(),
    }
    program_hash=hashlib.sha256(files['skills/bottle_recovery/program.py']).hexdigest()
    if program_hash!=existing['program_sha256']:raise ValueError('admitted skill hash mismatch')
    config={'schema':'openarm-real-harness/1','robot_service':None,'program_sha256':program_hash,
            'skill_version':version,'robot_contract':'openarm-body-v4/1',
            'right_contact_fraction_max':.5,'real_validation_passed':False,'api_storage':'.private/api.json'}
    files['config.json']=(json.dumps(config,indent=2)+'\n').encode()
    files['skills/bottle_recovery/manifest.json']=(json.dumps({
        'version':version,'program_sha256':program_hash,'admitted_in':'candidate simulation only',
        'real_validation_passed':False,'real_tool_implementations_included':False,
        'robot_contract':'openarm-body-v4/1','right_contact_fraction_max':.5,
        'placement_contract':'object center and upright axis preserved; checked axial-yaw alternatives',
        'simulation_backend_included':False},indent=2)+'\n').encode()
    hashes={}
    for name in RUNTIME_FILES:
        data=subprocess.check_output(['git','-C',str(upstream),'show',UPSTREAM_REV+':'+name])
        files['third_party/aspire/'+name]=data;hashes[name]=hashlib.sha256(data).hexdigest()
    files['third_party/aspire/ORIGIN.json']=(json.dumps({'url':'https://github.com/NVlabs/ASPIRE',
        'revision':UPSTREAM_REV,'files':hashes,'scope':'executor and registry only; no simulator or robot backend'},indent=2)+'\n').encode()
    manifest={'schema':'openarm-real-only/1','source_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'simulator_included':False,'hardware_driver_included':False,'skills_real_validated':False,
        'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}}
    files['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as out:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name=='run' else 0o100644)<<16;out.writestr(info,data)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.sha256').write_text(digest+'  '+destination.name+'\n')
    print(json.dumps({'archive':str(destination),'sha256':digest,'bytes':destination.stat().st_size,'simulator_included':False}))


if __name__=='__main__':main()
