"""Reproducible source/skill package with an official pinned ASPIRE runtime.

No credentials, weights, API profiles, environment, raw runs or Git history.
The archive prepares simulation/review and real-adapter development; it does
not contain an operational hardware driver.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM_REV='fa00f0ef2bc1d91d594b9aa502c22133c9f538f7'
RUNTIME_FILES=('src/cap_harness/__init__.py','src/cap_harness/runtime.py',
    'src/cap_harness/registry.py','src/cap_harness/contracts.py','src/cap_harness/errors.py',
    'LICENSE','NOTICE','pyproject.toml')


def eligible(name):
    path=PurePosixPath(name)
    if path.parts[0] in ('.git','.venv','.runtime','runs','handoff','models'):return False
    if any(p in ('__pycache__','node_modules','state') for p in path.parts):return False
    if name in ('harness/policy.json','harness/llm.json','harness/api.json'):return False
    if path.name.startswith('.env') or path.suffix in ('.pt','.ckpt','.safetensors','.key','.pem','.db','.sqlite','.log','.pyc'):return False
    return True


def build(destination, upstream):
    from harness.aspire.runtime import SOURCE
    upstream=(upstream or SOURCE).resolve()
    origin_path=upstream/'ORIGIN.json'
    origin=json.loads(origin_path.read_text()) if origin_path.exists() else None
    revision=(origin['revision'] if origin else subprocess.check_output(
        ['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip())
    if revision!=UPSTREAM_REV:raise ValueError('ASPIRE revision differs from validated runtime')
    if (ROOT/'.git').exists():
        names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    else:
        package=json.loads((ROOT/'PACKAGE-MANIFEST.json').read_text())
        names=[entry['path'] for entry in package['entries']]
    files={name:(ROOT/name).read_bytes() for name in names if name and eligible(name) and (ROOT/name).is_file()}
    if 'skills/bottle_relay/program.py' not in files:raise ValueError('stage the admitted skill before packaging')
    # Read only the transitive runtime dependencies. Archiving all of
    # cap_harness can trigger lazy fetching of unrelated real-robot assets.
    runtime_hashes={}
    for name in RUNTIME_FILES:
        data=((upstream/name).read_bytes() if origin else subprocess.check_output(
            ['git','-C',str(upstream),'show',UPSTREAM_REV+':'+name]))
        digest=hashlib.sha256(data).hexdigest()
        if origin and origin.get('files',{}).get(name)!=digest:
            raise ValueError('bundled official runtime hash mismatch: '+name)
        files['third_party/aspire/'+name]=data
        runtime_hashes[name]=digest
    files['third_party/aspire/ORIGIN.json']=(json.dumps({'url':'https://github.com/NVlabs/ASPIRE',
        'revision':UPSTREAM_REV,'files':runtime_hashes,
        'scope':'official ProgramExecutor and transitive runtime dependencies, license and metadata; no robot backends/models'},indent=2)+'\n').encode()
    # Include concise measured evidence; native large recordings stay local.
    for name in ('protocol.json','assessment.json','audit.json','model-review.json'):
        path=ROOT/'runs/20261024-relay'/name
        if path.exists():files['evidence/'+name]=path.read_bytes()
    files['harness/policy.json']=files['harness/policy.example.json']
    files['README-FIRST.md']=(ROOT/'docs/真机部署.md').read_bytes()
    entries=[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        for name,data in sorted(files.items())]
    manifest={'schema':'openarmsim-colleague-package/2','files':len(entries),'entries':entries,
        'source_commit':(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            if (ROOT/'.git').exists() else package.get('source_commit')),
        'hardware_driver_included':False,'hardware_enabled':False,'transfer_ready':False}
    files['PACKAGE-MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    destination=destination.resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('refuse overwriting an existing package')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('OpenArmSim/'+name,date_time=(2026,10,24,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name=='run' else 0o100644)<<16
            out.writestr(info,data)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.sha256').write_text(digest+'  '+destination.name+'\n')
    print(json.dumps({'package':str(destination),'sha256':digest,'files':len(entries),
        'bytes':destination.stat().st_size,'hardware_access':False},ensure_ascii=False))
    return destination


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'handoff/OpenArmSkill-20261024.zip')
    p.add_argument('--upstream',type=Path)
    args=p.parse_args(argv)
    if not args.output.resolve().is_relative_to(ROOT/'handoff'):p.error('package belongs in handoff/')
    build(args.output,args.upstream)
    return 0


if __name__=='__main__':raise SystemExit(main())
