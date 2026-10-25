"""Install only a real-only archive, within the single authorized SSH root."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
INSTALL=r'''
from pathlib import Path,PurePosixPath
import hashlib,io,json,os,sys,zipfile
root=Path('/home/sj/Desktop/wuji/lerobot_20260116/LLMHarness').resolve()
archive=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read()))
names=archive.namelist()
for name in names:
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or not (root/name).resolve().is_relative_to(root):raise ValueError('archive path escape')
 if any(x in p.parts for x in ['.private','.venv','.runtime','sim','models','web','assets']):raise ValueError('forbidden archive content')
manifest=json.loads(archive.read('records/manifest.json'))
if manifest.get('simulator_included') is not False:raise ValueError('not a real-only archive')
payload={**manifest['files'],**manifest.get('configuration_templates',{})}
if len(names)!=len(set(names)) or set(names)!=set(payload)|{'records/manifest.json'}:raise ValueError('unexpected/duplicate archive files')
for name,digest in payload.items():
 if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('archive hash mismatch')
baseline_path=root/'records/protected-before.json'
baseline=json.loads(baseline_path.read_text())
private=root/'.private/api.json';private_before=hashlib.sha256(private.read_bytes()).hexdigest()
def snapshot():
 result={}
 for name,old in baseline.items():
  p=Path(name);st=p.stat();row={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'mode':st.st_mode}
  if 'sha256' in old:row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  result[name]=row
 return result
before=snapshot()
old_manifest_path=root/'records/manifest.json'
if not old_manifest_path.exists():raise ValueError('expected compact release; inspect older layout first')
old=json.loads(old_manifest_path.read_text());old_names=list(old['files'])+list(old.get('mutable_configuration',[]))+['records/manifest.json']
for name in old_names:
 if not (root/name).resolve().is_relative_to(root):raise ValueError('old manifest path escape')
 if any(part in Path(name).parts for part in ['.private','.venv','.runtime']):raise ValueError('private/runtime path in old manifest')
for name,digest in old['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('existing release modified: '+name)
import time
backup=root/('records/versions/r'+str(old.get('runtime_release',5))+'-'+time.strftime('%Y%m%d-%H%M%S')+'.zip')
if backup.exists():raise ValueError('version backup already exists; do not overwrite')
backup.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(set(old_names)):
  p=root/name
  if p.is_file():z.write(p,name)
# Every overwritten file was either a known old release file or newly created
# by this release. Private settings, environment and unrelated files stay put.
for name in names:
 p=root/name
 if p.exists() and name not in old_names:raise ValueError('refuse overwriting unknown path: '+name)
written=[]
def merge(current,template):
 if isinstance(current,dict) and isinstance(template,dict):return {k:merge(current[k],v) if k in current else v for k,v in template.items()}|{k:v for k,v in current.items() if k not in template}
 return current
try:
 for name in names:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);data=archive.read(name)
  if name in manifest.get('mutable_configuration',[]) and p.is_file():
   template=json.loads(data);merged=merge(json.loads(p.read_text()),template)
   merged['schema']=template['schema']
   data=(json.dumps(merged,ensure_ascii=False,indent=2)+'\n').encode()
  p.write_bytes(data);written.append(name)
  if name=='run':p.chmod(0o755)
 for name,digest in manifest['files'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('installed hash mismatch')
except Exception:
 with zipfile.ZipFile(backup) as saved:
  for name in written:
   p=root/name
   if name in saved.namelist():p.write_bytes(saved.read(name))
   elif p.is_file():p.unlink()
 raise
after=snapshot();changed=[p for p in before if before[p]!=after[p]]
if not private.is_file() or private.stat().st_mode&0o077:raise ValueError('private API missing or not private')
same_api=private_before==hashlib.sha256(private.read_bytes()).hexdigest()
if not same_api:raise ValueError('private API changed')
receipt={'schema':'openarm-install/3','runtime_release':manifest.get('runtime_release'),'source_commit':manifest['source_commit'],
 'installed_files':len(manifest['files']),'simulator_included':False,'hardware_access':False,
 'devices_started':False,'motion_commands':0,'protected_files_checked':len(before),
 'protected_files_changed':changed,'same_api_preserved':same_api,
 'root_entries':sorted(p.name for p in root.iterdir()),
 'real_tool_implementation_included':True,'real_perception_connected':False,'real_motion_provider_admitted':False,
 'previous_release':str(backup.relative_to(root))}
(root/'records/deployment.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
if changed:raise ValueError('protected file changed: '+str(changed))
print(json.dumps(receipt,ensure_ascii=False))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);a=p.parse_args()
    data=a.archive.read_bytes()
    command='PYTHONDONTWRITEBYTECODE=1 python3 -B -c '+shlex.quote(INSTALL)
    r=subprocess.run(['ssh','-o','BatchMode=yes','data-cs',command],input=data,capture_output=True,check=True)
    receipt=json.loads(r.stdout)
    folder=ROOT/'runs/20261025-real';folder.mkdir(exist_ok=True)
    (folder/'deployment.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
