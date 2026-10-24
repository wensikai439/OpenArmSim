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
for name,digest in manifest['files'].items():
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
old_manifest_path=root/'MANIFEST.json'
if not old_manifest_path.exists():raise ValueError('expected v4 layout; refuse repeated migration')
old=json.loads(old_manifest_path.read_text());old_names=list(old['files'])+['MANIFEST.json','DEPLOYMENT.json','AGENTS.md']
for name in old_names:
 if not (root/name).resolve().is_relative_to(root):raise ValueError('old manifest path escape')
backup=root/'records/versions/v4.zip'
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
for name in names:
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(name))
 if name=='run':p.chmod(0o755)
# Remove only prior manifest-owned files now replaced/moved, after the backup.
for name in old_names:
 if name not in names:
  p=root/name
  if p.is_file():p.unlink()
parents={ (root/name).parent for name in old_names }
for p in sorted(parents,key=lambda p:len(p.parts),reverse=True):
 while p!=root and p.is_relative_to(root):
  try:p.rmdir()
  except OSError:break
  p=p.parent
for name,digest in manifest['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('installed hash mismatch')
after=snapshot();changed=[p for p in before if before[p]!=after[p]]
if not private.is_file() or private.stat().st_mode&0o077:raise ValueError('private API missing or not private')
same_api=private_before==hashlib.sha256(private.read_bytes()).hexdigest()
if not same_api:raise ValueError('private API changed')
receipt={'schema':'openarm-install/2','source_commit':manifest['source_commit'],
 'installed_files':len(manifest['files']),'simulator_included':False,'hardware_access':False,
 'devices_started':False,'motion_commands':0,'protected_files_checked':len(before),
 'protected_files_changed':changed,'same_api_preserved':same_api,
 'root_entries':sorted(p.name for p in root.iterdir()),
 'real_perception_connected':False,'real_motion_provider_admitted':False}
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
    (ROOT/'runs/20261024-ready/deployment.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
