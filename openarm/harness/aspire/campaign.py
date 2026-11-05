"""ASPIRE actor -> fix loop -> frozen evaluation -> independent skill verifier.

OpenArm is a custom geometry-oracle adapter, not an official paper benchmark.
The actor and verifier are separate API conversations with the user-selected model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from .client import ModelClient, image_part

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path('/home/wsk/ASPIRE/libero-pro/source')

API = '''Allowed injected Python functions (no imports except math, no files/network/private attributes):
get_observation() -> read-only mapping {source:'simulation_geometry_oracle', held_by:list of arm names,
 bottle_center_m:xyz metres, upright:bool, tick:int, sim_seconds:float}.
pick(arm='right', hold_seconds=2, next_handover=None) -> ExecutionResult.
 next_handover, if present, must be exactly {'to_arm':'left'} for this task. This declares the downstream receiver to the geometric candidate search; it is not a success assertion.
handover(from_arm='right',to_arm='left',hold_seconds=2) -> ExecutionResult.
place(arm='left',receptacle_id='basket') -> ExecutionResult.
move_relative(arm,x_m,y_m,z_m) -> ExecutionResult; world axes +X forward,+Y robot-left,+Z up, norm<=.15m.
All motion results have .ok (actual primitive execution), .steps_executed, .error (None or .message),
 .diagnostics.get('failure_code') and .diagnostics.get('current_holder'). Check .ok and re-observe.
There is no reset, home, object pose assignment, force/physics editing, task-success query or full-task shortcut.
The backend enforces contacts, lift, collision, receiver-only hold, release and stable basket support.
You may branch on current observations, never on seed or a stored global target coordinate.
At most12motion calls and120simulated seconds. hold_seconds should be between2and10.
Use top-level code/helpers, and set result to a compact report. Use ValueError for unsuccessful completion.
Returning normally is NOT task success. Do not change the required right->left direction.
'''

ACTOR = '''You are the ASPIRE coding actor for a new OpenArm task: use the RIGHT arm to pick the upright
black bottle from its CURRENT table position, hand it to the LEFT arm, and use the LEFT arm to put it
inside the basket, releasing it and withdrawing. The bottle location changes between independent resets.
Generate a PARAMETERIZED robot program using the allowed tools, not a trajectory or stored coordinates.
Tools use declared simulator geometry: do not describe this as vision-only or real-robot transfer.
You will receive native stage feedback from development runs and may repair the program.
Preserve failures; distinguish API/config errors from physical failure. Never invent new tool arguments.
Save a skill only from demonstrated behavior. Do not claim improvement that paired evaluation did not show.
''' + API

PROGRAM_SCHEMA = {'type': 'object', 'properties': {
    'code': {'type': 'string'}, 'hypothesis': {'type': 'string'},
    'changes': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['code', 'hypothesis', 'changes']}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures(folder):
    result = {'development': [], 'unseen': [], 'heldout': []}
    for partition, seeds in [('development', range(51,56)), ('unseen', range(66,71)), ('heldout', range(1,21))]:
        for seed in seeds:
            rng = random.Random(seed)
            xy = [.503+rng.uniform(-.035,.035), -.173+rng.uniform(-.04,.04)]
            if seed == 51:
                xy = [.503, -.173]  # one declared nominal development scene
            row = {'seed': seed, 'bottle_xy_m': xy}
            path = folder/'cases'/partition/f'{seed:03d}.json'
            write(path, row)
            result[partition].append(str(path))
    return result


def trial(program, case, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl', OPENARM_SCENE_OBJECTS='[]')
    log = output.with_suffix('.log')
    start = time.monotonic()
    with log.open('w') as stream:
        try:
            completed = subprocess.run([sys.executable, '-B', '-m', 'harness.aspire.runtime',
                '--program', str(program), '--case', str(case), '--output', str(output)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                timeout=200, check=False)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
    path = output/'result.json'
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = {'task_success': False, 'program_ok': False, 'protocol_success': False,
                  'error': 'trial timeout' if returncode == -1 else 'adapter process failed',
                  'failure_stage': 'runtime', 'returncode': returncode}
        write(output/'host-error.json', result)
    result.update(case=json.loads(Path(case).read_text()), wall_seconds=time.monotonic()-start,
                  trial=str(output), program_sha256=sha(program))
    return result


def feedback(row):
    return {k:row.get(k) for k in ('case','task_success','program_ok','protocol_success',
                                  'failure_stage','failure_code','error','stages','calls')}


def batch(program, cases, folder):
    rows = []
    for case in cases:
        seed = json.loads(Path(case).read_text())['seed']
        row = trial(program, case, folder/f'{seed:03d}')
        rows.append(row)
        write(folder/'results.json', rows)
        print(json.dumps({'batch':str(folder.relative_to(ROOT)), 'seed':seed,
              'passed':row.get('task_success'), 'failure':row.get('failure_code',row.get('error'))}), flush=True)
    return rows


def count(rows):
    return sum(r.get('task_success') is True for r in rows)


def scene_parts(folder):
    parts=[]
    for camera in ('head','left','right'):
        matches=list(folder.rglob(f'*initial*{camera}*.jpg'))
        if not matches:
            matches=list(folder.rglob(f'*{camera}*.jpg'))[:1]
        if matches:
            parts.append(image_part(matches[0]))
    return parts


def failure_parts(rows):
    """Selective post-primitive sensor evidence, not entire videos or native truth."""
    failed=next((r for r in rows if not r.get('task_success')),None)
    if failed is None:return []
    folder=Path(failed['trial'])/'media'
    parts=[]
    for camera in ('head','right'):
        paths=sorted(folder.glob(f'*-after-*-{camera}.jpg'))
        if paths:parts.append(image_part(paths[-1]))
    return parts


def propose(client, messages, path):
    submission = client.ask(messages, tool_name='submit_program',
        description='Submit the complete robot program and a short testable repair hypothesis',
        schema=PROGRAM_SCHEMA)
    code=submission['code'].strip()
    if code.startswith('```'):
        code='\n'.join(code.splitlines()[1:-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code+'\n')
    write(path.with_suffix('.proposal.json'), submission)
    return submission


def run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        raise RuntimeError('Campaign already exists; do not overwrite immutable experiments')
    cases=fixtures(folder)
    protocol={'schema':'openarm-aspire/1', 'upstream_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),
        'upstream_modules':{name:sha(UPSTREAM/'src/cap_harness'/name) for name in ('runtime.py','registry.py')},
        'task':'right bottle pick -> right-to-left handover -> left basket placement',
        'adapter':'custom OpenArm geometry-oracle tools with actual NVIDIA ProgramExecutor/ToolRegistry',
        'partitions':cases,'bottle_workspace_m':{'x':[.468,.538],'y':[-.213,-.133]},
        'fixture':'synthetic upright reset only; no pose edits after first control command',
        'budget':{'actor_versions':6,'verifier_versions':2,'api_requests':24,'api_tokens':60000,
                  'per_trial_sim_seconds':120,'per_trial_wall_seconds':200},
        'original_paper_protocol':False,'hardware_access':False,'transfer_ready':False,
        'control_mode':'paused simulation; model coding latency is outside the executed robot program',
        'baseline':'same existing TAMP primitives, straight-line call sequence; not supplied to the actor'}
    write(folder/'protocol.json',protocol)
    client=ModelClient(folder/'api')
    probe=folder/'probe.py';probe.write_text('result = get_observation()\n')
    trial(probe,Path(cases['development'][0]),folder/'probe')
    observations=scene_parts(folder/'probe')
    messages=[{'role':'system','content':ACTOR},{'role':'user','content':[
        {'type':'text','text':'Inspect this one initial scene and submit your initial program. Use no fixed world-coordinate grasp or release target.'},*observations]}]
    attempts=[];best=None
    for version in range(1,7):
        program=folder/'programs'/f'v{version:02d}.py'
        submitted=propose(client,messages,program)
        rows=batch(program,cases['development'],folder/'development'/f'v{version:02d}')
        attempts.append({'version':version,'program':str(program),'passes':count(rows),
                         'rows':rows,'hypothesis':submitted['hypothesis']})
        write(folder/'attempts.json',attempts)
        if best is None or count(rows)>best['passes']:
            best=attempts[-1]
        if count(rows)==len(rows):
            break
        messages += [{'role':'assistant','content':'Submitted program:\n'+program.read_text()},
            {'role':'user','content':[{'type':'text','text':json.dumps({'development_feedback':[feedback(r) for r in rows],
             'instruction':'Diagnose by failure mode and submit a complete repair. No new arguments or reset. Keep the required direction and final basket goal. Attached images show sensor evidence after a failing primitive.'},ensure_ascii=False)},*failure_parts(rows)]}]
    selected=folder/'skill'/'program.py';selected.parent.mkdir(exist_ok=True)
    shutil.copyfile(best['program'],selected)
    frozen={'program_sha256':sha(selected),'selected_version':best['version'],
            'development_passes':best['passes'],'frozen_before_unseen_and_heldout':True,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'harness/aspire').glob('*.py'))}}
    write(folder/'frozen.json',frozen)
    # All these results are measurement, never fed back into the actor.
    unseen=batch(selected,cases['unseen'],folder/'unseen')
    heldout=batch(selected,cases['heldout'],folder/'heldout')
    baseline=folder/'baseline.py'
    baseline.write_text('pick(arm="right")\nhandover(from_arm="right", to_arm="left")\nplace(arm="left", receptacle_id="basket")\n')
    baseline_rows=batch(baseline,cases['heldout'],folder/'baseline')
    # Distill only development evidence. No evaluation failures, seeds or score leak into this request.
    proposal=client.ask([{'role':'system','content':ACTOR}, {'role':'user','content':
        'Distill a compact reusable skill from the program and DEVELOPMENT evidence below. '
        'Include applicability, failure signatures, contact/holder checks, recovery, limitations and a short code sketch. '
        'Do not claim generality or improved success without evidence. Describe existing planner capabilities as inherited.\n'+
        json.dumps({'program':selected.read_text(),'development':[feedback(r) for r in best['rows']],
                    'earlier_attempts':[{'version':a['version'],'hypothesis':a['hypothesis'],'passes':a['passes']} for a in attempts]},ensure_ascii=False)}],
        tool_name='propose_skill',description='Submit reusable knowledge distilled from verified development behavior',
        schema={'type':'object','properties':{'markdown':{'type':'string'},'claims':{'type':'array','items':{'type':'string'}}},
                'required':['markdown','claims']},max_tokens=3200)
    (folder/'skill'/'proposal.md').write_text(proposal['markdown']+'\n')
    write(folder/'skill'/'proposal.json',proposal)
    # Separate clean model context: no winning program, chat history or development solution files.
    verifier_messages=[{'role':'system','content':ACTOR+'\nYou are an INDEPENDENT skill verifier. '
       'Reconstruct a program from the proposed skill alone. Do not invent improvements to hide defects.'},
       {'role':'user','content':[{'type':'text','text':'Proposed skill:\n'+proposal['markdown']},*observations]}]
    verify=[]
    for version in range(1,3):
        program=folder/'verification'/f'v{version:02d}.py'
        propose(client,verifier_messages,program)
        rows=batch(program,cases['development'],folder/'verification'/f'v{version:02d}')
        verify.append({'attempt':version,'passes':count(rows),'rows':rows})
        if count(rows)>=4:
            break
        verifier_messages += [{'role':'assistant','content':program.read_text()},
            {'role':'user','content':[{'type':'text','text':'Only your own verification feedback follows. Submit the final repair; then stop.\n'+
             json.dumps([feedback(r) for r in rows],ensure_ascii=False)},*failure_parts(rows)]}]
    verification={'verdict':'verified' if max(v['passes'] for v in verify)>=4 else 'refuted',
                  'scope':'skill transmission on source task, not independent generalization evidence',
                  'proposer_code_provided':False,'attempts':verify}
    write(folder/'verification.json',verification)
    if verification['verdict']=='verified':
        shutil.copyfile(folder/'skill'/'proposal.md',folder/'skill'/'SKILL.md')
    result={'complete':True,'development':{'passes':best['passes'],'total':5},
            'unseen':{'passes':count(unseen),'total':len(unseen)},
            'heldout':{'passes':count(heldout),'total':len(heldout)},
            'baseline':{'passes':count(baseline_rows),'total':len(baseline_rows)},
            'verification':verification['verdict'],'api':client.snapshot(),'frozen':frozen,
            'oracle_geometry':True,'vision_only':False,'hardware_access':False,'transfer_ready':False}
    write(folder/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/20261023-aspire')
    args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'runs'):
        parser.error('output must stay inside runs/')
    run(output)
