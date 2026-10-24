import math

s = observe()
if list(s['held_by']) != []:
    raise ValueError('unexpected hold state; stop for review')
if s['touching']['right'] or s['touching']['left']:
    raise ValueError('unexpected contact; stop for review')

gid = s['current_grasp_id']
attempts = 0
closed = False
while attempts < 2 and not closed:
    if attempts > 0:
        cands = propose_grasps(exclude_ids=(gid,))
        if not cands['candidates']:
            raise ValueError('no alternative grasp candidates')
        gid = cands['candidates'][0]['id']
        r = pregrasp(gid)
        if not r['ok'] or not r['done']:
            raise ValueError('pregrasp failed: ' + str(r['error_code']))
    r = approach(gid)
    if not r['ok'] or not r['done']:
        raise ValueError('approach failed: ' + str(r['error_code']))
    c = close_gripper(arm='right')
    if c['ok'] and c['done']:
        v = verify_hold(arm='right')
        if v['ok']:
            closed = True
            break
    else:
        if c.get('error_code') != 'grasp_not_confirmed':
            raise ValueError('close failed: ' + str(c.get('error_code')))
        r = open_gripper(arm='right')
        if not r['ok']:
            raise ValueError('open failed: ' + str(r['error_code']))
    attempts += 1

if not closed:
    raise ValueError('no confirmed right hold after recovery attempts')

r = lift()
if not r['ok'] or not r['done']:
    raise ValueError('lift failed: ' + str(r['error_code']))

r = start_transfer(placement_aware=True)
if not r['ok']:
    raise ValueError('start_transfer failed: ' + str(r['error_code']))

steps = 0
while True:
    a = advance()
    steps += 1
    if not a['ok']:
        raise ValueError('transfer advance failed: ' + str(a['error_code']))
    if a['done']:
        break
    if steps >= 16:
        raise ValueError('transfer did not complete within 16 advances')

s = observe()
if list(s['held_by']) != ['left']:
    raise ValueError('left hold not confirmed after transfer')

r = start_place()
if not r['ok']:
    raise ValueError('start_place failed: ' + str(r['error_code']))

steps = 0
while True:
    a = advance()
    steps += 1
    if not a['ok']:
        raise ValueError('place advance failed: ' + str(a['error_code']))
    if a['done']:
        break
    if steps >= 16:
        raise ValueError('placement did not complete within 16 advances')

final = observe()
if list(final['held_by']) != []:
    raise ValueError('hands not empty at end')
result = {'status': 'complete', 'final_observation': final}