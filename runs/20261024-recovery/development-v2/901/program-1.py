held = list(observe()['held_by'])
if held != []:
    raise ValueError('unexpected hold at start: ' + str(held))

s = observe()
if s['last_failure'] is not None and s['last_failure']['code'] != 'grasp_not_confirmed':
    raise ValueError('unexpected failure: ' + str(s['last_failure']))

cands = propose_grasps(exclude_ids=('g0-000',))
if not cands['candidates']:
    raise ValueError('no grasp candidates')
gid = cands['candidates'][0]['id']

r = pregrasp(gid)
if not r['ok'] or not r['done']:
    raise ValueError('pregrasp failed: ' + str(r['error_code']))

r = approach(gid)
if not r['ok'] or not r['done']:
    raise ValueError('approach failed: ' + str(r['error_code']))

r = close_gripper(arm='right')
if not r['ok'] or not r['done']:
    raise ValueError('close_gripper failed: ' + str(r['error_code']))

v = verify_hold(arm='right')
if not v['ok']:
    raise ValueError('verify_hold failed')

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
    raise ValueError('start_place failed')

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