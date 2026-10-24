import math

s = observe()
if list(s['held_by']) != []:
    raise ValueError('unexpected hold state; stop for review')
if s['touching']['right'] or s['touching']['left']:
    raise ValueError('unexpected contact; stop for review')

lf = s['last_failure']
if lf is None or lf['code'] != 'grasp_not_confirmed':
    raise ValueError('unexpected failure history; stop for review')

cands = propose_grasps(exclude_ids=())
if not cands['candidates']:
    raise ValueError('no grasp candidates')
gid = cands['candidates'][0]['id']

r = pregrasp(gid)
if not r['ok'] or not r['done']:
    raise ValueError('pregrasp failed: ' + str(r['error_code']))

r = approach(gid)
if not r['ok'] or not r['done']:
    raise ValueError('approach failed: ' + str(r['error_code']))

c = close_gripper(arm='right')
if not (c['ok'] and c['done']):
    raise ValueError('close failed: ' + str(c.get('error_code')))

v = verify_hold(arm='right')
if not v['ok']:
    raise ValueError('no verified right hold')

s = observe()
if list(s['held_by']) != ['right']:
    raise ValueError('right hold not confirmed in observation')

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