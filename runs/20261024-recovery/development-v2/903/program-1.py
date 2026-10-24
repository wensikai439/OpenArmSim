s = observe()
if list(s['held_by']) != ['right']:
    raise ValueError('expected right-only hold, got ' + str(list(s['held_by'])))
if not s['touching']['right']:
    raise ValueError('right touch not confirmed')
v = verify_hold(arm='right')
if not v['ok']:
    raise ValueError('right hold not verified')
r = pregrasp(s['current_grasp_id'])
if not r['ok'] or not r['done']:
    raise ValueError('reposition pregrasp failed')
fresh = observe()
if list(fresh['held_by']) != []:
    raise ValueError('unexpected hold state after retreat check')
result = {'status': 'recovery_stopped_for_review', 'final_observation': fresh}
