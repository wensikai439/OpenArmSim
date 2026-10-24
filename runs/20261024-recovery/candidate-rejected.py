s = observe()
if list(s['held_by']) != []:
    raise ValueError('unexpected hold at start; stop for review')
if s['touching']['right'] or s['touching']['left']:
    raise ValueError('unexpected contact at start; stop for review')
if s['operation'] is not None:
    raise ValueError('operation in progress; stop for review')

def fail_stop(msg):
    raise ValueError(msg)

def check_done(r, name):
    if not r['ok'] or not r['done']:
        fail_stop(name + ' failed: ' + str(r.get('error_code')))
    return r

def secure_right_grasp(max_attempts):
    excluded = ()
    cands = propose_grasps(exclude_ids=())
    if not cands['candidates']:
        fail_stop('no grasp candidates')
    gid = cands['candidates'][0]['id']
    rev = cands['pose_revision']
    attempt = 0
    while attempt < max_attempts:
        check_done(pregrasp(gid), 'pregrasp')
        check_done(approach(gid), 'approach')
        c = close_gripper(arm='right')
        if c['ok'] and c['done']:
            v = verify_hold(arm='right')
            if v['ok'] and list(observe()['held_by']) == ['right']:
                return True
            fail_stop('hold not confirmed after close')
        if c.get('error_code') != 'grasp_not_confirmed':
            fail_stop('close failed: ' + str(c.get('error_code')))
        o = observe()
        motor = o['gripper_motor_feedback_rad']['right']
        open_rad = o['gripper_open_rad']['right']
        if list(o['held_by']) != []:
            fail_stop('grasp_not_confirmed but hold present; stop for review')
        if motor - open_rad < 0.12 and attempt == 0:
            c2 = close_gripper(arm='right')
            if c2['ok'] and c2['done'] and verify_hold(arm='right')['ok'] and list(observe()['held_by']) == ['right']:
                return True
        check_done(open_gripper(arm='right'), 'open')
        check_done(retreat(observe()['checkpoint_id']), 'retreat')
        excluded = excluded + (gid,)
        cands = propose_grasps(exclude_ids=excluded)
        if not cands['candidates']:
            fail_stop('no alternative grasp candidates')
        gid = cands['candidates'][0]['id']
        attempt += 1
    fail_stop('no confirmed right hold within attempt budget')

s = observe()
if s['last_failure'] is not None and s['last_failure']['code'] != 'grasp_not_confirmed':
    fail_stop('historical failure unrelated to grasp; stop for review')

secure_right_grasp(3)
if list(observe()['held_by']) != ['right']:
    fail_stop('right-only hold not confirmed before lift')
check_done(lift(), 'lift')
if list(observe()['held_by']) != ['right']:
    fail_stop('right hold lost after lift')

t = start_transfer(placement_aware=True)
if not t['ok']:
    fail_stop('start_transfer failed: ' + str(t.get('error_code')))
steps = 0
done = False
while not done:
    a = advance()
    steps += 1
    if not a['ok']:
        s2 = observe()
        code = a['error_code']
        if code in ('dual_hold_not_confirmed', 'receiver_hold_not_confirmed') and list(s2['held_by']) == ['right'] and s2['touching']['right']:
            fail_stop('handover recovery needed: close LEFT at aligned receiver pose, verify both holds, finish_transfer, then place')
        fail_stop('transfer advance failed: ' + str(code))
    done = a['done']
    if not done and steps >= 16:
        fail_stop('transfer did not complete within 16 advances')

if list(observe()['held_by']) != ['left']:
    fail_stop('left-only hold not confirmed after transfer')

check_done(start_place(), 'start_place')
steps = 0
done = False
while not done:
    a = advance()
    steps += 1
    if not a['ok']:
        fail_stop('place advance failed: ' + str(a['error_code']))
    done = a['done']
    if not done and steps >= 16:
        fail_stop('placement did not complete within 16 advances')

final = observe()
if list(final['held_by']) != []:
    fail_stop('hands not empty at end')
result = {'status': 'complete', 'final_observation': final}
