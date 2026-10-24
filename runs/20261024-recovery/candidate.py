def fail_stop(msg):
    raise ValueError(msg)

def check_done(r, name):
    if not r['ok'] or not r['done']:
        fail_stop(name + ' failed: ' + str(r.get('error_code')))
    return r

def hold_tuple(s):
    return list(s['held_by'])

def secure_right_grasp(max_attempts):
    cands = propose_grasps()
    if not cands['candidates']:
        fail_stop('no grasp candidates')
    excluded = ()
    gid = cands['candidates'][0]['id']
    attempt = 0
    while attempt < max_attempts:
        check_done(pregrasp(gid), 'pregrasp')
        check_done(approach(gid), 'approach')
        c = close_gripper(arm='right')
        if c['ok'] and c['done']:
            v = verify_hold(arm='right')
            if v['ok'] and hold_tuple(observe()) == ['right']:
                return True
            fail_stop('hold not confirmed after close')
        if c.get('error_code') != 'grasp_not_confirmed':
            fail_stop('close failed: ' + str(c.get('error_code')))
        o = observe()
        if hold_tuple(o) != []:
            fail_stop('grasp_not_confirmed but hold present; stop for review')
        motor = o['gripper_motor_feedback_rad']['right']
        open_rad = o['gripper_open_rad']['right']
        if abs(motor - open_rad) < 0.12 and attempt == 0:
            c2 = close_gripper(arm='right')
            if c2['ok'] and c2['done']:
                v2 = verify_hold(arm='right')
                if v2['ok'] and hold_tuple(observe()) == ['right']:
                    return True
            if c2.get('error_code') != 'grasp_not_confirmed':
                fail_stop('retry close failed: ' + str(c2.get('error_code')))
        check_done(open_gripper(arm='right'), 'open')
        check_done(retreat(observe()['checkpoint_id']), 'retreat')
        excluded = excluded + (gid,)
        cands = propose_grasps(exclude_ids=excluded)
        if not cands['candidates']:
            fail_stop('no alternative grasp candidates')
        gid = cands['candidates'][0]['id']
        attempt += 1
    fail_stop('no confirmed right hold within attempt budget')

def recover_receiver_failure():
    s = observe()
    if s['terminal'] or s['invalid']:
        fail_stop('terminal or invalid state; stop')
    if hold_tuple(s) != ['right'] or not s['touching']['right']:
        fail_stop('receiver recovery precondition not met')
    check_done(close_gripper(arm='left'), 'close_left')
    vr = verify_hold(arm='right')
    if not vr['ok']:
        fail_stop('right hold not verified during recovery')
    vl = verify_hold(arm='left')
    if not vl['ok']:
        fail_stop('left hold not verified during recovery')
    held = hold_tuple(observe())
    if 'right' not in held or 'left' not in held:
        fail_stop('dual hold not observed during recovery')
    check_done(finish_transfer(), 'finish_transfer')
    return True

s = observe()
if hold_tuple(s) != []:
    fail_stop('unexpected hold at start; stop for review')
if s['touching']['right'] or s['touching']['left']:
    fail_stop('unexpected contact at start; stop for review')
if s['operation'] is not None:
    fail_stop('operation in progress; stop for review')
if s['terminal'] or s['invalid']:
    fail_stop('terminal or invalid at start')
if s['last_failure'] is not None and s['last_failure']['code'] != 'grasp_not_confirmed':
    fail_stop('historical failure unrelated to grasp; stop for review')

secure_right_grasp(3)
if hold_tuple(observe()) != ['right']:
    fail_stop('right-only hold not confirmed before lift')
check_done(lift(), 'lift')
if hold_tuple(observe()) != ['right']:
    fail_stop('right hold lost after lift')

t = start_transfer(placement_aware=True)
if not t['ok']:
    fail_stop('start_transfer failed: ' + str(t.get('error_code')))

steps = 0
done = False
transfer_recovered = False
while not done:
    if steps >= 16:
        fail_stop('transfer did not complete within 16 advances')
    a = advance()
    steps += 1
    if not a['ok']:
        code = a['error_code']
        s2 = observe()
        if s2['terminal'] or s2['invalid']:
            fail_stop('terminal or invalid after advance failure: ' + str(code))
        if code in ('dual_hold_not_confirmed', 'receiver_hold_not_confirmed') and hold_tuple(s2) == ['right'] and s2['touching']['right']:
            recover_receiver_failure()
            transfer_recovered = True
            done = True
            break
        fail_stop('transfer advance failed: ' + str(code))
    done = bool(a['done'])

if not transfer_recovered and hold_tuple(observe()) != ['left']:
    fail_stop('left-only hold not confirmed after transfer')
if transfer_recovered and hold_tuple(observe()) != ['left']:
    fail_stop('left-only hold not confirmed after recovered transfer')

sp = start_place()
if not sp['ok']:
    fail_stop('start_place failed: ' + str(sp.get('error_code')))

steps = 0
done = False
while not done:
    if steps >= 16:
        fail_stop('placement did not complete within 16 advances')
    a = advance()
    steps += 1
    if not a['ok']:
        s3 = observe()
        if s3['terminal'] or s3['invalid']:
            fail_stop('terminal or invalid during placement')
        fail_stop('place advance failed: ' + str(a['error_code']))
    done = bool(a['done'])

final = observe()
if hold_tuple(final) != []:
    fail_stop('hands not empty at end')
result = {'status': 'complete', 'final_observation': final}
