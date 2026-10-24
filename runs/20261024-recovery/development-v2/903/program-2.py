s = observe()
if list(s['held_by']) != ['right']:
    raise ValueError('expected right-only hold, got ' + str(list(s['held_by'])))
if not s['touching']['right']:
    raise ValueError('right touch not confirmed')
v = verify_hold(arm='right')
if not v['ok']:
    raise ValueError('right hold not verified')
if 'lift' not in list(s['completed_steps']):
    raise ValueError('lift not recorded as completed')
hand = observe()
if list(hand['held_by']) != ['right'] or not hand['touching']['right']:
    raise ValueError('right hold changed before transfer')
t = start_transfer(placement_aware=True)
if not t['ok']:
    raise ValueError('start_transfer failed: ' + str(t['error_code']))
steps = 0
done = False
while steps < 16 and not done:
    a = advance()
    steps += 1
    if not a['ok']:
        raise ValueError('advance failed: ' + str(a['error_code']))
    done = a['done']
if not done:
    raise ValueError('transfer did not finish within 16 advances')
fin = observe()
if list(fin['held_by']) != ['left']:
    raise ValueError('expected left-only hold after transfer, got ' + str(list(fin['held_by'])))
p = start_place()
if not p['ok']:
    raise ValueError('start_place failed: ' + str(p['error_code']))
steps = 0
done = False
while steps < 16 and not done:
    a = advance()
    steps += 1
    if not a['ok']:
        raise ValueError('place advance failed: ' + str(a['error_code']))
    done = a['done']
if not done:
    raise ValueError('placement did not finish within 16 advances')
result = {'status': 'transfer_and_place_done', 'final_observation': observe()}
