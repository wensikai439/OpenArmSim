obs = get_observation()
held = list(obs['held_by'])

if held not in ([], ['right'], ['left']):
    raise ValueError('invalid holds ' + str(held))

if held == []:
    if obs['upright'] is not True:
        raise ValueError('unsafe precondition: upright is not True')
    r = pick(arm='right', hold_seconds=2, next_handover={'to_arm': 'left'})
    if not r.ok:
        raise ValueError('pick failed: ' + str(r.diagnostics.get('failure_code')))
    obs = get_observation()
    if list(obs['held_by']) != ['right']:
        raise ValueError('post-pick holder: ' + str(list(obs['held_by'])))
    held = list(obs['held_by'])

if held == ['right']:
    r = handover(from_arm='right', to_arm='left', hold_seconds=2, placement_aware=True)
    if not r.ok:
        raise ValueError('handover failed: ' + str(r.diagnostics.get('failure_code')))
    obs = get_observation()
    if list(obs['held_by']) != ['left']:
        raise ValueError('post-handover holder: ' + str(list(obs['held_by'])))

obs = get_observation()
if list(obs['held_by']) != ['left']:
    raise ValueError('pre-place holder: ' + str(list(obs['held_by'])))

r = place(arm='left', receptacle_id='basket', shape_clearance=True)
if not r.ok:
    raise ValueError('place failed: ' + str(r.diagnostics.get('failure_code')))
obs = get_observation()
if list(obs['held_by']) != []:
    raise ValueError('post-place holder: ' + str(list(obs['held_by'])))

result = 'receipt: place ok; held_by=' + str(list(obs['held_by']))
