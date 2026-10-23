obs = get_observation()
report = {'initial_held_by': obs.get('held_by'), 'initial_bottle_center_m': obs.get('bottle_center_m'), 'initial_upright': obs.get('upright')}

if obs.get('upright') is not True:
    raise ValueError('bottle not upright at start; cannot proceed safely')

p = pick(arm='right', hold_seconds=2, next_handover={'to_arm': 'left'})
report['pick_ok'] = p.ok
report['pick_steps'] = p.steps_executed
report['pick_failure_code'] = p.diagnostics.get('failure_code') if p.diagnostics else None
report['pick_holder'] = p.diagnostics.get('current_holder') if p.diagnostics else None
if not p.ok:
    raise ValueError('pick failed: ' + str(report['pick_failure_code']))

obs = get_observation()
report['post_pick_held_by'] = obs.get('held_by')
if 'right' not in (obs.get('held_by') or []):
    raise ValueError('right arm does not hold bottle after pick')

h = handover(from_arm='right', to_arm='left', hold_seconds=2)
report['handover_ok'] = h.ok
report['handover_failure_code'] = h.diagnostics.get('failure_code') if h.diagnostics else None
report['handover_holder'] = h.diagnostics.get('current_holder') if h.diagnostics else None
if not h.ok:
    raise ValueError('handover failed: ' + str(report['handover_failure_code']))

obs = get_observation()
report['post_handover_held_by'] = obs.get('held_by')
if 'left' not in (obs.get('held_by') or []):
    raise ValueError('left arm does not hold bottle after handover')

pl = place(arm='left', receptacle_id='basket')
report['place_ok'] = pl.ok
report['place_failure_code'] = pl.diagnostics.get('failure_code') if pl.diagnostics else None
if not pl.ok:
    raise ValueError('place failed: ' + str(report['place_failure_code']))

obs = get_observation()
report['final_held_by'] = obs.get('held_by')
report['final_bottle_center_m'] = obs.get('bottle_center_m')
result = report
