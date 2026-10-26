label(name='right_reach')
s = observe()
shared_y = .5 * (s['robot']['left']['tcp_m'][1] + s['robot']['right']['tcp_m'][1])
targets = s['image_targets']['bottle_lower']
set_gripper(arm='right', opened=True)
orient(arm='right', yaw_deg=0, pitch_deg=-15)
label(name='view_calibration')
calibrate_views()
# RGB/FK-derived target. Keep a visible side standoff before approaching.
label(name='coarse_rgb_relative_pregrasp')
p = locate_pixels(pixels=targets)['estimated_position_m']
s = observe();r = s['robot']['right']['rotation']
move_tcp(arm='right',position_m=[p[i]-.09*r[i][2] for i in range(3)])
# Refresh the estimate after the larger camera/robot baseline.
p = locate_pixels(pixels=targets)['estimated_position_m']
s = observe();r = s['robot']['right']['rotation']
label(name='coarse_rgb_relative_approach')
move_tcp(arm='right',position_m=[p[i]-.022*r[i][2] for i in range(3)])
label(name='visual_correction_6mm')
a = visual_align(max_steps=24,target_pixels=targets,step_m=.006,tolerance_px=4.)
if not a['ok']:raise ValueError('visual alignment failed')

# Close, with measured-feedback check and at most one retry.
# Baseline is the fresh measured motor state, not the commanded open endpoint.
label(name='gripper_close')
s = observe()
baseline = s['robot']['right']['gripper_motor_rad']
c = set_gripper(arm='right', opened=False)
if abs(c['motor_rad'] - baseline) < .10:
    c = set_gripper(arm='right', opened=False)
    if abs(c['motor_rad'] - baseline) < .10:
        raise ValueError('gripper close failed: no measured closure after retry')

# Lift and verify with fresh RGB + robot lift.
label(name='right_lift_verify')
move_relative(arm='right', delta_m=[0, 0, .075])
wait(seconds=1.1)
v = check_grasp()
if not v['ok']:
    raise ValueError('RGB hold verification failed')

label(name='right_transport')
s=observe();r=s['robot']['right']['tcp_m']
try:
    move_tcp(arm='right',position_m=[r[0],.5*(r[1]+shared_y),r[2]+.095])
except Exception as err:
    if 'TCP tracking error exceeds' not in str(err):raise
    s=observe()
    if not grip_evidence(arm='right')['blocked_closure']:raise ValueError('right grip lost during transport')
    label(name='transport_replan_from_actual')
s=observe();r=s['robot']['right']['tcp_m']
try:
    move_tcp(arm='right',position_m=[r[0]-.015,shared_y,r[2]])
except Exception as err:
    if 'TCP tracking error exceeds' not in str(err):raise
    s=observe()
    if not grip_evidence(arm='right')['blocked_closure']:raise ValueError('right grip lost during transport')
    label(name='transport_replan_from_actual')
label(name='left_receive')
set_gripper(arm='left',opened=True)
s=observe();r=s['robot']['right']['tcp_m']
a=approach_options(arm='left',target_m=[r[0],r[1],r[2]+.09],yaws_deg=[-85,-75,-60,-45,0],pitch_deg=-5)
if not a['ok']:raise ValueError('no receiver approach')
move_tcp(arm='left',position_m=a['pre_m'],yaw_deg=a['yaw_deg'],pitch_deg=a['pitch_deg'])
move_tcp(arm='left',position_m=a['target_m'])
set_gripper(arm='left',opened=False)
label(name='receiver_closure_settle')
wait(seconds=.2)
# Fresh proprioceptive check: receiver closure must be blocked before donor release.
e_pre = grip_evidence(arm='left')
if not e_pre['blocked_closure']:raise ValueError('receiver closure not blocked before donor release')
label(name='release_right')
set_gripper(arm='right',opened=True)
move_relative(arm='right',delta_m=[0,0,-.13])
move_relative(arm='right',delta_m=[0,-.12,0])
label(name='donor_withdraw_settle')
wait(seconds=.5)
# Fresh proprioceptive check after donor withdrawal: receiver must still block closure.
e_post = grip_evidence(arm='left')
if not e_post['blocked_closure']:raise ValueError('receiver closure not blocked after donor withdrawal')

label(name='basket_localisation')
b=locate_basket()['estimated_position_m']
s=observe();l=s['robot']['left']['tcp_m']
move_tcp(arm='left',position_m=[b[0],b[1],l[2]+.02],yaw_deg=0,pitch_deg=-5)
label(name='lower_into_basket')
move_tcp(arm='left',position_m=[b[0],b[1],b[2]-.084+.14+.025])
set_gripper(arm='left',opened=True)
move_relative(arm='left',delta_m=[-.065,0,.04])
label(name='final_native_support_confirmation')
wait(seconds=1.1)
