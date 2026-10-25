s = observe()
targets = s['image_targets']['bottle_lower']
set_gripper(arm='right', opened=True)
orient(arm='right', yaw_deg=0, pitch_deg=-15)

# Stage 1: reach a ~9 cm side standoff from the bottle, holding commanded orientation.
a = visual_align(max_steps=24, target_pixels=targets, offset_m=[-.087, 0, .023], tolerance_px=12., yaw_deg=0, pitch_deg=-15)
if not a['ok']:
    raise ValueError('visual pre-approach failed')

# Stage 2: small image-servo steps onto the lower body.
s = observe()
a = visual_align(max_steps=24, target_pixels=s['image_targets']['bottle_lower'], step_m=.006, tolerance_px=4., yaw_deg=0, pitch_deg=-15)
if not a['ok']:
    raise ValueError('visual alignment failed')

# Close, with measured-feedback check and at most one retry.
open_rad = s['robot']['right']['gripper_open_rad']
c = set_gripper(arm='right', opened=False)
if abs(c['motor_rad'] - open_rad) < .10:
    c = set_gripper(arm='right', opened=False)
    if abs(c['motor_rad'] - open_rad) < .10:
        raise ValueError('gripper close failed: no measured closure after retry')

# Lift and verify with fresh RGB + robot lift.
move_relative(arm='right', delta_m=[0, 0, .075])
wait(seconds=1.0)
v = check_grasp()
if not v['ok']:
    raise ValueError('RGB hold verification failed')

