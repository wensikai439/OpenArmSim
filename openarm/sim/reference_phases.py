"""Command landmarks for development diagnostics, not physical success labels."""
import numpy as np

PROTOCOL = {'id': 'command_landmarks_v1', 'close_threshold_rad': -.2,
            'open_threshold_rad': -.7, 'after_frames': 10, 'before_frames': 5,
            'scope': 'recorded command changes; grasp/contact/success are not inferred'}


def landmarks(reference):
    left = np.asarray(reference['left_action_gripper'], float)
    right = np.asarray(reference['right_action_gripper'], float)
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2 or not np.isfinite([left,right]).all():
        raise ValueError('invalid recorded gripper commands')
    def first(values, after, closing):
        mask = values > PROTOCOL['close_threshold_rad'] if closing else values < PROTOCOL['open_threshold_rad']
        transitions = mask & ~np.r_[True, mask[:-1]]
        candidates = np.flatnonzero(transitions & (np.arange(len(values)) > after))
        if not len(candidates):
            raise ValueError('recording does not contain the expected ordered command landmarks')
        return int(candidates[0])
    rc = first(right, -1, True)
    lc = first(left, rc, True)
    ro = first(right, lc, False)
    lo = first(left, ro, False)
    ahead, behind = PROTOCOL['after_frames'], PROTOCOL['before_frames']
    plan = [(max(0,rc-behind), 'before_right_close', '右爪关闭前'),
            (min(rc+ahead,lc-1), 'after_right_close', '右爪关闭后'),
            ((lc+ro)//2, 'both_closed_command', '双爪闭合指令'),
            (min(ro+ahead,lo-1), 'after_right_open', '右爪释放后'),
            (lo-1, 'before_left_open', '左爪释放前'),
            (min(lo+ahead,len(left)-1), 'after_left_open', '左爪释放后'),
            (len(left)-1, 'record_end', '记录结束')]
    samples = {}
    for frame, key, label in plan:
        samples.setdefault(frame, []).append({'id': key, 'label': label})
    return {'events': {'right_close': rc, 'left_close': lc, 'right_open': ro, 'left_open': lo},
            'samples': samples, 'protocol': PROTOCOL}
