"""Object goal precision, fixed before motion and shared with native scoring."""
import math


def placement_tolerance(surface, target_xy, anchor_xy=None):
    if surface == 'basket':
        return .025
    if surface != 'table':
        raise ValueError('unknown support surface')
    if anchor_xy is None:
        return .010
    distance = math.dist(target_xy, anchor_xy)
    if distance < 1e-9:
        return .010
    if distance < .002:
        raise ValueError('桌面位移小于 2 毫米，超出当前仿真技能已声明的精度范围')
    # An unchanged object must fail even for a small requested translation.
    return min(.010, distance * .25)
