"""Reuse a matching grasp lookahead's intermediate height, as a proposal only."""
import numpy as np


def clearance(future, target_xy, *, default=.20):
    if not isinstance(future, dict): future = {}
    xy = future.get('target_xy_m')
    height = future.get('transit_height_m')
    valid = (isinstance(xy, list) and len(xy) == 2 and all(type(x) in (int, float) for x in xy) and np.isfinite(xy).all()
        and isinstance(target_xy, (list, tuple)) and len(target_xy) == 2 and all(type(x) in (int, float) for x in target_xy) and np.isfinite(target_xy).all()
        and np.max(abs(np.asarray(xy)-target_xy)) < 1e-8
        and type(height) in (int, float) and np.isfinite(height) and .05 <= height <= .30)
    return (float(height), {'used': True, 'height_m': float(height), 'target_xy_m': list(target_xy),
        'source': 'matching grasp lookahead; intermediate geometry proposal only',
        'native_path_and_stable_placement_required': True}) if valid else (default, {
        'used': False, 'height_m': default, 'source': 'original default; no matching valid lookahead'})
