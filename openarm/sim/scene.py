"""Named, explicit-oracle scene references, separate from robot observations.

Primitive bounds are derived from current native geometry. They are not
measurements, detections or grasp capability claims. New free bodies appear in
the directory without adding a bottle-specific perception branch.
"""
import hashlib
import json
import re

import mujoco
import numpy as np

CONTRACT = {'schema': 'scene-directory/1', 'source': 'simulator_ground_truth',
    'contains_privileged_object_state': True,
    'pose_source': 'current MuJoCo body/geom transforms; not camera perception',
    'geometry_source': 'compiled candidate primitives; not physical measurement',
    'reference_points': ['origin', 'bounds_center', 'top_center'],
    'frames': 'world axes or the referenced body/geom local axes',
    'scope': 'scene references and pose proposals; motion, grasp and placement require separate receipts',
    'hardware_access': False, 'transfer_ready': False}


def bounds(model, data, geom):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    kind = int(model.geom_type[geom])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = rotation[:, 2]
        extent = size[0] * np.sqrt(np.maximum(0., 1. - axis ** 2)) + size[1] * np.abs(axis)
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.repeat(size[0], 3)
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * np.abs(rotation[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.sqrt((rotation ** 2) @ (size ** 2))
    else:
        raise ValueError('reference bounds are unavailable for this geometry; no proxy box invented')
    center = data.geom_xpos[geom]
    return center - extent, center + extent


def quaternion(rotation):
    value = np.empty(4)
    mujoco.mju_mat2Quat(value, np.asarray(rotation).reshape(9))
    return np.r_[value[1:], value[0]].tolist()


class SceneDirectory:
    def __init__(self, env):
        self.env = env
        model = env.model
        self.rows = {'table': {'geom': model.geom('cell_table_col').id, 'label': '桌面', 'kind': 'support'},
            'basket': {'body': model.body('target_box').id, 'label': '篮子', 'kind': 'receptacle'}}
        for body in range(1, model.nbody):
            name = model.body(body).name or ''
            if name.startswith('openarm_') or not int(model.body_jntnum[body]): continue
            joint = int(model.body_jntadr[body])
            if int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE: continue
            ident = name[5:] if name.startswith('task_') else name
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', ident):
                raise ValueError('free scene body needs a stable lowercase semantic ID')
            if ident in self.rows: raise ValueError('duplicate scene reference ID: ' + ident)
            self.rows[ident] = {'body': body, 'label': '瓶子' if ident == 'bottle' else ident,
                'kind': 'rigid_object', 'free_joint': model.joint(joint).name}

    def snapshot(self, observation_id):
        model, data = self.env.model, self.env.data
        objects = []
        for ident, entry in self.rows.items():
            if 'geom' in entry:
                geoms = [entry['geom']]
                position = data.geom_xpos[entry['geom']]
                rotation = data.geom_xmat[entry['geom']].reshape(3, 3)
                body = int(model.geom_bodyid[entry['geom']])
            else:
                body = entry['body']
                geoms = np.flatnonzero(model.geom_bodyid == body).tolist()
                position = data.xpos[body]
                rotation = data.xmat[body].reshape(3, 3)
            geoms = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
            shapes = [{'geom': model.geom(g).name, 'type': mujoco.mjtGeom(int(model.geom_type[g])).name,
                'size_m': model.geom_size[g].tolist()} for g in geoms]
            box = None; error = None
            try:
                if not geoms: raise ValueError('no collision geometry')
                boxes = [bounds(model, data, geom) for geom in geoms]
                box = {'min_m': np.min([b[0] for b in boxes], axis=0).tolist(),
                    'max_m': np.max([b[1] for b in boxes], axis=0).tolist()}
            except ValueError as exc: error = str(exc)
            objects.append({'object_id': ident, 'label': entry['label'], 'kind': entry['kind'],
                'origin_m': position.tolist(), 'rotation_matrix': rotation.tolist(),
                'quaternion_xyzw': quaternion(rotation), 'world_bounds': box, 'bounds_error': error,
                'geometry': shapes, 'dynamic': 'free_joint' in entry,
                'mass_kg': float(model.body_mass[body]) if 'free_joint' in entry else None,
                'affordances': {'pose_reference': True, 'bounds_reference': box is not None,
                    'pick_provider': 'declared_bottle_tamp' if ident == 'bottle' else None,
                    'placement_surface': ident if ident in ('table', 'basket') else None},
                'physical_calibration': 'unverified'})
        return {**CONTRACT, 'observation_id': observation_id, 'objects': objects}


def target(directory, arguments, tcp):
    """Pure transform; its output is NOT a motion/grasp/placement receipt."""
    selected = next((row for row in directory['objects'] if row['object_id'] == arguments['object_id']), None)
    if selected is None: raise ValueError('unknown scene object ID; inspect_scene first')
    reference = arguments['reference_point']
    if reference == 'origin': point = np.asarray(selected['origin_m'])
    else:
        box = selected['world_bounds']
        if box is None: raise ValueError('this object has no supported primitive bounds')
        point = (np.asarray(box['min_m']) + box['max_m']) / 2
        if reference == 'top_center': point[2] = box['max_m'][2]
        elif reference != 'bounds_center': raise ValueError('unknown reference point')
    offset = np.asarray(arguments['offset_m'], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .5:
        raise ValueError('scene offset must be finite and at most 0.5 m')
    frame = arguments['offset_frame']
    if frame == 'object': offset = np.asarray(selected['rotation_matrix']) @ offset
    elif frame != 'world': raise ValueError('unknown scene offset frame')
    rotation = arguments.get('quaternion_xyzw', tcp['quaternion_xyzw'])
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (4,) or not np.isfinite(rotation).all() or abs(np.linalg.norm(rotation) - 1.) > .02:
        raise ValueError('target orientation must be a unit xyzw quaternion')
    rotation /= np.linalg.norm(rotation)
    values = {'observation_id': directory['observation_id'], 'arm': arguments['arm'],
        'frame': 'world', 'position_m': (point + offset).tolist(), 'quaternion_xyzw': rotation.tolist()}
    if 'planner' in arguments: values['planner'] = arguments['planner']
    if 'speed_scale' in arguments: values['speed_scale'] = arguments['speed_scale']
    ident = hashlib.sha256(json.dumps({'directory': directory, 'arguments': arguments,
        'target': values}, sort_keys=True).encode()).hexdigest()[:20]
    return {'target_id': ident, 'source_observation_id': directory['observation_id'],
        'reference': selected, 'reference_point': reference, 'offset_frame': frame,
        'offset_m': arguments['offset_m'], 'motion_request': {**values, 'scene_target_id': ident},
        'source': 'simulator_ground_truth', 'motion_started': False,
        'contains_privileged_object_state': True,
        'scope': 'fixed pose proposal from this observation; reachability/collision/contact are not established',
        'hardware_access': False, 'transfer_ready': False}
