"""cuRobo worker with explicit native inventory for optional scene objects.

Legacy requests retain the old three-object validation. Versioned requests bind
their full current registry; every present unselected geometry must remain a
world obstacle. No physics or GPU collision representation changes here.
"""
import copy
import hashlib
import logging
from http.server import HTTPServer
from pathlib import Path

import numpy as np

import tools.object_gpu as original
from sim.curobo_server import Handler, validate_scene

INVENTORY = {**original.GEOMS, 'can_short': ['can_short_shape'], 'book': ['book_shape'],
    'cup': ['cup_bottom', *[f'cup_wall_{i}' for i in range(12)], 'cup_handle_top', 'cup_handle_outer', 'cup_handle_bottom']}
FIXED_WORLD = {'floor', 'cell_table_col', 'box_floor', 'box_front', 'box_back', 'box_left', 'box_right'}


def validate_inventory(inventory, attached, scene):
    validate_scene(scene)
    if not FIXED_WORLD <= set(scene['cuboid']):
        raise ValueError('native fixed floor/table/basket obstacles must remain')
    if type(inventory) is not dict or not inventory or len(inventory) > len(INVENTORY) or 'bottle' not in inventory:
        raise ValueError('complete bounded native registry including base bottle required')
    for ident, names in inventory.items():
        if ident not in INVENTORY or names != INVENTORY[ident]:
            raise ValueError('native scene identity/geometry inventory mismatch')
    for other, names in INVENTORY.items():
        if other not in inventory and any(name in scene['cuboid'] for name in names):
            raise ValueError('world geometry contradicts absent native object')
    if attached is None:
        if any(name not in scene['cuboid'] for names in inventory.values() for name in names):
            raise ValueError('all present native objects must remain world obstacles for an empty hand')
        return None
    if type(attached) is not dict or set(attached) != {'object_id', 'spheres_tcp', 'removed_world_geoms', 'representation'}:
        raise ValueError('bounded registered-object attachment fields required')
    ident = attached['object_id']
    if ident not in original.GEOMS or inventory.get(ident) != original.GEOMS[ident] or attached['removed_world_geoms'] != original.GEOMS[ident]:
        raise ValueError('selected object and full native inventory must agree')
    spheres = np.asarray(attached['spheres_tcp'], dtype=np.float32)
    if (spheres.ndim != 2 or spheres.shape[1] != 4 or not 1 <= len(spheres) <= 32
            or not np.isfinite(spheres).all() or np.max(abs(spheres[:, :3])) > .5
            or np.any(spheres[:, 3] <= 0) or np.any(spheres[:, 3] > .1)):
        raise ValueError('invalid bounded attachment spheres')
    if any(name in scene['cuboid'] for name in inventory[ident]):
        raise ValueError('selected carried object cannot also be a static world obstacle')
    for other, names in inventory.items():
        if other != ident and not all(name in scene['cuboid'] for name in names):
            raise ValueError('every PRESENT unselected native object must remain a world obstacle')
    return spheres


class Planner(original.Planner):
    def __init__(self):
        super().__init__()
        self.metadata.update(scene_protocol='native-scene-inventory/1',
            scene_worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scene_inventory_contract='complete current native registry; all present nonselected geometry retained',
            supported_scene_inventory=INVENTORY)

    def plan(self, data):
        inventory = data.get('native_scene_inventory')
        if inventory is None:
            return super().plan(data)  # Original strict legacy contract.
        validate_inventory(inventory, data.get('attachment'), data['scene'])
        previous = original.validate_attachment
        original.validate_attachment = lambda attached, scene: validate_inventory(inventory, attached, scene)
        try:
            result = super().plan(data)
            result['native_scene_inventory'] = copy.deepcopy(inventory)
            return result
        finally:
            original.validate_attachment = previous


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    server = HTTPServer(('127.0.0.1', 8004), Handler)
    try:
        server.planner = Planner(); print('Scene cuRobo · 127.0.0.1:8004 · no hardware', flush=True)
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if getattr(server, 'planner', None) and server.planner.cached: server.planner.cached[1].destroy()
        server.server_close()


if __name__ == '__main__': main()
