"""Cold-compiled optional scene objects, owned by the isolated workbench.

Removing an item removes its native body and collision geometry. Applying a
selection explicitly creates a new scene; it is never a task recovery action.
"""
import copy
import math
import uuid
from unittest.mock import patch

import mujoco

from . import objects as objects_module
from .objects import ASSETS
from .task_scene import TaskScene


ITEMS = [
    {'object_id': 'can', 'label': '高圆柱', 'icon': 'cylinder', 'description': '圆柱体 · 18 cm',
        'manipulation': 'registered'},
    {'object_id': 'can_short', 'label': '短圆柱', 'icon': 'short-cylinder', 'description': '圆柱体 · 10 cm',
        'manipulation': 'scene_only'},
    {'object_id': 'block', 'label': '方块', 'icon': 'block', 'description': '长方体 · 18 cm',
        'manipulation': 'registered'},
    {'object_id': 'cup', 'label': '杯子', 'icon': 'cup', 'description': '带把手的空心杯',
        'manipulation': 'scene_only'},
    {'object_id': 'book', 'label': '书本', 'icon': 'book', 'description': '平放的小书本',
        'manipulation': 'scene_only'},
]
DEFAULT = ['block', 'can']
EXTRAS = [
    {'object_id': 'can_short', 'label': '短圆柱', 'type': 'cylinder', 'size_m': [.028, .05, 0.],
        'geom_pos_m': [0., 0., .05], 'position_m': [.64, -.35, 1.005], 'mass_kg': .15,
        'rgba': [.64, .68, .73, 1.]},
    {'object_id': 'book', 'label': '书本', 'type': 'box', 'size_m': [.055, .0425, .0125],
        'geom_pos_m': [0., 0., .0125], 'position_m': [.72, -.06, 1.005], 'mass_kg': .20,
        'rgba': [.48, .56, .62, 1.]},
]
CONTRACT = {'schema': 'optional-scene-library/1',
    'source': 'declared synthetic dimensions and materials; not scanned or calibrated assets',
    'base_scene': ['bottle', 'basket', 'table'],
    'application': 'explicit fresh cold-compiled scene; no removal/teleport/model mutation within a task',
    'additional_assets': 'native free rigid bodies included in scene references and collision planning',
    'untested_manipulation': ['can_short', 'cup', 'book'],
    'hardware_access': False, 'transfer_ready': False}


def validate_selection(values):
    if not isinstance(values, list) or any(type(x) is not str for x in values):
        raise ValueError('物品列表格式错误')
    ids = [item['object_id'] for item in ITEMS]
    if len(values) != len(set(values)) or any(x not in ids for x in values):
        raise ValueError('只能选择物品库里的物品，每件最多一份')
    return [ident for ident in ids if ident in values]


def add_cup(spec):
    body = spec.worldbody.add_body(name='task_cup', pos=[.655, .185, 1.005])
    body.add_freejoint(name='cup_free')
    common = {'rgba': [.80, .73, .61, 1.], 'friction': [1., .01, .01],
        'condim': 4, 'contype': 1, 'conaffinity': 1}
    body.add_geom(name='cup_bottom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[.035, .003, 0.], pos=[0, 0, .003], mass=.06, **common)
    for index in range(12):
        angle = 2 * math.pi * index / 12
        body.add_geom(name=f'cup_wall_{index}', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[.0025, .0088, .0425], pos=[.0325*math.cos(angle), .0325*math.sin(angle), .0485],
            quat=[math.cos(angle/2), 0, 0, math.sin(angle/2)], mass=.01, **common)
    for name, pos, size in [('top', [.05, 0, .08], [.02, .004, .004]),
        ('outer', [.07, 0, .055], [.004, .004, .025]), ('bottom', [.05, 0, .03], [.02, .004, .004])]:
        body.add_geom(name='cup_handle_'+name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos, size=size, mass=.008, **common)


class LibraryEnv(TaskScene):
    def __init__(self, *, selection=None, **kwargs):
        self.library_selection = validate_selection(DEFAULT if selection is None else selection)
        self.scene_revision = uuid.uuid4().hex[:12]
        assets = [copy.deepcopy(row) for row in [*ASSETS, *EXTRAS]
            if row['object_id'] in self.library_selection]
        original = objects_module.CandidateSpec
        selected = self.library_selection
        class Spec:
            @staticmethod
            def from_string(*args, **values):
                spec = original.from_string(*args, **values)
                if 'cup' in selected: add_cup(spec)
                return spec
        with patch.object(objects_module, 'CandidateSpec', Spec):
            super().__init__(assets=assets, **kwargs)
        labels = {item['object_id']: item['label'] for item in ITEMS}
        for ident, row in self.directory.rows.items():
            if ident in labels: row['label'] = labels[ident]

    def library(self):
        return {**CONTRACT, 'items': copy.deepcopy(ITEMS), 'selected': list(self.library_selection),
            'revision': self.scene_revision, 'editable': True}

    def manifest(self):
        value = super().manifest()
        value['scene_library'] = self.library()
        return value


def commit_scene(owner, candidate):
    old = owner.env
    previous = {key: getattr(owner, key, None) for key in
        ('epoch', 'keyframes', 'last_run', 'decision_receipt', 'frame_history', 'cache', 'frames', 'applied_view_revision')}
    with owner.lock:
        owner.scene_transition = {'cache': previous['cache'], 'frames': previous['frames'],
            'history': previous['frame_history']}
    try:
        owner.env = candidate; owner.epoch = uuid.uuid4().hex[:12]
        owner.keyframes = []; owner.last_run = None; owner.decision_receipt = None
        owner.frame_history = {}; owner._publish()
    except Exception:
        owner.env = old
        with owner.lock:
            for key, value in previous.items(): setattr(owner, key, value)
            owner.scene_transition = None
        raise
    with owner.lock: owner.scene_transition = None
    old.close()


def service_factory(base):
    """Only the physics owner compiles/commits scene changes; failures are atomic."""
    import sim.service as service
    service.OpenArmEnv = LibraryEnv
    class Commands:
        def __init__(self, original, owner): self.original, self.owner = original, owner
        def put(self, value): return self.original.put(value)
        def get(self, *args, **kwargs):
            while True:
                command, values, future = self.original.get(*args, **kwargs)
                if command != 'configure_scene': return command, values, future
                if not future.set_running_or_notify_cancel(): continue
                owner = self.owner; candidate = None
                try:
                    if owner.job or owner.decision_wait: raise ValueError('任务运行中不能修改物品')
                    if values['observation_id'] != f'{owner.epoch}:{owner.env.tick}':
                        raise ValueError('场景已变化，请重新打开物品库')
                    owner.env.require_current_sources()
                    for row in owner.env.registry.values():
                        if any(row['contacts'].read(owner.env.data)['touching'].values()):
                            raise ValueError('机械臂仍接触物品，请先完成放置并撤离')
                    selection = validate_selection(values['object_ids'])
                    if selection == owner.env.library_selection:
                        future.set_result({'changed': False, 'observation_id': owner.cache['observation_id']})
                        continue
                    candidate = LibraryEnv(selection=selection, images=False)
                    candidate.reset(seed=0)
                    from .oriented_grasp import initial_conflicts
                    conflicts = initial_conflicts(candidate)
                    if conflicts: raise ValueError('所选物品初始布局有碰撞，保留当前场景')
                    # Verify every model camera before releasing the old EGL
                    # environment. Nothing is committed on compile/validation failure.
                    for camera in ('operator', 'head', 'left', 'right'): candidate.render(camera)
                    commit_scene(owner, candidate); candidate = None
                    future.set_result({'changed': True, 'new_scene': True, 'physical_state_preserved': False,
                        'fresh_observation_required': True, 'selected': selection,
                        'observation_id': owner.cache['observation_id']})
                except Exception as exc:
                    if candidate is not None: candidate.close()
                    future.set_exception(exc)
    class LibraryService(base):
        def snapshot(self):
            with self.lock:
                transition = getattr(self, 'scene_transition', None)
                if transition: return copy.deepcopy(transition['cache'])
            return super().snapshot()
        def frame(self, camera, observation_id=None):
            with self.lock:
                transition = getattr(self, 'scene_transition', None)
                if transition:
                    frames = transition['frames'] if not observation_id else transition['history'].get(observation_id)
                    if frames is None or camera not in frames: raise ValueError('camera observation expired')
                    return frames[camera]
            return super().frame(camera, observation_id)
        def call(self, command, **arguments):
            if command == 'policy':
                snapshot = self.snapshot()
                if snapshot['diagnostics']['manipulation_scene']['object_id'] != 'bottle':
                    raise ValueError('当前 ACT 训练技能只覆盖瓶子右手交左手；请选择标准瓶子场景')
                if not snapshot.get('policy_configured'): raise ValueError('ACT 权重或本机推理接口未匹配')
                # ACT still receives all three RGB images at inference time.
                # Keep state/action traces and evidence keyframes without
                # duplicating all three camera files at every control tick.
                return service.SimulationService.call(self, command, **{**arguments, 'record_images': False})
            return super().call(command, **arguments)
        def _run(self, kind, args):
            if kind == 'policy': return service.SimulationService._run(self, kind, args)
            return super()._run(kind, args)
        def _worker(self):
            self.commands = Commands(self.commands, self)
            return super()._worker()
        def _publish(self):
            super()._publish()
            with self.lock:
                self.cache['scene_library'] = self.env.library()
                self.cache['scene_library']['observation_id'] = self.cache['observation_id']
                self.cache['scene_library']['editable'] = not bool(self.job or self.decision_wait)
                connected = bool(self.policy_config and self.policy_config.get('policy_type') == 'act'
                    and self.policy_status.get('ready'))
                admitted = connected and self.env.object_id == 'bottle'
                self.cache['policy_configured'] = admitted
                self.cache['policy_status'] = {**self.policy_status, 'ready': admitted,
                    'connection_ready': connected, 'performance_verified': False,
                    'capability': 'atomic_right_to_left_handover',
                    'reason': None if admitted else 'ACT 需匹配本机权重接口并操作瓶子',
                    'scope': 'ACT interface available; candidate scene transfer performance remains experimental'}
    return LibraryService
