"""Shape-frame invariance and original fixed-demo native action equivalence."""
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from sim.environment import ROOT
from sim.grasp import GeometricPlanner, cylinder_bands, frames
from sim.tamp import TampPlanner
from sim.recording import dump
from tools.contact_candidate import ContactCandidate

OUT = ROOT / 'runs/20261008-ik/contract'


def main():
    OUT.mkdir(exist_ok=False)
    protocol = {'schema': 'grasp-contract/1', 'seed': 0, 'episode': 0,
        'fixed_task': 'handover_to_basket', 'physical_model': 'same cold contact candidate',
        'comparison': 'two independent native owners; every command and qpos/qvel endpoint, terminal status and predicate',
        'geometry_checks': 'proper rotations, transverse closing direction, object translation/yaw equivariance',
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hardware_access': False, 'main_page_commands': 0, 'transfer_ready': False}
    dump(OUT / 'protocol.json', protocol)
    baseline = ContactCandidate(images=False)
    candidate = ContactCandidate(images=False)
    try:
        baseline.reset(seed=0); candidate.reset(seed=0)
        bands = cylinder_bands(candidate.model, candidate.data, candidate.bottle_id)
        addr = candidate.model.joint('bottle_free').qposadr[0]
        q = candidate.data.qpos.copy(); v = candidate.data.qvel.copy()
        # Pose edits belong only to scratch MjData, never to either live owner.
        scratch = mujoco.MjData(candidate.model)
        mujoco.mj_copyData(scratch, candidate.model, candidate.data)
        scratch.qpos[addr:addr + 3] += [.2, .1, .05]
        angle = .37
        scratch.qpos[addr + 3:addr + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        mujoco.mj_forward(candidate.model, scratch)
        shifted = cylinder_bands(candidate.model, scratch, candidate.bottle_id)
        for old, new in zip(bands, shifted):
            assert old['geom'] == new['geom']
            np.testing.assert_allclose(new['center_m'] - old['center_m'], [.2, .1, .05], atol=1e-12)
            np.testing.assert_allclose(new['axis'], old['axis'], atol=1e-12)
        for theta in np.arange(24) * np.pi / 12:
            for pitch in np.deg2rad([0, -15, -30, 15]):
                rotation = frames(np.array([0., 0., 1.]), theta, pitch)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)
                np.testing.assert_allclose(rotation[:, 1] @ [0, 0, 1], 0., atol=1e-12)
        np.testing.assert_array_equal(q, candidate.data.qpos)
        np.testing.assert_array_equal(v, candidate.data.qvel)
        p = TampPlanner(baseline, 'handover_to_basket')
        g = GeometricPlanner(candidate, 'handover_to_basket')
        assert not g.parameterized_pick
        first, second = p.actions(), g.actions()
        digest = hashlib.sha256(); count = 0; status = None
        started = time.monotonic()
        while True:
            outcomes = []
            for generator in (first, second):
                try: outcomes.append(('action', next(generator)))
                except StopIteration: outcomes.append(('completed', None))
                except Exception as exc: outcomes.append(('error', {'code': getattr(exc, 'code', type(exc).__name__), 'message': str(exc)}))
            assert outcomes[0][0] == outcomes[1][0], outcomes
            if outcomes[0][0] != 'action':
                assert outcomes[0][1] == outcomes[1][1]
                status = outcomes[0]; break
            a, b = outcomes[0][1], outcomes[1][1]
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1] == b[1]
            digest.update(np.asarray(a[0], dtype='<f8').tobytes())
            left = baseline.step_tamp(a[0]); right = candidate.step_tamp(b[0])
            np.testing.assert_array_equal(baseline.data.qpos, candidate.data.qpos)
            np.testing.assert_array_equal(baseline.data.qvel, candidate.data.qvel)
            assert left[2:4] == right[2:4]
            count += 1
            if left[2] or left[3]: status = ('native_terminal', baseline.evaluation); break
        assert baseline.evaluation == candidate.evaluation
        result = {'passed': True, 'geometry_checks': 96, 'fixed_demo_commands_compared': count,
            'fixed_demo_action_sha256': digest.hexdigest(), 'every_native_qpos_qvel_endpoint_identical': True,
            'fixed_demo_terminal': status, 'fixed_demo_baseline_result': p.result,
            'fixed_demo_extension_result': g.result, 'wall_seconds': time.monotonic() - started,
            'scope': 'fixed-demo equivalence and cylinder-frame checks; not physical calibration or arbitrary objects',
            'hardware_access': False, 'transfer_ready': False}
        dump(OUT / 'result.json', result)
        print(json.dumps({k: result[k] for k in ('passed', 'geometry_checks', 'fixed_demo_commands_compared', 'fixed_demo_terminal')}, ensure_ascii=False), flush=True)
    finally: baseline.close(); candidate.close()


if __name__ == '__main__': main()
