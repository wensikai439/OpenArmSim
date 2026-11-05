"""Independently audit the GPU receipt without rewriting native experiments."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpu_evidence(row, worker_hash):
    matches = [s for s in row['steps'] if s['task'] == 'move_to_pose']
    if len(matches) != 1: return False
    step = matches[0]; f = step['feedback']; backend = f.get('backend_metadata', {})
    return bool(step['parameters'].get('planner') == 'curobo'
        and f.get('planner') == 'official cuRobo V2 GPU IK/trajectory optimization'
        and f.get('worker_port') == 8004 and f.get('native_path_revalidation') is True
        and f.get('fallback') is False and backend.get('ready') is True
        and backend.get('worker_source_sha256') == worker_hash and bool(f.get('plans'))
        and f.get('primitive_success') is True and f.get('verification_source') == 'robot_forward_kinematics'
        and f.get('position_error_m', 1) <= .012 and f.get('rotation_error_rad', 1) <= .08)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', required=True); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple run name required')
    folder = ROOT / 'runs/20261009-bottle' / args.name
    protocol = json.loads((folder / 'protocol.json').read_text())
    expected = protocol['source_sha256']['tools/object_gpu.py']
    assert hashlib.sha256((folder/'source/tools/object_gpu.py').read_bytes()).hexdigest() == expected
    original = json.loads((folder/'results.json').read_text()); assert original['complete']
    assessments = []
    for row in original['cases']:
        checks = dict(row['checks']); checks['actual_GPU_transport'] = gpu_evidence(row, expected)
        raw = folder/row['case']['name']/'result.json'
        assessment = {'case': row['case'], 'passed': all(checks.values()), 'checks': checks,
            'original_collector_passed': row['passed'], 'raw_result_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
            'correction': 'Only GPU identity is rechecked against the actual planner/backend_metadata contract. All other original checks remain unchanged.',
            'no_new_robot_actions': True, 'no_goal_or_acceptance_changes': True, 'raw_result_unchanged': True}
        (raw.parent/'assessment.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2))
        assessments.append(assessment)
    positives = [r for r in original['cases'] if gpu_evidence(r, expected)]
    negative_checks = []
    if positives:
        for name, key, value in [('wrong_planner','planner','other'), ('fallback','fallback',True),
            ('no_path_recheck','native_path_revalidation',False), ('no_plans','plans',[]),
            ('bad_position','position_error_m',.013), ('bad_rotation','rotation_error_rad',.081)]:
            altered = copy.deepcopy(positives[0])
            next(s for s in altered['steps'] if s['task']=='move_to_pose')['feedback'][key] = value
            passed = not gpu_evidence(altered, expected); assert passed
            negative_checks.append({'name':name,'rejected':passed})
        assert not gpu_evidence(positives[0], 'incorrect-worker-hash')
        negative_checks.append({'name':'wrong_worker_source','rejected':True})
    (folder/'assessment.json').write_text(json.dumps({'complete': True, 'cases': assessments,
        'grader_negative_checks':negative_checks}, ensure_ascii=False, indent=2))
    print([(r['case']['name'],r['passed']) for r in assessments])


if __name__ == '__main__': main()
