"""Correct one collector field lookup; never modify raw tasks or thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpu_evidence(record, worker_hash):
    receipts = record['receipts']
    matches = [r for r in receipts if r['execution']['skill'] == 'move_to_pose']
    if len(matches) != 1: return False
    receipt = matches[0]; feedback = receipt['feedback']; metadata = feedback.get('backend_metadata', {})
    events = [e for turn in record['turns'] for e in turn['trace'] if e['kind'] == 'tool']
    hint = any(e['name'] == 'resolve_placement_transit' and e['status'] == 'completed'
        and e['detail']['result'].get('lookahead_execution_id') for e in events)
    return bool(hint and receipt['arguments'].get('planner') == 'curobo'
        and feedback.get('planner') == 'official cuRobo V2 GPU IK/trajectory optimization'
        and feedback.get('worker_port') == 8004 and feedback.get('native_path_revalidation') is True
        and feedback.get('fallback') is False and metadata.get('ready') is True
        and metadata.get('worker_source_sha256') == worker_hash and feedback.get('plans')
        and feedback.get('primitive_success') is True
        and feedback.get('verification_source') == 'robot_forward_kinematics'
        and feedback.get('position_error_m', 1.) <= .012 and feedback.get('rotation_error_rad', 1.) <= .08)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='api-1'); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple experiment name required')
    folder = ROOT/'runs/20261009-poses'/args.name
    protocol = json.loads((folder/'protocol.json').read_text())
    source = folder/'source/tools/object_gpu.py'; worker_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert worker_hash == protocol['source_sha256']['tools/object_gpu.py']
    records = json.loads((folder/'results.json').read_text()); assert records['complete']
    assessments = []
    for row in records['cases']:
        checks = dict(row.get('checks', {}))
        if checks: checks['fresh_pick_hint_and_real_GPU'] = gpu_evidence(row, worker_hash)
        path = folder/row['case']['name']/'result.json'
        result = {'case':row['case'], 'passed':bool(checks) and all(checks.values()), 'checks':checks,
            'original_collector_passed':row['passed'], 'raw_result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'correction':'Only GPU identity lookup corrected: native feedback uses planner/backend_metadata, not a top-level backend field. All other original checks retained.',
            'no_new_actions_or_changed_goals':True, 'original_raw_result_unchanged':True}
        (path.parent/'assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        assessments.append(result)
    (folder/'assessment.json').write_text(json.dumps({'complete':True, 'cases':assessments},ensure_ascii=False,indent=2))
    print([(a['case']['name'], a['passed']) for a in assessments])


if __name__ == '__main__': main()
