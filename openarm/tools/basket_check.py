"""Isolated native basket-chain diagnostic; never an API or hardware result."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['OPENARM_SCENE_OBJECTS'] = '[]'

from tools.object_workbench import build_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--composite', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'runs/20261014-demo' / args.name
    if out.exists() or not out.resolve().is_relative_to(root / 'runs/20261014-demo'):
        raise ValueError('Use a new diagnostic name')
    out.mkdir()
    factory, _ = build_backend()
    service = factory()
    report = {'scope': 'isolated native primitive chain; no LLM, no hardware, no live-workbench mutation', 'cases': []}
    try:
        service.call('task_budget', seconds=120)
        sequence = (
            ('pick_bottle', {'object_id': 'bottle', 'arm': 'right', 'hold_seconds': 10}),
            ('handover_bottle', {'object_id': 'bottle', 'from_arm': 'right', 'to_arm': 'left'}),
            ('place_bottle', {'object_id': 'bottle', 'arm': 'left', 'receptacle_id': 'basket'}))
        if args.composite:
            sequence = (('relocate_object', {'object_id': 'bottle', 'arm': 'right', 'via_arm': 'left',
                        'receptacle_id': 'basket', 'transport_planner': 'tamp'}),)
        for task, parameters in sequence:
            service.call('tamp', task=task, parameters=parameters,
                         observation_id=service.snapshot()['observation_id'], realtime=False,
                         initiator={'type': 'isolated-demo-diagnostic'})
            deadline = time.monotonic() + 180
            while service.snapshot().get('job') and time.monotonic() < deadline:
                time.sleep(.1)
            snapshot = service.snapshot()
            if snapshot.get('job'):
                raise TimeoutError('Isolated diagnostic budget exceeded; live workbench is untouched')
            summary = json.loads((root / 'runs' / snapshot['last_run']['run_id'] / 'summary.json').read_text())
            feedback = summary['metrics']['tamp']
            row = {'task': task, 'parameters': parameters, 'run_id': snapshot['last_run']['run_id'],
                   'passed': bool(feedback.get('primitive_success')), 'summary': summary, 'snapshot': snapshot}
            report['cases'].append(row)
            print(json.dumps({'task': task, 'passed': row['passed'], 'failure': feedback.get('failure_code'),
                              'phase': feedback.get('phase')}, ensure_ascii=False), flush=True)
            if not row['passed']:
                break
        final = service.snapshot()['diagnostics']['evaluation']
        report['passed'] = (len(report['cases']) == len(sequence) and all(row['passed'] for row in report['cases'])
                            and final['inside_basket'] and final['supported_by_basket'] and final['placement']['confirmed'])
        report['final_evaluation'] = final
    finally:
        service.close()
        (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0 if report.get('passed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
