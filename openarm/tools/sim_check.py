"""One bounded native grasp/hold check, isolated from the live workbench."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/' + time.strftime('%Y%m%d-%H%M%S') + '-grasp-check')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / args.output).resolve()
    if not out.is_relative_to(root / 'runs') or out.exists():
        parser.error('choose a new directory inside runs/')
    out.mkdir(parents=True)
    os.environ['OPENARM_SCENE_OBJECTS'] = '[]'
    from tools.object_workbench import build_backend
    from PIL import Image

    factory, _ = build_backend()
    service = factory()
    report = {'scope': 'nominal bottle grasp and 10-second hold; no LLM or hardware',
              'physical_parameters_changed': False, 'live_workbench_changed': False,
              'physical_calibration_validated': False, 'passed': False}
    try:
        service.call('task_budget', seconds=120)
        service.call('tamp', task='pick_bottle',
                     parameters={'object_id': 'bottle', 'arm': 'right', 'hold_seconds': 10},
                     observation_id=service.snapshot()['observation_id'], realtime=False,
                     initiator={'type': 'isolated-grasp-check'})
        deadline = time.monotonic() + 180
        while service.snapshot().get('job'):
            if time.monotonic() > deadline:
                raise TimeoutError('isolated check exceeded its wall-time budget')
            time.sleep(.2)
        snapshot = service.snapshot()
        run_id = snapshot['last_run']['run_id']
        native = root / 'runs' / run_id
        summary = json.loads((native / 'summary.json').read_text())
        private = json.loads((native / 'internal-summary.json').read_text())
        feedback = summary['metrics']['tamp']
        report.update(run_id=run_id, summary=summary, internal=private,
                      final_diagnostics=snapshot['diagnostics'])
        report['passed'] = bool(feedback.get('primitive_success')
            and private['max_two_finger_hold_seconds']['right'] >= 10
            and private['max_bottle_lift_m'] >= .03
            and not summary['evaluation']['invalid'])
        # Reuse the native recorded operator keyframe, with no new UI widgets.
        frames = sorted(p for p in (native / 'keyframes').glob('*.jpg')
                        if len(p.stem.split('-')) == 2)
        if frames:
            Image.open(frames[-1]).convert('RGB').save(out / 'grasp.jpg', quality=90)
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        service.close()
        (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'run_id': report.get('run_id'),
                      'error': report.get('error'), 'result': str(out / 'result.json')}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
