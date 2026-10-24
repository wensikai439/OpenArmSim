#!/usr/bin/env python3
"""Record a frozen case by rerunning native physics; never reconstruct poses.

The observer camera and encoder do not enter policy inputs. Compare every
recorded action/state with the original trial before labelling a video matched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def transitions(folder, result):
    for stage in result['stages']:
        for line in (folder / stage['native'] / 'transitions.jsonl').read_text().splitlines():
            row = json.loads(line)
            yield {key: row[key] for key in ('observation', 'action', 'next_observation',
                                            'terminated', 'truncated')}


def trace_digest(rows):
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\n')
        count += 1
    return {'sha256': digest.hexdigest(), 'transitions': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--encoder', required=True, type=Path)
    parser.add_argument('--backend', choices=['precision','body','placement','stable'],
                        help='Explicit recorded provider; otherwise infer admitted donor-support evidence')
    args = parser.parse_args()
    source, output = args.trial.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'runs') or output.exists():
        parser.error('output must be a new directory under runs/')
    from PIL import Image, ImageDraw, ImageFont
    original = json.loads((source / 'result.json').read_text())
    backend=args.backend or ('stable' if 'donor_support' in original else
        'body' if json.loads((source/'case.json').read_text()).get('grasp_region')=='body' else 'precision')
    import importlib
    names={'precision':'PrecisionWorld','body':'BodyWorld','placement':'PlacementWorld','stable':'StableWorld'}
    PrecisionWorld=getattr(importlib.import_module('harness.aspire.'+backend),names[backend])
    from harness.aspire.recovery_campaign import run_trial
    from harness.aspire.runtime import dump

    class VideoWorld(PrecisionWorld):
        def __init__(self, case, folder):
            super().__init__(case, folder)
            self.frames = []
            self.next_frame = 1 / 6
            self.font = ImageFont.truetype('DejaVuSans.ttf', 18)
            self.encoder_log = (self.output / 'encoder.log').open('wb')
            self.encoder = subprocess.Popen([
                str(args.encoder), '-hide_banner', '-loglevel', 'warning',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x600',
                '-framerate', '6', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '22', '-pix_fmt', 'yuv420p', '-r', '6', '-fps_mode', 'cfr',
                '-g', '12', '-movflags', '+faststart',
                str(self.output / 'video.mp4')], stdin=subprocess.PIPE, stderr=self.encoder_log)
            self._frame('initial', 'initial')

        def _frame(self, kind, phase):
            canvas = Image.new('RGB', (1280, 600), '#252926')
            canvas.paste(Image.fromarray(self.env.render('operator')), (0, 0))
            for camera, y in (('right', 0), ('left', 270)):
                canvas.paste(Image.fromarray(self.env.render(camera)).resize((320, 240)), (960, y + 30))
            draw = ImageDraw.Draw(canvas)
            for label, y in (('RIGHT WRIST', 5), ('LEFT WRIST', 275)):
                draw.text((974, y), label, font=self.font, fill='#f4f2ed')
            now = float(self.env.data.time)
            draw.text((20, 555), f'{now:05.1f}s   {kind} / {phase}', font=self.font, fill='#f4f2ed')
            held = ', '.join(s for s, value in self.env.evaluation['held'].items() if value) or 'none'
            draw.text((850, 555), f'CONTACT HOLD: {held}', font=self.font, fill='#f4f2ed')
            if not self.frames:
                canvas.save(self.output / 'poster.jpg', quality=90)
            if kind == 'final':
                canvas.save(self.output / 'last.jpg', quality=90)
            self.encoder.stdin.write(canvas.tobytes())
            self.frames.append({'frame': len(self.frames), 'sim_seconds': now,
                                'tick': self.env.tick, 'tool': kind, 'phase': phase})

        def _step(self, action, diagnostic):
            done = super()._step(action, diagnostic)
            if float(self.env.data.time) + 1e-9 >= self.next_frame:
                self._frame(self.active['kind'], diagnostic.get('phase', ''))
                self.next_frame += 1 / 6
            return done

        def close(self):
            try:
                self._frame('final', 'completed' if self.result()['task_success'] else 'stopped')
                self.encoder.stdin.close()
                if self.encoder.wait(timeout=60):
                    raise RuntimeError('video encoder failed; inspect encoder.log')
                dump(self.output / 'frames.json', self.frames)
            finally:
                self.encoder_log.close()
                super().close()

    case = json.loads((source / 'case.json').read_text())
    program = source / 'program-0.py'
    result = run_trial(case, output, program, 'frozen', world_type=VideoWorld)
    old = trace_digest(transitions(source, original))
    new = trace_digest(transitions(output, result))
    stages = lambda r: [{k: s[k] for k in ('tool', 'ok', 'failure_code', 'start_tick', 'end_tick')}
                        for s in r['stages']]
    matched = old == new and stages(original) == stages(result)
    proof = {'schema': 'skill-video/1', 'recording': 'new_native_physics_rerun',
             'original': str(source.relative_to(ROOT)), 'case': case,
             'original_trace': old, 'recorded_trace': new, 'exact_trace_match': matched,
             'original_success': original['task_success'], 'recorded_success': result['task_success'],
             'fps': 6, 'frame_count': len(json.loads((output / 'frames.json').read_text())),
             'playback': '1x simulation time; planning pauses omitted; endpoint retained',
             'observer_views': ['operator', 'right', 'left'], 'hardware_access': False,
             'program_sha256': result['program_sha256'],
             'video_sha256': hashlib.sha256((output / 'video.mp4').read_bytes()).hexdigest()}
    proof['runtime_backend']=backend
    dump(output / 'recording.json', proof)
    print(json.dumps({'recording': str(output), 'exact_trace_match': matched}), flush=True)
    if not matched:
        raise SystemExit('Video differs from original trace; do not label it identical.')


if __name__ == '__main__':
    main()
