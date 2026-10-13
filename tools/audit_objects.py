"""Read-only live-state/source audit; prune only fully covered own progress files."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261009-objects'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name): return json.loads((OUT / name).read_text())
def get(port, endpoint):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/{endpoint}', timeout=10) as response:
        return json.load(response)


def main():
    target = OUT / 'audit.json'
    if target.exists(): raise ValueError('preserve prior audit')
    sources = []
    for protocol, folder in [('protocol.json', 'source'), ('api.protocol.json', 'source/api'),
        ('workbench.protocol.json', 'source/workbench')]:
        for name, expected in read(protocol)['source_sha256'].items():
            frozen = OUT / folder / name
            match = digest(frozen) == expected
            if not match: raise ValueError('frozen experiment source changed: ' + str(frozen))
            sources.append({'protocol': protocol, 'path': name, 'frozen_sha256': expected,
                'frozen_matches_protocol': match, 'current_matches_frozen': digest(ROOT / name) == expected})
    for name in ['tools/audit_objects.py', 'tools/report_objects.py', 'tools/check_objects_ui.py',
        'tools/check_object_score.py', 'tools/check_object_contract.py',
        'tools/check_object_score_contract.py', 'README.md']:
        frozen = OUT / 'source/qa' / name; frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists(): raise ValueError('preserve prior QA source')
        frozen.write_bytes((ROOT / name).read_bytes())
        sources.append({'path': name, 'sha256': digest(frozen), 'capture': 'post-task QA/report source, not predeclared experiment code'})
    api = read('api.json'); ui = read('workbench.json'); verified = read('workbench-checked.json')
    assert digest(OUT / verified['original_input']) == verified['input_sha256']
    assert digest(ROOT / 'tools/check_object_score.py') == verified['scorer_sha256']
    from tools.check_object_score import score
    assert score(ui, read('workbench.protocol.json'))['passed']
    services = {}
    for port, expected_id in [(8765, '94cbe546b265:319'), (8766, '4fc9e7c32b75:640'), (8767, ui['terminal']['observation_id'])]:
        state = get(port, 'status'); manifest = get(port, 'manifest')
        assert state['observation_id'] == expected_id and not state.get('job') and not state['agent']['busy']
        assert manifest['source_status']['matches_disk'] and not manifest['source_status']['changed_files']
        assert not manifest['model_evidence']['runtime_parameters']['changed_fields']
        services[str(port)] = {'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
            'state': state['state'], 'idle': True, 'manifest': manifest,
            'scope': 'runtime guard covers only sources captured by this process at import; new injected modules are separately frozen'}
    models = []
    for case in read('results.json')['cases']:
        models.append(read(case['case']['name'] + '/initial.json')['manifest']['model_evidence']['compiled']['sha256'])
        assert case['model_evidence_unchanged']
    for record in [api, ui]:
        for receipt in record['receipts']:
            manifest = json.loads((ROOT / 'runs' / receipt['evidence']['run_id'] / 'manifest.json').read_text())
            models.append(manifest['model_evidence']['compiled']['sha256'])
    models.append(services['8767']['manifest']['model_evidence']['compiled']['sha256'])
    assert len(set(models)) == 1
    cleanup = []
    for progress_name, record in [('api.progress.json', api), ('workbench.progress.json', ui)]:
        path = OUT / progress_name; progress = json.loads(path.read_text())
        final_turns = {t['id']: t for t in record['turns']}
        assert progress['turn_id'] == record['turn_id']
        assert record['stage_history'][:len(progress['stage_history'])] == progress['stage_history']
        for turn in progress['turns']:
            final = final_turns[turn['id']]; events = {e['id']: e for e in final['trace']}
            for event in turn['trace']:
                assert event['id'] in events
                if event['status'] not in ('running', 'waiting'):
                    assert events[event['id']] == event
        cleanup.append({'path': progress_name, 'sha256': digest(path), 'bytes': path.stat().st_size,
            'reason': 'own intermediate cache; final retains turn/event IDs, all completed event details and complete stage-history prefix'})
    result = {'scope': 'declared upright primitive development cases; not arbitrary objects or physical calibration',
        'goal_status': 'active; general cuRobo carrying, new-shape handover and physical calibration remain',
        'source_audit': sources, 'services': services, 'same_compiled_multi_object_model_across_native_API_UI': True,
        'multi_object_compiled_sha256': models[0], 'raw_records_not_overwritten': True,
        'scoring_input_sha256': verified['input_sha256'], 'thresholds_changed': False,
        'progress_cache_cleanup': cleanup, 'company_SSH_commands': 0, 'hardware_access': False,
        'transfer_ready': False, 'new_automations': 0, 'fixed_demo_compatible_but_task_failed': read('legacy.json')}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for item in cleanup: (OUT / item['path']).unlink()
    print(json.dumps({'source_protocols_verified': True, 'services_idle_and_expected_observations': True,
        'compiled_model_identical': True, 'own_redundant_progress_removed': [item['path'] for item in cleanup]}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
