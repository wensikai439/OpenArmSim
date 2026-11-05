"""Sample only this platform's process tree and report whole selected-GPU telemetry."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]


def sample(seconds):
    record = json.loads((ROOT / '.runtime/server.json').read_text())
    parent = psutil.Process(record['pid'])
    processes = [parent, *parent.children(recursive=True)]
    for process in processes:
        try: process.cpu_percent()
        except psutil.NoSuchProcess: pass
    rows = []
    for _ in range(seconds):
        time.sleep(1)
        cpu = 0; rss = 0; pss = 0; details = []
        for process in processes:
            try:
                percent = process.cpu_percent(); memory = process.memory_info().rss
                cpu += percent; rss += memory
                pss += getattr(process.memory_full_info(), 'pss', memory)
                details.append({'pid': process.pid, 'cpu_one_core_percent': percent,
                                'rss_mib': round(memory / 2**20, 1), 'command': process.cmdline()})
            except psutil.NoSuchProcess: pass
        result = subprocess.check_output(['nvidia-smi', '--id=1',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = [float(value.strip()) for value in result.strip().split(',')]
        rows.append({'cpu_one_core_percent': cpu, 'rss_mib': rss / 2**20, 'pss_mib': pss / 2**20,
                     'gpu_total_mib': memory, 'gpu_utilization_percent': utilization, 'processes': details})
    return {'seconds': seconds, 'logical_cpus': psutil.cpu_count(),
        'cpu_mean_one_core_percent': round(statistics.mean(row['cpu_one_core_percent'] for row in rows), 1),
        'cpu_peak_one_core_percent': round(max(row['cpu_one_core_percent'] for row in rows), 1),
        'rss_peak_mib': round(max(row['rss_mib'] for row in rows), 1),
        'pss_peak_mib': round(max(row['pss_mib'] for row in rows), 1),
        'gpu_total_peak_mib': max(row['gpu_total_mib'] for row in rows),
        'gpu_utilization_peak_percent': max(row['gpu_utilization_percent'] for row in rows),
        'gpu_scope': 'selected physical GPU 1 total; shared-host usage can include other workloads', 'samples': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=15)
    parser.add_argument('--output', default='resources-idle.json')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300: parser.error('seconds must be 1..300')
    if Path(args.output).name != args.output: parser.error('output must be a filename')
    result = sample(args.seconds)
    destination = ROOT / '.runtime' / args.output
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, ensure_ascii=False))
