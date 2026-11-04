"""Nonpreemptive single-GPU admission under Enmao's host-RAM policy."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

PEAK_RSS = 17305047040
RESERVE = 2147483648


def observe(initial=False):
    process = psutil.Process()
    allowed = {os.getpid(), *(p.pid for p in process.parents())}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    used, free, util = map(int, gpu.splitlines()[0].split(','))
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    foreign = []
    for line in rows.splitlines():
        fields = line.split(',', 1)
        if len(fields) == 2 and any(word in fields[1].lower() for word in ('python', 'wsl', 'vmmem', 'ollama', 'llama')):
            if int(fields[0]) not in allowed:
                foreign.append({'pid': int(fields[0]), 'process': fields[1].strip()})
    available = psutil.virtual_memory().available
    current_rss = process.memory_info().rss
    # Initial peak budget includes this worker's already resident host allocation.
    ram_ok = available + current_rss >= PEAK_RSS + RESERVE if initial else available >= RESERVE
    passed = not foreign and ram_ok and (not initial or (free >= 20131 and util <= 20))
    return {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': passed,
            'initial': initial, 'foreign_gpu_processes': foreign, 'free_vram_mib': free,
            'used_vram_mib': used, 'gpu_utilization_percent': util, 'host_available_bytes': available,
            'current_worker_rss_bytes': current_rss, 'measured_host_peak_rss_bytes': PEAK_RSS,
            'host_reserve_bytes': RESERVE, 'host_capacity_passed': ram_ok,
            'policy': 'host measured peak plus2GiB; actual GPU nonpreemption; no CPU-script whitelist'}


def wait_for_admission(assets, initial=False):
    assets = Path(assets)
    while True:
        record = observe(initial)
        (assets / 'resource_admission_latest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if record['passed']:
            if initial:
                (assets / 'resource_admission_initial.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            return record
        print('Waiting for actual GPU vacancy/host RAM', json.dumps(record), flush=True)
        time.sleep(5)
