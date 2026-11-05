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


def apply_capacity(record, capacity, initial):
    """Apply the unchanged measured 4B budget to an observed resource record."""
    assert capacity['complete'] and capacity['decode']['argmax_checked_tokens'] == 256
    assert capacity['decode']['dtype'] == 'torch.bfloat16'
    peak = capacity['peak_process_tree_rss_bytes']
    assert peak > 0
    required_vram = (capacity['decode']['cuda_peak_reserved_bytes'] + (1 << 20) - 1) // (1 << 20) + 1024
    ram_ok = record['host_available_bytes'] + record['current_worker_rss_bytes'] >= peak + RESERVE if initial else record['host_available_bytes'] >= RESERVE
    return dict(record, initial=initial, passed=not record['foreign_gpu_processes'] and ram_ok and
                (not initial or record['free_vram_mib'] >= required_vram), host_capacity_passed=ram_ok,
                measured_host_peak_rss_bytes=peak, host_reserve_bytes=RESERVE,
                required_free_vram_mib=required_vram, model_capacity='Qwen3-4B BF16 fixed-input calibration',
                policy='Measured4B peak plus2GiB host RAM; actual GPU nonpreemption; no CPU-process whitelist')


def wait_for_admission(assets, initial=False, *, capacity=None, observer=None):
    """Wait using an explicit observer; never patch another caller's globals."""
    assets = Path(assets)
    while True:
        selected = observe if observer is None else observer
        record = selected(False if capacity is not None and observer is None else initial)
        if capacity is not None:
            measured = json.loads(Path(capacity).read_text(encoding='utf-8'))
            original_policy = record['policy']
            record = apply_capacity(record, measured, initial)
            if observer is not None:
                record['process_classification_policy'] = original_policy
        (assets / 'resource_admission_latest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if record['passed']:
            if initial:
                (assets / 'resource_admission_initial.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            return record
        print('Waiting for actual GPU vacancy/host RAM', json.dumps(record), flush=True)
        time.sleep(5)


def direction_admission(assets, *, initial=True, wait=False, observer=None):
    """Preserve the direction 14-GiB/foreign-process policy and its explicit recovery wait."""
    from studies._shared.structure.geometry_4b import admission__diagnose_4b_mask_windows
    inspect = admission__diagnose_4b_mask_windows if observer is None else observer
    waiting = False
    path = Path(assets)/'direction_resource_wait.json'
    while True:
        check = inspect(initial=initial)
        if not wait:
            assert check['passed'], f'Nonpreemptive GPU admission deferred: {check}'
            return check
        if check['passed']:
            if waiting:
                path.write_text(json.dumps({'status': 'admitted', 'pid': os.getpid(), **check}, indent=2)+'\n', encoding='utf-8')
            return check
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'status': 'waiting', 'pid': os.getpid(), **check}, indent=2)+'\n', encoding='utf-8')
        if not waiting:
            print('GPU admission waiting:', check, flush=True)
        waiting = True
        time.sleep(5)
