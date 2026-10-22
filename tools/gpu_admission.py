"""Read-only CUDA/display classification; no process-control APIs.

Usage: python tools/gpu_admission.py (import-only helper; no admission or process mutation).
Call observe() through the prepared Study admission wrapper or queue.
"""
import datetime
import os
from pathlib import PureWindowsPath
import subprocess
import xml.etree.ElementTree as ET

import psutil

DISPLAY = frozenset(('figma.exe', 'chrome.exe', 'msedge.exe', 'firefox.exe', '360se.exe',
                     'msedgewebview2.exe', 'dwm.exe', 'explorer.exe', 'winword.exe', 'excel.exe',
                     'powerpnt.exe', 'outlook.exe', 'onenote.exe', 'raylink.exe', 'raylinkcapturer.exe',
                     'todesk.exe', 'sunlogin.exe', 'sunloginclient.exe', 'parsec.exe', 'parsecd.exe',
                     'teamviewer.exe', 'teamviewer_service.exe'))
CUDA_RUNTIMES = frozenset(('python.exe', 'pythonw.exe', 'python', 'python3', 'wsl.exe', 'vmmemwsl',
                           'ollama.exe', 'ollama', 'llama-server.exe', 'llama-server'))


def memory_mib(value):
    if value is None or str(value).strip() in ('N/A', '[N/A]', ''):
        return None
    parts = str(value).split()
    factor = {'MiB': 1, 'GiB': 1024}.get(parts[1] if len(parts) > 1 else 'MiB')
    assert factor is not None, 'Unrecognized NVIDIA memory unit'
    number = float(parts[0]) * factor
    assert 0 <= number < float('inf')
    return number


def classify(row, allowed=()):
    result = dict(row, blocking=False)
    name = PureWindowsPath(row['process_name']).name.lower()
    kind, memory = row['type'], row.get('cuda_memory_mib')
    if row['pid'] in allowed:
        category = 'own_process'
    elif kind == 'G' or (name in DISPLAY and 'G' in kind and (memory is None or memory <= 16)):
        category = 'display'
    elif memory is not None and memory > 0 and any(t in kind for t in ('C', 'M')):
        category = 'cuda_compute'
        result['blocking'] = True
    elif kind in ('C', 'M', 'M+C') and name in CUDA_RUNTIMES and memory is None:
        # WDDM hides per-process bytes; a known runtime's pure CUDA context still defers another experiment.
        category = 'known_cuda_context_memory_unavailable'
        result['blocking'] = True
    else:
        category = 'unconfirmed_no_cuda_allocation_evidence'
    return dict(result, category=category)


def parse_xml(text, allowed=()):
    gpu = ET.fromstring(text).find('gpu')
    assert gpu is not None
    rows = []
    for p in gpu.findall('./processes/process_info'):
        kind = p.findtext('type', '')
        used = memory_mib(p.findtext('used_memory'))
        row = dict(pid=int(p.findtext('pid')), process_name=p.findtext('process_name', ''), type=kind,
                   gpu_memory_mib=used, cuda_memory_mib=used if kind in ('C', 'M', 'M+C') else None)
        rows.append(classify(row, allowed))
    return dict(free_vram_mib=memory_mib(gpu.findtext('./fb_memory_usage/free')),
                used_vram_mib=memory_mib(gpu.findtext('./fb_memory_usage/used')),
                gpu_utilization_percent=int(gpu.findtext('./utilization/gpu_util').split()[0]),
                classified_processes=rows, foreign_gpu_processes=[r for r in rows if r['blocking']],
                unconfirmed_processes=[r for r in rows if r['category'].startswith('unconfirmed')])


def observe(initial=False):
    process = psutil.Process()
    allowed = {os.getpid(), *(p.pid for p in process.parents())}
    result = parse_xml(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=10), allowed)
    assert result['free_vram_mib'] is not None
    return dict(result, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), initial=initial,
                host_available_bytes=psutil.virtual_memory().available, current_worker_rss_bytes=process.memory_info().rss,
                passed=not result['foreign_gpu_processes'], policy='Whitelist plus NVIDIA process type; unknown allocation requires positive CUDA memory evidence; desktop contexts never paused/killed')
