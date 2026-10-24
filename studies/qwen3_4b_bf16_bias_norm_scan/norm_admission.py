"""Use the measured4B capacity, retaining actual GPU nonpreemption."""
import json
from pathlib import Path
from resource_admission import observe as original_observe
from resource_admission import wait_for_admission as original_wait
from unittest.mock import patch
import resource_admission

CAPACITY = Path(__file__).resolve().parent / 'docs/capacity_calibration_v1.json'
RESERVE = 2147483648


def apply_capacity(record, capacity, initial):
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


def observe(initial=False):
    capacity = json.loads(CAPACITY.read_text(encoding='utf-8'))
    return apply_capacity(original_observe(False), capacity, initial)


def wait_for_admission(assets, initial=False):
    with patch.object(resource_admission, 'observe', observe):
        return original_wait(assets, initial=initial)
