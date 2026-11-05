"""Scalar resource budgets and explicit observer isolation; no CUDA workloads."""
import json

from studies._shared.structure import resource_admission as admission


def test_explicit_capacity_observer_wait_does_not_mutate_the_shared_observer(tmp_path, monkeypatch):
    capacity = {'complete': True, 'peak_process_tree_rss_bytes': 8 << 30,
        'decode': {'argmax_checked_tokens': 256, 'dtype': 'torch.bfloat16', 'cuda_peak_reserved_bytes': 10 << 30}}
    path = tmp_path / 'capacity.json'
    path.write_text(json.dumps(capacity))
    original = admission.observe
    calls = []
    def observer(initial):
        calls.append(initial)
        return {'host_available_bytes': 9 << 30, 'current_worker_rss_bytes': 1 << 30,
            'foreign_gpu_processes': [{'pid': 2}] if len(calls) == 1 else [],
            'free_vram_mib': 11264, 'policy': 'actual classified CUDA allocation'}
    sleeps = []
    monkeypatch.setattr(admission.time, 'sleep', sleeps.append)
    record = admission.wait_for_admission(tmp_path, initial=True, capacity=path, observer=observer)
    assert calls == [True, True] and sleeps == [5]
    assert admission.observe is original
    assert record['passed'] and record['process_classification_policy'] == 'actual classified CUDA allocation'
    assert json.loads((tmp_path / 'resource_admission_latest.json').read_text()) == record
    assert json.loads((tmp_path / 'resource_admission_initial.json').read_text()) == record


def test_norm_capacity_observes_original_policy_without_the_superseded_8b_initial_budget(tmp_path, monkeypatch):
    capacity = {'complete': True, 'peak_process_tree_rss_bytes': 8 << 30,
        'decode': {'argmax_checked_tokens': 256, 'dtype': 'torch.bfloat16', 'cuda_peak_reserved_bytes': 10 << 30}}
    path = tmp_path / 'capacity.json'; path.write_text(json.dumps(capacity))
    calls = []
    def original(initial):
        calls.append(initial)
        return {'host_available_bytes': 9 << 30, 'current_worker_rss_bytes': 1 << 30,
                'foreign_gpu_processes': [], 'free_vram_mib': 11264, 'policy': 'original'}
    monkeypatch.setattr(admission, 'observe', original)
    record = admission.wait_for_admission(tmp_path, initial=True, capacity=path)
    assert calls == [False] and record['passed']
    assert 'process_classification_policy' not in record
