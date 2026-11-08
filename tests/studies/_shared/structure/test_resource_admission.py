"""Scalar resource policy, explicit observer isolation and optimized-mode guards.

NVIDIA/process observations and sleeps are stubbed; no GPU/model workload.
"""
import json
import sys
from types import SimpleNamespace

import pytest

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


def test_host_peak_plus_margin_and_actual_foreign_gpu(monkeypatch):
    m = admission
    pid = m.os.getpid()
    own = SimpleNamespace(pid=pid, parents=lambda: [], memory_info=lambda: SimpleNamespace(rss=1000))
    monkeypatch.setattr(m.psutil, 'Process', lambda: own)
    available = [m.PEAK_RSS + m.RESERVE - 1000]
    monkeypatch.setattr(m.psutil, 'virtual_memory', lambda: SimpleNamespace(available=available[0]))
    foreign = [False]
    def query(argv, **kwargs):
        if 'memory.used,memory.free,utilization.gpu' in argv[1]:
            return '2500, 21500, 3\n'
        return '987654, python.exe\n' if foreign[0] else '7, dwm.exe\n'
    monkeypatch.setattr(m.subprocess, 'check_output', query)
    assert m.observe(initial=True)['passed']  # No obsolete +2GiB VRAM margin.
    available[0] -= 1
    assert not m.observe(initial=True)['host_capacity_passed']
    available[0] += 1
    foreign[0] = True
    assert not m.observe(initial=True)['passed']
    assert not m.observe(initial=False)['passed']


def test_measured_4b_admission_keeps_reserve_and_gpu_exclusion(monkeypatch):
    helper = admission
    cap = {'complete': True, 'peak_process_tree_rss_bytes': 8 << 30,
           'decode': {'argmax_checked_tokens': 256, 'dtype': 'torch.bfloat16', 'cuda_peak_reserved_bytes': 10 << 30}}
    base = dict(host_available_bytes=9 << 30, current_worker_rss_bytes=1 << 30,
                foreign_gpu_processes=[], free_vram_mib=11264)
    assert helper.apply_capacity(base, cap, True)['passed']
    assert not helper.apply_capacity(dict(base, host_available_bytes=(9 << 30)-1), cap, True)['passed']
    assert not helper.apply_capacity(dict(base, foreign_gpu_processes=[{'pid': 1}]), cap, True)['passed']
    assert not helper.apply_capacity(dict(base, free_vram_mib=11263), cap, True)['passed']
    assert not helper.apply_capacity(dict(base, host_available_bytes=(2 << 30)-1), cap, False)['passed']
    with pytest.raises(AssertionError):
        helper.apply_capacity(base, dict(cap, complete=False), True)


@pytest.mark.parametrize('initial', [False, True])
def test_recovery_waits_without_restarting_or_changing_controls(monkeypatch, tmp_path, initial):
    from types import SimpleNamespace
    from studies._shared.structure import resource_admission as m
    calls, sleeps = [], []
    checks = iter([{'passed': False, 'foreign_gpu_python_pids': [42]},
                   {'passed': False, 'foreign_gpu_python_pids': [43]},
                   {'passed': True, 'foreign_gpu_python_pids': []}])
    def admission(initial):
        calls.append(initial)
        return next(checks)
    monkeypatch.setattr(m, 'time', SimpleNamespace(sleep=sleeps.append))
    assert m.direction_admission(tmp_path, initial=initial, wait=True, observer=admission)['passed']
    assert calls == [initial] * 3 and sleeps == [5, 5]
    assert json.loads((tmp_path / 'direction_resource_wait.json').read_text())['status'] == 'admitted'


@pytest.mark.parametrize('passed', [False, True])
def test_ordinary_direction_admission_is_one_observation_without_wait_output(tmp_path, monkeypatch, passed):
    calls = []
    monkeypatch.setattr(admission.time, 'sleep', lambda seconds: pytest.fail('ordinary policy never waits'))
    def observe(initial):
        calls.append(initial)
        return {'passed': passed}
    if passed:
        assert admission.direction_admission(tmp_path, observer=observe)['passed']
    else:
        with pytest.raises(AssertionError, match='deferred'):
            admission.direction_admission(tmp_path, observer=observe)
    assert calls == [True] and not (tmp_path/'direction_resource_wait.json').exists()


_OPTIMIZED_CHILD = r'''import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

if __debug__:
    raise SystemExit('the child must use optimized Python')
case, assets = sys.argv[1], Path(sys.argv[2])
reject = case not in ('direction_valid', 'capacity_valid')

def forbidden(*args, **kwargs):
    raise RuntimeError('hardware observation, sleep or model import is forbidden')

geometry = ModuleType('studies._shared.structure.geometry_4b')
geometry.admission__diagnose_4b_mask_windows = forbidden
sys.modules[geometry.__name__] = geometry
from studies._shared.structure import resource_admission as policy
policy.observe = forbidden
policy.subprocess = SimpleNamespace(check_output=forbidden)
policy.time = SimpleNamespace(sleep=forbidden)
calls = []

def observer(initial):
    calls.append(initial)
    return {'passed': case == 'direction_valid'}

record = dict(host_available_bytes=9 << 30, current_worker_rss_bytes=1 << 30,
              foreign_gpu_processes=[], free_vram_mib=11264)
capacity = {'complete': True, 'peak_process_tree_rss_bytes': 8 << 30,
            'decode': {'argmax_checked_tokens': 256, 'dtype': 'torch.bfloat16',
                       'cuda_peak_reserved_bytes': 10 << 30}}
if case == 'capacity_incomplete': capacity['complete'] = False
elif case == 'capacity_wrong_argmax': capacity['decode']['argmax_checked_tokens'] = 255
elif case == 'capacity_wrong_dtype': capacity['decode']['dtype'] = 'torch.float32'
elif case == 'capacity_nonpositive_peak': capacity['peak_process_tree_rss_bytes'] = 0

try:
    if case.startswith('direction_'):
        result = policy.direction_admission(assets, initial=True, wait=False, observer=observer)
    else:
        result = policy.apply_capacity(record, capacity, True)
except AssertionError as error:
    if not reject:
        raise SystemExit('a valid control was rejected')
    if case == 'direction_rejected' and str(error) != "Nonpreemptive GPU admission deferred: {'passed': False}":
        raise SystemExit('ordinary rejection changed its diagnostic')
else:
    if reject:
        raise SystemExit('resource admission accepted an invalid optimized-mode input')
    if result.get('passed') is not True:
        raise SystemExit('valid admission did not return passed=True')
if calls != ([True] if case.startswith('direction_') else []):
    raise SystemExit('observer calls or initial policy changed')
if assets.exists() or 'torch' in sys.modules:
    raise SystemExit('optimized scalar check wrote assets or imported Torch')
print(json.dumps({'case': case, 'rejected': reject, 'optimized': True}))
'''


@pytest.mark.parametrize('case', [
    'direction_valid', 'direction_rejected', 'capacity_valid', 'capacity_incomplete',
    'capacity_wrong_argmax', 'capacity_wrong_dtype', 'capacity_nonpositive_peak',
])
def test_optimized_python_keeps_resource_rejection_and_valid_controls(tmp_path, case):
    from tests import _process_tree

    script = _OPTIMIZED_CHILD
    assets = tmp_path/'admission-assets'
    result = _process_tree.sh([sys.executable, '-B', '-O', '-c', script, case, str(assets)],
        timeout=15, capture_output=True, text=True, encoding='utf-8', check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'case': case,
        'rejected': case not in ('direction_valid', 'capacity_valid'), 'optimized': True}
    assert not assets.exists()
