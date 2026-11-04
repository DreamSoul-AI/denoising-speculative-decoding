"""Host RAM, not the superseded VRAM margin, governs nonofficial admission."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def resource_module():
    path = ROOT / 'studies/esp_table1_nonofficial_revalidation/resource_admission.py'
    spec = importlib.util.spec_from_file_location('sd_resource_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_peak_plus_margin_and_actual_foreign_gpu(monkeypatch):
    m = resource_module()
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


def test_copied_untimed_and_argmax_helpers_are_original_bytes():
    for name in ('untimed_bf16.py', 'argmax_audit.py'):
        assert (ROOT / 'studies/esp_table1_nonofficial_revalidation' / name).read_bytes() == (
            ROOT / 'studies/esp_table1_bf16_full_argmax' / name).read_bytes()
