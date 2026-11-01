"""studies/_shared/structure/profile_cpu.py runs end to end on a very small model and keeps exact match."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('profile_cpu', ROOT / 'studies' / '_shared' / 'structure' / 'profile_cpu.py')
profile_cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_cpu)


def test_component_buckets():
    assert profile_cpu.component('/x/spec_mtp/structure/algorithm/tree_attention.py') == 'tree attention'
    assert profile_cpu.component('/x/site-packages/torch/nn/modules/linear.py').startswith('model forward')
    assert profile_cpu.component('~', "<built-in method torch.cat>") == 'torch kernels (C calls)'
    assert profile_cpu.component('~', '<built-in method builtins.len>') == 'other builtins'
    assert profile_cpu.component('/x/y.py') == 'other Python'


def test_profile_small(tmp_path):
    code = profile_cpu.main(['--variant', 'esp_static', '--variant', 'qwen3/esp_bc30', '--hidden', '32',
                             '--layers', '1', '--vocab', '64', '--prompts', '1', '--new-tokens', '8',
                             '--out', str(tmp_path)])
    report = json.loads((tmp_path / 'profile.json').read_text(encoding='utf-8'))
    assert code == 0
    assert [r['exact_match_rate'] for r in report['results']] == [1.0, 1.0]
    for r in report['results']:
        assert abs(sum(r['self_time_share'].values()) - 1.0) < 0.01
    assert '| `qwen3/esp_bc30` |' in (tmp_path / 'profile.md').read_text(encoding='utf-8')
