
from spec_mtp.masks import build_mask_provider
import importlib.util
import ast
import json
from pathlib import Path
import sys

import pytest
import torch

from spec_mtp.masks import EMAVelocityMaskProvider, ESPMaskProvider

ROOT = Path(__file__).resolve().parents[2]


def tool(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    spec = importlib.util.spec_from_file_location('full_direction_control', ROOT / 'studies/qwen3_4b_bf16_directions/direction_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['template', 'random0', 'random1', 'random2', 'content'])
def test_full_direction_preserves_native_mean_and_declared_velocity(monkeypatch, mode):
    m = tool(monkeypatch)
    e = torch.randn(9, 16, generator=torch.Generator().manual_seed(7)).bfloat16()
    provider = EMAVelocityMaskProvider(2, e, beta=.9999, level='order0', lam=.1)
    info = {}
    m.control(provider, mode, 3, info).init_from_prompt(e[:7])
    zero = ESPMaskProvider(2, e, lam=.1)
    zero.init_from_prompt(e[:7])
    velocity = provider._velocity.clone()
    assert torch.equal(provider._level, zero.masks()[0])
    if mode == 'template':
        assert torch.equal(velocity, e[1] - e[0])
    if mode.startswith('random'):
        assert abs(info['norm_ratio'] - 1) < .01
        repeat = EMAVelocityMaskProvider(2, e, beta=.9999, level='order0', lam=.1)
        m.control(repeat, mode, 3, {}).init_from_prompt(e[:7])
        assert torch.equal(velocity, repeat._velocity)
    if mode == 'content':
        expected = EMAVelocityMaskProvider(2, e, beta=.9999, level='order0', lam=.1)
        expected.init_from_prompt(e[3:7])
        assert torch.equal(velocity, expected._velocity)
    previous = e[6]
    for new in e[7:]:
        provider.on_token_committed(new, previous)
        zero.on_token_committed(new, previous)
        assert torch.equal(provider._level, zero.masks()[0])
        if mode != 'content':
            assert torch.equal(provider._velocity, velocity)
        previous = new


def test_native_tiny_bf16_direction_controls_remain_argmax_safe(monkeypatch):
    import spec_mtp.decode as decoding
    from spec_mtp.model import HFCausalLM
    from studies._shared.structure.config import _convert
    from types import SimpleNamespace
    m = tool(monkeypatch)
    torch.manual_seed(17)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2,
        'max_position_embeddings': 128}, torch_dtype='bfloat16', attn_implementation='eager')
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    spec = _convert({'method': 'ema_velocity', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
        'tree': {'mode': 'dynamic', 'pruning': True}, 'ema': {'level': 'order0', 'beta': .9999, 'lam': .1}})
    original = build_mask_provider
    monkeypatch.setattr(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.))
    for mode in ['template', 'random0', 'random1', 'random2', 'content']:
        policy = m.control(original(spec, model.embedding_table), mode, 2, {})
        output, stats = decoding.SpecDecoder(model, spec, _convert({'temperature': 0., 'max_new_tokens': 12}), mask_policy=policy).generate(prompt)
        assert output.shape[1] == 17 and stats['verification_argmax_audit']['passed']
        assert stats['verification_argmax_audit']['checked_committed_tokens'] == 12


def test_tau_pairs_use_ratios_and_category_pooling(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    existing_readout = sys.modules.get('analyze_results')
    spec = importlib.util.spec_from_file_location('full_direction_analysis', ROOT / 'tools/analyze_4b_full_directions.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert sys.modules.get('analyze_results') is existing_readout
    def row(qid, category, committed, decode_calls):
        return {'question_id': qid, 'category': category, 'accepted': committed // 2,
                'model_calls': decode_calls + 1, 'new_tokens': committed + 1,
                'committed': committed, 'decode_calls': decode_calls}
    first = [row(1, 'a', 1, 2), row(2, 'a', 6, 4), row(3, 'b', 2, 2)]
    second = [row(1, 'a', 2, 4), row(2, 'a', 4, 4), row(3, 'b', 4, 2)]
    assert m.compare(first, second)['tau'] == {'win': 1, 'tie': 1, 'loss': 1}
    assert m.compare(second, first)['tau'] == {'win': 1, 'tie': 1, 'loss': 1}
    assert m.category_mean(first) == pytest.approx(13 / 12)
    assert m.category_mean(first) != sum(r['committed'] / r['decode_calls'] for r in first) / 3
    for error in ('Foreign GPU work appeared: 42', 'Nonpreemptive GPU admission deferred'):
        m.require_resource_failure({'native_status': 'failed'}, error)
        for status in ('succeeded', 'running', 'pending'):
            with pytest.raises(AssertionError, match='Never replace'):
                m.require_resource_failure({'native_status': status}, error)
    with pytest.raises(AssertionError, match='argmax mismatch'):
        m.require_resource_failure({'native_status': 'failed'}, 'argmax mismatch')


@pytest.mark.parametrize('initial', [False, True])
def test_recovery_waits_without_restarting_or_changing_controls(monkeypatch, tmp_path, initial):
    from types import SimpleNamespace
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    path = ROOT / 'studies/qwen3_4b_bf16_directions_recovery/direction_control.py'
    spec = importlib.util.spec_from_file_location('recovery_direction_control', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    original = ROOT / 'studies/qwen3_4b_bf16_directions/direction_control.py'
    def control_source(file):
        source = file.read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'control')
        return ast.get_source_segment(source, node)
    assert control_source(path) == control_source(original)
    calls, sleeps = [], []
    checks = iter([{'passed': False, 'foreign_gpu_python_pids': [42]},
                   {'passed': False, 'foreign_gpu_python_pids': [43]},
                   {'passed': True, 'foreign_gpu_python_pids': []}])
    def admission(initial):
        calls.append(initial)
        return next(checks)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'admission', admission)
    monkeypatch.setattr(m, 'time', SimpleNamespace(sleep=sleeps.append))
    assert m.wait_admission(initial=initial)['passed']
    assert calls == [initial] * 3 and sleeps == [5, 5]
    assert json.loads((tmp_path / '.tmp/full-direction-resource-wait.json').read_text())['status'] == 'admitted'
