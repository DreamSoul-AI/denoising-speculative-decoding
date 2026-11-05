
from spec_mtp.structure.algorithm.mask import build_mask_provider
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from spec_mtp.structure.algorithm.mask import EMAVelocityMaskProvider, ESPMaskProvider

ROOT = Path(__file__).resolve().parents[2]


def tool(monkeypatch):
    from types import SimpleNamespace
    from studies._shared.structure.geometry_4b import direction_control
    return SimpleNamespace(control=direction_control)


def cuda_device():
    assert torch.cuda.is_available(), 'CUDA is required; no CPU substitute'
    device = torch.device('cuda:0')
    print(f'direction correctness device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    return device


@pytest.mark.parametrize('mode', ['template', 'random0', 'random1', 'random2', 'content'])
def test_full_direction_preserves_native_mean_and_declared_velocity(monkeypatch, mode):
    m = tool(monkeypatch)
    device = cuda_device()
    e = torch.randn(9, 16, generator=torch.Generator().manual_seed(7)).bfloat16().to(device)
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
        cpu_vector = torch.randn(e.shape[-1], generator=torch.Generator().manual_seed(int(mode[-1])))
        cpu_vector *= (e[1]-e[0]).float().norm().cpu()/cpu_vector.norm()
        assert torch.equal(velocity, cpu_vector.to(device=e.device, dtype=e.dtype))
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


@pytest.mark.parametrize('mode', ['template', 'random0', 'random1', 'random2', 'content'])
def test_native_tiny_bf16_direction_controls_remain_argmax_safe(monkeypatch, mode):
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from studies._shared.structure.config import _convert
    from types import SimpleNamespace
    m = tool(monkeypatch)
    torch.manual_seed(17)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2,
        'max_position_embeddings': 128}, torch_dtype='bfloat16', attn_implementation='eager').to(cuda_device())
    prompt = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
    spec = _convert({'method': 'ema_velocity', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
        'tree': {'mode': 'dynamic', 'pruning': True}, 'ema': {'level': 'order0', 'beta': .9999, 'lam': .1}})
    original = build_mask_provider
    evaluation = _convert({'temperature': 0., 'max_new_tokens': 12})
    expected, timed = decoding.SpecDecoder(model, spec, evaluation,
        mask_policy=m.control(original(spec, model.embedding_table), mode, 2, {})).generate(prompt)
    monkeypatch.setattr(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.))
    output, stats = decoding.SpecDecoder(model, spec, evaluation,
        mask_policy=m.control(original(spec, model.embedding_table), mode, 2, {})).generate(prompt)
    assert torch.equal(expected, output)
    clocks = {'prefill_time', 'decode_time', 'draft_seconds', 'verify_seconds'}
    assert {k:v for k,v in timed.items() if k not in clocks} == {k:v for k,v in stats.items() if k not in clocks}
    assert output.shape[1] == 17 and stats['verification_argmax_audit']['passed']
    assert stats['verification_argmax_audit']['checked_committed_tokens'] == 12


def test_tau_pairs_use_ratios_and_category_pooling(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    existing_readout = sys.modules.get('analyze_results')
    spec = importlib.util.spec_from_file_location('full_direction_analysis', ROOT / 'studies/_shared/structure/analyze_4b_full_directions.py')
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
