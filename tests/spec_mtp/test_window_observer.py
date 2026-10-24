"""Observation must preserve the real decoder and remove its hooks on failure."""

from spec_mtp.masks import build_mask_provider

import importlib.util
from pathlib import Path

import pytest
import torch

from spec_mtp.decode import SpecDecoder
from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM
from spec_mtp.masks import OutputHiddenMaskProvider

ROOT = Path(__file__).resolve().parents[2]
module_spec = importlib.util.spec_from_file_location('window_observer', ROOT / 'tools/diagnose_output_windows.py')
observer = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(observer)


@pytest.fixture
def model():
    torch.manual_seed(7)
    return HFCausalLM(tiny_config={'vocab_size': 64, 'hidden_size': 32, 'num_layers': 1,
                                  'num_heads': 2, 'num_kv_heads': 1, 'tie_word_embeddings': True})


def spec(order):
    body = {'num_masks': 2, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [1, 1], 'pruning': False}}
    if order is None:
        body.update(method='esp', mask={'init': 'mean_prompt', 'update': False, 'lam': 0})
    else:
        body.update(method='output_hidden', output={'order': order, 'beta': .9, 'beta_prime': .99})
    return _convert(body)


@pytest.mark.parametrize('order', [None, 0, 1])
def test_observation_preserves_tokens_trace_and_temporal_seed(model, order):
    prompt = torch.tensor([[1, 2, 3, 4]])
    cfg = _convert({'max_new_tokens': 16, 'temperature': 0., 'stop_on_eos': False})
    output, stats = SpecDecoder(model, spec(order), cfg, mask_policy=build_mask_provider(spec(order), model.embedding_table)).generate(prompt)
    targets = [t['root_position'] for t in stats['trace'][:3]]
    samples, replay = observer.observe(model, prompt, spec(order), cfg, [],
                                       {'output_ids': output[0].tolist(), 'stats': stats}, targets)
    assert replay['replayed_output_equal'] and replay['semantic_trace_equal']
    assert replay['observer_target_forward_calls'] == replay['expected_target_forward_calls']
    for position, sample in samples.items():
        assert sample['seed_positions'][0] == position
        assert sample['seed_positions'][1] == position + 1
        assert sample['seed_logits'].shape == (2, 64)
        assert sample['seed_mask_inputs'].shape == (2, 32)
        assert torch.isfinite(sample['seed_hidden']).all()
        if position == prompt.shape[1]:
            assert sample['source_root'] is None
        else:
            assert sample['source_root'] < position
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_observer_cleanup_and_provider_restore_on_model_failure(model, monkeypatch):
    prompt = torch.tensor([[1, 2, 3, 4]])
    cfg = _convert({'max_new_tokens': 10, 'temperature': 0., 'stop_on_eos': False})
    output, stats = SpecDecoder(model, spec(1), cfg, mask_policy=build_mask_provider(spec(1), model.embedding_table)).generate(prompt)
    original = OutputHiddenMaskProvider.masks_per_owner
    def fail(*args, **kwargs):
        raise RuntimeError('injected forward failure')
    monkeypatch.setattr(model.hf.model.norm, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        observer.observe(model, prompt, spec(1), cfg, [],
                         {'output_ids': output[0].tolist(), 'stats': stats}, [stats['trace'][0]['root_position']])
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_target_margin_sign_and_best_alternative():
    row = torch.tensor([1., 3., 2.])
    assert observer.margin(row, 1)['target_margin_vs_best_other'] == 1.
    assert observer.margin(row, 2)['target_margin_vs_best_other'] == -1.


def test_crossed_norms_preserve_observed_vectors_and_change_only_magnitude(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    import diagnose_mask_scale
    a, b = torch.tensor([3., 4.]), torch.tensor([0., 10.])
    cases = diagnose_mask_scale.crossed_masks(a, b)
    assert len(cases) == 4
    lookup = {(c['direction'], c['norm_source']): c for c in cases}
    assert torch.equal(lookup['input', 'input']['vector'], a)
    assert torch.equal(lookup['output', 'output']['vector'], b)
    assert torch.equal(lookup['input', 'output']['vector'], 2 * a)
    assert torch.equal(lookup['output', 'input']['vector'], .5 * b)
    with pytest.raises(AssertionError):
        diagnose_mask_scale.crossed_masks(torch.zeros(2), b)
