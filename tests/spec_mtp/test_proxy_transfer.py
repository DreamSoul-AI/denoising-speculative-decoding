"""The one-vector diagnostic must preserve losslessness and clean up its patch."""

from spec_mtp.masks import build_mask_provider

import pytest
import torch

from spec_mtp.decode import SpecDecoder
from spec_mtp.masks import OutputHiddenMaskProvider
from test_window_observer import ROOT, model as model, observer, spec


def baseline(model):
    from studies._shared.structure.config import _convert
    prompt = torch.tensor([[1, 2, 3, 4]])
    cfg = _convert({'max_new_tokens': 16, 'temperature': 0., 'stop_on_eos': False})
    output, stats = SpecDecoder(model, spec(1), cfg, mask_policy=build_mask_provider(spec(1), model.embedding_table)).generate(prompt)
    reference = {'output_ids': output[0].tolist(), 'stats': stats}
    targets = [t['root_position'] for t in stats['trace']]
    root = targets[1]
    samples, _ = observer.observe(model, prompt, spec(1), cfg, [], reference, [root])
    return prompt, cfg, reference, targets, root, samples[root]['current_masks']


@pytest.mark.parametrize('placebo', [True, False])
def test_one_slot_scope_lossless_actual_owner_and_placebo(model, monkeypatch, placebo):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_proxy_transfer import transfer
    prompt, cfg, reference, targets, root, masks = baseline(model)
    donor = masks[0, 1].clone() if placebo else masks[0, 1].roll(1)
    samples, result, intervention = transfer(model, prompt, spec(1), cfg, [], reference,
                                           donor, masks, root, targets, placebo=placebo)
    assert result['output_equal_independent_AR'] and result['pre_intervention_trace_equal']
    assert result['modified_provider_calls'] == 1 and intervention['actual_root'] == root
    assert result['observer_forward_calls'] == 1 + result['stats']['decode_calls']
    changed = samples[root]['current_masks']
    assert torch.equal(changed[0, 1], donor)
    assert torch.equal(changed[0, 0], masks[0, 0]) and torch.equal(changed[1:], masks[1:])
    if placebo:
        assert result['placebo_full_semantic_equal'] and not result['unvisited_targets']
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_transfer_cleanup_after_intervention_forward_failure(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_proxy_transfer import transfer
    prompt, cfg, reference, targets, root, masks = baseline(model)
    original_provider = OutputHiddenMaskProvider.masks_per_owner
    original_norm = model.hf.model.norm.forward
    calls = 0
    target_call = next(i + 2 for i, t in enumerate(reference['stats']['trace']) if t['root_position'] == root)
    def fail_at_intervention(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == target_call:
            raise RuntimeError('injected after transfer')
        return original_norm(*args, **kwargs)
    monkeypatch.setattr(model.hf.model.norm, 'forward', fail_at_intervention)
    with pytest.raises(RuntimeError, match='injected after transfer'):
        transfer(model, prompt, spec(1), cfg, [], reference, masks[0, 1], masks, root, targets, placebo=True)
    assert OutputHiddenMaskProvider.masks_per_owner is original_provider
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks
