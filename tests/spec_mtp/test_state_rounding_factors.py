"""Arithmetic precision factors and selective state patches stay independently scoped."""

import pytest
import torch

from test_proxy_transfer import baseline
from test_window_observer import ROOT, model as model, spec


def test_phi_native_endpoint_and_scope(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_rounding_factors import phi_factors
    slots = torch.tensor([[.1, 2., -.3], [.9, -.5, 1.7]])
    scale = torch.tensor(.925)
    before = slots.clone()
    values, _ = phi_factors(slots, scale)
    assert len(values) == 8 and torch.equal(slots, before)
    assert torch.equal(values['000'], (scale * slots / slots.norm(dim=-1, keepdim=True))[1])
    exact = (scale.double() * slots[1].double() / slots[1].double().norm()).float()
    assert torch.equal(values['111'], exact)


def test_rms_native_endpoint_and_inputs_held(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_rounding_factors import rms_factors
    inputs = torch.arange(27, dtype=torch.float32).reshape(1, 9, 3) / 7
    weight = torch.tensor([.7, 1., 1.5])
    epsilon = torch.tensor(1e-6)
    before = inputs.clone()
    value, _ = rms_factors(inputs, weight, epsilon, (0, 0, 0, 0, 0))
    row = inputs[:, 4:5]
    expected = weight * (row * torch.rsqrt(row.pow(2).mean(-1, keepdim=True) + epsilon))
    assert torch.equal(value, expected) and torch.equal(inputs, before)
    with pytest.raises(AssertionError):
        rms_factors(inputs, weight, epsilon, (0, 0))


@pytest.mark.parametrize('coordinate', ['anchor', 'velocity'])
def test_selective_state_real_cache_and_cleanup(model, monkeypatch, coordinate):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_numerics import canonical_states
    from diagnose_q241_state_factors import selective_transfer
    prompt, cfg, reference, targets, root, masks = baseline(model)
    ids = torch.tensor([reference['output_ids']])
    teacher = model(model.embed(ids), torch.arange(ids.shape[1])[None], last_hidden_only=True)
    states = canonical_states(teacher.last_hidden_state[0], model.embedding_table, 1)
    samples, replay, _, records = selective_transfer(model, states, coordinate, prompt=prompt, spec=spec(1),
        eval_cfg=cfg, eos=[], reference=reference, donor=masks[0, 1], original_masks=masks,
        root=root, targets=targets, placebo=False)
    assert replay['output_equal_independent_AR'] and records
    assert all(r['other_state_bit_equal'] and r['incoming_proxies_unchanged'] for r in records)
    for position, sample in samples.items():
        if position > root:
            assert torch.equal(sample['current_state'][coordinate], states[position][coordinate])
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_selective_cleanup_on_failure(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q241_state_factors import selective_transfer
    from spec_mtp.masks import OutputHiddenMaskProvider
    prompt, cfg, reference, targets, root, masks = baseline(model)
    original = OutputHiddenMaskProvider.masks_per_owner
    def fail(*args, **kwargs):
        raise RuntimeError('injected selective failure')
    monkeypatch.setattr(model.hf.model.norm, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected selective'):
        selective_transfer(model, {}, 'anchor', prompt=prompt, spec=spec(1), eval_cfg=cfg, eos=[],
            reference=reference, donor=masks[0, 1], original_masks=masks, root=root, targets=targets)
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_equal_counts_do_not_hide_path_changes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from studies.output_space_cross_task.analyze_state_factors import trace_comparison
    left = {'stats': {'accepted': 1}, 'trace': [
        {'root_position': 10, 'accepted_nodes': [], 'committed_tokens': [1]},
        {'root_position': 11, 'accepted_nodes': [1], 'committed_tokens': [2, 3]}]}
    right = {'stats': {'accepted': 1}, 'trace': [
        {'root_position': 10, 'accepted_nodes': [1], 'committed_tokens': [1, 2]},
        {'root_position': 12, 'accepted_nodes': [], 'committed_tokens': [3]}]}
    result = trace_comparison(left, right)
    assert result['equal_accepted_totals'] and not result['full_semantic_equal']
    assert result['first_chronological_divergence']['index'] == 0
    assert result['common_root_gains'] == [10] and result['left_only_roots'] == [11]
    assert result['right_only_roots'] == [12]


def test_state_interaction_can_falsify_additivity(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from studies.output_space_cross_task.analyze_state_factors import interaction
    assert interaction({'native': 11, 'anchor': 13, 'velocity': 13, 'both': 14}) == -1
    assert interaction({'native': 12, 'anchor': 13, 'velocity': 13, 'both': 14}) == 0
