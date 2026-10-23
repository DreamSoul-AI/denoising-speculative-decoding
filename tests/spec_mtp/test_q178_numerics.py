"""Canonical-prefix state and cache-derived roots stay scoped to diagnostics."""

import pytest
import torch

from spec_mtp.masks import OutputHiddenMaskProvider
from test_proxy_transfer import baseline
from test_window_observer import ROOT, model as model, spec


def test_canonical_prefix_state_ends_at_root_minus_one(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_numerics import canonical_states
    hidden = torch.tensor([[1., 0.], [0., 1.], [2., 1.]])
    table = torch.ones(3, 2)
    for order in (0, 1):
        actual = canonical_states(hidden, table, order)
        provider = OutputHiddenMaskProvider(2, table, order=order)
        for position, y in enumerate(hidden):
            provider.observe_committed(y)
            assert torch.equal(actual[position + 1]['anchor'], y)
            assert torch.equal(actual[position + 1]['velocity'], provider.velocity)


def test_synchronization_reads_actual_cache_and_restores_hooks(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_numerics import canonical_states, synchronized_transfer
    prompt, cfg, reference, targets, root, masks = baseline(model)
    ids = torch.tensor([reference['output_ids']])
    teacher = model(model.embed(ids), torch.arange(ids.shape[1])[None], last_hidden_only=True)
    states = canonical_states(teacher.last_hidden_state[0], model.embedding_table, 1)
    samples, replay, _, records = synchronized_transfer(model, states, prompt=prompt, spec=spec(1),
        eval_cfg=cfg, eos=[], reference=reference, donor=masks[0, 1], original_masks=masks,
        root=root, targets=targets, placebo=False)
    assert replay['output_equal_independent_AR'] and records
    assert all(r['incoming_proxies_unchanged'] and r['canonical_state_bit_equal'] for r in records)
    for position, sample in samples.items():
        if position > root:
            assert all(torch.equal(sample['current_state'][k], states[position][k]) for k in ('anchor', 'velocity'))
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_synchronization_cleanup_after_forward_failure(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_numerics import synchronized_transfer
    prompt, cfg, reference, targets, root, masks = baseline(model)
    original = OutputHiddenMaskProvider.masks_per_owner
    def fail(*args, **kwargs):
        raise RuntimeError('injected synchronization failure')
    monkeypatch.setattr(model.hf.model.norm, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected synchronization'):
        synchronized_transfer(model, {}, prompt=prompt, spec=spec(1), eval_cfg=cfg, eos=[], reference=reference,
                              donor=masks[0, 1], original_masks=masks, root=root, targets=targets, placebo=False)
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks
