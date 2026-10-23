"""Operator controls retain native inputs/state and scoped cleanup."""

import torch
import pytest

from spec_mtp.masks import OutputHiddenMaskProvider
from test_proxy_transfer import baseline
from test_window_observer import ROOT, model as model, spec


def test_cpu_norm32_mapping_equals_native_and_preserves_inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_operators import map_masks
    provider = OutputHiddenMaskProvider(2, torch.randn(13, 7), order=1)
    provider.anchor, provider.velocity = torch.randn(7), torch.randn(7)
    proxies = torch.randn(2, 7)
    before = [provider.anchor.clone(), provider.velocity.clone(), proxies.clone()]
    assert torch.equal(map_masks(provider, proxies, 'norm32cpu'), provider.masks_per_owner(proxies))
    for mode in ('slots64', 'norm64'):
        actual = map_masks(provider, proxies, mode)
        assert actual.shape == (3, 2, 7) and actual.dtype == torch.float32
        assert torch.isfinite(actual).all()
    assert all(torch.equal(a, b) for a, b in zip(before, [provider.anchor, provider.velocity, proxies]))
    with pytest.raises(AssertionError):
        map_masks(provider, proxies, 'unsupported')


def test_operator_replay_restores_provider_and_hooks(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_numerics import canonical_states
    from diagnose_q178_operators import NATIVE, replay
    prompt, cfg, reference, targets, root, masks = baseline(model)
    ids = torch.tensor([reference['output_ids']])
    teacher = model(model.embed(ids), torch.arange(ids.shape[1])[None], last_hidden_only=True)
    states = canonical_states(teacher.last_hidden_state[0], model.embedding_table, 1)
    call = next(i + 1 for i, t in enumerate(reference['stats']['trace']) if t['root_position'] == root)
    _, evidence, records, differences = replay(model, states, 'norm32cpu', call,
        prompt=prompt, spec=spec(1), eval_cfg=cfg, eos=[], reference=reference,
        donor=masks[0, 1], original_masks=masks, root=root, targets=targets, placebo=False)
    assert evidence['output_equal_independent_AR'] and len(differences) == len(records)
    assert all(r['incoming_proxies_unchanged'] and r['canonical_state_bit_equal'] for r in records)
    assert OutputHiddenMaskProvider.masks_per_owner is NATIVE
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks
