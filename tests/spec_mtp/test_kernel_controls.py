"""Controls change one operation and retain independent cache/state."""

import copy
import torch
from test_window_observer import ROOT, model as model
from spec_mtp.masks import OutputHiddenMaskProvider
from spec_mtp import model as kv_cache


def test_norm_graph_native_replication_and_unchanged_state(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_norm_kernel import mapped
    p = OutputHiddenMaskProvider(2, torch.randn(13, 7), order=1)
    p.anchor, p.velocity = torch.randn(7), torch.randn(7)
    proxies = torch.randn(2, 7)
    saved = [p.anchor.clone(), p.velocity.clone(), proxies.clone()]
    original = p.masks_per_owner(proxies)
    native, _ = mapped(p, proxies, alternative=False)
    alternate, details = mapped(p, proxies)
    assert torch.equal(original, native)
    assert alternate.shape == original.shape and torch.isfinite(alternate).all()
    assert len(details) == 3
    assert all(torch.equal(a, b) for a, b in zip(saved, [p.anchor, p.velocity, proxies]))


def test_fixed_linear_row_reference_preserves_shape(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_qproj import linear_row
    module = torch.nn.Linear(7, 11)
    inputs = torch.randn(1, 9, 7)
    original = inputs.clone()
    for mode in ('singleton', 'reference64'):
        out = linear_row(module, inputs, 4, mode)
        assert out.shape == (1, 1, 11) and out.dtype == inputs.dtype
        assert torch.isfinite(out).all() and torch.equal(inputs, original)
        assert torch.allclose(out, module(inputs)[:, 4:5], atol=1e-6, rtol=1e-5)


def test_exact_block_uses_fresh_cache(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_qproj import exact_block
    cache = kv_cache.new_cache()
    model(model.embed(torch.tensor([[1, 2]])), torch.tensor([[0, 1]]), past_key_values=cache)
    frame = {'inputs': model.embed(torch.tensor([[3]])).detach(), 'positions': torch.tensor([[2]]),
             'attention': torch.zeros(1, 1, 1, 3), 'cache': copy.deepcopy(cache)}
    output = model(frame['inputs'], frame['positions'], attention_mask_4d=frame['attention'],
                   past_key_values=copy.deepcopy(cache), last_hidden_only=True)
    frame.update(logits=output.logits.clone(), hidden=output.last_hidden_state.clone())
    result = exact_block(model, frame)
    assert result['saved_cache_unchanged'] and frame['cache'].get_seq_length() == cache.get_seq_length() == 2
