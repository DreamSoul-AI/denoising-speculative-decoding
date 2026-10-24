"""Higher precision Phi holds the native slot/scale; RMS references hold row/weight."""

import torch
from test_window_observer import ROOT
from spec_mtp.masks import OutputHiddenMaskProvider
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def test_phi64_uses_native_slot_and_scale_without_state_mutation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_phi_rms import phi64
    for order in (0, 1):
        p = OutputHiddenMaskProvider(2, torch.randn(9, 7), order=order)
        p.anchor, p.velocity = torch.randn(7), torch.randn(7) if order else torch.zeros(7)
        proxies = torch.randn(2, 7)
        saved = [p.anchor.clone(), p.velocity.clone(), proxies.clone(), p.scale.clone()]
        actual = phi64(p, proxies)
        y = proxies[0]
        v = .99 * p.velocity + (1 - .99) * (y - p.anchor) if order else p.velocity
        slot = (y + v).double()
        expected = (p.scale.double() * slot / slot.norm()).float()
        assert torch.equal(actual, expected) and torch.isfinite(actual).all()
        assert all(torch.equal(a, b) for a, b in zip(saved, [p.anchor, p.velocity, proxies, p.scale]))


def test_rms_singleton_and_high_precision_keep_input_and_shape(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q178_phi_rms import norm_row
    module = LlamaRMSNorm(7, eps=1e-6)
    inputs = torch.randn(1, 9, 7)
    before = inputs.clone()
    native = module(inputs)[:, 4:5]
    for mode in ('singleton', 'reference64'):
        result = norm_row(module, inputs, mode)
        assert result.shape == (1, 1, 7) and result.dtype == inputs.dtype and torch.isfinite(result).all()
        assert torch.allclose(result, native, atol=1e-6, rtol=1e-5)
    assert torch.equal(inputs, before)
