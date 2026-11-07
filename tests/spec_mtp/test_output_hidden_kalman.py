"""Output-hidden hybrid Kalman: trajectory identity, FP32 correction, root update."""

import torch

from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, ar_generate
from spec_mtp.structure.algorithm.mask import OutputHiddenHybridKalman, OutputHiddenMaskProvider, build_mask_provider
from spec_mtp.structure.model.hf_causal import HFCausalLM
from studies._shared.structure.config import _convert


def _base():
    provider = OutputHiddenMaskProvider(1, torch.tensor([[3., 4.]]), beta=.5, beta_prime=.5)
    provider.init_hidden(torch.tensor([[0., 0.]]))
    return provider


def test_trajectory_matches_output_hidden_and_correction_stays_fp32():
    base = _base()
    proxy = torch.tensor([[4., 0.]])
    reference = OutputHiddenMaskProvider(1, torch.tensor([[3., 4.]]), beta=.5, beta_prime=.5)
    reference.init_hidden(torch.tensor([[0., 0.]]))
    plain = OutputHiddenHybridKalman(base, 'trajectory')
    plain.init_from_prompt(torch.zeros(2, 2))
    plain.init_hidden(torch.tensor([[0., 0.]]))
    assert torch.equal(plain.masks_per_owner(proxy), reference.masks_per_owner(proxy))
    live = OutputHiddenHybridKalman(OutputHiddenMaskProvider(1, torch.tensor([[3., 4.]]), beta=.5, beta_prime=.5),
                                    'hybrid', {'k': [0.25, 0.0], 'p': [0.01, 0.01], 'r': [0.04, 0.04],
                                               'q': [0.0001, 0.0001], 'boost': 64})
    live.init_from_prompt(torch.zeros(2, 2))
    live.init_hidden(torch.tensor([[1., 0.]]))
    live.base.velocity = torch.zeros(2)
    assert abs(float(live._p[0]) - 0.64) < 1e-6
    turned = torch.tensor([[0., 2.]])
    masks = live.masks_per_owner(turned)
    slots, observations = live.base.trajectory(turned)
    corrected = slots + (observations.unsqueeze(1) - slots) * live._k.view(1, 1, -1)
    assert masks.dtype == torch.float32
    assert torch.allclose(masks, live.base._map_hidden_slots(corrected))
    bare = OutputHiddenMaskProvider(1, torch.tensor([[3., 4.]]), beta=.5, beta_prime=.5)
    bare.init_hidden(torch.tensor([[1., 0.]]))
    bare.velocity = torch.zeros(2)
    assert not torch.allclose(masks, bare.masks_per_owner(turned))


def test_root_update_is_delayed_and_deep_error_is_separate():
    live = OutputHiddenHybridKalman(OutputHiddenMaskProvider(1, torch.ones(1, 1), beta=.5, beta_prime=.5),
                                    'hybrid', {'k': [0.25], 'p': [0.01], 'r': [0.04], 'q': [0.0001], 'boost': 64})
    live.init_from_prompt(torch.zeros(2, 1))
    live.init_hidden(torch.zeros(1, 1))
    live.base.anchor = torch.tensor([0.])
    live.base.velocity = torch.tensor([0.])
    live.masks_per_owner(torch.tensor([[4.]]))
    first = live._current[0]
    live.note_block_hiddens(torch.tensor([9.]), [])
    assert live._root_n == 0 and live._k.allclose(torch.tensor([0.25]))
    live.masks_per_owner(torch.tensor([[5.]]))
    live.note_block_hiddens(torch.tensor([2.]), [])
    gain = first['k']
    regressor = first['z'] - first['h']
    innov = (torch.tensor([2.]) - first['h']) - gain * regressor
    predicted = torch.tensor([0.6464])
    step = predicted * regressor / (torch.tensor([0.04]) + predicted * regressor * regressor)
    expect = gain + step * innov
    assert torch.allclose(live._k, expect)
    assert live.diagnostics()['root_n'] == 1 and live.diagnostics()['deep_n'] == 0
    assert live.diagnostics()['root_error'] != live.diagnostics()['root_base_error']
    frozen = OutputHiddenHybridKalman(OutputHiddenMaskProvider(1, torch.ones(1, 1)), 'frozen',
                                      {'k': [0.25], 'p': [0.01], 'r': [0.04], 'q': [0.0001]})
    frozen.init_from_prompt(torch.zeros(2, 1))
    frozen.init_hidden(torch.zeros(1, 1))
    frozen.base.anchor = torch.tensor([0.])
    frozen.base.velocity = torch.tensor([0.])
    frozen.masks_per_owner(torch.tensor([[4.]]))
    before = frozen._k.clone()
    frozen.note_block_hiddens(torch.tensor([1.]), [torch.tensor([2.])])
    frozen.note_block_hiddens(torch.tensor([3.]), [])
    assert torch.equal(frozen._k, before)
    assert frozen.diagnostics()['root_n'] == 1 and frozen.diagnostics()['deep_n'] == 1
    with pytest_raises():
        build_mask_provider({'method': 'esp', 'output_kalman': {'mode': 'trajectory'}}, torch.zeros(2, 2))


def pytest_raises():
    import pytest
    return pytest.raises(ValueError, match='output hidden')


def test_zero_gain_matches_output_hidden_and_nonzero_stays_lossless():
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 64, 'hidden_size': 32, 'num_layers': 2,
                                    'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 128,
                                    'tie_word_embeddings': True})
    model.hf.lm_head.weight = model.hf.model.embed_tokens.weight
    width = model.embedding_table.shape[-1]
    zeros = [0.0] * width
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    eval_cfg = _convert({'max_new_tokens': 8, 'temperature': 0., 'stop_on_eos': False})
    spec = {'method': 'output_hidden', 'num_masks': 1, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [2], 'pruning': False},
            'output': {'order': 1, 'beta': .9, 'beta_prime': .99}}
    plain = build_mask_provider(spec, model.embedding_table)
    zero = build_mask_provider({**spec, 'output_kalman': {'mode': 'hybrid', 'k': zeros, 'p': [1.0] * width,
                                                          'r': [1.0] * width, 'q': [0.01] * width, 'boost': 64}},
                               model.embedding_table)
    out_plain, _ = SpecDecoder(model, spec, eval_cfg, mask_policy=plain).generate(prompt)
    out_zero, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=zero).generate(prompt)
    assert torch.equal(out_plain, out_zero)
    assert stats['verification_argmax_audit']['passed']
    assert zero.diagnostics()['root_n'] > 0
    shifted = [0.2] * width
    hybrid = build_mask_provider({**spec, 'output_kalman': {'mode': 'hybrid', 'k': shifted, 'p': [0.01] * width,
                                                            'r': [0.04] * width, 'q': [0.0001] * width, 'boost': 64}},
                                 model.embedding_table)
    out, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=hybrid).generate(prompt)
    ar, _ = ar_generate(model, prompt, {'max_new_tokens': 8, 'temperature': 0., 'stop_on_eos': False})
    assert torch.equal(out, ar) and stats['verification_argmax_audit']['passed']
    rows = build_mask_provider({**spec, 'output_kalman': {'mode': 'collect'}}, model.embedding_table)
    SpecDecoder(model, spec, eval_cfg, mask_policy=rows).generate(prompt)
    harvested = rows.calibration_rows()
    assert harvested['h'].shape[0] == rows.diagnostics()['root_n']
    assert harvested['h'].shape[1] == width
