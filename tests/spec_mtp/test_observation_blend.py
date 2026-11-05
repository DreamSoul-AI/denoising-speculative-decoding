"""Protocol trajectory-observation-blend-v1: identity, losslessness and algebra on CPU."""

import pytest
import torch

from spec_mtp.structure.algorithm.counts import round_split
from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, ar_generate
from spec_mtp.structure.algorithm.mask import (BonusSlotMaskProvider, ESPMaskProvider,
                            PreforwardSignalCollector, apply_lowrank_delta,
                            build_mask_provider, centered_r2, fit_lowrank_delta)
from spec_mtp.structure.model.hf_causal import HFCausalLM
from spec_mtp.structure.algorithm.decode.tree import DraftTree


def _model():
    torch.manual_seed(7)
    return HFCausalLM(tiny_config={'vocab_size': 199, 'hidden_size': 64, 'intermediate_size': 128,
                                   'num_layers': 2, 'num_heads': 4, 'num_kv_heads': 2,
                                   'max_position_embeddings': 1024})


def _spec(method, k, mode, gain=None, condition='none', form='mean'):
    spec = {'method': method, 'num_masks': k, 'impl': 'efficient' if mode == 'static' else 'naive',
            'block_complexity': None if mode == 'static' else 30,
            'tree': {'mode': mode, 'branches': [7, 2][:k] if mode == 'static' else None, 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
            'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False, 'update': True}}
    if gain is not None:
        spec['observation'] = {'gain': gain, 'condition': condition, 'form': form}
    return spec


def _decode(model, spec, prompt, temperature, seed):
    eval_cfg = {'max_new_tokens': 20, 'temperature': temperature, 'stop_on_eos': False}
    decoder = SpecDecoder(model, spec, eval_cfg, generator=torch.Generator().manual_seed(seed),
                          mask_policy=build_mask_provider(spec, model.embedding_table))
    return decoder.generate(prompt)


PROMPTS = [torch.randint(0, 199, (1, n), generator=torch.Generator().manual_seed(n)) for n in (12, 31)]
CONFIGS = [('esp', 1, 'static'), ('esp', 2, 'static'), ('ema_velocity', 2, 'static'), ('esp', 2, 'dynamic')]


@pytest.mark.parametrize('method,k,mode', CONFIGS)
@pytest.mark.parametrize('condition', ['none', 'rejection'])
def test_zero_gain_is_the_trajectory_provider_token_for_token(method, k, mode, condition):
    model = _model()
    for prompt in PROMPTS:
        base_out, base_stats = _decode(model, _spec(method, k, mode), prompt, 0.0, 1)
        out, stats = _decode(model, _spec(method, k, mode, gain=0, condition=condition), prompt, 0.0, 1)
        assert torch.equal(out, base_out)
        assert stats['trace'] == base_stats['trace']
        assert stats['accepted'] == base_stats['accepted'] and stats['decode_calls'] == base_stats['decode_calls']


@pytest.mark.parametrize('method,k,mode', CONFIGS)
@pytest.mark.parametrize('gain', [0.5, 1.0])
@pytest.mark.parametrize('temperature', [0.0, 1.0])
@pytest.mark.parametrize('condition', ['none', 'rejection'])
@pytest.mark.parametrize('form', ['mean', 'top1'])
def test_blended_masks_keep_ar_output_and_argmax_audit(method, k, mode, gain, temperature, condition, form):
    model = _model()
    for index, prompt in enumerate(PROMPTS):
        out, stats = _decode(model, _spec(method, k, mode, gain, condition, form), prompt, temperature, 50 + index)
        ar_out, _ = ar_generate(model, prompt, {'max_new_tokens': 20, 'temperature': temperature, 'stop_on_eos': False},
                                generator=torch.Generator().manual_seed(50 + index))
        assert torch.equal(out, ar_out)
        if temperature == 0:
            assert stats['verification_argmax_audit']['passed'] is True




def test_bonus_slot_replaces_only_the_immediate_token_and_stays_lossless():
    torch.manual_seed(6)
    table = torch.randn(10, 4)
    logits = torch.randn(2, 10)
    logits[0, 4] = 9.0
    logits[1, 8] = 9.0
    base = ESPMaskProvider(2, table)
    slot = BonusSlotMaskProvider(base, table)
    slot.init_from_prompt(torch.randn(5, 4))
    before = base.masks().clone()
    slot.observe_mask_logits(logits)
    tree = DraftTree(root_token=0, tokens=[1, 2], depths=[1, 2], parents=[0, 1])
    owners = slot.masks_for_tree(tree)
    assert slot.protocol == 'bonus-slot-hard-v1'
    assert torch.equal(owners[0, 0], table[4]) and torch.equal(owners[1, 0], table[8])
    assert torch.equal(owners[2, 0], before[0])
    assert torch.equal(owners[0, 1], before[1]) and torch.equal(owners[1, 1], before[1])
    assert torch.equal(base.masks(), before)
    spec = {**_spec('esp', 2, 'static'), 'bonus_slot': {'mode': 'hard'}}
    model = _model()
    prompt = torch.randint(0, 199, (1, 6))
    reference = ar_generate(model, prompt, {'max_new_tokens': 8, 'temperature': 0, 'stop_on_eos': False})[0]
    output, stats = SpecDecoder(model, spec, {'max_new_tokens': 8, 'temperature': 0, 'stop_on_eos': False},
                                mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
    assert torch.equal(output, reference) and stats['verification_argmax_audit']['passed']
    topk = BonusSlotMaskProvider(ESPMaskProvider(2, table), table, mode='topk', widths=[2, 2])
    topk.init_from_prompt(torch.randn(5, 4))
    topk.observe_mask_logits(logits)
    vals, idx = torch.topk(logits[0], 2)
    expect = torch.softmax(vals.float(), dim=0).to(table.dtype) @ table[idx]
    assert topk.protocol == 'bonus-slot-topk-v1'
    assert torch.allclose(topk.masks_for_tree(tree)[0, 0], expect)
    assert torch.equal(topk.masks_for_tree(tree)[0, 1], topk.base.masks()[1])
    spec_k = {**_spec('esp', 2, 'static'), 'bonus_slot': {'mode': 'topk'}}
    output_k, stats_k = SpecDecoder(model, spec_k, {'max_new_tokens': 8, 'temperature': 0, 'stop_on_eos': False},
                                    mask_policy=build_mask_provider(spec_k, model.embedding_table)).generate(prompt)
    assert torch.equal(output_k, reference) and stats_k['verification_argmax_audit']['passed']
    with pytest.raises(ValueError, match='mode'):
        build_mask_provider({**spec, 'bonus_slot': {'mode': 'soft'}}, table)
    with pytest.raises(ValueError, match='combine'):
        build_mask_provider({**spec, 'observation': {'gain': 1}}, table)




def test_decoder_observes_prefill_and_every_continuing_round():
    model = _model()
    spec = _spec('esp', 2, 'static', gain=0.5)
    provider = build_mask_provider(spec, model.embedding_table)
    seen = []
    original = provider.observe_mask_logits
    provider.observe_mask_logits = lambda logits: (seen.append(logits.clone()), original(logits))
    decoder = SpecDecoder(model, spec, {'max_new_tokens': 20, 'temperature': 0.0, 'stop_on_eos': False},
                          generator=torch.Generator().manual_seed(3), mask_policy=provider)
    _, stats = decoder.generate(PROMPTS[0])
    continuing = sum(not row['stopped'] for row in stats['trace'])
    assert len(seen) == 1 + continuing
    assert all(row.shape == (2, model.vocab_size) for row in seen)






def test_scalar_kalman_step_steady_gain_and_lossless_decode():
    from spec_mtp.structure.algorithm.mask import KalmanMaskProvider

    provider = KalmanMaskProvider(2, torch.zeros(4, 3), mode='online', forget=1.0, r_min=1e-8, q_min=1e-12)
    provider.init_from_prompt(torch.zeros(2, 3))
    provider._level = torch.zeros(3)
    provider._p, provider._q, provider._r = 1.0, 0.25, 1.0
    provider._step(torch.tensor([2.0, 0.0, 0.0]))
    gain = 1.25 / 2.25
    assert torch.allclose(provider._level, torch.tensor([2.0, 0.0, 0.0]) * gain)
    assert abs(provider._p - ((1 - gain) ** 2 * 1.25 + gain ** 2)) < 1e-6
    phi = provider._q / provider._r
    k_star = (-phi + (phi * phi + 4 * phi) ** 0.5) / 2
    for _ in range(50):
        provider._step(provider._level.detach().clone())
    assert abs(provider.diagnostics()['k'] - k_star) < 1e-4
    frozen = KalmanMaskProvider(1, torch.zeros(4, 3), mode='frozen')
    frozen.init_from_prompt(torch.randn(6, 3))
    held = frozen.diagnostics()['k']
    frozen.on_token_committed(torch.randn(3), torch.zeros(3))
    assert frozen.diagnostics()['k'] == held
    spec = {**_spec('kalman', 1, 'static'), 'kalman': {'mode': 'online'}}
    model, prompt = _model(), PROMPTS[0]
    reference, _ = ar_generate(model, prompt, {'max_new_tokens': 20, 'temperature': 0, 'stop_on_eos': False})
    output, stats = _decode(model, spec, prompt, 0, 0)
    assert torch.equal(output, reference) and stats['verification_argmax_audit']['passed']
    with pytest.raises(ValueError, match='online or frozen'):
        KalmanMaskProvider(1, torch.zeros(4, 3), mode='soft')


def test_verify_correction_recovers_scalar_and_edits_only_slot_zero():
    from spec_mtp.structure.algorithm.mask import ESPMaskProvider, VerifyCorrectionMaskProvider, fit_mask_correction

    mhat = torch.zeros(6, 3)
    observed = torch.tensor([[1.0, 0.0, 0.0]]).repeat(6, 1)
    target = 0.4 * observed
    fit = fit_mask_correction(mhat, observed, target, 'scalar')
    assert abs(fit['k'] - 0.4) < 1e-6
    provider = VerifyCorrectionMaskProvider(ESPMaskProvider(2, torch.zeros(4, 3)), 'scalar', {'k': 0.5})
    provider.init_from_prompt(torch.zeros(2, 3))
    provider.base._masks = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    provider._z = torch.tensor([3.0, 1.0, 1.0])
    provider._masks_calls = 1
    corrected = provider.masks()
    assert torch.allclose(corrected[0], torch.tensor([2.0, 1.0, 1.0]))
    assert torch.allclose(corrected[1], torch.tensor([2.0, 2.0, 2.0]))
    reject = VerifyCorrectionMaskProvider(ESPMaskProvider(1, torch.zeros(4, 2)), 'reject_token')
    reject.init_from_prompt(torch.ones(2, 2))
    reject.masks()
    reject.on_token_committed(torch.tensor([4.0, 0.0]), torch.zeros(2))
    reject.masks()
    reject.on_token_committed(torch.tensor([5.0, 0.0]), torch.zeros(2))
    assert torch.allclose(reject.masks()[0], torch.tensor([5.0, 0.0]))
    affine = fit_mask_correction(mhat, observed, target, 'affine')
    assert len(affine['A']) == 3 and len(affine['gram']) == 4
    online = VerifyCorrectionMaskProvider(ESPMaskProvider(1, torch.zeros(4, 3)), 'affine', affine, online=True)
    online.init_from_prompt(torch.zeros(2, 3))
    online.base._masks = torch.zeros(1, 3)
    online._z = observed[0].clone()
    online._masks_calls = 1
    online.masks()
    before = online._A.clone()
    online.on_token_committed(target[0].clone(), torch.zeros(3))
    assert online._A.shape == before.shape and online.diagnostics()['online_updates'] == 1
    generator = torch.Generator().manual_seed(0)
    zed = torch.randn(30, 4, generator=generator)
    guess = torch.randn(30, 4, generator=generator)
    delta = zed - guess
    truth = guess + delta * 0.3
    from spec_mtp.structure.algorithm.mask import fit_affine_ridge, select_affine_ridge
    recovered = fit_affine_ridge(guess, zed, truth, 1e-6)
    assert torch.allclose(torch.tensor(recovered['A']), torch.eye(4) * 0.3, atol=1e-4)
    blocks = [(guess[start:start + 6], zed[start:start + 6], truth[start:start + 6]) for start in range(0, 30, 6)]
    chosen = select_affine_ridge(blocks, 0)
    assert chosen['ridge'] > 0 and chosen['prompts'] == 5 and 'eval' not in chosen
    ridge_online = VerifyCorrectionMaskProvider(ESPMaskProvider(1, torch.zeros(4, 4)), 'affine', chosen, online=True)
    ridge_online.init_from_prompt(torch.zeros(2, 4))
    ridge_online.base._masks = guess[:1]
    ridge_online._z = zed[0].clone()
    ridge_online._masks_calls = 1
    frozen_A = ridge_online._A.clone()
    ridge_online.masks()
    assert torch.equal(ridge_online._A, frozen_A) and ridge_online.diagnostics()['online_updates'] == 0
    ridge_online.on_token_committed(truth[0].clone(), torch.zeros(4))
    ridge_online.init_from_prompt(torch.zeros(2, 4))
    assert torch.equal(ridge_online._A, frozen_A) and ridge_online.diagnostics()['online_updates'] == 0
    from spec_mtp.structure.algorithm.mask import fit_lowrank_ridge, lowrank_replay_mse, select_lowrank_ridge
    gen = torch.Generator().manual_seed(1)
    raw = torch.randn(48, 6, generator=gen)
    guess_l = torch.randn(48, 6, generator=gen)
    zed_l = guess_l + raw
    truth_l = guess_l + (raw @ torch.randn(6, 2, generator=gen)) @ torch.randn(2, 6, generator=gen)
    low = fit_lowrank_ridge(guess_l, zed_l, truth_l, 2, 1e-6, iters=8)
    assert lowrank_replay_mse(guess_l, zed_l, truth_l, low) < 1e-3
    pieces = [(guess_l[start:start + 8], zed_l[start:start + 8], truth_l[start:start + 8]) for start in range(0, 48, 8)]
    picked = select_lowrank_ridge(pieces, 2, 0, iters=3)
    assert picked['rank'] == 2 and picked['prompts'] == 6 and picked['ridge'] > 0
    holder = VerifyCorrectionMaskProvider(ESPMaskProvider(1, torch.zeros(4, 6)), 'lowrank', picked, online=True)
    holder.init_from_prompt(torch.zeros(2, 6))
    holder.base._masks = guess_l[:1]
    holder._z = zed_l[0].clone()
    holder._masks_calls = 1
    frozen_u = holder._U.clone()
    holder.masks()
    assert torch.equal(holder._U, frozen_u) and holder.diagnostics()['online_updates'] == 0
    holder.on_token_committed(truth_l[0].clone(), torch.zeros(6))
    assert holder.diagnostics()['online_updates'] == 1
    holder.init_from_prompt(torch.zeros(2, 6))
    assert torch.equal(holder._U, frozen_u) and holder.diagnostics()['online_updates'] == 0
    spec = {**_spec('esp', 1, 'static'), 'correction': {'kind': 'scalar', 'k': 0.0, 'num': 0.0, 'den': 1.0}}
    model, prompt = _model(), PROMPTS[0]
    reference, _ = ar_generate(model, prompt, {'max_new_tokens': 20, 'temperature': 0, 'stop_on_eos': False})
    output, stats = _decode(model, spec, prompt, 0, 0)
    assert torch.equal(output, reference) and stats['verification_argmax_audit']['passed']
    from spec_mtp.structure.algorithm.mask import OnlineVerifyKalman
    p0, q0, r0, forget = 1.0, 0.01, 1.0, 0.95
    p_pred = p0 + q0
    gain = p_pred / (p_pred + r0)
    corrected = gain * 2.0
    innov2 = 16.0
    posterior = (1 - gain) ** 2 * p_pred + gain ** 2 * r0
    weight = (1 - forget) / (1 - forget ** 2)
    expect_r = max((1 - weight) * r0 + weight * (innov2 - p_pred), 1e-6)
    expect_q = max((1 - weight) * q0 + weight * (gain * gain * innov2), 1e-8)
    live = OnlineVerifyKalman(ESPMaskProvider(2, torch.zeros(4, 1)), 'full')
    live.init_from_prompt(torch.zeros(2, 1))
    live.base._masks = torch.tensor([[0.0], [9.0]])
    live._z = torch.tensor([2.0])
    merged = live.masks()
    assert torch.allclose(merged[0], torch.tensor([corrected]))
    assert torch.allclose(merged[1], torch.tensor([9.0]))
    live.on_token_committed(torch.tensor([4.0]), torch.zeros(1))
    assert abs(float(live._r) - expect_r) < 1e-5 and abs(float(live._q) - expect_q) < 1e-5
    assert abs(float(live._p) - posterior) < 1e-5
    live.init_from_prompt(torch.ones(3, 1) * 50)
    assert float(live._r) == 1.0 and abs(float(live._q) - 0.01) < 1e-8 and live._z is None
    untouched = OnlineVerifyKalman(ESPMaskProvider(1, torch.zeros(2, 1)), 'trajectory')
    untouched.init_from_prompt(torch.zeros(2, 1))
    untouched.base._masks = torch.tensor([[7.0]])
    assert float(untouched.masks()) == 7.0
    from spec_mtp.structure.algorithm.mask import AlignedResidualKalman
    aligned = AlignedResidualKalman(ESPMaskProvider(1, torch.zeros(4, 2)), 'full')
    aligned.init_from_prompt(torch.zeros(2, 2))
    aligned.base._masks = torch.zeros(1, 2)
    assert torch.equal(aligned.masks(), torch.zeros(1, 2))
    aligned.on_token_committed(torch.tensor([3.0, 0.0]), torch.zeros(2))
    assert aligned._f == 0.0 and aligned._nu is not None
    aligned.base._masks = torch.zeros(1, 2)
    assert torch.equal(aligned.masks(), torch.zeros(1, 2))
    aligned.on_token_committed(torch.tensor([3.0, 0.0]), torch.zeros(2))
    gain = 1.01 / 2.01
    assert abs(aligned._f - gain) < 1e-5
    assert abs(aligned._r - 1.0) < 1e-5
    aligned.base._masks = torch.zeros(1, 2)
    corrected = aligned.masks()
    assert torch.allclose(corrected[0], torch.tensor([gain * 3.0, 0.0]))
    aligned.init_from_prompt(torch.ones(2, 2))
    assert aligned._f == 0.0 and aligned._nu is None
    from spec_mtp.structure.algorithm.mask import HybridGainKalman, calibrate_diagonal_gain
    guessed = torch.zeros(6, 2)
    observed = torch.tensor([[2.0, 0.0]]).repeat(6, 1)
    target = guessed + 0.25 * (observed - guessed)
    prior = calibrate_diagonal_gain(guessed, observed, target)
    assert abs(prior['k'][0] - 0.25) < 1e-5 and abs(prior['k'][1]) < 1e-5
    assert abs(prior['p'][0] - prior['r'][0] / 24.0) < 1e-6
    assert prior['p'][0] < 1.0
    live = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'hybrid',
                            {'k': [0.2], 'p': [0.01], 'r': [0.04], 'q': [0.0001]})
    live.init_from_prompt(torch.zeros(2, 1))
    live.base._masks = torch.zeros(1, 1)
    live._z = torch.tensor([2.0])
    assert torch.allclose(live.masks()[0], torch.tensor([0.4]))
    live.on_token_committed(torch.tensor([1.0]), torch.zeros(1))
    predicted = 0.0101
    step = predicted * 2.0 / (0.04 + predicted * 4.0)
    expect = 0.2 + step * (1.0 - 0.2 * 2.0)
    assert abs(float(live._k) - expect) < 1e-5
    live.init_from_prompt(torch.zeros(2, 1))
    assert abs(float(live._k) - 0.2) < 1e-6
    frozen = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'frozen', prior)
    frozen.init_from_prompt(torch.zeros(2, 2))
    frozen.base._masks = torch.zeros(1, 2)
    frozen._z = torch.tensor([1.0, 1.0])
    frozen.masks()
    before = frozen._k.clone()
    frozen.on_token_committed(torch.tensor([0.0, 0.0]), torch.zeros(2))
    assert torch.equal(frozen._k, before)
    opened = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'opened',
                              {'k': [0.2], 'p': [0.01], 'r': [0.04], 'q': [0.0001], 'boost': 64})
    opened.init_from_prompt(torch.zeros(2, 1))
    assert abs(float(opened._p) - 0.64) < 1e-6 and abs(float(opened._q) - 0.0064) < 1e-6
    opened.base._masks = torch.zeros(1, 1)
    opened._z = torch.tensor([2.0])
    opened.masks()
    opened.on_token_committed(torch.tensor([1.0]), torch.zeros(1))
    wide = 0.6464
    wide_step = wide * 2.0 / (0.04 + wide * 4.0)
    assert abs(float(opened._k) - (0.2 + wide_step * (1.0 - 0.4))) < 1e-5
    assert abs(opened.diagnostics()['weight_mean'] - wide_step * 2.0) < 1e-5
    branch = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'branch',
                              {'k': [0.2], 'p': [0.01], 'r': [0.04], 'q': [0.0001]})
    branch.init_from_prompt(torch.zeros(2, 1))
    branch.base._masks = torch.tensor([[3.0]])
    branch._z = torch.tensor([1.0])
    assert torch.allclose(branch.masks(), torch.tensor([[3.0]]))
    assert torch.allclose(branch.consume_probe(), torch.tensor([3.0 + 0.2 * (1.0 - 3.0)]))
    from spec_mtp.structure.algorithm.decode.tree import DraftTree, add_depth1_branch, replace_weakest_depth1
    tree = DraftTree(root_token=5, tokens=[1, 2], depths=[1, 1], parents=[0, 0], probs=[0.1, 0.1])
    assert add_depth1_branch(tree, 1) is False
    assert add_depth1_branch(tree, 5) is False
    assert add_depth1_branch(tree, 9) is True and tree.tokens == [1, 2, 9]
    same = DraftTree(root_token=5, tokens=[1, 2, 3], depths=[1, 1, 1], parents=[0, 0, 0], probs=[0.5, 0.2, 0.2])
    assert replace_weakest_depth1(same, 1) is False and same.tokens == [1, 2, 3]
    assert replace_weakest_depth1(same, 5) is False and same.tokens == [1, 2, 3]
    assert replace_weakest_depth1(same, 9) is True and same.tokens == [1, 2, 9] and len(same.tokens) == 3
    parented = DraftTree(root_token=0, tokens=[1, 8], depths=[1, 2], parents=[0, 1], probs=[0.2, 0.1])
    assert replace_weakest_depth1(parented, 9) is False and parented.tokens == [1, 8]
    mixed = DraftTree(root_token=0, tokens=[1, 7, 8], depths=[1, 1, 2], parents=[0, 0, 1], probs=[0.1, 0.4, 0.05])
    assert replace_weakest_depth1(mixed, 9) is True and mixed.tokens == [1, 9, 8]
    model = _model()
    spec = _spec('esp', 1, 'static')
    spec['tree'] = {'mode': 'static', 'branches': [4], 'pruning': True}
    width = model.embedding_table.shape[-1]
    zeros = [0.0] * width
    spec['hybrid_gain'] = {'mode': 'branch', 'k': zeros, 'p': [1.0] * width, 'r': [1.0] * width, 'q': [0.01] * width}
    _output, stats = _decode(model, spec, PROMPTS[0], 0, 0)
    assert stats['verification_argmax_audit']['passed']
    spec['hybrid_gain'] = {'mode': 'replace', 'k': zeros, 'p': [1.0] * width, 'r': [1.0] * width, 'q': [0.01] * width}
    replaced = build_mask_provider(spec, model.embedding_table)
    replaced.init_from_prompt(torch.zeros(2, width))
    assert replaced.extra_edit == 'replace'
    replaced.record_branch(True, 4)
    replaced.record_branch(False, 3)
    assert replaced.diagnostics()['max_nodes'] == 4 and replaced.extra_added == 1
    _output, stats = _decode(model, spec, PROMPTS[0], 0, 0)
    assert stats['verification_argmax_audit']['passed']


def test_bonus_aligned_gain_learns_the_last_commit_and_stays_lossless():
    from spec_mtp.structure.algorithm.mask import HybridGainKalman
    payload = {'k': [0.2], 'p': [0.01], 'r': [0.04], 'q': [0.0001], 'boost': 64, 'observe': 'last'}
    aligned = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'opened', payload)
    aligned.init_from_prompt(torch.zeros(2, 1))
    aligned.base._masks = torch.zeros(1, 1)
    aligned._z = torch.tensor([2.0])
    assert torch.allclose(aligned.masks()[0], torch.tensor([0.4]))
    aligned.on_token_committed(torch.tensor([1.0]), torch.zeros(1))
    aligned.on_token_committed(torch.tensor([5.0]), torch.zeros(1))
    assert abs(float(aligned._k) - 0.2) < 1e-6
    aligned.finish_round()
    wide = 0.6464
    step = wide * 2.0 / (0.04 + wide * 4.0)
    assert abs(float(aligned._k) - (0.2 + step * (5.0 - 0.4))) < 1e-5
    aligned.finish_round()
    assert abs(float(aligned._k) - (0.2 + step * (5.0 - 0.4))) < 1e-5
    early = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'opened',
                             {'k': [0.2], 'p': [0.01], 'r': [0.04], 'q': [0.0001], 'boost': 64})
    early.init_from_prompt(torch.zeros(2, 1))
    early.base._masks = torch.zeros(1, 1)
    early._z = torch.tensor([2.0])
    early.masks()
    early.on_token_committed(torch.tensor([1.0]), torch.zeros(1))
    kept = float(early._k)
    early.on_token_committed(torch.tensor([5.0]), torch.zeros(1))
    early.finish_round()
    assert abs(float(early._k) - kept) < 1e-6
    collected = HybridGainKalman(ESPMaskProvider(1, torch.zeros(4, 1)), 'collect')
    collected.init_from_prompt(torch.zeros(2, 1))
    collected.base._masks = torch.tensor([[0.0]])
    collected._z = torch.tensor([2.0])
    assert torch.equal(collected.masks(), torch.tensor([[0.0]]))
    collected.on_token_committed(torch.tensor([1.0]), torch.zeros(1))
    collected.on_token_committed(torch.tensor([5.0]), torch.zeros(1))
    collected.finish_round()
    rows = collected.calibration_rows()
    assert torch.allclose(rows['m'], torch.tensor([[0.0]]))
    assert torch.allclose(rows['z'], torch.tensor([[2.0]]))
    assert torch.allclose(rows['first'], torch.tensor([[1.0]]))
    assert torch.allclose(rows['last'], torch.tensor([[5.0]]))
    with pytest.raises(ValueError, match='observe'):
        build_mask_provider({'method': 'esp', 'num_masks': 1, 'hybrid_gain': {'mode': 'collect', 'observe': 'bonus'}},
                            torch.zeros(4, 1))
    model = _model()
    spec = _spec('esp', 1, 'static')
    spec['tree'] = {'mode': 'static', 'branches': [4], 'pruning': True}
    width = model.embedding_table.shape[-1]
    spec['hybrid_gain'] = {'mode': 'opened', 'observe': 'last', 'boost': 64, 'k': [0.0] * width,
                           'p': [0.01] * width, 'r': [0.04] * width, 'q': [0.0001] * width}
    provider = build_mask_provider(spec, model.embedding_table)
    prompt = PROMPTS[0]
    out, stats = SpecDecoder(model, spec, {'max_new_tokens': 20, 'temperature': 0.0, 'stop_on_eos': False},
                             generator=torch.Generator().manual_seed(1), mask_policy=provider).generate(prompt)
    ar_out, _ = ar_generate(model, prompt, {'max_new_tokens': 20, 'temperature': 0.0, 'stop_on_eos': False},
                            generator=torch.Generator().manual_seed(1))
    assert torch.equal(out, ar_out)
    assert stats['verification_argmax_audit']['passed'] is True
    assert provider.protocol == 'hybrid-gain-bonus-aligned-v1'


def test_round_split_groups_by_previous_outcome():
    trace = [{'accepted_nodes': [1, 2], 'committed_tokens': [5, 6, 7]},
             {'accepted_nodes': [], 'committed_tokens': [8]},
             {'accepted_nodes': [3], 'committed_tokens': [9, 10]},
             {'accepted_nodes': [], 'committed_tokens': [11]},
             {'accepted_nodes': [4], 'committed_tokens': [12]}]
    split = round_split(trace)
    assert split['after_prefill'] == {'rounds': 1, 'accepted': 2, 'committed': 3, 'tau': 3.0}
    assert split['after_accept'] == {'rounds': 2, 'accepted': 0, 'committed': 2, 'tau': 1.0}
    assert split['after_reject'] == {'rounds': 2, 'accepted': 2, 'committed': 3, 'tau': 1.5}
    assert round_split([])['after_reject']['tau'] is None
    with pytest.raises(ValueError, match='bonus'):
        round_split([{'accepted_nodes': [], 'committed_tokens': [1, 2]}])


def test_lowrank_delta_recovers_rank1_and_rejects_noise():
    generator = torch.Generator().manual_seed(0)
    width = 8
    train = torch.randn(60, width, generator=generator)
    hold = torch.randn(40, width, generator=generator)
    direction = torch.randn(width, generator=generator)
    loading = torch.tensor([0.4, -0.2, 0.1, 0.0, 0.3, -0.1, 0.2, 0.05])
    train_error = torch.outer(train @ direction, loading)
    hold_error = torch.outer(hold @ direction, loading)
    fit = fit_lowrank_delta(train, train_error, rank=1, ridge=1e-3, iters=8)
    center = train_error.mean(dim=0)
    assert centered_r2(hold_error, apply_lowrank_delta(hold, fit), center) > 0.98
    noise = torch.randn(60, width, generator=generator)
    noise_hold = torch.randn(40, width, generator=generator)
    null = fit_lowrank_delta(noise, train_error, rank=1, ridge=1.0, iters=8)
    assert centered_r2(hold_error, apply_lowrank_delta(noise_hold, null), center) < 0.2


def test_preforward_collector_matches_esp_tokens():
    model = _model()
    spec = _spec('esp', 1, 'static')
    prompt = PROMPTS[0]
    eval_cfg = {'max_new_tokens': 8, 'temperature': 0.0, 'stop_on_eos': False}
    plain = build_mask_provider(spec, model.embedding_table)
    reference, reference_stats = SpecDecoder(model, spec, eval_cfg, mask_policy=plain).generate(prompt)
    collector = PreforwardSignalCollector(ESPMaskProvider(1, model.embedding_table), model.embedding_table)
    original = model.forward

    def keeping_hidden(*args, **kwargs):
        kwargs['last_hidden_only'] = True
        return original(*args, **kwargs)

    model.forward = keeping_hidden
    try:
        output, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=collector).generate(prompt)
    finally:
        model.forward = original
    assert torch.equal(output, reference)
    assert stats['verification_argmax_audit']['passed'] is True
    assert reference_stats['verification_argmax_audit']['passed'] is True
    rows = collector.rows()
    assert rows
    assert rows[0]['hidden'].shape == (64,)
    assert rows[0]['expected'].shape == (64,)
    assert rows[0]['accepted'] >= 0
    bare = PreforwardSignalCollector(ESPMaskProvider(1, model.embedding_table), model.embedding_table)
    bare.init_from_prompt(model.embed(prompt)[0])
    bare.observe_mask_logits(torch.zeros(1, model.vocab_size))
    bare.on_token_committed(torch.zeros(64), torch.zeros(64))
    with pytest.raises(RuntimeError, match='mask hidden'):
        bare.masks()
