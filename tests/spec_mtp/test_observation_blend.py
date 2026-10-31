"""Protocol trajectory-observation-blend-v1: identity, losslessness and algebra on CPU."""

import pytest
import torch

from spec_mtp.analysis import round_split
from spec_mtp.decode import SpecDecoder, ar_generate
from spec_mtp.masks import ESPMaskProvider, ObservationBlendMaskProvider, OutputHiddenMaskProvider, build_mask_provider
from spec_mtp.model import HFCausalLM
from spec_mtp.tree import DraftTree


def _model():
    torch.manual_seed(7)
    return HFCausalLM(tiny_config={'vocab_size': 199, 'hidden_size': 64, 'intermediate_size': 128,
                                   'num_layers': 2, 'num_heads': 4, 'num_kv_heads': 2,
                                   'max_position_embeddings': 1024})


def _spec(method, k, mode, gain=None, condition='none'):
    spec = {'method': method, 'num_masks': k, 'impl': 'efficient' if mode == 'static' else 'naive',
            'block_complexity': None if mode == 'static' else 30,
            'tree': {'mode': mode, 'branches': [7, 2][:k] if mode == 'static' else None, 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
            'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False, 'update': True}}
    if gain is not None:
        spec['observation'] = {'gain': gain, 'condition': condition}
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
def test_blended_masks_keep_ar_output_and_argmax_audit(method, k, mode, gain, temperature, condition):
    model = _model()
    for index, prompt in enumerate(PROMPTS):
        out, stats = _decode(model, _spec(method, k, mode, gain, condition), prompt, temperature, 50 + index)
        ar_out, _ = ar_generate(model, prompt, {'max_new_tokens': 20, 'temperature': temperature, 'stop_on_eos': False},
                                generator=torch.Generator().manual_seed(50 + index))
        assert torch.equal(out, ar_out)
        if temperature == 0:
            assert stats['verification_argmax_audit']['passed'] is True


def test_masks_follow_the_blend_and_never_write_back():
    torch.manual_seed(0)
    table = torch.randn(40, 6)
    prompt = torch.randn(9, 6)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(2, table, lam=0.2), table, 0.25)
    reference = ESPMaskProvider(2, table, lam=0.2)
    blend.init_from_prompt(prompt)
    reference.init_from_prompt(prompt)
    assert torch.equal(blend.masks(), reference.masks())
    logits = torch.randn(2, 40)
    blend.observe_mask_logits(logits)
    z = torch.softmax(logits, dim=-1) @ table
    assert torch.allclose(blend.observation(), z)
    assert torch.allclose(blend.masks(), reference.masks() + 0.25 * (z - reference.masks()))
    assert torch.equal(blend.base.masks(), reference.masks())
    token = torch.randn(6)
    blend.on_token_committed(token, prompt[-1])
    reference.on_token_committed(token, prompt[-1])
    assert torch.equal(blend.base.masks(), reference.masks())
    assert torch.allclose(blend.masks(), reference.masks() + 0.25 * (z - reference.masks()))
    full = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 1)
    full.init_from_prompt(prompt)
    full.observe_mask_logits(logits)
    assert torch.allclose(full.masks(), z)


def test_owner_depth_shifts_the_observation_and_falls_back_past_k():
    torch.manual_seed(1)
    table = torch.randn(30, 5)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(3, table), table, 0.5)
    blend.init_from_prompt(torch.randn(6, 5))
    trajectory = blend.base.masks()
    assert torch.equal(blend.masks_by_depth(), trajectory.expand(4, 3, 5))
    blend.observe_mask_logits(torch.randn(3, 30))
    z = blend.observation()
    by_depth = blend.masks_by_depth()
    for depth in range(4):
        for slot in range(3):
            target = depth + slot
            expected = trajectory[slot] + 0.5 * (z[target] - trajectory[slot]) if target < 3 else trajectory[slot]
            assert torch.allclose(by_depth[depth, slot], expected)
    assert torch.equal(blend.masks(), by_depth[0])


def test_rejection_condition_renormalizes_slot_one_without_child_tokens():
    torch.manual_seed(2)
    table = torch.randn(12, 4)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 0.5, condition='rejection')
    assert blend.protocol == 'trajectory-observation-rejection-v1'
    blend.init_from_prompt(torch.randn(5, 4))
    trajectory = blend.base.masks()
    logits = torch.randn(2, 12)
    blend.observe_mask_logits(logits)
    probs = torch.softmax(logits, dim=-1)
    # root -> nodes 1 (token 3), 2 (token 5); node 1 -> node 3 (token 7); nodes 2, 3 are leaves.
    tree = DraftTree(root_token=0, tokens=[3, 5, 7], depths=[1, 1, 2], parents=[0, 0, 1])
    owners = blend.masks_for_tree(tree)
    by_depth = blend.masks_by_depth()

    def conditioned(depth, excluded):
        p = probs[depth].clone()
        p[excluded] = 0
        return (p / p.sum()) @ table

    assert torch.allclose(owners[0, 0], trajectory[0] + 0.5 * (conditioned(0, [3, 5]) - trajectory[0]), atol=1e-5)
    assert torch.allclose(owners[0, 1], by_depth[0, 1])
    assert torch.allclose(owners[1, 0], trajectory[0] + 0.5 * (conditioned(1, [7]) - trajectory[0]), atol=1e-5)
    assert torch.allclose(owners[2], by_depth[1]) and torch.allclose(owners[3], by_depth[2])
    marginal = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 0.5)
    marginal.init_from_prompt(torch.randn(5, 4))
    marginal.observe_mask_logits(logits)
    assert torch.allclose(marginal.masks_for_tree(tree), marginal.masks_by_depth()[[0, 1, 1, 2]])


def test_rejection_condition_falls_back_when_children_hold_all_mass():
    table = torch.randn(6, 3)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(1, table), table, 1.0, condition='rejection')
    blend.init_from_prompt(torch.randn(4, 3))
    logits = torch.full((1, 6), -1e9)
    logits[0, 2] = 0
    blend.observe_mask_logits(logits)
    owners = blend.masks_for_tree(DraftTree(root_token=0, tokens=[2], depths=[1], parents=[0]))
    assert torch.equal(owners[0, 0], blend.base.masks()[0])
    with pytest.raises(ValueError, match='condition'):
        ObservationBlendMaskProvider(ESPMaskProvider(1, table), table, 0.5, condition='kalman')


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


@pytest.mark.parametrize('spec_observation,match', [
    ({'gain': 1.5}, 'gain'), ({'gain': True}, 'gain'), ({'gain': 0.5, 'K': 1}, 'only gain')])
def test_invalid_observation_settings_are_rejected(spec_observation, match):
    spec = {**_spec('esp', 1, 'static'), 'observation': spec_observation}
    with pytest.raises(ValueError, match=match):
        build_mask_provider(spec, torch.randn(10, 4))


def test_observation_rejects_hidden_providers_and_bad_logits():
    table = torch.randn(10, 4)
    with pytest.raises(ValueError, match='input-embedding'):
        ObservationBlendMaskProvider(OutputHiddenMaskProvider(1, table), table, 0.5)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 0.5)
    blend.init_from_prompt(torch.randn(3, 4))
    with pytest.raises(ValueError, match='one logit row'):
        blend.observe_mask_logits(torch.randn(3, 10))
    with pytest.raises(ValueError, match='finite candidate'):
        blend.observe_mask_logits(torch.full((2, 10), float('nan')))


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
