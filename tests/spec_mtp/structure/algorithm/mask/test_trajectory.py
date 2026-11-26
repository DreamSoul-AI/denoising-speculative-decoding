"""Trajectory algebra, observation timing and tree-position contracts, on CPU."""

import pytest
import torch

from spec_mtp.structure.algorithm.decode.tree import DraftTree
from spec_mtp.structure.algorithm.decode.tree_attention import build_block_attention, mask_indices_of
from spec_mtp.structure.algorithm.mask import ESPMaskProvider, InputTrajectoryMaskProvider, build_mask_provider
from spec_mtp.structure.algorithm.mask.trajectory import covariance_step, steady_state_gain


def test_zero_order_matches_esp_under_common_initialization():
    table = torch.randn(16, 5, generator=torch.Generator().manual_seed(3))
    prompt = table[:4]
    ours = InputTrajectoryMaskProvider(3, alpha=.1)
    esp = ESPMaskProvider(3, table, lam=.1)
    for provider in (ours, esp):
        provider.init_from_prompt(prompt)
    for y in table[4:]:
        ours.on_token_committed(y, prompt[-1])
        esp.on_token_committed(y, prompt[-1])
        torch.testing.assert_close(ours.masks(), esp.masks())
    assert ours.committed_steps == 12
    assert torch.count_nonzero(ours.V) == 0


def test_holt_uses_prediction_residual_not_raw_displacement():
    prompt = torch.tensor([[0.], [4.]])
    holt = InputTrajectoryMaskProvider(2, kind='holt', alpha=.25, beta=.5)
    difference = InputTrajectoryMaskProvider(2, kind='difference', alpha=.25, beta=.5)
    for provider in (holt, difference):
        provider.init_from_prompt(prompt)
        provider.on_token_committed(torch.tensor([6.]), prompt[-1])
    torch.testing.assert_close(holt.M, torch.tensor([3.]))
    torch.testing.assert_close(holt.V, torch.tensor([.5]))
    torch.testing.assert_close(difference.V, torch.tensor([1.]))
    holt.on_token_committed(torch.tensor([8.]), torch.tensor([6.]))
    torch.testing.assert_close(holt.M, torch.tensor([4.625]))
    torch.testing.assert_close(holt.V, torch.tensor([1.0625]))


def test_alpha_one_holt_equals_difference_only_with_matching_initial_anchor():
    # Constant prompt ensures mean(prompt) == last(prompt); otherwise the
    # first innovation includes a level error missing from raw differences.
    providers = [InputTrajectoryMaskProvider(3, kind=kind, alpha=1, beta=.7)
                 for kind in ('difference', 'holt')]
    for provider in providers:
        provider.init_from_prompt(torch.tensor([[2., 1.], [2., 1.]]))
    for y in (torch.tensor([4., -1.]), torch.tensor([1., 3.]), torch.tensor([5., 6.])):
        for provider in providers:
            provider.on_token_committed(y, torch.zeros(2))
        torch.testing.assert_close(providers[0].masks(), providers[1].masks())


@pytest.mark.parametrize('horizon', ['owner_depth', 'slot'])
def test_tree_readout_is_pure_and_matches_position_ids(horizon):
    provider = InputTrajectoryMaskProvider(2, kind='holt', alpha=.5, beta=.5, horizon=horizon)
    provider.init_from_prompt(torch.tensor([[0., 0.], [2., 4.]]))
    provider.on_token_committed(torch.tensor([5., 8.]), torch.tensor([2., 4.]))
    tree = DraftTree(1, tokens=[2, 3, 4], depths=[1, 1, 2], parents=[0, 0, 1])
    before = (provider.M.clone(), provider.V.clone(), provider.last_observation.clone(), provider.committed_steps)
    masks = provider.masks_for_tree(tree)
    _, positions = build_block_attention(tree, 2, 7, 7, torch.float32, torch.device('cpu'), -1e30)
    for owner, depth in enumerate([0] + tree.depths):
        for slot, row in enumerate(mask_indices_of(tree, 2, owner), start=1):
            assert positions[0, row].item() - 7 == depth + slot
            h = depth + slot if horizon == 'owner_depth' else slot
            torch.testing.assert_close(masks[owner, slot - 1], provider.M + h * provider.V)
    torch.testing.assert_close(masks[1], masks[2])
    tree.tokens[:] = [9, 8, 7]  # Draft identity is not a committed observation.
    torch.testing.assert_close(provider.masks_for_tree(tree), masks)
    provider.masks_for_tree(tree).zero_()  # Returned storage must not alias state.
    provider.masks().zero_()
    for actual, expected in zip((provider.M, provider.V, provider.last_observation), before[:3]):
        assert torch.equal(actual, expected)
    assert provider.committed_steps == before[3]


def test_difference_last_anchor_is_an_explicit_separate_control():
    level = InputTrajectoryMaskProvider(2, kind='difference', alpha=.1)
    last = InputTrajectoryMaskProvider(2, kind='difference', alpha=.1, anchor='last')
    for provider in (level, last):
        provider.init_from_prompt(torch.tensor([[0.], [4.]]))
        provider.on_token_committed(torch.tensor([8.]), torch.tensor([4.]))
    torch.testing.assert_close(level.V, last.V)
    torch.testing.assert_close(last.masks() - level.masks(), torch.full((2, 1), 5.4))


def test_kalman_first_gain_matches_closed_form_and_holt_equivalent_step():
    provider = InputTrajectoryMaskProvider(2, kind='kalman', q_level=.1, q_velocity=.2, r=.5)
    prompt = torch.tensor([[0., 1.], [4., 3.]])
    provider.init_from_prompt(prompt)
    prior_M = provider.M.clone()
    y = torch.tensor([6., 9.])
    provider.on_token_committed(y, prompt[-1])
    torch.testing.assert_close(provider.gain, torch.tensor([2.1 / 2.6, 1 / 2.6], dtype=torch.float64))
    torch.testing.assert_close(provider.M, prior_M + (2.1 / 2.6) * (y - prior_M))
    torch.testing.assert_close(provider.V, (1 / 2.6) * (y - prior_M))
    holt = InputTrajectoryMaskProvider(2, kind='holt', alpha=2.1 / 2.6, beta=1 - 1 / 2.1)
    holt.init_from_prompt(prompt)
    holt.on_token_committed(y, prompt[-1])
    torch.testing.assert_close(holt.masks(), provider.masks())


@pytest.mark.parametrize('kind', ['kalman', 'fixed_gain'])
def test_covariance_is_psd_and_gain_is_not_residual_adaptive(kind):
    providers = [InputTrajectoryMaskProvider(2, kind=kind, q_level=1e-4, q_velocity=.01) for _ in range(2)]
    for provider in providers:
        provider.init_from_prompt(torch.tensor([[0., 1.], [2., 5.]]))
    frozen_gain = providers[0].gain.clone() if kind == 'fixed_gain' else None
    for t in range(200):
        for index, provider in enumerate(providers):
            provider.on_token_committed(torch.tensor([float(t), float(index * 1000 - t)]), torch.zeros(2))
            assert torch.isfinite(provider.covariance).all()
            torch.testing.assert_close(provider.covariance, provider.covariance.T)
            assert torch.linalg.eigvalsh(provider.covariance).min() >= -1e-12
        torch.testing.assert_close(providers[0].gain, providers[1].gain)
    if frozen_gain is not None:
        assert torch.equal(providers[0].gain, frozen_gain)
    posterior, gain = covariance_step(providers[0].covariance, providers[0].process_noise, 1.)
    torch.testing.assert_close(posterior, providers[0].covariance, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(gain, providers[0].gain, atol=1e-8, rtol=1e-8)


def test_steady_state_solver_rejects_nonconvergence():
    with pytest.raises(ValueError, match='did not converge'):
        steady_state_gain(torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64), 1., max_steps=1)


@pytest.mark.parametrize('patch', [
    {'kind': 'unknown'}, {'alpha': float('nan')}, {'beta': 1.1}, {'r': 0},
    {'q_velocity': -1}, {'num_masks': True}, {'anchor': 'last'}, {'horizon': 'flat'},
])
def test_invalid_configuration_is_rejected(patch):
    with pytest.raises(ValueError):
        InputTrajectoryMaskProvider(**({'num_masks': 2} | patch))


def test_factory_rejects_composite_experiments_and_unknown_fields():
    spec = {'method': 'input_trajectory', 'num_masks': 2, 'trajectory': {'kind': 'holt'}}
    assert isinstance(build_mask_provider(spec, torch.eye(4)), InputTrajectoryMaskProvider)
    for key in ('observation', 'bonus_slot', 'correction', 'kalman_merge', 'kalman_align', 'hybrid_gain', 'output_kalman'):
        with pytest.raises(ValueError, match='does not combine'):
            build_mask_provider(spec | {key: {}}, torch.eye(4))
    with pytest.raises(ValueError, match='unknown field'):
        build_mask_provider(spec | {'trajectory': {'alhpa': .2}}, torch.eye(4))


def test_reinitialization_and_invalid_observation_preserve_contract():
    provider = InputTrajectoryMaskProvider(2, kind='kalman')
    prompt = torch.tensor([[2., 4.]])  # A single-token prompt is sufficient.
    provider.init_from_prompt(prompt)
    before = provider.masks().clone()
    for y in (torch.tensor([float('nan'), 2.]), torch.zeros(3)):
        with pytest.raises(ValueError, match='finite matching'):
            provider.on_token_committed(y, prompt[-1])
        assert provider.committed_steps == 0
        assert torch.equal(provider.masks(), before)
    provider.on_token_committed(torch.tensor([7., 6.]), prompt[-1])
    provider.init_from_prompt(prompt)
    assert provider.committed_steps == 0 and torch.count_nonzero(provider.V) == 0
    assert torch.equal(provider.masks(), before)
    torch.testing.assert_close(provider.covariance, torch.eye(2, dtype=torch.float64))
