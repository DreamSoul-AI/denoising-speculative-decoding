"""Provider initialization and observation-blend tensor contracts."""
import pytest
import torch

from spec_mtp.structure.algorithm.decode.tree import DraftTree
from spec_mtp.structure.algorithm.mask.providers import (
    ObservationBlendMaskProvider, OutputHiddenMaskProvider, KalmanMaskProvider,
    OnlineVerifyKalman, AlignedResidualKalman, HybridGainKalman,
)
from spec_mtp.structure.algorithm.mask import ESPMaskProvider, EMAVelocityMaskProvider

@pytest.mark.parametrize('init, prompt_len, match', [
    ('nope', 4, 'Unsupported mask.init'),
    ('last_k', 1, 'last_k init needs'),
])
def test_esp_mask_init_errors(init, prompt_len, match, cuda_tiny_model, cuda_decode, record_property):
    tiny = cuda_tiny_model()
    table = tiny.embedding_table
    provider = ESPMaskProvider(2, table, init=init, lam=0.1)
    with pytest.raises(ValueError, match=match):
        provider.init_from_prompt(torch.zeros(prompt_len, table.size(-1), device=cuda_decode.device))


def test_ema_rejects_short_prompt_and_bad_level(cuda_tiny_model, cuda_decode, record_property):
    tiny = cuda_tiny_model()
    table = tiny.embedding_table
    with pytest.raises(ValueError, match='ema.level'):
        EMAVelocityMaskProvider(1, table, level='order2')
    provider = EMAVelocityMaskProvider(1, table)
    with pytest.raises(ValueError, match='EMA velocity needs'):
        provider.init_from_prompt(torch.zeros(1, table.size(-1), device=cuda_decode.device))


def test_masks_follow_the_blend_and_never_write_back(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    generator = torch.Generator(device=device).manual_seed(0)
    table = torch.randn(40, 6, generator=generator, device=device)
    prompt = torch.randn(9, 6, generator=generator, device=device)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(2, table, lam=0.2), table, 0.25)
    reference = ESPMaskProvider(2, table, lam=0.2)
    blend.init_from_prompt(prompt)
    reference.init_from_prompt(prompt)
    assert torch.equal(blend.masks(), reference.masks())
    logits = torch.randn(2, 40, generator=generator, device=device)
    blend.observe_mask_logits(logits)
    z = torch.softmax(logits, dim=-1) @ table
    assert torch.allclose(blend.observation(), z)
    assert torch.allclose(blend.masks(), reference.masks() + 0.25 * (z - reference.masks()))
    assert torch.equal(blend.base.masks(), reference.masks())
    token = torch.randn(6, generator=generator, device=device)
    blend.on_token_committed(token, prompt[-1])
    reference.on_token_committed(token, prompt[-1])
    assert torch.equal(blend.base.masks(), reference.masks())
    assert torch.allclose(blend.masks(), reference.masks() + 0.25 * (z - reference.masks()))
    full = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 1)
    full.init_from_prompt(prompt)
    full.observe_mask_logits(logits)
    assert torch.allclose(full.masks(), z)
    devices = {'table': str(table.device), 'logits': str(logits.device),
               'output': str(full.observation().device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_owner_depth_shifts_the_observation_and_falls_back_past_k(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    generator = torch.Generator(device=device).manual_seed(1)
    table = torch.randn(30, 5, generator=generator, device=device)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(3, table), table, 0.5)
    blend.init_from_prompt(torch.randn(6, 5, generator=generator, device=device))
    trajectory = blend.base.masks()
    assert torch.equal(blend.masks_by_depth(), trajectory.expand(4, 3, 5))
    logits = torch.randn(3, 30, generator=generator, device=device)
    blend.observe_mask_logits(logits)
    z = blend.observation()
    by_depth = blend.masks_by_depth()
    for depth in range(4):
        for slot in range(3):
            target = depth + slot
            expected = trajectory[slot] + 0.5 * (z[target] - trajectory[slot]) if target < 3 else trajectory[slot]
            assert torch.allclose(by_depth[depth, slot], expected)
    assert torch.equal(blend.masks(), by_depth[0])
    devices = {'table': str(table.device), 'logits': str(logits.device),
               'output': str(by_depth.device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_rejection_condition_renormalizes_slot_one_without_child_tokens(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    generator = torch.Generator(device=device).manual_seed(2)
    table = torch.randn(12, 4, generator=generator, device=device)
    blend = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 0.5, condition='rejection')
    assert blend.protocol == 'trajectory-observation-rejection-v1'
    blend.init_from_prompt(torch.randn(5, 4, generator=generator, device=device))
    trajectory = blend.base.masks()
    logits = torch.randn(2, 12, generator=generator, device=device)
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
    marginal.init_from_prompt(torch.randn(5, 4, generator=generator, device=device))
    marginal.observe_mask_logits(logits)
    assert torch.allclose(marginal.masks_for_tree(tree), marginal.masks_by_depth()[[0, 1, 1, 2]])
    devices = {'table': str(table.device), 'logits': str(logits.device),
               'output': str(owners.device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_rejection_condition_falls_back_when_children_hold_all_mass(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        table = torch.randn(6, 3, device=device)
        blend = ObservationBlendMaskProvider(ESPMaskProvider(1, table), table, 1.0, condition='rejection')
        blend.init_from_prompt(torch.randn(4, 3, device=device))
        logits = torch.full((1, 6), -1e9, device=device)
        logits[0, 2] = 0
        blend.observe_mask_logits(logits)
        owners = blend.masks_for_tree(DraftTree(root_token=0, tokens=[2], depths=[1], parents=[0]))
        assert torch.equal(owners[0, 0], blend.base.masks()[0])
        with pytest.raises(ValueError, match='condition'):
            ObservationBlendMaskProvider(ESPMaskProvider(1, table), table, 0.5, condition='kalman')
    devices = {'table': str(table.device), 'logits': str(logits.device),
               'output': str(owners.device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


@pytest.fixture
def table(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    g = torch.Generator(device=device).manual_seed(0)
    value = torch.randn(32, 8, generator=g, device=device)
    assert value.device == device
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('mask_table_device', str(value.device))
    return value


def test_esp_update_false_leaves_masks_unchanged(table, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        provider = ESPMaskProvider(2, table, init='mean_prompt', lam=0.5, update=False)
        prompt = torch.randn(5, 8, device=device)
        provider.init_from_prompt(prompt)
        before = provider.masks().clone()
        provider.on_token_committed(torch.randn(8, device=device), prompt[-1])
        assert torch.equal(provider.masks(), before)
    devices = {'table': str(table.device), 'prompt': str(prompt.device), 'mask': str(provider.masks().device)}
    assert set(devices.values()) == {str(device)}
    record_property('provider_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_esp_offset_and_sample_inits(table, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        prompt = torch.randn(4, 8, device=device)
        offset = ESPMaskProvider(2, table, init='offset_embedding', offset_scale=5.0)
        offset.init_from_prompt(prompt)
        mu, sigma = table.mean(0), table.std(0)
        assert torch.allclose(offset.masks()[0], mu + 5.0 * sigma)
        assert torch.equal(offset.masks()[0], offset.masks()[1])

        gen = torch.Generator(device=device).manual_seed(3)
        sample = ESPMaskProvider(2, table, init='sample_embedding', generator=gen)
        sample.init_from_prompt(prompt)
        assert sample.masks().shape == (2, 8)
        # Re-seeding the same generator reproduces the same noise.
        again = ESPMaskProvider(2, table, init='sample_embedding', generator=torch.Generator(device=device).manual_seed(3))
        again.init_from_prompt(prompt)
        assert torch.equal(sample.masks(), again.masks())
    devices = {'table': str(table.device), 'prompt': str(prompt.device), 'offset_mask': str(offset.masks().device), 'sample_mask': str(sample.masks().device), 'reseeded_mask': str(again.masks().device)}
    assert set(devices.values()) == {str(device)}
    record_property('provider_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_ema_update_false_freezes_velocity_but_moves_anchor(table, cuda_decode, record_property):
    """``update: false`` freezes the velocity EMA; the anchor still tracks the
    last committed token (documented in mask_providers), so masks() still moves."""
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        prompt = torch.randn(6, 8, device=device)
        frozen = EMAVelocityMaskProvider(2, table, update=False, extrapolate_ema=False, level='last')
        frozen.init_from_prompt(prompt)
        velocity_before = frozen._velocity.clone()
        ref_before = frozen._ref_norm.clone()
        new_embed = torch.randn(8, device=device)
        frozen.on_token_committed(new_embed, prompt[-1])
        assert torch.equal(frozen._velocity, velocity_before)
        assert torch.equal(frozen._ref_norm, ref_before)
        assert torch.equal(frozen._anchor, new_embed)
        # masks use the new anchor as origin → values change even with frozen velocity.
        assert not torch.allclose(frozen.masks(), velocity_before)  # shape differs anyway; pin anchor effect:
        assert torch.allclose(frozen.masks()[0], new_embed + frozen._step_vector(velocity_before, ref_before))
    devices = {'table': str(table.device), 'prompt': str(prompt.device), 'new_embed': str(new_embed.device), 'mask': str(frozen.masks().device)}
    assert set(devices.values()) == {str(device)}
    record_property('provider_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_ema_extrapolate_ema_rebuilds_masks_after_commit(table, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        prompt = torch.randn(6, 8, device=device)
        moving = EMAVelocityMaskProvider(2, table, update=True, extrapolate_ema=True, beta=0.5, step_scale=1.0)
        moving.init_from_prompt(prompt)
        first = moving.masks().clone()
        moving.on_token_committed(prompt[-1] + 1.0, prompt[-1])
        assert first.shape == moving.masks().shape
        assert not torch.equal(first, moving.masks())
    devices = {'table': str(table.device), 'prompt': str(prompt.device), 'mask': str(moving.masks().device)}
    assert set(devices.values()) == {str(device)}
    record_property('provider_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_top1_form_uses_the_most_likely_allowed_token(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        generator = torch.Generator(device=device).manual_seed(5)
        table = torch.randn(12, 4, generator=generator, device=device)
        logits = torch.randn(2, 12, generator=generator, device=device)
        logits[0, 3], logits[0, 9], logits[1, 7] = 9.0, 8.0, 9.0
        tree = DraftTree(root_token=0, tokens=[3, 5, 7], depths=[1, 1, 2], parents=[0, 0, 1])
        plain = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 1.0, form='top1')
        plain.init_from_prompt(torch.randn(5, 4, generator=generator, device=device))
        plain.observe_mask_logits(logits)
        assert plain.protocol == 'trajectory-observation-top1-v1'
        assert torch.allclose(plain.masks(), table[[3, 7]])
        rejecting = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 1.0, condition='rejection', form='top1')
        rejecting.init_from_prompt(torch.randn(5, 4, generator=generator, device=device))
        rejecting.observe_mask_logits(logits)
        assert rejecting.protocol == 'trajectory-observation-top1-rejection-v1'
        owners = rejecting.masks_for_tree(tree)
        assert torch.allclose(owners[0, 0], table[9])
        second = torch.softmax(logits[1], dim=-1)
        second[7] = -1
        assert torch.allclose(owners[1, 0], table[int(second.argmax())])
        assert torch.allclose(owners[0, 1], table[7]) and torch.allclose(owners[2, 0], table[7])
        with pytest.raises(ValueError, match='form'):
            ObservationBlendMaskProvider(ESPMaskProvider(1, table), table, 0.5, form='median')
    devices = {'table': str(table.device), 'logits': str(logits.device),
               'plain_output': str(plain.masks().device), 'output': str(owners.device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


def test_observation_rejects_hidden_providers_and_bad_logits(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        table = torch.randn(10, 4, device=device)
        with pytest.raises(ValueError, match='input-embedding'):
            ObservationBlendMaskProvider(OutputHiddenMaskProvider(1, table), table, 0.5)
        blend = ObservationBlendMaskProvider(ESPMaskProvider(2, table), table, 0.5)
        prompt = torch.randn(3, 4, device=device)
        blend.init_from_prompt(prompt)
        wrong_row_logits = torch.randn(3, 10, device=device)
        with pytest.raises(ValueError, match='one logit row'):
            blend.observe_mask_logits(wrong_row_logits)
        nan_logits = torch.full((2, 10), float('nan'), device=device)
        with pytest.raises(ValueError, match='finite candidate'):
            blend.observe_mask_logits(nan_logits)
    devices = {'table': str(table.device), 'prompt': str(prompt.device),
               'wrong_row_logits': str(wrong_row_logits.device), 'nan_logits': str(nan_logits.device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)

@pytest.mark.parametrize('level', ['last', 'order0'])
def test_ema_cuda_matches_independent_weighted_history(level, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    prompt = torch.tensor([[0., 1.], [2., 0.], [1., 4.], [4., 2.]], device=device)
    provider = EMAVelocityMaskProvider(2, torch.zeros(3, 2, device=device),
                                     beta=0.5, lam=0.25, step_scale=1.5, level=level)
    provider.init_from_prompt(prompt)
    # Closed-form weights on the three prompt differences are 1/4, 1/4, 1/2.
    velocities = torch.tensor([[1.75, -0.25], [1.375, 1.875], [-0.3125, 0.4375]],
                              device=device, dtype=torch.float64)
    anchors = torch.tensor([[4., 2.], [5., 6.], [3., 5.]], device=device, dtype=torch.float64)
    means = torch.tensor([[1.75, 1.75], [2.5625, 2.8125], [2.671875, 3.359375]],
                         device=device, dtype=torch.float64)
    for index in range(3):
        if index:
            provider.on_token_committed(anchors[index].float(), anchors[index - 1].float())
        origin = anchors[index] if level == 'last' else means[index]
        expected = origin + torch.tensor([[1.5], [3.]], device=device, dtype=torch.float64) * velocities[index]
        actual = provider.masks()
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(provider._velocity.double(), velocities[index], rtol=0, atol=0)
        torch.testing.assert_close(provider._level.double(), means[index] if level == 'order0' else means[0],
                                   rtol=0, atol=0)
    provider.init_from_prompt(prompt)
    origin = anchors[0] if level == 'last' else means[0]
    reset_expected = origin + torch.tensor([[1.5], [3.]], device=device, dtype=torch.float64) * velocities[0]
    torch.testing.assert_close(provider.masks().double(), reset_expected, rtol=0, atol=0)
    record_property('reference_tensor_devices', {'input': str(prompt.device), 'reference': str(expected.device),
                                                'output': str(actual.device)})
    record_property('reference_contract', 'EMA beta=.5 closed-form history; last/order0 anchors; two commits; reset')
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


@pytest.mark.parametrize('mode', ['online', 'frozen'])
def test_scalar_kalman_cuda_matches_independent_joseph_and_reset(mode, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    prompt = torch.tensor([[0., 1.], [2., 3.]], device=device)
    provider = KalmanMaskProvider(2, torch.zeros(3, 2, device=device), mode=mode,
                                 forget=0.8, r_min=0.05, q_min=0.03, q_scale=0.25)
    provider.init_from_prompt(prompt)
    level = prompt.double().mean(0)
    noise = (prompt.double() - level).square().mean().clamp_min(0.05)
    variance, process = noise.clone(), (noise * 0.25).clamp_min(0.03)
    gain = torch.zeros((), device=device, dtype=torch.float64)
    updates = 0

    # Conditional-Gaussian covariance is an independent oracle for production Joseph updates.
    def reference_step(observation, locked):
        nonlocal level, noise, variance, process, gain, updates
        covariance_prediction = variance + process
        residual = observation.double() - level
        selected_gain = gain if locked else covariance_prediction / (covariance_prediction + noise)
        level = level + selected_gain * residual
        if not locked:
            gain = selected_gain
            variance = covariance_prediction * noise / (covariance_prediction + noise)
            if mode == 'online':
                updates += 1
                weight = 0.2 / (1 - 0.8 ** (updates + 1))
                innovation_energy = residual.square().mean()
                noise = ((1 - weight) * noise + weight * (innovation_energy - covariance_prediction)).clamp_min(0.05)
                process = ((1 - weight) * process + weight * gain.square() * innovation_energy).clamp_min(0.03)

    for observed in prompt:
        reference_step(observed, False)
    initial_level = level.clone()
    initial_covariances = torch.stack((variance, process, noise, gain)).clone()
    initial_updates = updates
    observations = torch.tensor([[1., 2.], [5., -1.], [-2., 4.]], device=device)
    for observed in observations:
        previous_level = level.clone()
        reference_step(observed, mode == 'frozen')
        provider.on_token_committed(observed, torch.zeros(2, device=device))
        actual = provider.masks()
        torch.testing.assert_close(actual.double(), level.expand(2, 2), rtol=1e-5, atol=2e-6)
        diagnostics = provider.diagnostics()
        actual_covariances = torch.tensor([diagnostics[key] for key in ['p', 'q', 'r', 'k']],
                                          device=device, dtype=torch.float64)
        torch.testing.assert_close(actual_covariances, torch.stack((variance, process, noise, gain)),
                                   rtol=1e-5, atol=2e-6)
        if mode == 'frozen':
            torch.testing.assert_close(actual_covariances, initial_covariances, rtol=1e-6, atol=1e-6)
            assert not torch.equal(previous_level, level)
    provider.init_from_prompt(prompt)
    torch.testing.assert_close(provider.masks().double(), initial_level.expand(2, 2), rtol=1e-5, atol=2e-6)
    reset_state = torch.tensor([provider.diagnostics()[key] for key in ['p', 'q', 'r', 'k']],
                               device=device, dtype=torch.float64)
    torch.testing.assert_close(reset_state, initial_covariances, rtol=1e-5, atol=2e-6)
    assert provider._t == initial_updates and provider._locked == (mode == 'frozen')
    record_property('reference_tensor_devices', {'input': str(prompt.device), 'reference': str(level.device),
                                                'output': str(actual.device)})
    record_property('reference_contract', 'scalar random walk/Joseph; adaptive noise floors or fixed gain/covariance; reset')
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


@pytest.mark.parametrize('forget,q0,r0', [
    (0.8, 0.2, 0.4), (1.0, 0.2, 0.4), (1.0, 0.001, 0.002),
], ids=['adaptive', 'fixed', 'fixed-below-floor'])
def test_online_verify_cuda_matches_independent_coordinate_filter(forget, q0, r0, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    prompt = torch.tensor([[0., 1.], [2., 3.]], device=device)
    base = ESPMaskProvider(2, torch.zeros(3, 2, device=device), update=False)
    provider = OnlineVerifyKalman(base, part='full', forget=forget, p0=0.7, q0=q0, r0=r0,
                                 r_min=0.05, q_min=0.03)
    provider.init_from_prompt(prompt)
    prior = torch.tensor([1., 2.], device=device, dtype=torch.float64)
    variance, process, noise = [torch.full((2,), value, device=device, dtype=torch.float64)
                                for value in [0.7, q0, r0]]
    observations = torch.tensor([[1.2, 5.], [-1., 2.5], [4., -2.]], device=device)
    previous_observation = None
    for index, observed in enumerate(observations):
        predicted = variance + process
        gain = predicted / (predicted + noise)
        corrected = prior if previous_observation is None else prior + gain * (previous_observation - prior)
        expected = torch.stack((corrected, prior))
        actual = provider.masks()
        torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(provider.masks(), actual, rtol=0, atol=0)
        residual = observed.double() - prior
        variance = predicted * noise / (predicted + noise)
        # forget=1 preserves supplied Q/R, including below-floor initial values.
        if forget < 1:
            weight = (1 - forget) / (1 - forget ** (index + 2))
            process = ((1 - weight) * process + weight * gain.square() * residual.square()).clamp_min(0.03)
            noise = ((1 - weight) * noise + weight * (residual.square() - predicted)).clamp_min(0.05)
        provider.on_token_committed(observed, prompt[-1])
        torch.testing.assert_close(provider._p.double(), variance, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(provider._q.double(), process, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(provider._r.double(), noise, rtol=1e-5, atol=2e-6)
        assert provider._t == index + 1 and provider._pending is None
        torch.testing.assert_close(provider._z, observed, rtol=0, atol=0)
        # A later commit in this round does not create another root-prior update.
        provider.on_token_committed(observed + 7, observed)
        assert provider._t == index + 1
        torch.testing.assert_close(provider._z, observed, rtol=0, atol=0)
        torch.testing.assert_close(provider._p.double(), variance, rtol=1e-5, atol=2e-6)
        previous_observation = observed.double()
    provider.masks()
    provider.init_from_prompt(prompt + 2)
    assert provider._pending is None
    torch.testing.assert_close(provider._q, torch.full((2,), q0, device=device), rtol=0, atol=0)
    torch.testing.assert_close(provider._r, torch.full((2,), r0, device=device), rtol=0, atol=0)
    torch.testing.assert_close(provider.masks(), (prior.float() + 2).expand(2, 2), rtol=0, atol=0)
    assert provider._z is None and provider._t == 0 and not provider._seen
    torch.testing.assert_close(provider._p, torch.full((2,), 0.7, device=device), rtol=0, atol=0)
    record_property('reference_tensor_devices', {'input': str(prompt.device), 'reference': str(expected.device),
                                                'output': str(actual.device), 'covariance': str(provider._p.device)})
    record_property('reference_contract', 'coordinate Joseph/adaptive floors; one update/first commit per round; reset')
    record_property('noise_adaptation', forget != 1)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


@pytest.mark.parametrize('forget,q0,r0', [
    (0.8, 0.2, 0.4), (1.0, 0.2, 0.4), (1.0, 0.001, 0.002),
], ids=['adaptive', 'fixed', 'fixed-below-floor'])
def test_aligned_residual_cuda_matches_independent_scalar_filter(forget, q0, r0, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    prompt = torch.tensor([[0., 1.], [2., 3.]], device=device)
    base = ESPMaskProvider(2, torch.zeros(3, 2, device=device), update=False)
    provider = AlignedResidualKalman(base, part='full', forget=forget, p0=0.7, q0=q0, r0=r0,
                                    r_min=0.05, q_min=0.03)
    provider.init_from_prompt(prompt)
    prior = torch.tensor([1., 2.], device=device, dtype=torch.float64)
    coefficient, variance, process, noise = [torch.tensor(value, device=device, dtype=torch.float64)
                                             for value in [0., 0.7, q0, r0]]
    previous_residual = None
    observations = torch.tensor([[3., 2.], [2., 4.], [0., 1.], [1., 2.]], device=device)
    for index, observed in enumerate(observations):
        corrected = prior if previous_residual is None else prior + coefficient * previous_residual
        expected = torch.stack((corrected, prior))
        actual = provider.masks()
        torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)
        residual = observed.double() - prior
        if previous_residual is not None:
            norm_product = torch.linalg.vector_norm(residual) * torch.linalg.vector_norm(previous_residual)
            cosine = torch.zeros((), device=device, dtype=torch.float64) if norm_product == 0 else (
                (residual * previous_residual).sum() / norm_product)
            predicted = variance + process
            gain = predicted / (predicted + noise)
            old_coefficient = coefficient
            coefficient = (old_coefficient + gain * (cosine - old_coefficient)).clamp(-1, 1)
            variance = predicted * noise / (predicted + noise)
            if forget < 1:
                weight = (1 - forget) / (1 - forget ** (index + 1))
                noise = ((1 - weight) * noise + weight * (cosine - old_coefficient).square()).clamp_min(0.05)
                process = ((1 - weight) * process + weight * (coefficient - old_coefficient).square()).clamp_min(0.03)
        provider.on_token_committed(observed, prompt[-1])
        diagnostics = provider.diagnostics()
        actual_state = torch.tensor([diagnostics[key] for key in ['f', 'p', 'q', 'r']],
                                    device=device, dtype=torch.float64)
        torch.testing.assert_close(actual_state, torch.stack((coefficient, variance, process, noise)),
                                   rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(provider._nu.double(), residual, rtol=0, atol=0)
        assert diagnostics['updates'] == index and provider._pending is None
        previous_residual = residual
    provider.masks()
    provider.init_from_prompt(prompt - 2)
    assert provider._nu is None and provider._pending is None and not provider._seen
    assert provider.diagnostics() == {'part': 'full', 'updates': 0, 'f': 0., 'p': 0.7, 'q': q0, 'r': r0}
    torch.testing.assert_close(provider.masks(), (prior.float() - 2).expand(2, 2), rtol=0, atol=0)
    record_property('reference_tensor_devices', {'input': str(prompt.device), 'reference': str(expected.device),
                                                'output': str(actual.device), 'state': str(provider._state.device)})
    record_property('reference_contract', 'scalar residual cosine/F/Joseph; zero residual; one update per round; reset')
    record_property('noise_adaptation', forget != 1)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)

@pytest.mark.parametrize('variant', ['first', 'last', 'frozen'])
def test_hybrid_gain_cuda_matches_independent_first_last_and_frozen(variant, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    prompt = torch.zeros(2, 2, device=device)
    payload = {'k': [0.25, -0.5], 'p': [0.5, 0.25], 'q': [0.125, 0.0625], 'r': [0.75, 0.5],
               'observe': 'last' if variant == 'last' else 'first'}
    base = ESPMaskProvider(2, torch.zeros(3, 2, device=device), update=False)
    provider = HybridGainKalman(base, mode='frozen' if variant == 'frozen' else 'hybrid', payload=payload)
    provider.init_from_prompt(prompt)
    gain0, variance0, process, noise = [torch.tensor(payload[key], device=device, dtype=torch.float64)
                                       for key in ['k', 'p', 'q', 'r']]
    prefill = torch.tensor([2., -1.], device=device)
    first = torch.tensor([1., 3.], device=device)
    last = torch.tensor([5., 2.], device=device)
    provider.on_token_committed(prefill, prompt[-1])
    actual = provider.masks()
    torch.testing.assert_close(actual[0].double(), gain0 * prefill.double(), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], torch.zeros(2, device=device), rtol=0, atol=0)
    # The regression covariate belongs to the pre-round z/prior snapshot.
    regressor = prefill.double()
    predicted_variance = variance0 + process
    update_step = predicted_variance * regressor / (noise + predicted_variance * regressor.square())
    target = last.double() if variant == 'last' else first.double()
    expected_gain = gain0 if variant == 'frozen' else gain0 + update_step * (target - gain0 * regressor)
    expected_variance = variance0 if variant == 'frozen' else (
        predicted_variance * noise / (noise + predicted_variance * regressor.square()))
    provider.on_token_committed(first, prefill)
    torch.testing.assert_close(provider._k.double(), expected_gain if variant == 'first' else gain0,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(provider._p.double(), expected_variance if variant == 'first' else variance0,
                               rtol=1e-6, atol=1e-6)
    provider.on_token_committed(last, first)
    torch.testing.assert_close(provider._z, last, rtol=0, atol=0)
    torch.testing.assert_close(provider._k.double(), expected_gain if variant == 'first' else gain0,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(provider._p.double(), expected_variance if variant == 'first' else variance0,
                               rtol=1e-6, atol=1e-6)
    provider.finish_round()
    torch.testing.assert_close(provider._k.double(), expected_gain, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(provider._p.double(), expected_variance, rtol=1e-6, atol=1e-6)
    assert provider._pending is None and provider._weight_n == (0 if variant == 'frozen' else 1)
    provider.finish_round()
    torch.testing.assert_close(provider._k.double(), expected_gain, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(provider._p.double(), expected_variance, rtol=1e-6, atol=1e-6)
    assert provider._weight_n == (0 if variant == 'frozen' else 1)
    next_masks = provider.masks()
    torch.testing.assert_close(next_masks[0].double(), expected_gain * last.double(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(provider._q.double(), process, rtol=0, atol=0)
    torch.testing.assert_close(provider._r.double(), noise, rtol=0, atol=0)
    provider.init_from_prompt(prompt)
    assert provider._z is None and provider._pending is None and provider._first is None
    assert provider._weight_n == 0
    torch.testing.assert_close(provider._k.double(), gain0, rtol=0, atol=0)
    torch.testing.assert_close(provider._p.double(), variance0, rtol=0, atol=0)
    torch.testing.assert_close(provider.masks(), torch.zeros(2, 2, device=device), rtol=0, atol=0)
    record_property('reference_tensor_devices', {'input': str(prompt.device), 'reference': str(expected_gain.device),
                                                'output': str(next_masks.device), 'covariance': str(provider._p.device)})
    record_property('reference_contract', 'diagonal regression KF; first/last/frozen; multi-commit/finish idempotence/reset')
    record_property('observation_target', variant)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)
