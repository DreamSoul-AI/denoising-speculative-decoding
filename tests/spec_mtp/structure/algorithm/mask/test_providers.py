"""Provider initialization and observation-blend tensor contracts."""
import pytest
import torch

from spec_mtp.structure.algorithm.decode.tree import DraftTree
from spec_mtp.structure.algorithm.mask.providers import ObservationBlendMaskProvider
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
