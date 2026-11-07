"""Factory method admission before any mask execution."""

import pytest

import torch

from spec_mtp.structure.algorithm.mask.providers import ESPMaskProvider, EMAVelocityMaskProvider

from studies._shared.structure.config import _convert

from spec_mtp.structure.algorithm.mask import build_mask_provider

def test_build_mask_provider_wires_esp_and_ema(cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    g = torch.Generator(device=device).manual_seed(0)
    table = torch.randn(32, 8, generator=g, device=device)
    esp = build_mask_provider(_convert({
        'method': 'esp', 'num_masks': 1, 'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
    }), table)
    assert isinstance(esp, ESPMaskProvider)
    ema = build_mask_provider(_convert({
        'method': 'ema_velocity', 'num_masks': 1,
        'ema': {'beta': 0.9, 'step_scale': 0, 'level': 'last', 'update': True},
    }), table)
    assert isinstance(ema, EMAVelocityMaskProvider)
    devices = {'table': str(table.device), 'esp_embedding': str(esp.embedding_table.device), 'ema_embedding': str(ema.embedding_table.device)}
    assert set(devices.values()) == {str(device)}
    record_property('provider_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)

@pytest.mark.parametrize('spec_observation,match', [
    ({'gain': 1.5}, 'gain'), ({'gain': True}, 'gain'), ({'gain': 0.5, 'K': 1}, 'only gain')])
def test_invalid_observation_settings_are_rejected(spec_observation, match, cuda_decode, record_property):
    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]):
        spec = {'method': 'esp', 'num_masks': 1, 'impl': 'efficient',
                'block_complexity': None,
                'tree': {'mode': 'static', 'branches': [7], 'pruning': True},
                'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
                'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False, 'update': True},
                'observation': spec_observation}
        table = torch.randn(10, 4, device=device)
        with pytest.raises(ValueError, match=match):
            build_mask_provider(spec, table)
    devices = {'table': str(table.device)}
    assert set(devices.values()) == {str(device)}
    record_property('observation_tensor_devices', devices)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)

@pytest.mark.parametrize('section,field', [
    ('kalman', 'r_min'), ('kalman', 'q_min'), ('kalman', 'q_scale'),
    ('kalman_merge', 'p0'), ('kalman_merge', 'q0'), ('kalman_merge', 'r0'),
    ('kalman_merge', 'r_min'), ('kalman_merge', 'q_min'),
    ('kalman_align', 'p0'), ('kalman_align', 'q0'), ('kalman_align', 'r0'),
    ('kalman_align', 'r_min'), ('kalman_align', 'q_min'),
])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')], ids=['nan', 'inf', 'negative-inf'])
def test_kalman_factory_rejects_nonfinite_variances(section, field, value, cuda_decode, record_property):
    table = torch.tensor([[0., 1.], [1., 2.]], device=cuda_decode.device)
    spec = {'method': 'kalman' if section == 'kalman' else 'esp', 'num_masks': 1,
            section: {field: value}}
    with pytest.raises(ValueError, match=field + ' must be a finite positive number'):
        build_mask_provider(spec, table)
    record_property('admission_tensor_device', str(table.device))
