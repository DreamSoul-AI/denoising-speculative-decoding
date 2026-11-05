"""Factory method admission before any mask execution."""
import pytest
import torch
from spec_mtp.structure.algorithm.mask.providers import ESPMaskProvider, EMAVelocityMaskProvider
from studies._shared.structure.config import _convert
from spec_mtp.structure.algorithm.mask import build_mask_provider


def test_build_mask_provider_rejects_unknown_method(cuda_tiny_model, cuda_decode, record_property):
    tiny = cuda_tiny_model()
    with pytest.raises(ValueError, match='Unsupported spec.method'):
        build_mask_provider(_convert({'method': 'magic', 'num_masks': 1}), tiny.embedding_table)


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
