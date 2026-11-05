"""Provider initialization and its tightly coupled factory admission."""
import pytest
import torch
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
