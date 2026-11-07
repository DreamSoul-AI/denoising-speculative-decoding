"""Bounded CUDA RNG and checkpoint contracts, independent of training orchestration."""




import pytest

import torch

from spec_mtp.structure.algorithm.resume import capture_rng, restore_rng

def test_unfrozen_cuda_rng_state_does_not_query_or_initialize_cuda(monkeypatch):
    original = capture_rng()
    saved = {**original, 'cuda': None}

    def forbidden(*args, **kwargs):
        pytest.fail('an unfrozen CUDA domain must remain untouched')

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'is_available', forbidden)
        patch.setattr(torch.cuda, 'device_count', forbidden)
        patch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
        restore_rng(saved)
    assert capture_rng()['python'] == original['python']
    assert torch.equal(capture_rng()['torch'], original['torch'])
