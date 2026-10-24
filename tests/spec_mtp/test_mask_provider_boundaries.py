"""mask_providers error paths and update / extrapolate edges (no weights).

Statement coverage is already full via other suites; these pin the contracts
explicitly so a regression in an error message or update=False path is obvious.
"""

from __future__ import annotations

import pytest
import torch

from spec_mtp.masks import (
    EMAVelocityMaskProvider, ESPMaskProvider, build_mask_provider,
)
from studies._shared.structure.config import _convert


@pytest.fixture
def table():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 8, generator=g)


def test_esp_update_false_leaves_masks_unchanged(table):
    provider = ESPMaskProvider(2, table, init='mean_prompt', lam=0.5, update=False)
    prompt = torch.randn(5, 8)
    provider.init_from_prompt(prompt)
    before = provider.masks().clone()
    provider.on_token_committed(torch.randn(8), prompt[-1])
    assert torch.equal(provider.masks(), before)


def test_esp_offset_and_sample_inits(table):
    prompt = torch.randn(4, 8)
    offset = ESPMaskProvider(2, table, init='offset_embedding', offset_scale=5.0)
    offset.init_from_prompt(prompt)
    mu, sigma = table.mean(0), table.std(0)
    assert torch.allclose(offset.masks()[0], mu + 5.0 * sigma)
    assert torch.equal(offset.masks()[0], offset.masks()[1])

    gen = torch.Generator().manual_seed(3)
    sample = ESPMaskProvider(2, table, init='sample_embedding', generator=gen)
    sample.init_from_prompt(prompt)
    assert sample.masks().shape == (2, 8)
    # Re-seeding the same generator reproduces the same noise.
    again = ESPMaskProvider(2, table, init='sample_embedding', generator=torch.Generator().manual_seed(3))
    again.init_from_prompt(prompt)
    assert torch.equal(sample.masks(), again.masks())


def test_ema_update_false_freezes_velocity_but_moves_anchor(table):
    """``update: false`` freezes the velocity EMA; the anchor still tracks the
    last committed token (documented in mask_providers), so masks() still moves."""
    prompt = torch.randn(6, 8)
    frozen = EMAVelocityMaskProvider(2, table, update=False, extrapolate_ema=False, level='last')
    frozen.init_from_prompt(prompt)
    velocity_before = frozen._velocity.clone()
    ref_before = frozen._ref_norm.clone()
    new_embed = torch.randn(8)
    frozen.on_token_committed(new_embed, prompt[-1])
    assert torch.equal(frozen._velocity, velocity_before)
    assert torch.equal(frozen._ref_norm, ref_before)
    assert torch.equal(frozen._anchor, new_embed)
    # masks use the new anchor as origin → values change even with frozen velocity.
    assert not torch.allclose(frozen.masks(), velocity_before)  # shape differs anyway; pin anchor effect:
    assert torch.allclose(frozen.masks()[0], new_embed + frozen._step_vector(velocity_before, ref_before))


def test_ema_extrapolate_ema_rebuilds_masks_after_commit(table):
    prompt = torch.randn(6, 8)
    moving = EMAVelocityMaskProvider(2, table, update=True, extrapolate_ema=True, beta=0.5, step_scale=1.0)
    moving.init_from_prompt(prompt)
    first = moving.masks().clone()
    moving.on_token_committed(prompt[-1] + 1.0, prompt[-1])
    assert first.shape == moving.masks().shape
    assert not torch.equal(first, moving.masks())


def test_build_mask_provider_wires_esp_and_ema(table):
    esp = build_mask_provider(_convert({
        'method': 'esp', 'num_masks': 1, 'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
    }), table)
    assert isinstance(esp, ESPMaskProvider)
    ema = build_mask_provider(_convert({
        'method': 'ema_velocity', 'num_masks': 1,
        'ema': {'beta': 0.9, 'step_scale': 0, 'level': 'last', 'update': True},
    }), table)
    assert isinstance(ema, EMAVelocityMaskProvider)
