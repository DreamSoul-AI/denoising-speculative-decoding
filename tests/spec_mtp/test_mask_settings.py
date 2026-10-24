"""Named mask settings used by a Study axis."""

import pytest

from studies._shared.structure.config import spec_for


def test_frozen_mean_does_not_update():
    spec = spec_for('frozen_mean')
    assert spec.method == 'esp'
    assert spec.mask['update'] is False
    assert spec.mask.init == 'mean_prompt'
    assert spec.tree.branches == [14]
    assert spec.num_masks == 1


def test_mean_plus_gamma_uses_study_overrides():
    spec = spec_for('mean_plus_gamma', beta=0.999, lam=0.1, gamma=1, branches=[7, 2], num_masks=2)
    assert spec.method == 'ema_velocity'
    assert spec.ema.level == 'order0'
    assert spec.ema.beta == 0.999
    assert spec.ema.lam == 0.1
    assert spec.ema.step_scale == 1
    assert spec.tree.branches == [7, 2]
    assert spec.num_masks == 2


def test_unknown_mask_is_rejected():
    with pytest.raises(ValueError, match='unknown mask setting'):
        spec_for('not_a_mask')
