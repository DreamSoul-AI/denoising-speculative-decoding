import dataclasses
import hashlib

import pytest
import torch

from studies.k_affine_residual.structure.correction import (
    ARMS, FrozenCorrection, apply_correction, fit_correction,
)


def fit(x, residual):
    return fit_correction(x, residual, model_space='a' * 64,
                          fit_documents=(('fit', 'b' * 64),), fit_data_sha='c' * 64)


def test_known_affine_residual_and_independent_bias():
    x = torch.tensor([[1., 2.], [2., -1.], [4., 3.], [-2., 1.]], dtype=torch.float64)
    a = torch.tensor([[2., -1.], [.5, 3.]], dtype=torch.float64)
    b = torch.tensor([4., -2.], dtype=torch.float64)
    residual = x @ a.T + b
    frozen = fit(x, residual)
    z = torch.zeros_like(residual)
    full = apply_correction(frozen, x, z, 'full', model_space='a' * 64)
    assert torch.max(torch.abs(full - residual)) < .002
    bias = apply_correction(frozen, x, z, 'bias', model_space='a' * 64)
    torch.testing.assert_close(bias, residual.mean(0).expand_as(residual))
    assert not torch.allclose(bias[0], b)


def test_centering_leaves_intercept_unpenalized():
    x = torch.ones(3, 2, dtype=torch.float64) * 900
    residual = torch.tensor([[4., 7.]] * 3, dtype=torch.float64)
    frozen = fit(x, residual)
    torch.testing.assert_close(apply_correction(frozen, x, x, 'full', model_space='a' * 64), x + residual)


@pytest.mark.parametrize('arm', ARMS)
def test_zero_residual_is_identity(arm):
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    frozen = fit(x, torch.zeros_like(x))
    torch.testing.assert_close(apply_correction(frozen, x, x, arm, model_space='a' * 64), x)


@pytest.mark.parametrize('x,residual', [
    ([[True, 1.]], [[1., 2.]]),
    ([[float('nan'), 1.]], [[1., 2.]]),
    ([[1., 2.]], [[float('inf'), 2.]]),
    ([], []), ([[1., 2.]], [[1., 2.], [3., 4.]]),
])
def test_invalid_fit_inputs_rejected(x, residual):
    with pytest.raises(ValueError):
        fit(x, residual)


def test_frozen_payload_is_immutable_and_hash_checked():
    frozen = fit([[1., 2.]], [[3., 4.]])
    assert frozen.sha256 == hashlib.sha256(frozen.payload.encode()).hexdigest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.payload = '{}'
    with pytest.raises(ValueError, match='SHA'):
        FrozenCorrection(frozen.payload + ' ', frozen.sha256)


@pytest.mark.parametrize('arm,space,x', [
    ('next-token', 'a' * 64, [[1., 2.]]),
    ('full', 'd' * 64, [[1., 2.]]),
    ('full', 'a' * 64, [[1.]]),
    ('full', 'a' * 64, [[True, 2.]]),
])
def test_apply_rejects_unknown_arm_space_and_bad_dimension(arm, space, x):
    with pytest.raises(ValueError):
        apply_correction(fit([[1., 2.]], [[3., 4.]]), x, [[0., 0.]], arm, model_space=space)
