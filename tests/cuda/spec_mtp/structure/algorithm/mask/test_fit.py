"""Affine fitting and prompt-fold replay use the actual CUDA input device."""

import pytest
import torch

from spec_mtp.structure.algorithm.mask import fit


def _cuda(record_property):
    assert torch.cuda.is_available(), 'affine correctness requires actual CUDA; no CPU fallback'
    device = torch.device('cuda:0')
    record_property('cuda_device', torch.cuda.get_device_name(device))
    record_property('torch_version', torch.__version__)
    return device


def _watch_solvers(monkeypatch):
    devices = []
    for name in ('solve', 'lstsq'):
        original = getattr(torch.linalg, name)

        def checked(left, right, *args, _original=original, **kwargs):
            assert left.is_cuda and right.is_cuda, 'affine solve moved off the CUDA input device'
            assert left.device == right.device
            devices.append(str(left.device))
            return _original(left, right, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, name, checked)
    return devices


@pytest.mark.parametrize('kind', ['affine', 'ridge'])
def test_cuda_affine_fit_recovers_known_map_and_residual_without_host_solve(monkeypatch, record_property, kind):
    device = _cuda(record_property)
    solves = _watch_solvers(monkeypatch)
    delta = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                          [0, 0, 1], [0, 0, -1], [1, 2, 3], [-2, 1, -3]], device=device, dtype=torch.float64)
    baseline = torch.full_like(delta, 0.2)
    weights = torch.tensor([[0.3, -0.2, 0.1], [0.4, 0.2, -0.5], [-0.1, 0.6, 0.8]],
                           device=device, dtype=torch.float64)
    bias = torch.tensor([0.8, -0.4, 0.2], device=device, dtype=torch.float64)
    observed = baseline + delta
    target = baseline + delta @ weights.T + bias
    original = tuple(value.clone() for value in (baseline, observed, target))
    fitted = (fit.fit_mask_correction(baseline, observed, target, 'affine') if kind == 'affine'
              else fit.fit_affine_ridge(baseline, observed, target, 1e-8))
    actual = torch.as_tensor(fitted['A'], device=device, dtype=torch.float64)
    intercept = torch.as_tensor(fitted['b'], device=device, dtype=torch.float64)
    assert torch.allclose(actual, weights, atol=1e-7, rtol=0)
    assert torch.allclose(intercept, bias, atol=1e-7, rtol=0)
    # The fixed offset has a known SSE and also tests reconstruction of serialized A/b on CUDA.
    shifted = target + torch.tensor([0.1, -0.2, 0.3], device=device, dtype=torch.float64)
    assert fit._affine_residual(baseline, observed, shifted, fitted) == pytest.approx(8 * 0.14, abs=1e-7)
    assert all(torch.equal(value, saved) for value, saved in zip((baseline, observed, target), original))
    assert solves and set(solves) == {'cuda:0'}
    torch.cuda.synchronize(device)
    record_property('actual_solver_devices', solves)


def test_cuda_affine_ridge_selection_scores_prompt_folds_on_device(monkeypatch, record_property):
    device = _cuda(record_property)
    solves = _watch_solvers(monkeypatch)
    generator = torch.Generator(device=device).manual_seed(617)
    baseline = torch.randn(24, 2, device=device, dtype=torch.float64, generator=generator)
    delta = torch.randn(24, 2, device=device, dtype=torch.float64, generator=generator)
    observed = baseline + delta
    weights = torch.tensor([[0.4, -0.3], [0.2, 0.6]], device=device, dtype=torch.float64)
    bias = torch.tensor([0.7, -0.2], device=device, dtype=torch.float64)
    target = baseline + delta @ weights.T + bias
    blocks = [(baseline[start:start + 4], observed[start:start + 4], target[start:start + 4])
              for start in range(0, 24, 4)]
    fitted = fit.select_affine_ridge(blocks, seed=11, folds=3)
    assert fitted['prompts'] == 6 and fitted['rows'] == 24
    assert fitted['ridge'] == fit.RIDGE_GRID[0]
    assert all(row['mse'] >= 0 for row in fitted['cv'])
    assert fitted['cv'][0]['mse'] < 1e-9
    # Unseen inputs must recover this known map, independent of the fitting residual implementation.
    held_delta = torch.tensor([[2.0, -1.0], [-3.0, 0.5]], device=device, dtype=torch.float64)
    prediction = held_delta @ torch.as_tensor(fitted['A'], device=device, dtype=torch.float64).T
    prediction += torch.as_tensor(fitted['b'], device=device, dtype=torch.float64)
    assert torch.allclose(prediction, held_delta @ weights.T + bias, atol=1e-4, rtol=0)
    assert len(solves) == len(fit.RIDGE_GRID) * 3 + 1 and set(solves) == {'cuda:0'}
    torch.cuda.synchronize(device)
    record_property('actual_solver_devices', solves)
