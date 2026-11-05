"""Bounded CUDA fit and R2 scoring preserve the preforward numerical path."""


def test_preforward_fit_and_scores_use_cuda_tensors(monkeypatch, record_property):
    import math
    import torch
    from studies.trajectory_observation.structure import preforward_signal as protocol

    assert torch.cuda.is_available(), 'preforward math requires real CUDA; no CPU fallback'
    device = torch.device('cuda:0')
    generator = torch.Generator(device=device).manual_seed(17)
    blocks = []
    for _ in range(5):
        delta = torch.randn((3, 3), generator=generator, device=device)
        error = delta[:, :1] @ torch.tensor([[1., 2., -1.]], device=device)
        blocks.append((delta, error))
    fit_native, r2_native = protocol.fit_lowrank_delta, protocol.centered_r2
    observed = []
    def fit(delta, error, *args, **kwargs):
        assert delta.is_cuda and error.is_cuda
        observed.append('fit')
        return fit_native(delta, error, *args, **kwargs)
    def r2(error, predicted, center):
        assert error.is_cuda and predicted.is_cuda and center.is_cuda
        observed.append('r2')
        return r2_native(error, predicted, center)
    monkeypatch.setattr(protocol, 'fit_lowrank_delta', fit)
    monkeypatch.setattr(protocol, 'centered_r2', r2)
    mse = protocol._cv_mse(blocks, rank=1, ridge=.01, device=device)
    assert math.isfinite(mse) and mse >= 0 and observed.count('fit') == 5
    delta = torch.randn((6, 3), generator=generator, device=device)
    exact = [(delta, 2 * delta)]
    factors = {'V': torch.eye(3, device=device), 'U': 2 * torch.eye(3, device=device),
               'b': torch.zeros(3, device=device)}
    for score in (protocol._score(exact, exact, factors, device), protocol._score_diagonal(exact, exact)):
        assert abs(score['train_r2'] - 1) < 1e-6 and abs(score['holdout_r2'] - 1) < 1e-6
    assert observed.count('r2') == 4
    torch.cuda.synchronize(device)
    record_property('cuda_device', torch.cuda.get_device_name(device))
    record_property('fit_scope', 'five local prompt folds, rank one, three features; exact CUDA R2 controls')
