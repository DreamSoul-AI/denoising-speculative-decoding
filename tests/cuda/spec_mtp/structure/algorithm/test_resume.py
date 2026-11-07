"""Bounded CUDA RNG and checkpoint contracts, independent of training orchestration."""

from copy import deepcopy

import random

import numpy as np

import pytest

import torch

from spec_mtp.structure.algorithm.resume import capture_rng, restore_rng

def _same_rng(left, right):
    assert left['python'] == right['python']
    assert left['numpy'][0] == right['numpy'][0]
    assert np.array_equal(left['numpy'][1], right['numpy'][1])
    assert left['numpy'][2:] == right['numpy'][2:]
    assert torch.equal(left['torch'].cpu(), right['torch'].cpu())
    assert len(left['cuda']) == len(right['cuda'])
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(left['cuda'], right['cuda']))

def _draw():
    return (random.random(), np.random.random(4), torch.rand(4), torch.rand(4, device='cuda:0'))

def _same_draw(left, right):
    assert left[0] == right[0] and np.array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2]) and torch.equal(left[3], right[3])
    assert right[3].is_cuda

def test_cuda_rng_roundtrip_restores_all_globals_but_not_explicit_generator(record_property):
    assert torch.cuda.is_available(), 'RNG validation requires real CUDA; no CPU fallback'
    torch.cuda.init()
    original = capture_rng()
    try:
        random.seed(314)
        np.random.seed(314)
        torch.manual_seed(314)
        torch.cuda.manual_seed_all(314)
        generator = torch.Generator(device='cuda:0').manual_seed(2718)
        explicit = generator.get_state().clone()
        saved = capture_rng()
        expected = _draw()
        explicit_draw = torch.rand(4, generator=generator, device='cuda:0')
        explicit_after = generator.get_state().clone()
        _draw()
        restore_rng(saved)
        _same_rng(capture_rng(), saved)
        assert torch.equal(generator.get_state(), explicit_after)
        _same_draw(expected, _draw())
        generator.set_state(explicit)
        assert torch.equal(torch.rand(4, generator=generator, device='cuda:0'), explicit_draw)
        torch.cuda.synchronize()
        record_property('cuda_device', torch.cuda.get_device_name(0))
        record_property('rng_scope', 'Python/NumPy/Torch CPU/all initialized CUDA globals; explicit generator separate')
    finally:
        restore_rng(original)

@pytest.mark.parametrize('fault', ['unavailable', 'device_count'])
def test_cuda_rng_device_mismatch_rejects_before_any_global_state_change(monkeypatch, fault, record_property):
    assert torch.cuda.is_available(), 'RNG validation requires real CUDA; no CPU fallback'
    torch.cuda.init()
    original = capture_rng()
    try:
        saved = capture_rng()
        _draw()
        before = capture_rng()
        device_count = torch.cuda.device_count()
        with monkeypatch.context() as patch:
            if fault == 'unavailable':
                patch.setattr(torch.cuda, 'is_available', lambda: False)
            else:
                patch.setattr(torch.cuda, 'device_count', lambda: device_count + 1)
            with pytest.raises(ValueError, match='CUDA RNG devices'):
                restore_rng(saved)
        _same_rng(before, capture_rng())
        record_property('cuda_device', torch.cuda.get_device_name(0))
    finally:
        restore_rng(original)

@pytest.mark.parametrize('fault', ['python', 'numpy', 'torch', 'cuda'])
def test_cuda_rng_malformed_domain_rejects_before_any_global_state_change(fault, record_property):
    assert torch.cuda.is_available(), 'RNG validation requires real CUDA; no CPU fallback'
    torch.cuda.init()
    original = capture_rng()
    try:
        saved = capture_rng()
        if fault == 'python':
            saved['python'] = ('invalid',)
        elif fault == 'numpy':
            saved['numpy'] = ('invalid',)
        elif fault == 'torch':
            saved['torch'] = torch.tensor([1], dtype=torch.uint8)
        else:
            saved['cuda'][0] = torch.tensor([1], dtype=torch.uint8)
        _draw()
        before = capture_rng()
        with pytest.raises(ValueError, match='invalid checkpoint RNG state'):
            restore_rng(saved)
        _same_rng(before, capture_rng())
        record_property('cuda_device', torch.cuda.get_device_name(0))
    finally:
        restore_rng(original)

def test_cuda_checkpoint_bundle_restores_one_controlled_optimizer_step(tmp_path, record_property):
    from spec_mtp.structure.system.factory import System
    from spec_mtp.structure.system.logger import Logger

    assert torch.cuda.is_available(), 'Checkpoint validation requires real CUDA; no CPU fallback'
    torch.cuda.init()
    original = capture_rng()
    assets = tmp_path / 'assets'
    system = System(device='cuda:0', logger=Logger(assets), assets_dir=assets, meta={})

    def objects():
        model = torch.nn.Linear(3, 2).to('cuda:0')
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7)
        return model, optimizer, scheduler

    def step(model, optimizer, scheduler, inputs, targets):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach().clone()

    try:
        torch.manual_seed(42)
        model, optimizer, scheduler = objects()
        step(model, optimizer, scheduler, torch.randn(4, 3, device='cuda:0'), torch.randn(4, 2, device='cuda:0'))
        payload = {'model': deepcopy(model.state_dict()), 'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict()), 'step': 1,
            'training_state': {'version': 1, 'rng': capture_rng()}}
        bundle = system.save_checkpoint(payload, 'bounded')
        saved_bytes = {path.relative_to(assets): path.read_bytes() for path in assets.rglob('*') if path.is_file()}
        draws = _draw()
        inputs, targets = torch.randn(4, 3, device='cuda:0'), torch.randn(4, 2, device='cuda:0')
        expected_loss = step(model, optimizer, scheduler, inputs, targets)
        expected_weights = deepcopy(model.state_dict())
        expected_optimizer, expected_scheduler = deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict())
        resumed, resumed_optimizer, resumed_scheduler = objects()
        restored = system.load_checkpoint(str(bundle))
        assert restored['step'] == 1 and restored['model']['weight'].is_cuda
        resumed.load_state_dict(restored['model'])
        resumed_optimizer.load_state_dict(restored['optimizer'])
        resumed_scheduler.load_state_dict(restored['scheduler'])
        restore_rng(restored['training_state']['rng'])
        _same_draw(draws, _draw())
        actual_inputs, actual_targets = torch.randn(4, 3, device='cuda:0'), torch.randn(4, 2, device='cuda:0')
        assert torch.equal(inputs, actual_inputs) and torch.equal(targets, actual_targets)
        actual_loss = step(resumed, resumed_optimizer, resumed_scheduler, actual_inputs, actual_targets)
        torch.cuda.synchronize()
        assert torch.equal(actual_loss, expected_loss)
        assert all(torch.equal(value, expected_weights[key]) for key, value in resumed.state_dict().items())
        actual_optimizer = resumed_optimizer.state_dict()
        assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
        assert all(torch.equal(value['momentum_buffer'], expected_optimizer['state'][key]['momentum_buffer'])
                   for key, value in actual_optimizer['state'].items())
        assert resumed_scheduler.state_dict() == expected_scheduler
        assert saved_bytes == {path.relative_to(assets): path.read_bytes() for path in assets.rglob('*') if path.is_file()}
        record_property('cuda_device', torch.cuda.get_device_name(0))
        record_property('checkpoint_scope', 'one controlled SGD/scheduler step and global RNG; no Data/Trainer restore')
    finally:
        restore_rng(original)
