"""Data v2 cursor admission and exact sampler replay, without model execution."""

from tests.conftest import cursor_data as _data, cursor_values_equal as _equal


import pytest


from spec_mtp.structure.data.resume import generator_states

@pytest.mark.parametrize('kind,explicit_loader', [('random', False), ('random', True), ('shared', True),
                                                ('sequential', False), ('sequential', True)])
def test_mid_epoch_cuda_batches_preserve_suffix_and_do_not_replay_transforms(kind, explicit_loader):
    import torch

    assert torch.cuda.is_available(), 'CUDA cursor integration requires an available device'
    device = torch.device('cuda', torch.cuda.current_device())
    print(f'data cursor device={device} name={torch.cuda.get_device_name(device)}')
    data, _, _ = _data(kind, explicit_loader=explicit_loader, device=device)
    iterator = data.iter_batches('train')
    prefix = [next(iterator), next(iterator)]
    saved = data.training_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    expected = list(iterator)
    expected_generators = generator_states(data._loaders['train'])
    restored, _, reads = _data(kind, explicit_loader=explicit_loader, device=device)
    restored.restore_training_state(saved)
    cpu_rng = torch.get_rng_state().clone()
    torch.cuda.set_rng_state(cuda_rng, device)
    actual = list(restored.iter_batches('train'))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert len(actual) == len(expected) == 3
    assert all(batch.device == device and torch.equal(batch, reference) for batch, reference in zip(actual, expected))
    ids = [int(value) for batch in prefix + actual for value in batch[:, 0].tolist()]
    assert sorted(ids) == list(range(13)) and len(reads) == 7
    assert reads == [int(value) for batch in actual for value in batch[:, 0].tolist()]
    assert _equal(expected_generators, generator_states(restored._loaders['train']))
