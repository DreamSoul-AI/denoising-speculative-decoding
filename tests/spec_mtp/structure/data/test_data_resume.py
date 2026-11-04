"""Data v2 cursor admission and exact sampler replay, without model execution."""

from copy import deepcopy

import pytest

from spec_mtp.structure.data.factory import Data
from spec_mtp.structure.data.resume import generator_states, restore_generators


def _data(kind='random', *, workers=0, seed=7, explicit_loader=True, device=None, size=13):
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    reads = []

    class Samples:
        def __len__(self):
            return size

        def __getitem__(self, index):
            reads.append(index)
            if device is None:
                return index
            return torch.stack((torch.tensor(index, device=device), torch.rand((), device=device)))

    dataset = Samples()
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = (SequentialSampler(dataset) if kind == 'sequential' else
               RandomSampler(dataset, generator=sampler_generator))
    loader_generator = (sampler_generator if kind == 'shared' else
                        torch.Generator().manual_seed(seed + 1) if explicit_loader else None)
    options = {} if device is not None else {'collate_fn': lambda values: values}
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, generator=loader_generator,
                        num_workers=workers, **options)
    data = Data(name='cursor-fixture', source='local', loaders={'train': loader},
                meta={'seed': seed, 'batch_size': 3, 'config': {'fixture': 'fixed-id-order'}})
    return data, dataset, reads


def _equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    return left == right


@pytest.mark.parametrize('kind', ['random', 'shared', 'sequential'])
@pytest.mark.parametrize('exhausted', [False, True])
@pytest.mark.parametrize('size', [12, 13])
def test_epoch_boundary_preserves_last_yield_and_next_epoch_order(kind, exhausted, size):
    import torch

    continuous, _, _ = _data(kind, size=size)
    iterator = continuous.iter_batches('train')
    batch_count = (size + 2) // 3
    batches = [next(iterator) for _ in range(batch_count)]
    if exhausted:
        with pytest.raises(StopIteration):
            next(iterator)
    state = continuous.training_state()
    assert state['version'] == 2 and state['consumed'] == batch_count and state['exhausted'] is exhausted
    assert sorted(item for batch in batches for item in batch) == list(range(size))
    if not exhausted:
        with pytest.raises(StopIteration):
            next(iterator)
    expected_next = list(continuous.iter_batches('train'))
    restored, _, reads = _data(kind, size=size)
    assert restored.restore_training_state(state) is (not exhausted)
    if not exhausted:
        before = torch.get_rng_state().clone()
        assert list(restored.iter_batches('train')) == []
        assert torch.equal(before, torch.get_rng_state()) and reads == []
    actual_next = list(restored.iter_batches('train'))
    assert actual_next == expected_next


@pytest.mark.parametrize('fault', ['legacy_version', 'boolean_version', 'extra_field', 'missing_after',
    'seed', 'identity', 'spec', 'negative_cursor', 'boolean_cursor', 'large_cursor', 'early_exhausted',
    'generator_schema', 'generator_missing', 'generator_type', 'generator_bytes', 'alias_bytes', 'nonjson_identity'])
def test_invalid_data_state_leaves_live_and_pending_state_unchanged(fault):
    import torch

    data, _, _ = _data('shared')
    iterator = data.iter_batches('train')
    next(iterator)
    saved = data.training_state()
    data.restore_training_state(saved)
    pending, loader, start = data._resume_data, data._loaders['train'], deepcopy(data._train_start)
    live, global_rng = generator_states(loader), torch.get_rng_state().clone()
    cursor = (loader.batch_sampler.consumed, loader.batch_sampler.exhausted)
    bad = deepcopy(saved)
    if fault == 'legacy_version': bad['version'] = 1
    elif fault == 'boolean_version': bad['version'] = True
    elif fault == 'extra_field': bad['unknown'] = True
    elif fault == 'missing_after': del bad['after']
    elif fault == 'seed': bad['seed'] = False
    elif fault == 'identity': bad['identity']['name'] = 'other'
    elif fault == 'spec': bad['spec']['batch_size'] = True
    elif fault == 'negative_cursor': bad['consumed'] = -1
    elif fault == 'boolean_cursor': bad['consumed'] = True
    elif fault == 'large_cursor': bad['consumed'] = 6
    elif fault == 'early_exhausted': bad['exhausted'] = True
    elif fault == 'generator_schema': bad['after']['unknown'] = None
    elif fault == 'generator_missing': bad['after']['sampler'] = None
    elif fault == 'generator_type': bad['after']['sampler'] = [1, 2]
    elif fault == 'generator_bytes': bad['after']['loader'] = torch.zeros(1, dtype=torch.uint8)
    elif fault == 'nonjson_identity': bad['identity']['config'] = {'unsupported': {1, 2}}
    else: bad['after']['loader'] = torch.Generator().manual_seed(999).get_state()
    for action in (data.validate_training_state, data.restore_training_state):
        with pytest.raises(ValueError):
            action(bad)
    assert data._resume_data is pending and data._loaders['train'] is loader
    assert _equal(live, generator_states(loader)) and _equal(start, data._train_start)
    assert cursor == (loader.batch_sampler.consumed, loader.batch_sampler.exhausted)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert _equal(saved, pending)


def test_restored_state_is_owned_and_generator_alias_mismatch_is_rejected():
    data, _, _ = _data('shared')
    next(data.iter_batches('train'))
    saved = data.training_state()
    target, _, _ = _data('shared')
    target.restore_training_state(saved)
    saved['consumed'] = 4
    saved['identity']['config']['fixture'] = 'changed'
    saved['start']['sampler'].zero_()
    assert target._resume_data['consumed'] == 1
    assert target.meta['config']['fixture'] == 'fixed-id-order'
    target.validate_training_state(target._resume_data)
    separate, _, _ = _data('random')
    with pytest.raises(ValueError, match='sampler/batch/seed'):
        separate.restore_training_state(target._resume_data)


@pytest.mark.parametrize('workers', [1, 2])
def test_unsupported_worker_loader_rejects_exact_cursor_without_starting_workers(workers):
    data, _, _ = _data()
    next(data.iter_batches('train'))
    state = data.training_state()
    unsupported, _, reads = _data(workers=workers)
    assert unsupported.training_state() is None
    with pytest.raises(ValueError, match='unsupported'):
        unsupported.restore_training_state(state)
    assert unsupported._resume_data is None and reads == []


def test_budget_admission_is_pure_before_full_budget_rebind():
    data, dataset, _ = _data(explicit_loader=False)
    data._train_set = dataset
    data.rebind_train_steps(step=0, num_steps=4, step_period=2)
    iterator = data.iter_batches('train')
    next(iterator)
    saved = data.training_state()
    expected = list(iterator)
    target, target_dataset, reads = _data(explicit_loader=False)
    target._train_set = target_dataset
    loader = target._loaders['train']
    before = generator_states(loader)
    target.validate_training_state(saved, num_steps=4, step_period=2)
    assert target._loaders['train'] is loader and target._resume_data is None and reads == []
    assert _equal(before, generator_states(loader))
    target.rebind_train_steps(step=0, num_steps=4, step_period=2)
    target.restore_training_state(saved)
    actual = [batch.tolist() for batch in target.iter_batches('train')]
    assert actual == [batch.tolist() for batch in expected]


def test_generator_restore_prevalidates_both_domains_before_mutation():
    import torch

    data, _, _ = _data()
    loader = data._loaders['train']
    original = generator_states(loader)
    bad = deepcopy(original)
    bad['sampler'] = torch.Generator().manual_seed(123).get_state()
    bad['loader'] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError):
        restore_generators(loader, bad)
    assert _equal(original, generator_states(loader))


def test_pending_restore_is_rechecked_before_loader_wrapping_or_rng_mutation():
    import torch

    donor, _, _ = _data()
    next(donor.iter_batches('train'))
    target, _, reads = _data()
    target.restore_training_state(donor.training_state())
    pending, loader = target._resume_data, target._loaders['train']
    loader.batch_sampler.drop_last = True
    rng, generators = torch.get_rng_state().clone(), generator_states(loader)
    with pytest.raises(ValueError):
        target.iter_batches('train')
    assert target._resume_data is pending and target._loaders['train'] is loader and reads == []
    assert torch.equal(rng, torch.get_rng_state()) and _equal(generators, generator_states(loader))


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
