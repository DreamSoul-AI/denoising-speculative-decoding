"""Data v2 cursor admission and exact sampler replay, without model execution."""

from tests.conftest import cursor_data as _data, cursor_values_equal as _equal

from copy import deepcopy

import pytest


from spec_mtp.structure.data.resume import generator_states, restore_generators

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
