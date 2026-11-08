"""Replay sampler indices, never consumed dataset transforms, on CPU-loader resume."""

from itertools import islice
from collections.abc import Sized
from typing import cast


class BatchCursor:
    def __init__(self, batches):
        self.batches = batches
        self.consumed = 0
        self.exhausted = False

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        start = self.consumed
        for batch in islice(iter(self.batches), start, None):
            self.consumed += 1
            yield batch
        self.exhausted = True


def loader_spec(loader):
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

    if type(loader) is not DataLoader or loader.num_workers != 0:
        return None
    batches = loader.batch_sampler
    if isinstance(batches, BatchCursor):
        batches = batches.batches
    if type(batches) is not BatchSampler or type(batches.sampler) not in (RandomSampler, SequentialSampler):
        return None
    sampler = batches.sampler
    if isinstance(sampler, RandomSampler) and sampler.generator is None:
        return None
    sampler_generator = getattr(sampler, 'generator', None)
    if any(value is not None and value.device.type != 'cpu' for value in (sampler_generator, loader.generator)):
        return None
    return {'dataset_size': len(cast(Sized, loader.dataset)), 'batch_size': batches.batch_size,
            'drop_last': batches.drop_last, 'sampler': type(sampler).__name__,
            'num_samples': len(cast(Sized, sampler)), 'replacement': getattr(sampler, 'replacement', False),
            'shared_generator': sampler_generator is not None and sampler_generator is loader.generator}


def generators(loader):
    sampler = loader.batch_sampler
    if isinstance(sampler, BatchCursor):
        sampler = sampler.batches
    return {'sampler': getattr(sampler.sampler, 'generator', None), 'loader': loader.generator}


def generator_states(loader):
    return {key: value.get_state().clone() if value is not None else None
            for key, value in generators(loader).items()}


def validate_generator_states(states, present, *, shared=False):
    """Trial-load serialized CPU sampler states without touching live generators."""
    import torch

    if not isinstance(states, dict) or set(states) != {'sampler', 'loader'}:
        raise ValueError('invalid checkpoint data generators')
    normalized: dict[str, torch.Tensor | None] = {}
    for key, state in states.items():
        if (state is not None) != present[key]:
            raise ValueError('checkpoint loader generator does not match current data')
        if state is None:
            normalized[key] = None
            continue
        if type(state) is not torch.Tensor:
            raise ValueError('invalid checkpoint generator state')
        try:
            trial = torch.Generator(device='cpu')
            trial.set_state(state.cpu())
            normalized[key] = trial.get_state()
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError('invalid checkpoint generator state') from error
    if shared:
        sampler_state, loader_state = normalized['sampler'], normalized['loader']
        if sampler_state is None or loader_state is None or not torch.equal(sampler_state, loader_state):
            raise ValueError('shared sampler/loader generator states differ')
    return normalized


def restore_generators(loader, states):
    actual = generators(loader)
    values = validate_generator_states(states, {key: value is not None for key, value in actual.items()},
                                       shared=actual['sampler'] is not None and actual['sampler'] is actual['loader'])
    for key, generator in actual.items():
        if generator is not None:
            generator.set_state(values[key])


def cursor_loader(loader):
    from torch.utils.data import DataLoader

    cursor = BatchCursor(loader.batch_sampler)
    return DataLoader(loader.dataset, batch_sampler=cursor, collate_fn=loader.collate_fn,
                      num_workers=0, pin_memory=loader.pin_memory, generator=loader.generator)
