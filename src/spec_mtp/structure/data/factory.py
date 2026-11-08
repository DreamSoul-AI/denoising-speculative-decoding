"""Runtime Data object, registry, and factory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator

from spec_mtp.structure.data.config import DataConfig
from spec_mtp.structure.origin import bind_endpoint, discard_bad_archive, endpoint_for


class Data:
    def __init__(
        self,
        *,
        name: str,
        source: str,
        loaders: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        self.name = name
        self.source = source
        self._loaders = loaders
        self.meta = meta
        self._train_set: Any = None
        self._train_start: dict[str, Any] | None = None
        self._resume_data: dict[str, Any] | None = None

    def iter_batches(self, split: str) -> Iterator[Any]:
        loader = self._loaders.get(split)
        if split == 'train' and self._resume_data is not None:
            self.validate_training_state(self._resume_data)
        if loader is None:
            return iter(())
        if split == 'train':
            from spec_mtp.structure.data.resume import (
                BatchCursor, cursor_loader, generator_states, loader_spec, restore_generators,
            )
            if loader_spec(loader) is not None:
                import torch

                if not isinstance(loader.batch_sampler, BatchCursor):
                    loader = cursor_loader(loader)
                    self._loaders[split] = loader
                cursor = loader.batch_sampler
                pending = self._resume_data
                inside = pending is not None and not pending['exhausted']
                if pending is not None:
                    restore_generators(loader, pending['after'] if pending['exhausted'] else pending['start'])
                self._resume_data = None
                cursor.consumed = 0 if pending is None or pending['exhausted'] else pending['consumed']
                cursor.exhausted = False
                self._train_start = generator_states(loader)
                # A replacement iterator must not consume another global base-seed draw.
                rng = torch.get_rng_state() if inside else None
                iterator = iter(loader)
                if rng is not None:
                    torch.set_rng_state(rng)
                return iterator
        return iter(loader)

    def training_state(self) -> dict[str, Any] | None:
        from spec_mtp.structure.data.resume import BatchCursor, generator_states, loader_spec

        loader = self._loaders.get('train')
        spec = loader_spec(loader)
        if loader is None or spec is None or not isinstance(loader.batch_sampler, BatchCursor) or self._train_start is None:
            return None
        cursor = loader.batch_sampler
        return deepcopy({'version': 2, 'spec': spec, 'seed': self.meta.get('seed'),
                'identity': {'name': self.name, 'source': self.source, 'config': self.meta.get('config')},
                'consumed': cursor.consumed, 'exhausted': cursor.exhausted,
                'start': self._train_start, 'after': generator_states(loader)})

    def validate_training_state(self, state: dict[str, Any], *, num_steps: int | None = None,
                                step_period: int = 1) -> None:
        """Admit the cursor and every RNG domain before mutating loader or pending state."""
        import json
        from spec_mtp.structure.data.resume import generators, loader_spec, validate_generator_states

        loader = self._loaders.get('train')
        expected = loader_spec(loader)
        if (not isinstance(state, dict) or expected is None
                or set(state) != {'version', 'spec', 'seed', 'identity', 'consumed', 'exhausted', 'start', 'after'}
                or type(state.get('version')) is not int or state['version'] != 2):
            raise ValueError('unsupported checkpoint data state; exact resume requires version 2')
        present = {key: value is not None for key, value in generators(loader).items()}
        if num_steps is not None and self._train_set is not None:
            if type(num_steps) is not int or num_steps < 1 or type(step_period) is not int or step_period < 1:
                raise ValueError('invalid checkpoint data step budget')
            size = int(self.meta.get('batch_size') or getattr(loader, 'batch_size', 1) or 1)
            expected = {**expected, 'dataset_size': len(self._train_set), 'batch_size': size,
                        'num_samples': size * num_steps * step_period, 'sampler': 'RandomSampler',
                        'replacement': False, 'drop_last': False, 'shared_generator': False}
            present = {'sampler': self.meta.get('seed') is not None, 'loader': False}
        def canonical(value):
            try:
                return json.dumps(value, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError('invalid checkpoint data metadata') from error

        if canonical(state.get('spec')) != canonical(expected) or canonical(state.get('seed')) != canonical(self.meta.get('seed')):
            raise ValueError('checkpoint data sampler/batch/seed does not match current data')
        if canonical(state.get('identity')) != canonical({'name': self.name, 'source': self.source, 'config': self.meta.get('config')}):
            raise ValueError('checkpoint data identity/config does not match current data')
        consumed = state.get('consumed')
        batches = expected['num_samples'] // expected['batch_size'] if expected['drop_last'] else (
            expected['num_samples'] + expected['batch_size'] - 1) // expected['batch_size']
        if (type(consumed) is not int or not 0 <= consumed <= batches or type(state.get('exhausted')) is not bool
                or (state['exhausted'] and consumed != batches)):
            raise ValueError('invalid checkpoint data cursor')
        for field in ('start', 'after'):
            validate_generator_states(state[field], present, shared=expected['shared_generator'])

    def restore_training_state(self, state: dict[str, Any]) -> bool:
        self.validate_training_state(state)
        self._resume_data = deepcopy(state)
        return not state['exhausted']

    def steps_per_epoch(self, split: str = 'train') -> int | None:
        loader = self._loaders.get(split)
        if loader is not None:
            try:
                value = int(len(loader))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        if split != 'train':
            return None
        train_size = self.meta.get('train_size')
        batch_size = self.meta.get('batch_size')
        if train_size and batch_size:
            import math

            return max(int(math.ceil(int(train_size) / int(batch_size))), 1)
        return None

    def rebind_train_steps(
        self,
        *,
        step: int,
        num_steps: int,
        step_period: int = 1,
    ) -> None:
        """Rebuild train loader for remaining optimizer steps (main ``make_data_loader``).

        ``num_samples = batch_size * (num_steps - step) * step_period``. Generator is
        re-seeded from ``meta['seed']`` each time, so a mid-run resume sees the
        **prefix** of the same shuffle, not the unseen suffix (same as main).
        Native exact resume instead binds the original full budget at step 0,
        then applies ``restore_training_state`` to select the unseen suffix.
        """
        dataset = self._train_set
        if dataset is None:
            return
        batch_size = int(self.meta.get('batch_size') or getattr(self._loaders.get('train'), 'batch_size', 1) or 1)
        period = max(int(step_period), 1)
        remaining = max(int(num_steps) - int(step), 0)
        num_samples = batch_size * remaining * period
        config = self.meta.get('config')
        opts = _loader_options(config if isinstance(config, dict) else {})
        self._loaders['train'] = _train_loader_for_samples(
            dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=self.meta.get('seed'),
            pin_memory=opts['pin_memory'],
            num_workers=opts['num_workers'],
        )

    def to_result_snapshot(self) -> dict[str, Any]:
        out = {'name': self.name, 'source': self.source, **self.meta}
        out.pop('train_loader', None)
        out.pop('test_loader', None)
        return out


class DataRegistry:
    _items: dict[tuple[str, str], Callable[..., Data]] = {}

    @classmethod
    def register(cls, name: str, source: str, builder: Callable[..., Data]) -> None:
        cls._items[(name, source)] = builder

    @classmethod
    def get(cls, name: str, source: str) -> Callable[..., Data] | None:
        return cls._items.get((name, source))

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        return sorted(cls._items)


class DataFactory:
    @staticmethod
    def build(
        data_config: DataConfig,
        assets_dir: Path | str,
        seed: int | None = None,
        origin: str | None = None,
    ) -> Data:
        name = data_config.name
        source = data_config.source if data_config.source is not None else 'torch'
        builder = DataRegistry.get(name, source) if name is not None else None
        if builder is None:
            raise ValueError(f'unknown data name/source: {name!r}/{source!r}')
        if builder is _build_torch_vision:
            return _build_torch_vision(data_config, Path(assets_dir), seed=seed, origin=origin)
        return builder(data_config, Path(assets_dir), seed=seed)


def _loader_options(cfg: dict[str, Any]) -> dict[str, Any]:
    pin = cfg.get('pin_memory')
    workers = cfg.get('num_workers', 0)
    return {
        'pin_memory': bool(pin) if pin is not None else False,
        'num_workers': int(workers or 0),
    }


def _empty_train_loader(dataset: Any, batch_size: int) -> Any:
    class _Empty:
        def __init__(self, dataset: Any, batch_size: int) -> None:
            self.dataset = dataset
            self.batch_size = batch_size

        def __len__(self) -> int:
            return 0

        def __iter__(self):
            return iter(())

    return _Empty(dataset, batch_size)


def _train_loader_for_samples(
    dataset: Any,
    *,
    batch_size: int,
    num_samples: int,
    seed: int | None,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> Any:
    from torch.utils.data import DataLoader, RandomSampler

    from spec_mtp.structure.system.runtime import make_generator, worker_init_fn

    if num_samples <= 0:
        return _empty_train_loader(dataset, batch_size)
    generator = make_generator(seed)
    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=int(num_samples),
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        pin_memory=bool(pin_memory),
        num_workers=int(num_workers),
        worker_init_fn=worker_init_fn if seed is not None else None,
    )


def _build_stub(data_config: DataConfig, assets_dir: Path, seed: int | None = None) -> Data:
    name = data_config.name or 'unknown'
    source = data_config.source or 'stub'
    cfg = dict(data_config.config)
    return Data(
        name=name,
        source=source,
        loaders={},
        meta={
            'ready': True,
            'assets_dir': str(assets_dir),
            'config': cfg,
            'stub': True,
            'seed': seed,
        },
    )


_VISION_TORCH: dict[str, dict[str, Any]] = {
    'MNIST': {
        'root': 'mnist',
        'ctor': 'MNIST',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (1, 28, 28),
        'target_size': 10,
        'mean': (0.1307,),
        'std': (0.3081,),
        'train_aug': None,
    },
    'FashionMNIST': {
        'root': 'fashionmnist',
        'ctor': 'FashionMNIST',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (1, 28, 28),
        'target_size': 10,
        'mean': (0.2860,),
        'std': (0.3530,),
        'train_aug': None,
    },
    'CIFAR10': {
        'root': 'cifar10',
        'ctor': 'CIFAR10',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (3, 32, 32),
        'target_size': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'train_aug': 'cifar',
    },
    'CIFAR100': {
        'root': 'cifar100',
        'ctor': 'CIFAR100',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (3, 32, 32),
        'target_size': 100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'train_aug': 'cifar',
    },
    'SVHN': {
        'root': 'svhn',
        'ctor': 'SVHN',
        'train_kw': {'split': 'train'},
        'test_kw': {'split': 'test'},
        'data_size': (3, 32, 32),
        'target_size': 10,
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
        'train_aug': 'svhn',
    },
}


def _with_recorded_stats(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    """Prefer mean / std from ``stats.yaml`` when a profile has been written."""
    from spec_mtp.structure.data.profile import load_stats

    recorded = load_stats(path)
    if recorded is None:
        return spec
    copied = dict(spec)
    copied['mean'] = tuple(float(x) for x in recorded['mean'])
    copied['std'] = tuple(float(x) for x in recorded['std'])
    return copied


def _train_transforms(spec: dict[str, Any], *, augment: bool) -> Any:
    """Pixels stay in 0–1. Flip / crop happen later, in the model, via kornia."""
    del spec, augment
    from torchvision import transforms

    return transforms.ToTensor()


def _eval_transforms(spec: dict[str, Any]) -> Any:
    from torchvision import transforms

    return transforms.ToTensor()


def vision_root(name: str) -> str | None:
    spec = _VISION_TORCH.get(name)
    if spec is None:
        return None
    return str(spec['root'])


def _build_torch_vision(
    data_config: DataConfig,
    assets_dir: Path,
    seed: int | None = None,
    origin: str | None = None,
) -> Data:
    import torch
    from torch.utils.data import DataLoader, Subset

    from spec_mtp.structure.system.runtime import make_generator, worker_init_fn
    from spec_mtp.structure.system.torchvision_load import import_torchvision

    import_torchvision()
    from torchvision import datasets

    name = data_config.name or 'unknown'
    spec = _VISION_TORCH[name]
    cfg = dict(data_config.config)
    root = assets_dir / str(spec['root'])
    root.mkdir(parents=True, exist_ok=True)
    ctor = getattr(datasets, str(spec['ctor']))
    augment = bool(cfg.get('augment', spec.get('train_aug') is not None))
    spec = _with_recorded_stats(spec, root / 'stats.yaml')
    train_tf = _train_transforms(spec, augment=augment)
    test_tf = _eval_transforms(spec)
    endpoint = endpoint_for(name, origin)
    discard_bad_archive(root, endpoint)
    with bind_endpoint(ctor, endpoint):
        train_full = ctor(root=str(root), download=True, transform=train_tf, **dict(spec['train_kw']))
        test_ds = ctor(root=str(root), download=True, transform=test_tf, **dict(spec['test_kw']))
    train_size = cfg.get('train_size')
    if train_size is not None:
        n = min(int(train_size), len(train_full))
        train_ds = Subset(train_full, list(range(n)))
    else:
        n = len(train_full)
        train_ds = train_full
    batch_size = int(cfg.get('batch_size', 64))
    if batch_size > n:
        batch_size = n
    generator = make_generator(seed)
    opts = _loader_options(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=opts['pin_memory'],
        num_workers=opts['num_workers'],
        worker_init_fn=worker_init_fn if seed is not None else None,
    )
    test_ratio = float(cfg.get('test_batch_ratio', 1) or 1)
    test_batch_size = max(int(batch_size * test_ratio), 1)
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=opts['pin_memory'],
        num_workers=opts['num_workers'],
    )
    data = Data(
        name=name,
        source='torch',
        loaders={'train': train_loader, 'test': test_loader},
        meta={
            'ready': True,
            'assets_dir': str(assets_dir),
            'config': cfg,
            'train_size': n,
            'test_size': len(test_ds),
            'batch_size': batch_size,
            'test_batch_size': test_batch_size,
            'data_size': list(spec['data_size']),
            'target_size': int(spec['target_size']),
            'mean': tuple(spec['mean']),
            'std': tuple(spec['std']),
            'train_aug': spec.get('train_aug') if augment else None,
            'augment': augment,
            'pin_memory': opts['pin_memory'],
            'num_workers': opts['num_workers'],
            'dtype': str(torch.float32),
            'seed': seed,
        },
    )
    data._train_set = train_ds
    return data


DataRegistry.register('Toy', 'stub', _build_stub)
DataRegistry.register('MNIST', 'stub', _build_stub)
DataRegistry.register('FashionMNIST', 'stub', _build_stub)
DataRegistry.register('CIFAR10', 'stub', _build_stub)
DataRegistry.register('CIFAR100', 'stub', _build_stub)
DataRegistry.register('SVHN', 'stub', _build_stub)
DataRegistry.register('MNIST', 'torch', _build_torch_vision)
DataRegistry.register('FashionMNIST', 'torch', _build_torch_vision)
DataRegistry.register('CIFAR10', 'torch', _build_torch_vision)
DataRegistry.register('CIFAR100', 'torch', _build_torch_vision)
DataRegistry.register('SVHN', 'torch', _build_torch_vision)
