"""Resume target resolution (algorithm-layer; system only loads files)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from spec_mtp.structure.algorithm.config import AlgorithmConfig


def capture_rng() -> dict[str, Any]:
    import random
    import numpy as np
    import torch

    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _validated_rng(state: dict[str, Any]) -> dict[str, Any]:
    import random
    import numpy as np
    import torch

    if not isinstance(state, dict) or not {'python', 'numpy', 'torch', 'cuda'} <= state.keys():
        raise ValueError('invalid checkpoint RNG state')
    cuda = state['cuda']
    if cuda is not None and (not isinstance(cuda, (list, tuple)) or not torch.cuda.is_available()
                             or len(cuda) != torch.cuda.device_count()):
        raise ValueError('checkpoint CUDA RNG devices do not match current runtime')
    # Validate every domain on private generators before changing any global state.
    try:
        python_rng = random.Random()
        python_rng.setstate(state['python'])
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state(state['numpy'])
        cpu_rng = torch.Generator(device='cpu')
        cpu_rng.set_state(state['torch'].cpu())
        cuda_states = []
        for index, value in enumerate(cuda or []):
            generator = torch.Generator(device=f'cuda:{index}')
            generator.set_state(value.cpu())
            cuda_states.append(generator.get_state())
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'invalid checkpoint RNG state: {exc}') from exc
    return {'python': python_rng.getstate(), 'numpy': numpy_rng.get_state(),
            'torch': cpu_rng.get_state(), 'cuda': cuda_states if cuda is not None else None}


def validate_rng(state: dict[str, Any]) -> None:
    """Admit serialized RNG states without changing any global generator."""
    _validated_rng(state)


def restore_rng(state: dict[str, Any]) -> None:
    import random
    import numpy as np
    import torch

    admitted = _validated_rng(state)
    random.setstate(admitted['python'])
    np.random.set_state(admitted['numpy'])
    torch.set_rng_state(admitted['torch'])
    if admitted['cuda'] is not None:
        torch.cuda.set_rng_state_all(admitted['cuda'])


def partial_resume_warning(logger=None) -> None:
    import warnings

    message = ('checkpoint resume is partial: RNG/data iterator equivalence is not guaranteed '
               'for legacy checkpoints or unsupported data/Trainer runtimes')
    if logger is not None:
        logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def resume_stem(config: AlgorithmConfig, *, mode: str) -> str | None:
    """Return checkpoint stem/path to load, or None to skip resume."""
    explicit = config.setting('resume_from')
    if explicit:
        return str(explicit)
    raw = config.setting('resume')
    if raw is None:
        return 'latest' if mode == 'train' else 'best'
    if raw is False:
        return None
    text = str(raw).strip().lower()
    if text in ('', 'false', '0', 'none', 'off'):
        return None
    return str(raw)


def study_dir_from_assets(assets_dir: Path | str | None) -> Path | None:
    if assets_dir is None:
        return None
    assets = Path(assets_dir).resolve()
    if assets.name != 'assets':
        return None
    # studies/<name>/runs/<id>/assets
    try:
        return assets.parent.parent.parent
    except Exception:
        return None


def sibling_train_checkpoint(
    system: Any,
    *,
    stem: str,
    current_id: str | None = None,
) -> Path | None:
    """Eval Run: ``best.pt`` lives on the matching train Run, not this assets dir."""
    assets_dir = getattr(system, 'assets_dir', None)
    study_dir = study_dir_from_assets(assets_dir)
    if study_dir is None or assets_dir is None:
        return None
    run_id = current_id or Path(assets_dir).resolve().parent.name
    name = stem if stem.endswith('.pt') else f'{stem}.pt'
    folder_name = stem[:-3] if stem.endswith('.pt') else stem
    for rid in sibling_train_ids(study_dir, run_id):
        base = study_dir / 'runs' / rid / 'assets' / 'checkpoints'
        path = base / name
        if path.is_file():
            return path
        folder = base / folder_name
        if folder.is_dir() and ((folder / 'model.pt').is_file() or (folder / 'meta.json').is_file()):
            return folder
    return None


def sibling_train_ids(study_dir: Path, run_id: str) -> list[str]:
    """Match the current index's factors and seed, even before checkpoints exist."""
    from spec_mtp.structure.artifact.index import load_index

    try:
        index = load_index(study_dir)
    except (OSError, TypeError, ValueError):
        return []
    current_factors: dict[str, Any] | None = None
    current_seed: Any = None
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            rid = str(run.get('run_dir') or run.get('id') or '')
            if rid == run_id:
                current_factors = dict(exp.get('factors') or {})
                current_seed = run.get('seed')
                break
        if current_factors is not None:
            break
    if current_factors is None:
        return []
    want = dict(current_factors)
    want['algorithm.mode'] = 'train'
    matched: list[str] = []
    for exp in index.get('experiments') or []:
        if dict(exp.get('factors') or {}) != want:
            continue
        for run in exp.get('runs') or []:
            if run.get('seed') != current_seed:
                continue
            rid = str(run.get('run_dir') or run.get('id') or '')
            if rid:
                matched.append(rid)
    return matched


def resolve_checkpoint_ref(
    ref: str,
    system: Any,
    *,
    mode: str,
) -> str:
    """Turn stem / relative path / ``sibling`` into a path or stem system can load."""
    text = str(ref).strip()
    if text.lower() in ('sibling', 'train'):
        stem = 'best' if mode == 'eval' else 'latest'
        found = sibling_train_checkpoint(system, stem=stem)
        return str(found) if found is not None else stem
    path = Path(text)
    if path.is_file() or path.is_dir():
        return str(path)
    assets_dir = getattr(system, 'assets_dir', None)
    study_dir = study_dir_from_assets(assets_dir)
    if study_dir is not None:
        relative = study_dir / text
        if relative.is_file():
            return str(relative)
        if not text.endswith('.pt'):
            alt = study_dir / f'{text}.pt'
            if alt.is_file():
                return str(alt)
    return text


def _has_checkpoint_material(path: Path) -> bool:
    """Recognize weight-bearing paths without loading tensors during planning."""
    if path.is_file():
        return True
    folder = path if path.is_dir() or path.suffix.lower() != '.pt' else path.with_suffix('')
    return folder.with_name(f'{folder.name}.pt').is_file() or any(
        (folder / name).is_file() for name in ('payload.pt', 'model.pt')
    )


def sibling_train_dependency(study_dir: Path, run_id: str) -> str | None:
    """Identify the train that launch must finish before this eval can run."""
    from spec_mtp.structure.artifact import load_config

    root = Path(study_dir) / 'runs' / run_id
    try:
        cfg = AlgorithmConfig.from_mapping(load_config(root / 'config.yaml').get('algorithm'))
    except (OSError, ValueError):
        return None
    if cfg.mode != 'eval':
        return None
    ref = resume_stem(cfg, mode='eval')
    if ref is None:
        return None
    parents = sibling_train_ids(study_dir, run_id)
    if not parents:
        return None
    target = Path(resolve_checkpoint_ref(ref, SimpleNamespace(assets_dir=root / 'assets'), mode='eval'))
    if target.exists() or target.suffix.lower() == '.pt' or '/' in str(target) or '\\' in str(target):
        if _has_checkpoint_material(target):
            # Explicit external/local weights do not depend on a sibling train.
            return next((rid for rid in parents if
                         (Path(study_dir) / 'runs' / rid / 'assets' / 'checkpoints').resolve()
                         in target.resolve().parents), None)
    else:
        own = root / 'assets' / 'checkpoints' / target
        if _has_checkpoint_material(own):
            return None
    return parents[0]


def apply_module_state(module: Any, payload: dict[str, Any] | None) -> None:
    if module is None or not payload:
        return
    state = payload.get('model')
    if state is None:
        return
    module.load_state_dict(state)


def admit_training_resume(payload, *, model, tracker, data, contract, num_steps, step_period, resume_sources=None):
    """Check persisted training state before touching live scientific objects."""
    import math
    from copy import copy, deepcopy
    import torch
    from spec_mtp.structure.algorithm.tracker import AlgorithmTracker

    if payload is None:
        return
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
        raise ValueError('checkpoint requires model state')
    expected = model.module.state_dict()
    if set(payload['model']) != set(expected):
        raise ValueError('checkpoint model state keys differ')
    for name, value in payload['model'].items():
        current = expected[name]
        if (not isinstance(value, torch.Tensor) or not isinstance(current, torch.Tensor)
                or value.shape != current.shape or value.dtype != current.dtype):
            raise ValueError(f'checkpoint model state is incompatible: {name}')
    for field in ('step', 'epoch'):
        value = payload.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'invalid checkpoint {field}')
    if payload.get('step', 0) is not None and payload.get('step', 0) > contract['num_steps']:
        raise ValueError('checkpoint step exceeds the declared budget')
    for field in ('best_value', 'best_accuracy'):
        value = payload.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f'invalid checkpoint {field}')
    state = payload.get('training_state')
    if state is not None:
        if not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] != 1:
            raise ValueError('unsupported checkpoint training_state version')
        if any(type(payload.get(field)) is not int or payload[field] < 0 for field in ('step', 'epoch')):
            raise ValueError('checkpoint training state requires step and epoch counters')
        if not isinstance(payload.get('tracker'), dict):
            raise ValueError('checkpoint training state requires tracker state')
        if state.get('data') is not None and not {'version', 'rng', 'data', 'contract', 'stall', 'already_percent', 'stopped'} <= state.keys():
            raise ValueError('checkpoint exact training state is incomplete')
        if state.get('data') is not None and not resume_sources:
            raise ValueError('exact recovery requires a committed checkpoint bundle with a bound SHA256')
        if (state.get('contract') is not None or state.get('data') is not None) and state.get('contract') != contract:
            raise ValueError('checkpoint training contract differs; keep budget/settings/runtime unchanged')
        if state.get('data') is not None:
            validate = getattr(data, 'validate_training_state', None)
            if not callable(validate):
                raise ValueError('data does not support checkpoint iterator admission')
            validate(state['data'], num_steps=num_steps, step_period=step_period)
        if 'rng' not in state:
            raise ValueError('checkpoint training state requires RNG state')
        validate_rng(state['rng'])
        if type(state.get('stall', 0)) is not int or state.get('stall', 0) < 0:
            raise ValueError('invalid checkpoint early-stop state')
        if type(state.get('stopped', False)) is not bool:
            raise ValueError('invalid checkpoint stop state')
        percents = state.get('already_percent', [])
        if not isinstance(percents, list) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 100 for v in percents):
            raise ValueError('invalid checkpoint save progress')
    optimizer = payload.get('optimizer')
    if state is not None and optimizer is None:
        raise ValueError('checkpoint training state requires optimizer state')
    if optimizer is not None:
        if not isinstance(optimizer, dict) or not isinstance(optimizer.get('param_groups'), list) or not isinstance(optimizer.get('state'), dict):
            raise ValueError('invalid checkpoint optimizer state')
        try:
            groups = optimizer['param_groups']
            if len(groups) != 1:
                raise ValueError('checkpoint optimizer parameter-group count differs')
            ids = [key for group in groups for key in group['params']]
            parameters = list(model.module.parameters())
            if (len(ids) != len(parameters) or len(set(ids)) != len(ids)
                    or any(type(key) is not int for key in ids) or not set(optimizer['state']) <= set(ids)):
                raise ValueError('checkpoint optimizer parameters differ')
            for group in groups:
                if not math.isfinite(group['lr']) or group['lr'] < 0:
                    raise ValueError('invalid checkpoint optimizer learning rate')
            for key, parameter in zip(ids, parameters):
                for value in optimizer['state'].get(key, {}).values():
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape != parameter.shape:
                        raise ValueError('checkpoint optimizer tensor shape differs')
            from spec_mtp.structure.algorithm.optim import make_optimizer

            private_parameters = [torch.nn.Parameter(torch.empty(parameter.shape, dtype=parameter.dtype, device='meta'))
                                  for parameter in parameters]
            trial = make_optimizer(SimpleNamespace(parameters=lambda: iter(private_parameters)),
                                   AlgorithmConfig.from_mapping(contract['settings']))
            expected_group = trial.state_dict()['param_groups'][0]
            if not set(expected_group) <= set(groups[0]):
                raise ValueError('checkpoint optimizer group schema differs')
            for key, value in expected_group.items():
                if key not in ('params', 'lr') and groups[0][key] != value:
                    raise ValueError('checkpoint optimizer settings differ')
            trial.load_state_dict(optimizer)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('invalid checkpoint optimizer state') from exc
    scheduler = payload.get('scheduler')
    if scheduler is not None:
        from spec_mtp.structure.algorithm.optim import make_optimizer, make_scheduler

        config = AlgorithmConfig.from_mapping(contract['settings'])
        parameters = [torch.nn.Parameter(torch.empty(parameter.shape, dtype=parameter.dtype, device='meta'))
                      for parameter in model.module.parameters()]
        private_optimizer = make_optimizer(SimpleNamespace(parameters=lambda: iter(parameters)), config)
        private_scheduler = make_scheduler(private_optimizer, config, contract['num_steps'] if contract['unit'] == 'step'
                                            else contract['num_epochs'])
        if private_scheduler is None or not isinstance(scheduler, dict):
            raise ValueError('invalid checkpoint scheduler state')
        expected_scheduler = private_scheduler.state_dict()

        def matching(saved, expected):
            if isinstance(expected, dict):
                return (type(saved) is type(expected) and saved.keys() == expected.keys()
                        and all(matching(saved[key], value) for key, value in expected.items()))
            if isinstance(expected, (list, tuple)):
                return (type(saved) is type(expected) and len(saved) == len(expected)
                        and all(matching(a, b) for a, b in zip(saved, expected)))
            if isinstance(expected, bool):
                return type(saved) is bool
            if isinstance(expected, (int, float)):
                return type(saved) in (int, float) and math.isfinite(saved)
            return type(saved) is type(expected)

        if not matching(scheduler, expected_scheduler):
            raise ValueError('invalid checkpoint scheduler schema or value types')
        for field in set(expected_scheduler) - {'last_epoch', '_step_count', '_last_lr', '_get_lr_called_within_step', '_is_initial'}:
            if scheduler[field] != expected_scheduler[field]:
                raise ValueError('checkpoint scheduler settings differ')
        if (type(scheduler.get('last_epoch', 0)) is not int or type(scheduler.get('_step_count', 0)) is not int
                or scheduler.get('last_epoch', 0) < -1 or scheduler.get('_step_count', 0) < 0):
            raise ValueError('invalid checkpoint scheduler progress')
        private_scheduler.load_state_dict(deepcopy(scheduler))
    elif state is not None:
        settings = AlgorithmConfig.from_mapping(contract['settings'])
        configured = settings.setting('scheduler', settings.setting('scheduler_name'))
        if str(configured).lower() not in ('none', '', 'constant', 'null'):
            raise ValueError('checkpoint requires configured scheduler state')
    if payload.get('logger') is not None and not isinstance(payload['logger'], dict):
        raise ValueError('invalid checkpoint logger state')
    if payload.get('tracker') is not None:
        tracker_state = payload['tracker']
        if not isinstance(tracker_state, dict) or type(tracker_state.get('step', 0)) is not int or tracker_state.get('step', 0) < 0:
            raise ValueError('invalid checkpoint tracker state')
        try:
            AlgorithmTracker.load_state_dict(copy(tracker), deepcopy(tracker_state))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError('invalid checkpoint tracker state') from exc
