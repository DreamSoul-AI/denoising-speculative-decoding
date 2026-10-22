"""Map an RPipe Run layer into the legacy NamedDict the decoders expect."""

from __future__ import annotations

from typing import Any



def legacy_cfg(data: Any, model: Any, system: Any, algorithm_config: Any, seed: int) -> Any:
    eval_cfg = dict(algorithm_config.setting('eval', {}) or {})
    eval_cfg['device'] = str(system.device)
    return _convert({
        'run': {'seed': int(seed)},
        'data': dict(data.meta.get('data_cfg') or {}),
        'model': dict(model.meta.get('model_cfg') or {}),
        'eval': eval_cfg,
        'spec': dict(algorithm_config.setting('spec', {}) or {}),
    })


def apply_variant(spec_cfg: dict[str, Any] | None, variant: str | None) -> Any:
    """Merge a smoke variant or a ``<group>/<cell>`` paper preset into ``spec``."""
    from studies._shared.structure.paper_variants import PAPER_VARIANTS
    from studies._shared.structure.smoke_variants import VARIANTS

    key = str(variant or '').lower()
    if not key:
        return spec_cfg
    presets = {**VARIANTS, **PAPER_VARIANTS}
    if key not in presets:
        import difflib

        close = difflib.get_close_matches(key, sorted(presets), n=3, cutoff=0.6)
        hint = f'; did you mean {", ".join(map(repr, close))}?' if close else (
            f'; smoke variants: {sorted(VARIANTS)}; paper presets are <llama3|qwen3>/<cell>, '
            'see spec_mtp/structure/algorithm/paper_variants.py')
        raise ValueError(f'unknown spec variant: {key!r}{hint}')
    merged = dict(spec_cfg or {})
    merged.update(presets[key])
    return _convert(merged)

import yaml

class NamedDict(dict):
    """Dict subclass with attribute-style access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = _convert(value)

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def _convert(obj):
    if isinstance(obj, dict) and not isinstance(obj, NamedDict):
        return NamedDict({k: _convert(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


def get_config(path):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    cfg = _convert(raw)
    cfg['cfg_file'] = path
    return cfg


def apply_override(cfg, dotted_key, raw_value):
    """Apply a `--set a.b.c=value` style override with YAML-typed value."""
    keys = dotted_key.split('.')
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = NamedDict()
        node = node[key]
    node[keys[-1]] = _convert(yaml.safe_load(raw_value))


_TREE = {
    'num_masks': 1,
    'impl': 'efficient',
    'tree': {'mode': 'static', 'branches': [14], 'pruning': True},
}


def spec_for(name: str, beta: float | None = None, lam: float | None = None,
             gamma: float | None = None, branches: list[int] | None = None,
             num_masks: int | None = None) -> Any:
    key = str(name or '').lower()
    if key == 'frozen_mean':
        body: dict = {
            **_TREE,
            'method': 'esp',
            'mask': {'init': 'mean_prompt', 'lam': 0, 'update': False},
        }
    elif key == 'last_token':
        body = _ema(level='last', step_scale=0, beta=0.9, lam=0.1)
    elif key == 'updating_mean':
        body = _ema(level='order0', step_scale=0, beta=0.9, lam=0.1 if lam is None else lam)
    elif key == 'gamma_last':
        body = _ema(
            level='last',
            step_scale=1.0 if gamma is None else gamma,
            beta=0.9 if beta is None else beta,
            lam=0.1,
            normalize=False,
        )
    elif key == 'gamma_last_unit':
        body = _ema(
            level='last',
            step_scale=1.0 if gamma is None else gamma,
            beta=0.9 if beta is None else beta,
            lam=0.1,
            normalize=True,
        )
    elif key == 'mean_plus_gamma':
        body = _ema(
            level='order0',
            step_scale=1.0 if gamma is None else gamma,
            beta=0.9999 if beta is None else beta,
            lam=0.1 if lam is None else lam,
            normalize=False,
        )
    else:
        raise ValueError(f'unknown mask setting: {name!r}')
    if branches is not None:
        body['tree']['branches'] = [int(x) for x in branches]
    if num_masks is not None:
        body['num_masks'] = int(num_masks)
    return _convert(body)


def _ema(level: str, step_scale: float, beta: float, lam: float, normalize: bool = False) -> dict[str, Any]:
    return {
        **_TREE,
        'method': 'ema_velocity',
        'ema': {
            'beta': float(beta),
            'step_scale': float(step_scale),
            'normalize': bool(normalize),
            'extrapolate_ema': False,
            'update': True,
            'level': level,
            'lam': float(lam),
        },
    }
