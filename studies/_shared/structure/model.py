"""RPipe Model builders for HF causal and tiny LLaMA smoke models."""

from __future__ import annotations

from pathlib import Path

from rpipe.structure.model.factory import Model, ModelRegistry

from studies._shared.structure.config import _convert
from spec_mtp.model import build_model


def _model_cfg(model_config, model_type, assets_dir):
    cfg = dict(model_config.config or {})
    cfg['type'] = model_type
    if model_type == 'hf_causal':
        cfg['name_or_path'] = cfg.get('name_or_path') or model_config.name
        cfg = _materialize_hub(cfg, Path(assets_dir))
    return cfg


def _materialize_hub(model_cfg, assets_dir: Path):
    """Land weights under the Study ``shared/model`` tree so Runs reuse one download."""
    resolved = dict(model_cfg)
    resolved['cache_dir'] = str(assets_dir / 'hf')
    hub = str(model_cfg.get('hub', '') or '').lower()
    if hub != 'modelscope':
        return resolved
    import torch  # noqa: F401  load before ModelScope on Windows so BLAS DLLs agree
    from modelscope import snapshot_download

    local = snapshot_download(str(model_cfg['name_or_path']), cache_dir=str(assets_dir / 'modelscope'),
                              local_files_only=bool(model_cfg.get('local_files_only', False)))
    resolved['name_or_path'] = local
    resolved['local_files_only'] = True
    return resolved


def _build_tiny(model_config, assets_dir, data_meta=None):
    del data_meta
    model_cfg = _model_cfg(model_config, 'tiny_llama', assets_dir)
    module = build_model(_convert({'model': model_cfg}))
    return Model(
        name=str(model_config.name or 'tiny_llama'),
        source='spec_mtp',
        module=module,
        meta={'model_cfg': model_cfg},
    )


def _build_hf(model_config, assets_dir, data_meta=None):
    del data_meta
    model_cfg = _model_cfg(model_config, 'hf_causal', assets_dir)
    module = build_model(_convert({'model': model_cfg}))
    return Model(
        name=str(model_config.name or 'hf_causal'),
        source='spec_mtp',
        module=module,
        meta={'model_cfg': model_cfg},
    )


def register_models():
    ModelRegistry.register('tiny_llama', 'spec_mtp', _build_tiny)
    ModelRegistry.register('hf_causal', 'spec_mtp', _build_hf)
