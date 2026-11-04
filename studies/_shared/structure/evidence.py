"""Actual decode inputs and runtime, checked again before finalisation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spec_mtp.structure.artifact.provenance import file_sha256 as file_hash


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def runtime() -> dict[str, Any]:
    import torch
    import transformers
    from spec_mtp.structure.artifact.provenance import capture_environment

    return {**capture_environment(), 'transformers': transformers.__version__,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_precision': torch.get_float32_matmul_precision()}


def model_digest(module: Any) -> str:
    result = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        result.update(name.encode())
        result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        result.update(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return result.hexdigest()


def capture(data: Any, model: Any, prompts: list[Any], eos_ids: Any, cfg: Any) -> dict[str, Any]:
    paths = []
    source = data.meta.get('source_path')
    if source:
        paths.append(Path(source).resolve())
    for block, key in ((data.meta.get('data_cfg') or {}, 'path'),
                       (model.meta.get('model_cfg') or {}, 'name_or_path')):
        value = block.get(key)
        if value and Path(value).exists():
            path = Path(value).resolve()
            paths.extend([path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file()))
    return {'runtime': runtime(), 'model_digest': model_digest(model.module),
            'dtype': str(model.module.dtype), 'device': str(model.module.device),
            'backend': model.module.hf.config._attn_implementation,
            'model_commit': getattr(model.module.hf.config, '_commit_hash', None),
            'tokenizer': data.meta.get('tokenizer'), 'eos_ids': list(eos_ids),
            'prompts': [{'category': p['category'], 'input_ids': p['input_ids'][0].tolist()} for p in prompts],
            'settings': cfg, 'files': {str(p): file_hash(p) for p in paths}}


def check_inputs(saved: dict[str, Any]) -> None:
    for name, expected in saved.get('files', {}).items():
        path = Path(name)
        if not path.is_file() or file_hash(path) != expected:
            raise RuntimeError(f'decode input changed: {path}')
