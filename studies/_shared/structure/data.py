"""RPipe Data builders for spec_mtp prompt sets."""

from __future__ import annotations

import shutil
from pathlib import Path

from rpipe.structure.data.factory import Data, DataRegistry

from spec_mtp.data import encode_prompts, load_records, needs_tokenizer

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_SPECBENCH = _REPO / 'datasets' / 'spec_bench' / 'question.jsonl'


def _data_cfg(data_config):
    cfg = dict(data_config.config or {})
    cfg['name'] = str(data_config.name or 'local_prompts').lower()
    return cfg


def _land_specbench(data_cfg, assets_dir: Path) -> dict:
    """Copy the repo SpecBench jsonl into ``shared/data/specbench`` once."""
    src = Path(data_cfg.get('path') or _DEFAULT_SPECBENCH)
    if not src.is_file():
        raise FileNotFoundError(f'SpecBench question file not found: {src}')
    dest_dir = assets_dir / 'specbench'
    dest = dest_dir / 'question.jsonl'
    dest_dir.mkdir(parents=True, exist_ok=True)
    from studies._shared.structure.evidence import file_hash

    if not dest.is_file() or file_hash(src) != file_hash(dest):
        shutil.copy2(src, dest)
    if not dest.is_file():
        raise FileNotFoundError(
            f'SpecBench question file not found: {src}\n'
            'Place question.jsonl at datasets/spec_bench/question.jsonl')
    landed = dict(data_cfg)
    landed['path'] = str(dest)
    landed['source_path'] = str(src.resolve())
    return landed


def _build(data_config, assets_dir, seed=None):
    del seed
    data_cfg = _data_cfg(data_config)
    if data_cfg['name'] == 'specbench':
        data_cfg = _land_specbench(data_cfg, Path(assets_dir))
    records = load_records(data_cfg)
    meta = {'data_cfg': data_cfg, 'source_path': data_cfg.get('source_path')}
    if needs_tokenizer(data_cfg):
        loaders = {'test': records}
        meta['record_count'] = len(records)
    else:
        prompts = encode_prompts(records, None, data_cfg)
        loaders = {'test': prompts}
        meta['prompt_count'] = len(prompts)
    label = str(data_config.name or data_cfg['name'])
    return Data(name=label, source='spec_mtp', loaders=loaders, meta=meta)


def register_data():
    for name in ('specbench', 'local_prompts', 'dolly', 'SpecBench', 'LocalPrompts', 'Dolly'):
        DataRegistry.register(name, 'spec_mtp', _build)
