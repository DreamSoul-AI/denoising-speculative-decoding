"""Study recipe for SpecMTP: construction checks, then source registration.

A Study opts in with ``recipe: recipe.py`` whose body is
``from studies._shared.structure.recipe import register``. RPipe calls ``register(ctx)`` in
prepare, after its own freeze check and before Data / Model / Algorithm are
built. Nothing here loads weights, reads prompts or decodes.

Registration targets RPipe's runtime registries under the ``spec_mtp`` source
only; RPipe's default sources are untouched.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rpipe.structure.make.recipe import RecipeContext

SOURCE = 'spec_mtp'
FLASH_ATTENTION = ('flash_attention_2', 'flash_attention_3')


class SpecMtpConfigError(ValueError):
    pass


def suggest(value: object, choices: list[str] | tuple[str, ...]) -> str:
    """``"; did you mean 'x'?"`` for a close match, else the sorted choices."""
    import difflib

    close = difflib.get_close_matches(str(value).lower(), [str(c) for c in choices], n=1, cutoff=0.6)
    return f'; did you mean {close[0]!r}?' if close else f'; known: {sorted(choices)}'


def _layer(config: dict[str, Any], name: str) -> dict[str, Any]:
    block = config.get(name) or {}
    if not isinstance(block, dict):
        raise SpecMtpConfigError(f'{name} must be a mapping')
    return block


def validate(config: dict[str, Any]) -> dict[str, Any]:
    """Reject configs outside the supported SpecMTP boundary before construction.

    Returns a small description of the checked Run. Only conditions the
    decoder already requires are checked here; nothing is relaxed or added.
    """
    from rpipe.structure.control.layers import AlgorithmConfig

    from studies._shared.structure.algorithm import TASKS, effective_spec, task_name
    from studies._shared.structure.config import _convert
    from spec_mtp.data import _REGISTRY as DATASETS

    data, model, algorithm = (_layer(config, name) for name in ('data', 'model', 'algorithm'))
    if algorithm.get('source') != SOURCE:
        raise SpecMtpConfigError(f'algorithm.source must be {SOURCE!r} in a SpecMTP Study')
    if algorithm.get('mode') != 'eval':
        raise SpecMtpConfigError('SpecMTP decoding is an eval Algorithm; set algorithm.mode: eval')
    if data.get('source') == SOURCE and str(data.get('name', '')).lower() not in DATASETS:
        raise SpecMtpConfigError(f'unknown SpecMTP data name: {data.get("name")!r}' + suggest(data.get('name'), list(DATASETS)))
    if model.get('source') == SOURCE:
        if model.get('name') not in ('tiny_llama', 'hf_causal'):
            raise SpecMtpConfigError(f'unknown SpecMTP model name: {model.get("name")!r}'
                                     + suggest(model.get('name'), ['tiny_llama', 'hf_causal']))
        attn = str((model.get('config') or {}).get('attn_implementation', 'eager')).lower()
        if attn in FLASH_ATTENTION:
            raise SpecMtpConfigError(
                f'attn_implementation={attn} cannot take the 4D tree attention mask; use eager or sdpa')
    algorithm_config = AlgorithmConfig.from_mapping(algorithm)
    task = task_name(algorithm_config)
    if task not in TASKS:
        raise SpecMtpConfigError(f'unknown spec_mtp task: {task!r}' + suggest(task, list(TASKS)))
    # Same resolution as the Algorithm (spec → variant → mask); pure config work.
    spec = effective_spec(_convert(dict(algorithm_config.setting('spec', {}) or {})), algorithm_config)
    info = {'task': task}
    if task == 'esp_mtp':  # only the tree decoder has the impl / tree boundary
        impl = str(spec.get('impl', 'efficient')).lower()
        mode = str((spec.get('tree') or {}).get('mode', 'dynamic')).lower()
        if impl not in ('naive', 'efficient'):
            raise SpecMtpConfigError(f'Unsupported spec.impl: {impl}')
        if impl == 'efficient' and mode != 'static':
            raise SpecMtpConfigError('spec.impl=efficient requires spec.tree.mode=static; '
                                     'use spec.impl=naive with dynamic trees')
        info.update(impl=impl, tree=mode, method=str(spec.get('method', 'esp')))
    return info


def register_sources() -> None:
    """Idempotent: (name, spec_mtp) Data / Model builders and the eval Algorithm."""
    from studies._shared.structure.algorithm import register_algorithms
    from studies._shared.structure.data import register_data
    from studies._shared.structure.model import register_models

    register_data()
    register_models()
    register_algorithms()


def register(ctx: RecipeContext) -> None:
    from rpipe.structure.make.expand import load_study_yaml

    from studies._shared.structure.sources import RUN_ASSET_NAME, check_frozen, manifest

    study = load_study_yaml(ctx.study_dir) if (ctx.study_dir / 'study.yaml').is_file() else {}
    flow = study.get('flow') or {}
    if flow.get('prepare_shared', True) is not False:
        raise SpecMtpConfigError(
            'RPipe make-time shared data only knows torchvision hosts; '
            'SpecMTP lands SpecBench and weights in shared/ during Run prepare. '
            'Set flow.prepare_shared: false')
    current = manifest()
    from studies._shared.structure.evidence import digest

    current['config_digest'] = digest(ctx.config)
    from rpipe.structure.artifact.provenance import source_files

    current['study_files'] = source_files(ctx.study_dir, study)
    check_frozen(ctx.study_dir, study, current)
    try:
        validate(ctx.config)
    except SpecMtpConfigError as error:
        where = ctx.study_dir / 'study.yaml'
        raise SpecMtpConfigError(f'{error} (Run config from {where}; checked before any weights load)') from error
    ctx.assets_dir.mkdir(parents=True, exist_ok=True)
    (ctx.assets_dir / RUN_ASSET_NAME).write_text(
        json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    register_sources()
