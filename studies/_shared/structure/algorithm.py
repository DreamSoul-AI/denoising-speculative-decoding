"""SpecMTP eval Algorithm ``(eval, spec_mtp)``: esp_mtp / ar_baseline / alignment_probe.

Registered on RPipe's AlgorithmRegistry by ``studies._shared.structure.recipe.register``.
Decoding is a dedicated Algorithm, not RPipe's classification eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.factory import AlgorithmRegistry
from rpipe.structure.algorithm.tracker import AlgorithmTracker

from studies._shared.structure import alignment_probe, ar_baseline, evaluate as esp_mtp
from studies._shared.structure.inputs import place_model, select_prompts
from studies._shared.structure.config import apply_variant, legacy_cfg
from studies._shared.structure.config import spec_for
from studies._shared.structure.prompts import resolve_eos, resolve_prompts

_TASKS = {
    'esp_mtp': esp_mtp,
    'ar_baseline': ar_baseline,
    'alignment_probe': alignment_probe,
}


def task_name(algorithm_config):
    return str(algorithm_config.setting('task', 'esp_mtp')).lower()


def effective_spec(spec, algorithm_config):
    """``spec`` block, then ``variant``, then a named ``mask`` setting (last wins)."""
    variant = algorithm_config.setting('variant', None)
    if variant:
        spec = apply_variant(dict(spec), variant)
    mask = algorithm_config.setting('mask', None)
    if mask:
        spec = spec_for(
            mask,
            beta=algorithm_config.setting('beta', None),
            lam=algorithm_config.setting('lam', None),
            gamma=algorithm_config.setting('gamma', None),
            branches=algorithm_config.setting('branches', None),
            num_masks=algorithm_config.setting('num_masks', None),
        )
    return spec


class SpecMtpEvalAlgorithm(Algorithm):
    def run(self, data, model, system, tracker: AlgorithmTracker) -> dict:
        task = task_name(self.config)
        impl = _TASKS.get(task)
        if impl is None:
            raise ValueError(f'unknown spec_mtp task: {task!r}')

        seed = int(system.meta.get('seed', 0))
        cfg = legacy_cfg(data, model, system, self.config, seed)
        cfg.spec = effective_spec(cfg.spec, self.config)

        module = place_model(model.module, system.device)
        prompts, eos_ids = select_prompts(
            resolve_prompts(data, model),
            resolve_eos(data, model, cfg.eval),
            cfg.eval,
        )
        from studies._shared.structure.evidence import capture, check_inputs, digest, model_digest, runtime

        assets = Path(system.assets_dir)
        assets.mkdir(parents=True, exist_ok=True)
        inputs = capture(data, model, prompts, eos_ids, cfg)
        (assets / 'decode_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n', encoding='utf-8')
        if task == 'ar_baseline':
            out = impl.evaluate(
                module, prompts, eos_ids, cfg.eval, seed, system.device, save_dir=None, aggregate=False)
        elif task == 'alignment_probe':
            # collect writes the layer metrics from these raw sums and counts.
            assets.mkdir(parents=True, exist_ok=True)
            out = impl.evaluate(
                module, prompts, eos_ids, cfg.spec, cfg.eval, save_dir=None, aggregate=False)
        else:
            out = impl.evaluate(
                module, prompts, eos_ids, cfg.spec, cfg.eval, seed, system.device,
                save_dir=None, aggregate=False)

        check_inputs(inputs)
        if inputs['model_digest'] != model_digest(module) or inputs['runtime'] != runtime():
            raise RuntimeError('model weights or runtime changed during decoding')
        raw = {'task': task, 'out': out, 'inputs_digest': digest(inputs)}
        (assets / 'decode_observations.json').write_text(json.dumps(raw, indent=2) + '\n', encoding='utf-8')
        return {'mode': 'eval', 'task': task}


TASKS = tuple(_TASKS)


def register_algorithms():
    AlgorithmRegistry.register('eval', 'spec_mtp', SpecMtpEvalAlgorithm)
