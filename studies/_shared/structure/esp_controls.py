"""Scoped ESP backend and model-local shape controls; no import-time registration."""
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import torch

from spec_mtp.structure.algorithm.factory import AlgorithmRegistry
from studies._shared.structure.algorithm import UntimedEval


class NumericsEval(UntimedEval):
    def run(self, data, model, system, tracker):
        controls = self.config.setting('numerics')
        assert model.module.dtype == torch.bfloat16
        old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        old_det = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(controls['deterministic'])
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = controls['bf16_reduced_precision_reduction']
            actual = {'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                      'deterministic': torch.are_deterministic_algorithms_enabled(), 'dtype': str(model.module.dtype)}
            assert actual['bf16_reduced_precision_reduction'] == controls['bf16_reduced_precision_reduction']
            assert actual['deterministic'] == controls['deterministic']
            path = Path(system.assets_dir) / 'numerics_controls.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(actual, indent=2) + '\n', encoding='utf-8')
            return super().run(data, model, system, tracker)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced
            torch.use_deterministic_algorithms(old_det, warn_only=old_warn)


def rowwise_linear(original, x):
    assert x.dtype == torch.bfloat16
    flat = x.reshape(-1, x.shape[-1])
    if not len(flat):
        return original(x)
    result = torch.cat([original(row) for row in flat.split(1)], dim=0)
    assert result.dtype == x.dtype
    return result.reshape(*x.shape[:-1], result.shape[-1])


class ShapeEval(NumericsEval):
    def run(self, data, model, system, tracker):
        enabled = self.config.setting('shape_control')['linear_rows']
        linears = [module for module in model.module.hf.modules() if isinstance(module, torch.nn.Linear)]
        assert linears and all(module.weight.dtype == torch.bfloat16 for module in linears)
        path = Path(system.assets_dir) / 'shape_controls.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'linear_rows': enabled, 'linear_modules': len(linears),
            'dtype': str(model.module.dtype)}, indent=2) + '\n', encoding='utf-8')
        with ExitStack() as stack:
            if enabled:
                for module in linears:
                    stack.enter_context(patch.object(module, 'forward', partial(rowwise_linear, module.forward)))
            return super().run(data, model, system, tracker)


def register_controls(*, shape=False):
    if type(shape) is not bool:
        raise ValueError('ESP shape policy must be an explicit boolean')
    AlgorithmRegistry.register('eval', 'spec_mtp', ShapeEval if shape else NumericsEval)
