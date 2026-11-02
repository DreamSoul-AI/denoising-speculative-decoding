"""Model-local linear geometry control; math and BF16 weights are unchanged."""
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch
import torch
from numerics import NumericsEval
from spec_mtp.structure.algorithm.factory import AlgorithmRegistry

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
        linears = [m for m in model.module.hf.modules() if isinstance(m, torch.nn.Linear)]
        assert linears and all(m.weight.dtype == torch.bfloat16 for m in linears)
        Path(system.assets_dir).mkdir(parents=True, exist_ok=True)
        (Path(system.assets_dir)/'shape_controls.json').write_text(json.dumps({'linear_rows': enabled, 'linear_modules': len(linears), 'dtype': str(model.module.dtype)}, indent=2)+'\n', encoding='utf-8')
        with ExitStack() as stack:
            if enabled:
                for module in linears:
                    stack.enter_context(patch.object(module, 'forward', partial(rowwise_linear, module.forward)))
            return super().run(data, model, system, tracker)

def register_shape():
    AlgorithmRegistry.register('eval', 'spec_mtp', ShapeEval)
