"""Scoped BF16 backend controls for the ESP reproduction gate."""
import json
from pathlib import Path
import torch
from untimed_4b import UntimedEval
from spec_mtp.structure.algorithm.factory import AlgorithmRegistry

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
            actual = {'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, 'deterministic': torch.are_deterministic_algorithms_enabled(), 'dtype': str(model.module.dtype)}
            assert actual['bf16_reduced_precision_reduction'] == controls['bf16_reduced_precision_reduction']
            assert actual['deterministic'] == controls['deterministic']
            path = Path(system.assets_dir) / 'numerics_controls.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(actual, indent=2) + '\n', encoding='utf-8')
            return super().run(data, model, system, tracker)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced
            torch.use_deterministic_algorithms(old_det, warn_only=old_warn)

def register_controls():
    AlgorithmRegistry.register('eval', 'spec_mtp', NumericsEval)
