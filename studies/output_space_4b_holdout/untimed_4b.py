"""Study-only untimed evaluation and direct CUDA materialization of unchanged weights."""
from types import SimpleNamespace
from unittest.mock import patch
import spec_mtp.decode as decoding
from studies._shared.structure.algorithm import SpecMtpEvalAlgorithm
from studies._shared.structure.model import _build_hf
from rpipe.structure.algorithm.factory import AlgorithmRegistry
from rpipe.structure.model.factory import ModelRegistry


class UntimedEval(SpecMtpEvalAlgorithm):
    def run(self, data, model, system, tracker):
        assert self.config.setting('eval', {}).get('warmup') is False
        # Timing fields remain zero sentinels required by native collection;
        # no decoding clock is read and the process-wide time module is intact.
        with patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
            return super().run(data, model, system, tracker)


def build_cuda(model_config, assets_dir, data_meta=None):
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained

    def load(*args, **kwargs):
        assert 'device_map' not in kwargs
        kwargs['device_map'] = 'cuda'
        return original(*args, **kwargs)

    # Direct placement avoids a full FP32 host copy when another task uses RAM.
    # Model architecture/dtype/weights, native wrapper and factory stay intact.
    with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=load):
        return _build_hf(model_config, assets_dir, data_meta)


def register_overrides():
    ModelRegistry.register('hf_causal','spec_mtp',build_cuda)
    AlgorithmRegistry.register('eval','spec_mtp',UntimedEval)
