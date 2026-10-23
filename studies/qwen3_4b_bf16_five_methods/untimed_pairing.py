"""Untimed native BF16 execution with frozen checkpoint and prompt checks."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import spec_mtp.decode as decoding
from studies._shared.structure.algorithm import SpecMtpEvalAlgorithm
from studies._shared.structure import evidence
from studies._shared.structure.model import _build_hf
from rpipe.structure.algorithm.factory import AlgorithmRegistry
from rpipe.structure.model.factory import ModelRegistry


def verify_inputs(saved, manifest, snapshot):
    assert saved['dtype'] == 'torch.bfloat16' and saved['backend'] == 'eager'
    assert saved['eos_ids'] == manifest['eos_token_ids']
    assert len(saved['prompts']) == len(manifest['rendered_prompts'])
    for actual, expected in zip(saved['prompts'], manifest['rendered_prompts']):
        assert actual['category'] == expected['category'] and actual['input_ids'] == expected['input_ids']
    landed = {Path(name).name: sha for name, sha in saved['files'].items()}
    assert landed['question.jsonl'] == manifest['data_sha256']
    assert all(landed[f['path']] == f['sha256'] for f in snapshot['files'])
    cfg = saved['settings']['eval']
    assert cfg['temperature'] == 0 and cfg['max_new_tokens'] == 256 and cfg['warmup'] is False
    assert cfg['stop_on_eos'] is True and cfg['run_ar_baseline'] is False


class PairedUntimedEval(SpecMtpEvalAlgorithm):
    def run(self, data, model, system, tracker):
        assert self.config.setting('eval', {}).get('warmup') is False
        docs = Path(__file__).resolve().parent / 'docs'
        snapshot = json.loads((docs/'modelscope_snapshot.json').read_text())
        manifest = json.loads((docs/'prompts_manifest.json').read_text())
        module = model.module
        assert module.dtype == torch.bfloat16 and module.device.type == 'cuda'
        assert module.embedding_table.data_ptr() == module.hf.get_output_embeddings().weight.data_ptr()
        assert all(getattr(module.hf.config, key) == value for key, value in snapshot['config'].items())
        contract = {'dtype': str(module.dtype), 'device': str(module.device),
            'tied_storage': True, 'embedding_shape': list(module.embedding_table.shape),
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'deterministic': torch.are_deterministic_algorithms_enabled(), 'manifest_checked': False}
        assert contract['bf16_reduced_precision_reduction'] and not contract['deterministic']
        capture = evidence.capture
        def checked_capture(*args, **kwargs):
            saved = capture(*args, **kwargs)
            verify_inputs(saved, manifest, snapshot)
            contract['manifest_checked'] = True
            (Path(system.assets_dir)/'pairing_contract.json').write_text(
                json.dumps(contract, indent=2)+'\n', encoding='utf-8')
            return saved
        with patch.object(evidence, 'capture', checked_capture), \
                patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
            return super().run(data, model, system, tracker)


def build_cuda(model_config, assets_dir, data_meta=None):
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained
    def load(*args, **kwargs):
        assert 'device_map' not in kwargs
        return original(*args, **dict(kwargs, device_map='cuda'))
    # Direct placement avoids a full host copy; weights and native wrapper stay intact.
    with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=load):
        return _build_hf(model_config, assets_dir, data_meta)


def register_overrides():
    ModelRegistry.register('hf_causal', 'spec_mtp', build_cuda)
    AlgorithmRegistry.register('eval', 'spec_mtp', PairedUntimedEval)
