"""SpecMTP eval Algorithm ``(eval, spec_mtp)``: esp_mtp / ar_baseline / alignment_probe.

Registered on RPipe's AlgorithmRegistry by ``studies._shared.structure.recipe.register``.
Decoding is a dedicated Algorithm, not RPipe's classification eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spec_mtp.structure.algorithm.base import Algorithm
from spec_mtp.structure.algorithm.factory import AlgorithmRegistry
from spec_mtp.structure.algorithm.tracker import AlgorithmTracker

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


class UntimedEval(SpecMtpEvalAlgorithm):
    """Use the native verifier and AR diagnostic with zero timing sentinels."""
    def run(self, data, model, system, tracker):
        from types import SimpleNamespace
        from unittest.mock import patch
        import spec_mtp.structure.algorithm.decode.loop as decoding

        assert self.config.setting('eval', {}).get('warmup') is False
        with patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
            return super().run(data, model, system, tracker)


TASKS = tuple(_TASKS)


def register_algorithms():
    AlgorithmRegistry.register('eval', 'spec_mtp', SpecMtpEvalAlgorithm)


class PairedUntimedEval(UntimedEval):
    pairing_root: Path
    def run(self, data, model, system, tracker):
        import torch
        from unittest.mock import patch
        from studies._shared.structure import evidence
        from studies._shared.structure.validation import verify_paired_inputs
        assert self.config.setting('eval', {}).get('warmup') is False
        docs = self.pairing_root / 'docs'
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
            verify_paired_inputs(saved, manifest, snapshot)
            contract['manifest_checked'] = True
            (Path(system.assets_dir)/'pairing_contract.json').write_text(
                json.dumps(contract, indent=2)+'\n', encoding='utf-8')
            return saved
        with patch.object(evidence, 'capture', checked_capture):
            return super().run(data, model, system, tracker)


class DirectionEval(PairedUntimedEval):
    study_root: Path
    execution_root: Path
    recovery = False
    def run(self, data, model, system, tracker):
        from unittest.mock import patch
        from studies._shared.structure import evaluate as assembly
        from studies._shared.structure.geometry_4b import direction_control as control, content_start_of
        from studies._shared.structure.validation import check_direction_refs
        from studies._shared.structure.resource_admission import direction_admission
        from spec_mtp.structure.artifact.provenance import file_sha256
        import spec_mtp.structure.algorithm.decode.loop as decoding
        BASE, ROOT = self.pairing_root, self.execution_root
        check_direction_refs(self.study_root, self.execution_root)
        from transformers import AutoTokenizer
        manifest = json.loads((BASE / 'docs/prompts_manifest.json').read_text())
        records = {p['question_id']: p for p in map(json.loads,
            (ROOT / 'datasets/spec_bench/question.jsonl').read_text(encoding='utf-8').splitlines())}
        assert file_sha256(ROOT / 'datasets/spec_bench/question.jsonl') == manifest['data_sha256']
        expected = manifest['rendered_prompts']
        assert len(expected) == 480
        tokenizer = AutoTokenizer.from_pretrained(BASE / 'shared/model/modelscope/Qwen/Qwen3-4B', local_files_only=True)
        starts = [content_start_of(tokenizer, records[p['question_id']]['turns'][0], p['input_ids']) for p in expected]
        mode = self.config.setting('direction')
        builder, generate = assembly.build_mask_provider, decoding.SpecDecoder.generate
        current: dict[str, Any] = {'index': 0, 'direction': {}}
        progress: list[dict[str, Any]] = []
        assets = Path(system.assets_dir)
        (assets / 'direction_admission.json').write_text(json.dumps(model.meta['direction_admission'], indent=2) + '\n')

        def controlled_builder(*args, **kwargs):
            return control(builder(*args, **kwargs), mode, lambda: starts[current['index']], current['direction'])

        def observed_generate(decoder, prompt):
            index = current['index']
            assert prompt[0].tolist() == expected[index]['input_ids']
            check = direction_admission(assets, initial=False, wait=self.recovery)
            current['direction'].clear()
            output, stats = generate(decoder, prompt)
            assert stats['verification_argmax_audit']['passed']
            assert stats['prefill_time'] == stats['decode_time'] == 0
            row = {'index': index, 'question_id': expected[index]['question_id'], 'direction': dict(current['direction']),
                'gpu_admission': check,
                'accepted': stats['accepted'], 'committed': stats['committed'], 'decode_calls': stats['decode_calls'],
                'new_tokens': stats['new_tokens'], 'argmax_checked_tokens': stats['verification_argmax_audit']['checked_committed_tokens']}
            progress.append(row)
            with (assets / 'direction_partial_private.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'output_ids': output[0].tolist(), 'stats': stats, **row}) + '\n')
            (assets / 'direction_progress.json').write_text(json.dumps({'direction': mode, 'completed': len(progress),
                'total': 480, 'rows': progress}, indent=2) + '\n', encoding='utf-8')
            if (index + 1) % 10 == 0:
                print('Direction', mode, 'completed', index + 1, '/ 480', flush=True)
            current['index'] += 1
            return output, stats

        with patch.object(assembly, 'build_mask_provider', side_effect=controlled_builder), \
                patch.object(decoding.SpecDecoder, 'generate', observed_generate):
            result = super().run(data, model, system, tracker)
        assert current['index'] == 480
        check_direction_refs(self.study_root, self.execution_root)
        return result


class NormEval(PairedUntimedEval):
    study_root: Path
    execution_root: Path
    admission_wait: Any
    control_kind = "norm"

    def check_refs(self):
        from studies._shared.structure.validation import check_control_refs
        return check_control_refs(self.study_root, self.execution_root, kind=self.control_kind)

    def run(self, data, model, system, tracker):
        self.check_refs()
        from unittest.mock import patch
        from studies._shared.structure import evaluate as assembly
        from studies._shared.structure.geometry_4b import observe_initial_bias
        import spec_mtp.structure.algorithm.decode.loop as decoding
        expected = json.loads((self.pairing_root / 'docs/prompts_manifest.json').read_text(encoding='utf-8'))['rendered_prompts']
        mode = self.config.setting('direction')
        scale = self.config.setting('spec')['ema']['step_scale']
        assert mode in ('template', 'random0', 'random1', 'random2') and scale in (.5, 2.)
        assert len(expected) == 480
        builder, generate = assembly.build_mask_provider, decoding.SpecDecoder.generate
        current: dict[str, Any] = {'index': 0, 'bias': {}}
        progress: list[dict[str, Any]] = []
        assets = Path(system.assets_dir)
        (assets / 'norm_admission_initial.json').write_text(json.dumps(model.meta['norm_scan_admission']) + '\n')

        def controlled_builder(*args, **kwargs):
            provider = builder(*args, **kwargs)
            assert provider.step_scale == scale
            return observe_initial_bias(provider, mode, current['bias'])

        def observed_generate(decoder, prompt):
            index = current['index']
            assert prompt[0].tolist() == expected[index]['input_ids']
            admission = self.admission_wait(assets, initial=False)
            current['bias'].clear()
            output, stats = generate(decoder, prompt)
            audit = stats['verification_argmax_audit']
            assert audit['passed'] and audit['checked_committed_tokens'] == stats['new_tokens']
            assert stats['prefill_time'] == stats['decode_time'] == 0
            row = {'index': index, 'question_id': expected[index]['question_id'], 'bias': dict(current['bias']),
                'accepted': stats['accepted'], 'committed': stats['committed'], 'decode_calls': stats['decode_calls'],
                'new_tokens': stats['new_tokens'], 'argmax_checked_tokens': audit['checked_committed_tokens'],
                'attempt_by_depth': stats['attempt_by_depth'], 'accept_by_depth': stats['accept_by_depth'],
                'first_block_accepted': len(stats['trace'][0]['accepted_nodes']) if stats['trace'] else None,
                'gpu_admission': admission}
            progress.append(row)
            with (assets / 'norm_partial_private.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'output_ids': output[0].tolist(), 'stats': stats, **row}) + '\n')
            (assets / 'norm_progress.json').write_text(json.dumps({'direction': mode, 'step_scale': scale,
                'completed': len(progress), 'total': 480, 'rows': progress}, indent=2) + '\n', encoding='utf-8')
            if (index + 1) % 10 == 0:
                print('Norm scan', mode, scale, index + 1, '/480', flush=True)
            current['index'] += 1
            return output, stats

        with patch.object(assembly, 'build_mask_provider', side_effect=controlled_builder), \
                patch.object(decoding.SpecDecoder, 'generate', observed_generate):
            result = super().run(data, model, system, tracker)
        assert current['index'] == 480
        self.check_refs()
        return result



class PairedRepeatEval(PairedUntimedEval):
    study_root: Path
    execution_root: Path
    admission_wait: Any
    def run(self, data, model, system, tracker):
        from unittest.mock import patch
        from studies._shared.structure.validation import check_control_refs
        import spec_mtp.structure.algorithm.decode.loop as decoding
        check_control_refs(self.study_root, self.execution_root, kind="paired")
        generate, ar = decoding.SpecDecoder.generate, ar_baseline.ar_generate
        def ready_generate(decoder, prompt):
            self.admission_wait(system.assets_dir, initial=False)
            return generate(decoder, prompt)
        def ready_ar(*args, **kwargs):
            self.admission_wait(system.assets_dir, initial=False)
            return ar(*args, **kwargs)
        with patch.object(decoding.SpecDecoder, 'generate', ready_generate), patch.object(ar_baseline, 'ar_generate', ready_ar):
            result = super().run(data, model, system, tracker)
        check_control_refs(self.study_root, self.execution_root, kind="paired")
        return result



class ContinuationEval(NormEval):
    """The unchanged norm protocol with its own source chain and paired resource observer."""
    control_kind = "continuation"
