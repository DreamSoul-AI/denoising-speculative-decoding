"""Scoped ESP backend and model-local shape controls; no import-time registration."""
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
import json
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from spec_mtp.structure.algorithm.factory import AlgorithmRegistry
from studies._shared.structure.algorithm import UntimedEval
import spec_mtp.structure.algorithm.decode.loop as decoding


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


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def bind_committed(prompt_ids, output_ids, stats, selections):
    """Check forward/row ownership and EOS/budget-trimmed commitment coverage."""
    assert len(selections) == 1 + sum(len(s['accepted_nodes']) + 1 for s in stats['trace'])
    assert selections[0]['forward'] == 0 and selections[0]['row'] == len(prompt_ids) - 1
    committed = [selections[0]]
    offset = 1
    for step, trace in enumerate(stats['trace'], 1):
        count = len(trace['accepted_nodes']) + 1
        block = selections[offset:offset + count]
        assert [s['forward'] for s in block] == [step] * count
        assert [s['row'] for s in block] == [0] + trace['accepted_nodes']
        assert trace['committed_tokens'] == [s['chosen'] for s in block[:len(trace['committed_tokens'])]]
        committed.extend(block[:len(trace['committed_tokens'])])
        offset += count
    assert output_ids[:len(prompt_ids)] == prompt_ids
    assert output_ids[len(prompt_ids):] == [s['chosen'] for s in committed]
    assert len(committed) == stats['new_tokens'] == 1 + stats['committed']
    assert all(s['chosen'] == s['argmax'] for s in selections)
    return {'passed': True, 'checked_committed_tokens': len(committed),
            'selected_tokens_including_trimmed_tail': len(selections),
            'input_sha256': token_digest(prompt_ids), 'output_sha256': token_digest(output_ids),
            'committed': committed}


@contextmanager
def observe_argmax(assets):
    original_generate = decoding.SpecDecoder.generate
    original_select = decoding.select_token
    audit_path = assets / 'argmax_audit.jsonl'
    assert not audit_path.exists(), 'never append a new execution to old audit evidence'
    with audit_path.open('x', encoding='utf-8') as stream:
        def generate(decoder, prompt_ids):
            assert decoder.temperature == 0 and decoder.model.dtype.__str__() == 'torch.bfloat16'
            latest = [None]
            forward = [-1]
            selections: list[dict[str, int]] = []
            def capture(_module, _args, output):
                latest[0] = output.logits[0]
                forward[0] += 1
            def select(row, temperature, generator):
                assert temperature == 0
                matrix = latest[0]
                assert matrix is not None and row.ndim == 1 and row.numel() == matrix.shape[1]
                offset = (row.data_ptr() - matrix.data_ptr()) // row.element_size()
                assert offset >= 0 and offset % matrix.stride(0) == 0
                index = offset // matrix.stride(0)
                assert index < matrix.shape[0] and row.stride(0) == matrix.stride(1)
                chosen = original_select(row, temperature, generator)
                expected = int(row.argmax(dim=-1).item())
                assert chosen == expected, 'verification choice differs from its forward argmax'
                selections.append({'forward': forward[0], 'row': index, 'chosen': chosen, 'argmax': expected})
                return chosen
            handle = decoder.model.register_forward_hook(capture)
            try:
                with patch.object(decoding, 'select_token', select):
                    output, stats = original_generate(decoder, prompt_ids)
                assert forward[0] == stats['decode_calls']
                audit = bind_committed(prompt_ids[0].tolist(), output[0].tolist(), stats, selections)
                stats['verification_argmax_audit'] = audit
                stream.write(json.dumps(audit) + '\n')
                stream.flush()
                return output, stats
            finally:
                primary = sys.exc_info()[1]
                latest[0] = None
                try:
                    handle.remove()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    setattr(primary, "cleanup_errors", (*getattr(primary, "cleanup_errors", ()), cleanup))
        with patch.object(decoding.SpecDecoder, 'generate', generate):
            yield


class NumericPolicyEval(UntimedEval):
    """Record the original BF16 backend policy while retaining untimed native decoding."""
    study_root: Path
    execution_root: Path
    audit_argmax = False

    def check_source_binding(self):
        from studies._shared.structure.validation import source_study_root
        if source_study_root(self.study_root).parents[1] != self.execution_root:
            raise ValueError('ESP source binding changed execution root')

    def run(self, data, model, system, tracker):
        self.check_source_binding()
        assert self.config.setting('eval', {}).get('warmup') is False
        assert model.module.dtype == torch.bfloat16
        controls = {'dtype': str(model.module.dtype),
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'deterministic': torch.are_deterministic_algorithms_enabled()}
        (Path(system.assets_dir)/'numeric_policy.json').write_text(json.dumps(controls, indent=2)+'\n', encoding='utf-8')
        context = observe_argmax(Path(system.assets_dir)) if self.audit_argmax else nullcontext()
        with context:
            result = super().run(data, model, system, tracker)
        self.check_source_binding()
        return result


class ArgmaxEval(NumericPolicyEval):
    audit_argmax = True


class RevalidationEval(ArgmaxEval):
    study_root: Path
    execution_root: Path
    baseline_root: Path
    def run(self, data, model, system, tracker):
        from studies._shared.structure import evidence
        from studies._shared.structure.validation import check_source_refs
        from studies._shared.structure.resource_admission import wait_for_admission
        check_source_refs(self.study_root, self.execution_root)
        (Path(system.assets_dir) / 'resource_admission_initial.json').write_text(
            json.dumps(model.meta['initial_resource_admission'], indent=2) + '\n', encoding='utf-8')
        original_capture, original_generate = evidence.capture, decoding.SpecDecoder.generate
        baseline = self.baseline_root/'docs'
        expected = json.loads((baseline / 'prompts_manifest.json').read_text(encoding='utf-8'))['rendered_prompts']
        formal = json.loads((baseline / 'protocol_evidence_v1.json').read_text(encoding='utf-8'))
        count = [0]

        def capture(*args, **kwargs):
            saved = original_capture(*args, **kwargs)
            assert saved['dtype'] == 'torch.bfloat16' and saved['backend'] == 'eager'
            assert saved['model_digest'] == formal['cells'][0]['model_digest']
            assert len(saved['prompts']) == len(expected) == 480
            assert all(a['input_ids'] == b['input_ids'] and a['category'] == b['category']
                       for a, b in zip(saved['prompts'], expected))
            return saved

        def generate(decoder, prompt):
            wait_for_admission(system.assets_dir, initial=False)
            assert prompt[0].tolist() == expected[count[0]]['input_ids']
            output, stats = original_generate(decoder, prompt)
            count[0] += 1
            progress = {'completed': count[0], 'total': 480, 'BC': decoder.nominal_block_complexity,
                        'last_committed': stats['committed'], 'last_decode_calls': stats['decode_calls']}
            (Path(system.assets_dir) / 'revalidation_progress.json').write_text(json.dumps(progress) + '\n')
            if count[0] % 10 == 0:
                print('Nonofficial Table1', progress, flush=True)
            return output, stats

        with patch.object(evidence, 'capture', capture), patch.object(decoding.SpecDecoder, 'generate', generate):
            result = super().run(data, model, system, tracker)
        assert count[0] == 480
        check_source_refs(self.study_root, self.execution_root)
        return result
