"""Audit the unchanged third-party ESP API under frozen Table1 inputs, without timing.

    python tools/audit_esp_third_party.py --study studies/esp_table1_bf16_full_argmax --model-cache D:/models/Qwen3-8B --out .tmp/esp-third-party-v1
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts'


def load_original():
    from analyze_4b_geometry import sha
    origin = json.loads((VENDOR / 'ORIGIN.json').read_text(encoding='utf-8'))
    assert origin['commit'] == 'ad76f8da29f3bfc7faa20488193c6aa83a9f708c' and not origin['author_official']
    for entry in origin['files']:
        assert sha(VENDOR / 'upstream' / entry['path']) == entry['sha256']
    spec = importlib.util.spec_from_file_location('esp_original_v6', VENDOR / 'upstream/code/esp_v6.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, origin


def audit_step(logits, prompt_length, draft_ids, result, remaining):
    """Check the positions preceding the emitted prefix, rather than draft-self/mask rows."""
    n_acc, acc_idx, bonus, _ = result
    assert prompt_length > 0 and logits.ndim == 2 and remaining > 0
    assert logits.shape[0] == prompt_length + len(draft_ids) + 1
    assert n_acc >= 0 and (n_acc == 0 or 0 <= acc_idx < len(draft_ids))
    proposed = (draft_ids[:acc_idx + 1] if n_acc else []) + [bonus]
    records = []
    for offset, token in enumerate(proposed[:remaining]):
        row = prompt_length - 1 + offset
        expected = int(logits[row].argmax())
        records.append({'offset': offset, 'verifier_row': row, 'chosen': int(token), 'argmax': expected,
            'passed': int(token) == expected,
            'argmax_minus_chosen_logit': float(logits[row, expected] - logits[row, int(token)])})
    return records


class ArgmaxRejected(Exception):
    pass


@torch.no_grad()
def main():
    from analyze_4b_geometry import sha
    from diagnose_4b_mask_windows import admission
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model-cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    report = {'status': 'preflight', 'author_official': False, 'no_timing': True, 'rows': [],
        'tool_sha256': sha(__file__), 'plan_sha256': sha((Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md')),
        'settings': {'model': 'Qwen/Qwen3-8B', 'dtype': 'bfloat16', 'temperature': 0,
                     'max_new_tokens': 256, 'block_complexities': [10, 30, 60], 'num_masks': 1,
                     'seed': 123, 'pilot_question_id': 121, 'forward_cap_per_bc': 24},
        'full_480_executed': False, 'quality_passed': False, 'admission': []}
    private = {}
    def save():
        (args.out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        original, origin = load_original()
        report.update(upstream_commit=origin['commit'], upstream_url=origin['source_url'],
            origin_sha256=sha(VENDOR / 'ORIGIN.json'), original_files=origin['files'])
        manifest_path = args.study / 'docs/prompts_manifest.json'
        prompts = json.loads(manifest_path.read_text())
        assert prompts['full_count'] == len(prompts['rendered_prompts']) == 480
        first = prompts['rendered_prompts'][0]
        assert first['question_id'] == 121
        data_path = ROOT / 'datasets/spec_bench/question.jsonl'
        assert sha(data_path) == prompts['dataset_sha256']
        data = {r['question_id']: r for r in map(json.loads, data_path.read_text(encoding='utf-8').splitlines())}
        snapshot_path = args.study / 'docs/modelscope_snapshot.json'
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot['model_id'] == 'Qwen/Qwen3-8B'
        for entry in snapshot['files']:
            assert sha(args.model_cache / entry['path']) == entry['sha256']
        tokenizer = AutoTokenizer.from_pretrained(args.model_cache, local_files_only=True)
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': data[121]['turns'][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert tokenizer.encode(rendered) == first['input_ids']
        report.update(prompts_manifest_sha256=sha(manifest_path), dataset_sha256=sha(data_path),
            model_snapshot_sha256=sha(snapshot_path), frozen_prompt_tokens=len(first['input_ids']))
        check = admission()
        check['required_free_mib'] = 20 * 1024
        check['passed'] = not check['foreign_gpu_python_pids'] and check['free_mib'] >= check['required_free_mib']
        report['admission'].append(check)
        if not check['passed']:
            report['status'] = 'deferred_nonpreemptive_admission'
            save()
            return
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = AutoModelForCausalLM.from_pretrained(args.model_cache, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        assert next(model.parameters()).is_cuda and all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point())
        report['runtime'] = {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                             'model_dtype': str(next(model.parameters()).dtype)}
        for bc in (10, 30, 60):
            check = admission(initial=False)
            report['admission'].append(check)
            if not check['passed']:
                report['status'] = 'deferred_between_budgets'
                break
            row = {'bc': bc, 'question_id': 121, 'status': 'running', 'completed_model_calls': 0,
                   'attempted_model_calls': 0, 'argmax_checked': 0, 'valid_tau': None, 'forward_trace': []}
            decisions, phase = [], ['mask_proposal']
            decoder = original.ESPDecoder(model, tokenizer, model.device)
            verify = decoder.verify_single_pass
            def before(_module, _args, _kwargs):
                row['attempted_model_calls'] += 1
                assert row['completed_model_calls'] < 24, 'Bounded preflight budget exhausted'
            def after(_module, _args, kwargs, _output):
                row['completed_model_calls'] += 1
                row['forward_trace'].append({'kind': phase[0], 'input_length': int(kwargs['inputs_embeds'].shape[1])})
            def audited_verify(prompt_ids, draft_ids, mask_embed):
                captured = []
                handle = model.register_forward_hook(lambda _m, _a, output: captured.append(output.logits[0].detach()))
                phase[0] = 'verification'
                try:
                    result = verify(prompt_ids, draft_ids, mask_embed)
                finally:
                    handle.remove()
                    phase[0] = 'mask_proposal'
                assert len(captured) == 1
                records = audit_step(captured[0], prompt_ids.shape[1], draft_ids, result, 256 - len(decisions))
                for record in records:
                    record['output_index'] = len(decisions)
                    decisions.append(record)
                    row['argmax_checked'] += 1
                    if not record['passed']:
                        row['first_rejection'] = {k: v for k, v in record.items() if k not in ('chosen', 'argmax')}
                        row['first_rejection']['chosen_sha256'] = hashlib.sha256(str(record['chosen']).encode()).hexdigest()
                        row['first_rejection']['argmax_sha256'] = hashlib.sha256(str(record['argmax']).encode()).hexdigest()
                        raise ArgmaxRejected('Proposed commit differs from the verification prefix-parent argmax')
                return result
            handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
            try:
                with patch.object(decoder, 'verify_single_pass', side_effect=audited_verify), \
                        patch.object(original, 'time', SimpleNamespace(time=lambda: 0.)), redirect_stdout(io.StringIO()):
                    result = decoder.generate_esp(rendered, block_complexity=bc, n_mask_tokens=1, max_new_tokens=256)
                row.update(status='pilot_completed', returned_tokens=len(result[0]), upstream_declared_tau=float(result[3]))
            except ArgmaxRejected as error:
                row.update(status='verifier_argmax_failed', error=str(error), stopped_before_bad_commit=True)
            except Exception as error:
                row.update(status='failed_preserved', error=repr(error))
            finally:
                for handle in handles:
                    handle.remove()
            report['rows'].append(row)
            private[str(bc)] = decisions
            save()
            print('BC', bc, row['status'], 'completed forwards', row['completed_model_calls'], flush=True)
        (args.out / 'private_decisions.json').write_text(json.dumps(private, indent=2) + '\n', encoding='utf-8')
        report['private_decisions_sha256'] = sha(args.out / 'private_decisions.json')
        load_original()
        if report['status'] != 'deferred_between_budgets':
            report['status'] = 'preflight_finished_full480_not_run'
    except Exception as error:
        report.update(status='failed_preserved', error=repr(error))
        save()
        raise
    save()


if __name__ == '__main__':
    main()
