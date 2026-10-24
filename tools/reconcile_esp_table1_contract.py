"""Reconcile fixed Table1 inputs and observed-path accounting on CPU only.

    python tools/reconcile_esp_table1_contract.py --tokenizer D:/models/Qwen3-8B
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from studies._shared.structure.evidence import digest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/esp_table1_bf16_full_argmax'


def accounting(rows):
    """Exact weighting identities; contributions are accounting, not causes."""
    n = len(rows)
    calls = sum(r['decode_calls'] for r in rows)
    assert n and calls > 0 and all(r['decode_calls'] > 0 for r in rows)
    ratios = [r['committed'] / r['decode_calls'] for r in rows]
    pooled = sum(r['committed'] for r in rows) / calls
    mean = sum(ratios) / n
    covariance_term = sum((r['decode_calls'] - calls / n) * (t - mean)
                          for r, t in zip(rows, ratios)) / calls
    within = between = category_mean = 0.0
    categories = sorted({r['category'] for r in rows})
    for category in categories:
        group = [r for r in rows if r['category'] == category]
        count = sum(r['decode_calls'] for r in group)
        tau = sum(r['committed'] for r in group) / count
        prompt_mean = sum(r['committed'] / r['decode_calls'] for r in group) / len(group)
        within += len(group) / n * (tau - prompt_mean)
        between += (count / calls - len(group) / n) * tau
        category_mean += tau / len(categories)
    assert math.isclose(pooled - mean, covariance_term, abs_tol=1e-12)
    assert math.isclose(pooled - mean, within + between, abs_tol=1e-12)
    return dict(pooled_tau=pooled, prompt_mean_tau=mean, category_mean_tau=category_mean,
                pooled_minus_prompt_mean=pooled - mean, covariance_over_mean_calls=covariance_term,
                within_category_call_weight_term=within, between_category_call_weight_term=between)


def prefix_accounting(block_sizes, limit=100):
    """Clip the observed path after its prefill token; not a new generation."""
    assert limit >= 1 and all(size > 0 for size in block_sizes)
    remaining = limit - 1
    committed = calls = partial = 0
    for size in block_sizes:
        if not remaining:
            break
        take = min(size, remaining)
        committed += take
        remaining -= take
        calls += 1
        partial += take < size
    return dict(committed=committed, decode_calls=calls, new_tokens=committed + 1,
                boundary_partial_blocks=partial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='Existing local tokenizer directory; no download or weights')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch  # Windows must load torch before transformers.
    from spec_mtp.data import load_tokenizer
    from spec_mtp.data import encode, load_records

    assert torch.empty(0).device.type == 'cpu'
    source_paths = ['studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json',
                    'docs/research/esp-table1-nonofficial-full480-recheck-2026-10-16.json']
    blobs = {name: (ROOT / name).read_bytes() for name in source_paths}
    formal, audit = (json.loads(blobs[name]) for name in source_paths)
    sources = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    for name in ('src/spec_mtp/structure/data/specbench.py', 'src/spec_mtp/structure/data/encode.py',
                 'src/spec_mtp/structure/data/datasets.py'):
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True)
    baseline = None
    cells = {}
    for cell, checked in zip(formal['cells'], audit['cells'], strict=True):
        assert cell['run_id'] == checked['run_id'] and cell['argmax_all_passed']
        for name, expected in checked['files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        run = STUDY / 'runs' / cell['run_id'] / 'assets'
        inputs = json.loads((run / 'decode_inputs.json').read_text(encoding='utf-8'))
        raw = json.loads((run / 'decode_observations.json').read_text(encoding='utf-8'))
        assert raw['inputs_digest'] == digest(inputs)
        stored = inputs['tokenizer']
        assert tokenizer.get_vocab() == stored['vocab'] and tokenizer.chat_template == stored['chat_template']
        assert tokenizer.__class__.__name__ == stored['class']
        assert all(getattr(tokenizer, name) == value for name, value in stored['special_tokens'].items())
        assert inputs['eos_ids'] == [tokenizer.eos_token_id]
        dataset = ROOT / 'datasets/spec_bench/question.jsonl'
        dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert hashlib.sha256((STUDY / 'shared/data/specbench/question.jsonl').read_bytes()).hexdigest() == dataset_sha
        bindings = [sha for name, sha in inputs['files'].items() if name.replace('\\', '/').endswith('/question.jsonl')]
        assert len(bindings) == 2 and all(sha == dataset_sha for sha in bindings)
        config = dict(inputs['settings']['data'], path=str(dataset))
        records = load_records(config)
        rendered = encode(records, tokenizer, config)
        ids = [r['question_id'] for r in records]
        token_lists = [p['input_ids'].tolist()[0] for p in rendered]
        assert len(records) == len(rendered) == len(cell['rows']) == 480
        assert ids == [r['question_id'] for r in cell['rows']]
        assert token_lists == [p['input_ids'] for p in inputs['prompts']]
        assert [r['category'] for r in records] == [r['category'] for r in cell['rows']] == [
            p['category'] for p in inputs['prompts']]
        contract = dict(dataset_sha256=dataset_sha, question_ids=ids,
                        token_ids_sha256=digest(token_lists), vocab_sha256=digest(stored['vocab']),
                        chat_template_sha256=hashlib.sha256(stored['chat_template'].encode()).hexdigest(),
                        data_settings={k: v for k, v in config.items() if k not in ('path', 'source_path')},
                        tokenizer_class=stored['class'], tokenizer_commit=stored['commit'],
                        eos_ids=inputs['eos_ids'], eval_settings=inputs['settings']['eval'])
        if baseline is None:
            baseline = contract
            lengths = sorted(map(len, token_lists))
            input_summary = dict(min_tokens=lengths[0], max_tokens=lengths[-1], total_tokens=sum(lengths),
                                 median_tokens=(lengths[239] + lengths[240]) / 2,
                                 questions_with_multiple_source_turns=sum(len(q['turns']) > 1 for q in
                                     (json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip())),
                                 selected_turn_index=0, all480_rerendered_exact=True,
                                 source_records=480, selected_records=480, filtered_records=0)
        assert contract == baseline
        stops = dict(eos_below_cap=0, eos_at_cap=0, cap_without_eos=0, other=0)
        prefix_rows = []
        for row, observation, prompt in zip(cell['rows'], raw['out']['evidence'], token_lists, strict=True):
            output = observation['output_ids']
            assert output[:len(prompt)] == prompt == observation['input_ids']
            generated = output[len(prompt):]
            assert len(generated) == row['new_tokens'] == row['committed'] + 1 and len(generated) <= 256
            eos_positions = [j for j, token in enumerate(generated) if token in inputs['eos_ids']]
            assert not eos_positions or eos_positions == [len(generated) - 1]
            reason = ('eos_at_cap' if len(generated) == 256 else 'eos_below_cap') if eos_positions else (
                'cap_without_eos' if len(generated) == 256 else 'other')
            stops[reason] += 1
            trace = observation['stats']['trace']
            assert [t for step in trace for t in step['committed_tokens']] == generated[1:]
            assert len(trace) == row['decode_calls'] and not row['hit_context_limit']
            prefix = prefix_accounting([len(step['committed_tokens']) for step in trace])
            prefix_rows.append(dict(prefix, category=row['category']))
        assert stops['other'] == 0 and sum(stops.values()) == 480
        result = accounting(cell['rows'])
        assert math.isclose(result['pooled_tau'], cell['pooled_tau'], abs_tol=1e-12)
        result.update(run_id=cell['run_id'], all480_input_contract_equal=True, stop_counts=stops,
                      observed_path_prefix100=dict(accounting(prefix_rows),
                          committed=sum(r['committed'] for r in prefix_rows),
                          decode_calls=sum(r['decode_calls'] for r in prefix_rows),
                          clipped_blocks=sum(r['boundary_partial_blocks'] for r in prefix_rows),
                          ended_before100=sum(r['new_tokens'] < 100 for r in prefix_rows)))
        cells[f"{cell['BC']}/{cell['num_masks']}"] = result
    report = dict(scope='CPU existing-path reconciliation, not author-code reproduction or a 100-token generation',
                  sources=sources, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_contract=baseline, input_summary=input_summary, cells=cells,
                  author_contract_available=False, GPU_forwards=0)
    path = ROOT / 'docs/research/esp-table1-cpu-contract-reconciliation-2026-10-16.json'
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'Never overwrite changed evidence'
    else:
        path.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps(dict(input_summary=input_summary, cells=cells), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
