"""Recompute Table1 gaps from existing public evidence; never loads a model.

    python tools/analyze_esp_table1_gap.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MT_CATEGORIES = {'coding', 'extraction', 'humanities', 'math', 'reasoning', 'roleplay', 'stem', 'writing'}
FILES = {
    'eight_b': 'studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json',
    'five_methods': 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json',
    'directions': 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json',
}


def summarize(rows, target=None):
    assert rows and len({r['question_id'] for r in rows}) == len(rows)
    totals = {k: sum(r[k] for r in rows) for k in ('committed', 'decode_calls', 'accepted', 'new_tokens')}
    committed, calls, accepted, new = (totals[k] for k in ('committed', 'decode_calls', 'accepted', 'new_tokens'))
    assert calls > 0 and new == committed + len(rows)
    assert all(r['decode_calls'] > 0 and r['accepted'] + r['decode_calls'] >= r['committed'] for r in rows)
    clipped = accepted + calls - committed
    result = dict(n=len(rows), **totals, pooled_tau=committed / calls,
        prompt_mean_tau=sum(r['committed'] / r['decode_calls'] for r in rows) / len(rows),
        accepted_per_decode_call=accepted / calls, end_clipped_tokens=clipped,
        unclipped_accounting_tau=(accepted + calls) / calls,
        clipping_tau_delta=clipped / calls,
        including_prefill_tau=new / (calls + len(rows)),
        output_cap_256_count=sum(r['new_tokens'] == 256 for r in rows),
        shorter_than_256_count=sum(r['new_tokens'] < 256 for r in rows),
        output_tokens_mean=new / len(rows),
        argmax_checked_tokens=sum(r['argmax_checked_tokens'] for r in rows))
    assert result['argmax_checked_tokens'] == new
    if target is not None:
        result.update(target=target, pooled_delta_to_target=committed / calls - target,
            relative_shortfall=(target - committed / calls) / target,
            extra_committed_at_same_calls=target * calls - committed)
    return result


def paired(first, second):
    assert [r['question_id'] for r in first] == [r['question_id'] for r in second]
    result = {key: dict(win=0, tie=0, loss=0) for key in ('accepted', 'calls', 'tau')}
    for a, b in zip(first, second):
        changes = {'accepted': b['accepted'] - a['accepted'], 'calls': a['model_calls'] - b['model_calls'],
                   'tau': b['committed'] * a['decode_calls'] - a['committed'] * b['decode_calls']}
        for key, delta in changes.items():
            result[key]['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    result['delta_second_minus_first'] = {k: sum(r[k] for r in second) - sum(r[k] for r in first)
                                         for k in ('accepted', 'model_calls', 'new_tokens')}
    return result


def group_summary(rows, key, target=None):
    groups = {value: [r for r in rows if key(r) == value] for value in sorted({key(r) for r in rows})}
    result = {value: summarize(rs, target) for value, rs in groups.items()}
    if target is not None:
        calls = sum(r['decode_calls'] for r in rows)
        for entry in result.values():
            entry['decode_call_weight'] = entry['decode_calls'] / calls
            entry['uniform_target_gap_contribution'] = entry['extra_committed_at_same_calls'] / calls
    return result


def main():
    blobs = {k: (ROOT / path).read_bytes() for k, path in FILES.items()}
    data = {k: json.loads(blob) for k, blob in blobs.items()}
    assert data['directions']['complete'] and data['eight_b']['native_finalisation_complete']
    ids = [r['question_id'] for r in data['eight_b']['cells'][0]['rows']]
    assert len(ids) == 480
    cells = {}
    for cell in data['eight_b']['cells']:
        assert cell['native_status'] == 'succeeded' and cell['argmax_all_passed']
        rows = cell['rows']
        assert [r['question_id'] for r in rows] == ids
        target = {30: 1.63, 60: 1.74}.get(cell['BC'])
        overall = summarize(rows, target)
        assert abs(overall['pooled_tau'] - cell['pooled_tau']) < 1e-12
        categories = group_summary(rows, lambda r: r['category'], target)
        assert sorted(e['n'] for e in categories.values()) == [10] * 8 + [80] * 5
        families = group_summary(rows, lambda r: 'mt_bench' if r['category'] in MT_CATEGORIES else r['category'], target)
        assert len(families) == 6 and all(v['n'] == 80 for v in families.values())
        overall['category13_mean_pooled_tau'] = sum(e['pooled_tau'] for e in categories.values()) / 13
        overall['source_family6_mean_pooled_tau_diagnostic'] = sum(e['pooled_tau'] for e in families.values()) / 6
        assert abs(overall['category13_mean_pooled_tau'] - cell['category_mean_pooled_tau']) < 1e-12
        for r in rows:
            r['model_calls'] = r['decode_calls'] + 1
        cells[f"{cell['BC']}/{cell['num_masks']}"] = dict(run_id=cell['run_id'], actual_spec=cell['actual_spec'],
            overall=overall, categories=categories, source_families_diagnostic=families)
    pairs = {}
    for first, second in [('10/1', '30/1'), ('30/1', '60/1'), ('60/2', '60/1')]:
        get = lambda name: next(c['rows'] for c in data['eight_b']['cells'] if f"{c['BC']}/{c['num_masks']}" == name)
        pairs[first + ' -> ' + second] = paired(get(first), get(second))
    old = data['five_methods']['cells']
    for budget in (30, 60):
        first, second = (old[f'{budget}/{name}']['rows'] for name in ('frozen_mean', 'input_zero'))
        assert [r['question_id'] for r in first] == ids
        pairs[f'4B {budget} frozen_mean -> input_zero'] = paired(first, second)
    direction = {}
    for name, cell in data['directions']['cells'].items():
        assert [r['question_id'] for r in cell['rows']] == ids
        direction[name] = summarize(cell['rows'])
    body = dict(scope='CPU readout only; original480 retained; no model, new prompt, GPU or timing',
        sources={k: dict(path=FILES[k], sha256=hashlib.sha256(blob).hexdigest()) for k, blob in blobs.items()},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_family6_is_diagnostic_not_confirmed_paper_weighting=True,
        uniform_target_contributions_are_accounting_not_paper_per_task_errors=True,
        eight_b=cells, paired=pairs, four_b_direction_context=direction)
    path = ROOT / 'docs/research/esp-table1-gap-analysis-2026-10-16.json'
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'Never overwrite changed evidence'
    else:
        path.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({k: c['overall'] for k, c in cells.items()}, indent=2))


if __name__ == '__main__':
    main()
