"""Paired analysis of retained E1 observations, without model calls."""

import random
import csv
import json
from pathlib import Path

from spec_mtp.audit import audit_run
from studies.trajectory_observation.structure.input_state import pooled, file_hash, write_json


def paired_comparison(rows, left, right, *, resamples=4000, seed=20261128):
    groups = {}
    for arm in (left, right):
        selected = [row for row in rows if row['split'] == 'eval' and row['arm'] == arm]
        group = {row['qid']: row for row in selected}
        if len(group) != len(selected) or not group:
            raise ValueError('missing or duplicate evaluation pairs')
        groups[arm] = group
    if groups[left].keys() != groups[right].keys():
        raise ValueError('evaluation pairs differ')
    keys = sorted(groups[left])
    a, b = [groups[left][k] for k in keys], [groups[right][k] for k in keys]
    def delta(x, y):
        lx, ry = pooled(x), pooled(y)
        if lx['tau'] is None or ry['tau'] is None:
            raise ValueError('zero-call evaluation pair')
        return lx['tau'] - ry['tau']
    generator = random.Random(seed)
    samples = []
    for _ in range(resamples):
        indices = [generator.randrange(len(keys)) for _ in keys]
        samples.append(delta([a[i] for i in indices], [b[i] for i in indices]))
    samples.sort()
    return {'left': left, 'right': right, 'prompts': len(keys), 'tau_delta': delta(a, b),
            'exploratory_paired_ci95': [samples[int(.025 * resamples)], samples[int(.975 * resamples)]],
            'equal_output_prompts': sum(x['tokens'] == y['tokens'] for x, y in zip(a, b)),
            'calls_lower_equal_higher': [sum(x['C'] < y['C'] for x, y in zip(a, b)),
                                        sum(x['C'] == y['C'] for x, y in zip(a, b)),
                                        sum(x['C'] > y['C'] for x, y in zip(a, b))],
            'equal_length_prompts': sum(x['N'] == y['N'] for x, y in zip(a, b))}


def analyze_run(run_root):
    """Reaudit actual raw observations, then export paired counts and tables."""
    root = Path(run_root)
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('complete') is not True:
        raise ValueError('incomplete E1 experiment')
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    model_manifest = json.loads(Path(config['model_manifest']).read_text(encoding='utf-8'))
    sources = {'code': file_hash(config['source_manifest']), 'inputs': file_hash(config['inputs']),
               'model': file_hash(config['model_manifest']), 'config': file_hash(root / 'config.json')}
    rows = [json.loads(line) for line in (root / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    keys = [(r['split'], r['qid'], r['arm']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate experiment row')
    for row in rows:
        raw = root / row['raw']
        verdict = audit_run(raw, sources, expected_model_sources=model_manifest)
        if not (verdict.source_integrity and verdict.argmax_coverage and verdict.argmax_correctness):
            raise ValueError('raw reaudit failed')
        end = json.loads(raw.read_text(encoding='utf-8').splitlines()[-1])
        actual = end['stats']['acceptance_metrics']
        if any(row[k] != actual[k] for k in ('A', 'C', 'S')) or row['tokens'] != end['committed']:
            raise ValueError('row metrics differ from raw committed evidence')
        if row['N'] != len(end['committed']):
            raise ValueError('row output length differs from evidence')
    selected = summary['selected']
    inputs = json.loads((root / 'inputs.json').read_text(encoding='utf-8'))
    evaluation = [r for r in rows if r['split'] == 'eval']
    for arm in selected:
        actual = {r['qid'] for r in evaluation if r['arm'] == arm}
        if actual != {r['qid'] for r in inputs['eval']}:
            raise ValueError('incomplete evaluation arm')
    totals = {a: pooled([r for r in evaluation if r['arm'] == a]) for a in selected}
    kalman = next(a for a in selected if a.startswith('kalman_'))
    holt = next(a for a in selected if a.startswith('holt_'))
    comparisons = [paired_comparison(rows, kalman, holt)]
    comparisons.extend(paired_comparison(rows, a, 'order0') for a in selected if a != 'order0')
    comparisons.append(paired_comparison(rows, kalman, 'fixed_gain'))
    categories = {c: {a: pooled([r for r in evaluation if r['arm'] == a and r['category'] == c])
                     for a in selected} for c in sorted({r['category'] for r in evaluation})}
    report = {'totals': totals, 'categories': categories, 'comparisons': comparisons,
              'audited_rows': len(rows), 'audited_tokens': sum(r['N'] for r in rows),
              'dev': {a: pooled([r for r in rows if r['split'] == 'dev' and r['arm'] == a])
                      for a in dict.fromkeys(r['arm'] for r in rows if r['split'] == 'dev')}}
    write_json(root / 'analysis.json', report)
    columns = ['arm', 'prompts', 'N', 'A', 'C', 'S', 'accepted_per_call', 'tau']
    with (root / 'results.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({'arm': a, **totals[a]} for a in selected)
    lines = ['| arm | n | output N | draft A | calls C | decode S | A/C | τ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in selected:
        p = totals[a]
        lines.append(f'| {a} | {p["prompts"]} | {p["N"]} | {p["A"]} | {p["C"]} | {p["S"]} | '
                     f'{p["accepted_per_call"]:.4f} | {p["tau"]:.4f} |')
    with (root / 'results.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return report
