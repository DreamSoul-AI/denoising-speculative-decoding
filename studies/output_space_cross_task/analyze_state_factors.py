"""Structured q241 path/state factor comparison without model forwards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, margin


def trace_comparison(left, right):
    a, b = left['trace'], right['trace']
    by_a = {t['root_position']: t for t in a}
    by_b = {t['root_position']: t for t in b}
    assert len(a) == len(by_a) and len(b) == len(by_b)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    changed = [p for p in sorted(by_a.keys() & by_b.keys()) if by_a[p] != by_b[p]]
    return {'full_semantic_equal': a == b, 'equal_accepted_totals': left['stats']['accepted'] == right['stats']['accepted'],
        'first_chronological_divergence': None if a == b else {
            'index': first, 'left': a[first] if first < len(a) else None,
            'right': b[first] if first < len(b) else None},
        'common_changed_roots': changed, 'left_only_roots': sorted(by_a.keys() - by_b.keys()),
        'right_only_roots': sorted(by_b.keys() - by_a.keys()),
        'common_root_gains': [p for p in changed if len(by_b[p]['accepted_nodes']) > len(by_a[p]['accepted_nodes'])],
        'common_root_losses': [p for p in changed if len(by_b[p]['accepted_nodes']) < len(by_a[p]['accepted_nodes'])]}


def interaction(counts):
    assert set(counts) == {'native', 'anchor', 'velocity', 'both'}
    return counts['both'] - counts['anchor'] - counts['velocity'] + counts['native']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--operator', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    study = Path(__file__).resolve().parent
    old = ROOT / '.tmp/q241-canonical-20261014-v1'
    paths = {'analysis': Path(__file__), 'state_report': args.state / 'diagnosis.json',
        'state_tensor': args.state / 'selected_tensors.pt', 'old_report': old / 'diagnosis.json',
        'old_tensor': old / 'selected_tensors.pt', 'operator_report': args.operator / 'diagnosis.json',
        'operator_tensor': args.operator / 'factor_tensors.pt'}
    binding = {k: file_hash(p) for k, p in paths.items()}
    previous, new, op = (read(paths[k]) for k in ('old_report', 'state_report', 'operator_report'))
    assert all(r['quality_passed'] for r in (previous, new, op))
    assert manifest()['digest'] == new['source_digest'] == op['source_digest'] == '670c744e0535dcad'
    for name, r in (('old', previous), ('state', new), ('operator', op)):
        assert binding[name + '_tensor'] == r['raw_tensor_sha256']
    assert file_hash(ROOT / 'tools/diagnose_q241_state_factors.py') == new['source_binding']['tool']
    assert file_hash(study / 'docs/STATE_FACTOR_PLAN.md') == new['source_binding']['plan']
    assert file_hash(ROOT / 'tools/diagnose_q178_rounding_factors.py') == op['source_binding']['tool']
    assert file_hash(ROOT / 'studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md') == op['source_binding']['plan']
    a = torch.load(paths['old_tensor'], weights_only=True, map_location='cpu', mmap=True)
    b = torch.load(paths['state_tensor'], weights_only=True, map_location='cpu', mmap=True)
    cases = {c['label']: c for c in previous['cases']}
    for c in new['cases']:
        if c['coordinate'] == 'native':
            assert c['replay'] == cases[c['label']]['replay']
        else:
            cases[c['label']] = c
            assert all(r['other_state_bit_equal'] and r['incoming_proxies_unchanged'] and r['selected_state_bit_equal']
                       for r in c['canonical_replacements'])
            for p in c['replay']['visited_targets']:
                if p > 716:
                    assert torch.equal(b[f'{c["label"]}:{p}']['current_state'][c['coordinate']],
                                       a['canonical_states'][p][c['coordinate']])
    native = read(study / 'docs/execution_v1.json')
    ar = next(r for r in native['runs'] if r['case'] == 'AR')['raw']['evidence'][0]['output_ids']
    eos = read(study / 'runs' / next(r['run_id'] for r in native['runs'] if r['case'] == 'AR')
               / 'assets/decode_inputs.json')['eos_ids']
    comparisons = []
    pairs = [('native_self', 'canonical_self'), ('native_graft', 'canonical_graft'),
             ('canonical_self', 'canonical_graft'), ('anchor_self', 'anchor_graft'),
             ('velocity_self', 'velocity_graft')]
    for left, right in pairs:
        detail = trace_comparison(cases[left]['replay'], cases[right]['replay'])
        windows = []
        for p in detail['common_changed_roots']:
            keys = [f'{label}:{p}' for label in (left, right)]
            source = [a if label.startswith(('native', 'canonical')) else b for label in (left, right)]
            if not all(k in data for k, data in zip(keys, source)):
                windows.append({'root': p, 'status': 'not_captured_in_original_fixed_targets'})
                continue
            x, y = (data[k] for k, data in zip(keys, source))
            state = {k: {'bit_equal': torch.equal(x['current_state'][k], y['current_state'][k]),
                'max_abs': float((x['current_state'][k] - y['current_state'][k]).abs().max())}
                for k in ('anchor', 'velocity', 'proxies')}
            ranks = []
            for s in (x, y):
                first = margin(s['seed_logits'][0], ar[p + 1])
                valid = first['top1_id'] == ar[p + 1] and ar[p + 1] not in eos and p + 2 < len(ar)
                ranks.append({'source_root': s['source_root'], 'owner': s['source_owner'],
                    'first': first, 'second': margin(s['seed_logits'][1], ar[p + 2]) if p + 2 < len(ar) else None,
                    'legal_second_AR_path': valid})
            windows.append({'root': p, 'states': state, 'slots': ranks,
                'seed_logits_max_abs': float((x['seed_logits'] - y['seed_logits']).abs().max()),
                'seed_mask_inputs_max_abs': float((x['seed_mask_inputs'] - y['seed_mask_inputs']).abs().max())})
        comparisons.append({'left': left, 'right': right, **detail, 'captured_changed_windows': windows})
    table = []
    for suffix in ('self', 'graft'):
        counts = {mode: cases[f'{label}_{suffix}']['replay']['stats']['accepted']
                  for mode, label in [('native', 'native'), ('anchor', 'anchor'), ('velocity', 'velocity'), ('both', 'canonical')]}
        table.append({'donor': suffix, 'accepted': counts, 'additivity_interaction': interaction(counts)})
    result = {'source_binding': binding, 'quality_passed': True, 'factor_table': table,
        'H5_local_count_additivity': 'falsified' if any(t['additivity_interaction'] for t in table) else 'not_falsified',
        'H6_equal_count_implies_equal_paths': 'falsified' if any(c['equal_accepted_totals'] and not c['full_semantic_equal'] for c in comparisons) else 'not_falsified',
        'comparisons': comparisons, 'new_tokens': sum(c['replay']['stats']['new_tokens'] for c in new['cases']),
        'new_generation_forwards': sum(c['replay']['observer_forward_calls'] for c in new['cases']),
        'teacher_forwards': 0, 'operator_model_forwards': 0,
        'attribution_boundary': 'Selective state synchronization is a controlled intervention, not a unique causal percentage for native microdifferences.'}
    def save(path, body):
        text = json.dumps(body, indent=2, allow_nan=False) + '\n'
        if path.exists():
            assert path.read_text(encoding='utf-8') == text
        else:
            path.write_text(text, encoding='utf-8')
    save(study / 'docs/state_factor_analysis_v1.json', result)
    save(study / 'docs/state_factor_diagnosis_v1.json', new)
    save(ROOT / 'studies/output_space_extension/docs/rounding_factor_diagnosis_v1.json', op)
    print(json.dumps({k: result[k] for k in ('factor_table', 'H5_local_count_additivity',
                       'H6_equal_count_implies_equal_paths', 'new_tokens', 'new_generation_forwards')}))


if __name__ == '__main__':
    main()
