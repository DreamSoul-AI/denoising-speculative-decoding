"""Verify six new sampled units and separately labelled native tail stress evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, TRACE_KEYS
from diagnose_rare_stress import event_summary
from studies.output_space_cross_task.analyze_sampling_expansion import summarize_unit
from studies.output_space_cross_task.analyze_first_delta import path_accounting


def bounded_sign_prediction(rows):
    by_unit = {(r['qid'], r['seed']): r for r in rows}
    expected = {(242, 126): 1, (244, 126): -1}
    results = []
    for unit, sign in expected.items():
        row = by_unit[unit]
        delta = row.get('accepted_delta')
        results.append({'qid': unit[0], 'seed': unit[1], 'expected_sign': sign,
            'actual_delta': delta, 'verdict': 'infeasible_not_scored' if delta is None else
                ('preserved' if delta * sign > 0 else 'falsified')})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion', required=True, type=Path)
    parser.add_argument('--rare', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    expansion, rare = (read(p / 'diagnosis.json') for p in (args.expansion, args.rare))
    assert expansion['quality_passed'] and rare['quality_passed']
    assert manifest()['digest'] == expansion['source_digest'] == rare['source_digest'] == '670c744e0535dcad'
    assert file_hash(args.expansion / 'selected_tensors.pt') == expansion['raw_tensor_sha256']
    for report, tool in ((expansion, 'diagnose_sampling_continuation.py'), (rare, 'diagnose_rare_stress.py')):
        assert file_hash(ROOT / 'tools' / tool) == report['source_binding']['tool']
        assert file_hash(study / 'docs/SAMPLING_CONTINUATION_PLAN.md') == report['source_binding']['plan']
    assert file_hash(study / 'docs/sampling_continuation_inputs_v1.json') == expansion['source_binding']['input_manifest']
    assert file_hash(ROOT / 'datasets/spec_bench/question.jsonl') == expansion['source_binding']['dataset']
    expected = [(242,126),(244,126),(245,126),(245,127),(246,126),(246,127)]
    assert [(u['qid'], u['seed']) for u in expansion['units']] == expected
    rows = [summarize_unit(u, args.expansion, False) for u in expansion['units']]
    data = torch.load(args.expansion / 'selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
    semantic = []
    for unit, row in zip(expansion['units'], rows):
        entry = unit['selection']['intervention_root']
        if entry is None:
            continue
        prefix = f'q{unit["qid"]}s{unit["seed"]}'
        original = data[f'{prefix}:baseline:output_order1:{entry}']['current_masks']
        donor = data[f'{prefix}:baseline:output_order0:{entry}']['current_masks'][0,1]
        actual = data[f'{prefix}:graft:{entry}']['current_masks']
        assert torch.equal(actual[0,1], donor) and torch.equal(actual[0,0], original[0,0]) and torch.equal(actual[1:], original[1:])
        cases = {c['label']: c['replay'] for c in unit['cases']}
        assert all(c['only_selected_vector_replaced'] and c['replacement_equal_donor'] and c['modified_provider_calls'] == 1 for c in cases.values())
        row['path_accounting'] = path_accounting(cases['self']['trace'], cases['graft']['trace'], row['self_accepted'], row['graft_accepted'])
        semantic.append({'qid': unit['qid'], 'seed': unit['seed'], 'AR': unit['AR'],
            'native': [{'case': r['case'], 'stats': {k: v for k,v in r['stats'].items() if k != 'trace'},
                        'trace': [{k:t[k] for k in TRACE_KEYS} for t in r['stats']['trace']]} for r in unit['native']],
            'observed': unit['observed'], 'cases': unit['cases']})
    context = torch.load(ROOT / '.tmp/q241-local-factors-20261015-v1/selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
    groups = []
    assert len(rare['groups']) == 3
    for group, (temperature, count) in zip(rare['groups'], ((.8,16384),(1.,16384),(2.,131072))):
        assert group['temperature'] == temperature and group['verifier_draws'] == count
        assert group['seeds'] == list(range(10000,10000+count))
        assert group['RNG_consumed'] and group['same_verifier_draws_for_all_cells']
        assert group['role'] == ('changed_temperature_stress' if temperature == 2. else 'original_temperature_control')
        probs = torch.softmax(context['frame:719']['logits'][0,0].cuda().float()/temperature, dim=-1)
        for cell in group['cells']:
            p = cell['exact_candidate_probability']
            assert float(probs[cell['candidate']]) == p
            expected_cell = event_summary(group['verifier_token_ids'], group['seeds'], cell['candidate'], p)
            assert all(cell[k] == v for k,v in expected_cell.items())
            assert cell['hits'] == cell['first_slot_hits']
        groups.append({k:v for k,v in group.items() if k not in ('seeds','verifier_token_ids')})
        groups[-1]['seed_range_inclusive'] = [10000, 9999+count]
        groups[-1]['distinct_verifier_tokens'] = len(set(group['verifier_token_ids']))
        groups[-1]['draw_bank_sha256'] = __import__('hashlib').sha256(json.dumps(group['verifier_token_ids']).encode()).hexdigest()
    result = {'quality_passed': True, 'source_digest': manifest()['digest'],
        'model_digest': expansion['model_digest'], 'runtime': expansion['runtime'],
        'analysis_sha256': file_hash(Path(__file__)),
        'plan_sha256': file_hash(study / 'docs/SAMPLING_CONTINUATION_PLAN.md'),
        'expansion_report_sha256': file_hash(args.expansion / 'diagnosis.json'),
        'rare_report_sha256': file_hash(args.rare / 'diagnosis.json'),
        'tensor_sha256': expansion['raw_tensor_sha256'],
        'source_binding': {'expansion': expansion['source_binding'], 'rare': rare['source_binding']},
        'rows': rows, 'full_semantic_paths': semantic, 'H8': bounded_sign_prediction(rows),
        'H9_changed_temperature_each_candidate_observed': all(c['hits'] > 0 for c in groups[-1]['cells']),
        'rare_groups': groups, 'new_generated_tokens': sum(r['generated_tokens'] for r in rows),
        'new_model_forwards': sum(r['model_forwards'] for r in rows),
        'rare_native_verifier_draws': sum(g['verifier_draws'] for g in groups), 'teacher_forwards': 0,
        'scope': 'Six fixed new units; conditional T=2 stress is not original-temperature tail reachability or full-path evidence.'}
    target = study / 'docs/sampling_continuation_analysis_v1.json'
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        target.write_text(text, encoding='utf-8')
    print(json.dumps({'rows': [{k:r[k] for k in ('qid','seed','entry','native_zero_accepted','self_accepted','graft_accepted','accepted_delta') if k in r} for r in rows],
        'H8': result['H8'], 'H9': result['H9_changed_temperature_each_candidate_observed'],
        'new_tokens': result['new_generated_tokens'], 'forwards': result['new_model_forwards'],
        'rare': [(g['temperature'],[(c['candidate'],c['hits'],c['first_hit_seed']) for c in g['cells']]) for g in groups]}))


if __name__ == '__main__':
    main()
