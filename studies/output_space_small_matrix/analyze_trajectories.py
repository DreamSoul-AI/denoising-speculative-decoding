"""Rebuild paired-prefix diagnosis from the saved evidence; no model forwards.

    python studies/output_space_small_matrix/analyze_trajectories.py
"""

import itertools
import json
from pathlib import Path


def diagnose(data):
    bycase = {r['case']: r for r in data['runs']}
    methods = ['frozen_mean_chain', 'output_order0', 'output_order1']
    pairs = []
    for i, qid in enumerate(data['question_ids']):
        ar = bycase['AR']['raw']['evidence'][i]
        for case in methods:
            assert bycase[case]['raw']['evidence'][i]['output_ids'] == ar['output_ids']
        for first, second in itertools.combinations(methods, 2):
            first_ev = bycase[first]['raw']['evidence'][i]
            second_ev = bycase[second]['raw']['evidence'][i]
            a = {t['root_position']: t for t in first_ev['stats']['trace']}
            b = {t['root_position']: t for t in second_ev['stats']['trace']}
            assert len(a) == len(first_ev['stats']['trace']) and len(b) == len(second_ev['stats']['trace'])
            common = sorted(set(a) & set(b))
            changed, same, mask_changed = [], 0, 0
            for position in common:
                x, y = a[position], b[position]
                assert x['cache_before'] == y['cache_before'] == position
                assert first_ev['input_ids'] == second_ev['input_ids'] == ar['input_ids']
                assert first_ev['output_ids'][:position + 1] == second_ev['output_ids'][:position + 1]
                if len(x['accepted_nodes']) == len(y['accepted_nodes']):
                    same += 1
                else:
                    changed.append({'root_position': position, 'generation_root_index': position - len(ar['input_ids']),
                                    'accepted_first': len(x['accepted_nodes']), 'accepted_second': len(y['accepted_nodes']),
                                    'first_committed': x['committed_tokens'], 'second_committed': y['committed_tokens'],
                                    'stopped_first': x['stopped'], 'stopped_second': y['stopped']})
                if first.startswith('output') and second.startswith('output'):
                    mask_changed += (x['output_hidden']['last_owner_diff_from_root']
                                     != y['output_hidden']['last_owner_diff_from_root'])
            pairs.append({'question_id': qid, 'first': first, 'second': second,
                          'common_root_positions': common, 'same_acceptance_length_at_common_roots': same,
                          'changed_acceptance_length_at_common_roots': changed,
                          'first_only_root_positions': sorted(set(a) - set(b)),
                          'second_only_root_positions': sorted(set(b) - set(a)),
                          'output_owner_difference_stat_changed_at_common_roots': (
                              mask_changed if first.startswith('output') and second.startswith('output') else None),
                          'observed_prefixes_verified_equal': True})
    return {'comparison_basis': 'same question and absolute committed root position, verified common AR prefix; no step-number matching',
            'new_model_forwards_for_analysis': 0, 'pairs': pairs,
            'limits': 'No draft logits/full masks/exact-proxy hidden pairs saved; observed acceptance differences are not causal attribution.'}


def main():
    study = Path(__file__).resolve().parent
    data = json.loads((study / 'docs/execution_v1.json').read_text(encoding='utf-8'))
    assert data['study_quality_passed'] and data['question_ids'] == [172, 173, 174]
    diagnosis = diagnose(data)
    target = study / 'docs/trajectory_diagnosis.json'
    if target.exists():
        # Existing recorded analysis must reproduce exactly, never overwrite a discrepancy.
        assert json.loads(target.read_text(encoding='utf-8')) == diagnosis
    target.write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2) + '\n',encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
