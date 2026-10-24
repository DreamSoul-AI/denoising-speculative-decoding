"""Offline first/second-slot ranking and actual-path audit of saved transfers.

    python studies/output_space_extension/analyze_results.py --raw .tmp/extension-transfer-20261014-v1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, margin


def slots(sample, tokens, position):
    import torch
    first_target = tokens[position + 1]
    first = margin(sample['seed_logits'][0], first_target)
    valid = int(sample['seed_logits'][0].argmax()) == first_target and position + 2 < len(tokens)
    second = margin(sample['seed_logits'][1], tokens[position + 2]) if position + 2 < len(tokens) else None
    return {'first': first, 'second': second, 'second_AR_path_valid': valid,
            'seed_logits_finite': bool(torch.isfinite(sample['seed_logits']).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest

    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    report = read(args.raw / 'diagnosis.json')
    data = read(study / 'docs/execution_v1.json')
    assert report['quality_passed'] and manifest()['digest'] == report['source_digest'] == '670c744e0535dcad'
    assert file_hash(args.raw / 'selected_tensors.pt') == report['raw_tensor_sha256']
    assert file_hash(study / 'docs/execution_v1.json') == report['source_binding']['matrix_sha256']
    raw = torch.load(args.raw / 'selected_tensors.pt', weights_only=True)
    native = {r['case']: r for r in data['runs']}
    findings = []
    for i, prompt in enumerate(report['prompts']):
        qid, root = prompt['question_id'], prompt['intervention_root']
        if root is None:
            findings.append({'question_id': qid, 'status': prompt['status']})
            continue
        tokens = native['AR']['raw']['evidence'][i]['output_ids']
        baselines = {b['case']: b for b in prompt['baselines']}
        natural = []
        baseline_samples = {case: {t['root_position']: raw[f'q{qid}:baseline:{case}:' + str(t['root_position'])]
            for t in row['trace'] if t['root_position'] >= root} for case, row in baselines.items()}
        for position in sorted(set(baseline_samples['output_order0']) & set(baseline_samples['output_order1'])):
            zero, one = (slots(baseline_samples[c][position], tokens, position) for c in ('output_order0', 'output_order1'))
            natural.append({'root_position': position, 'zero': zero, 'one': one,
                'valid_second_slot_opposite_signs': bool(zero['second_AR_path_valid'] and one['second_AR_path_valid']
                    and zero['second']['target_margin_vs_best_other'] * one['second']['target_margin_vs_best_other'] < 0)})
        cases = []
        for case in prompt['cases']:
            label, recipient, donor = case['label'], case['recipient'], case['donor']
            sample = lambda r: raw[f'q{qid}:{label}:{r}']
            original = baseline_samples[recipient]
            actual_trace = case['replay']['trace']
            selected = sample(root)['current_masks']
            assert torch.equal(selected[0, 1], baseline_samples[donor][root]['current_masks'][0, 1])
            assert torch.equal(selected[0, 0], original[root]['current_masks'][0, 0])
            assert torch.equal(selected[1:], original[root]['current_masks'][1:])
            assert torch.equal(sample(root)['seed_mask_inputs'], original[root]['seed_mask_inputs'])
            windows = []
            for recorded in case['windows']:
                position = recorded['root_position']
                current = sample(position)
                w = {'root_position': position, 'actual': slots(current, tokens, position),
                     'source_root': current['source_root'], 'source_owner': current['source_owner']}
                assert w['actual']['seed_logits_finite']
                if position in original:
                    w['baseline'] = slots(original[position], tokens, position)
                    for state in ('anchor', 'velocity', 'proxies'):
                        assert torch.equal(current['current_state'][state], original[position]['current_state'][state]) == recorded['states'][state]['bit_equal']
                    w['anchor_and_velocity_bit_equal'] = all(recorded['states'][k]['bit_equal'] for k in ('anchor', 'velocity'))
                if position in baseline_samples[donor]:
                    w['second_seed_bit_equal_to_donor'] = torch.equal(current['seed_mask_inputs'][1], baseline_samples[donor][position]['seed_mask_inputs'][1])
                windows.append(w)
            edges = []
            start = next(j for j, t in enumerate(actual_trace) if t['root_position'] == root)
            visited = {w['root_position'] for w in windows}
            for first, second in zip(actual_trace[start:start + 2], actual_trace[start + 1:start + 3]):
                a, b = first['root_position'], second['root_position']
                if a not in visited or b not in visited:
                    edges.append({'source_root': a, 'destination_root': b, 'status': 'not both predeclared tensor targets'})
                    continue
                owner = first['accepted_nodes'][-1] if first['accepted_nodes'] else 0
                assert torch.equal(sample(a)['current_mask_hidden'][owner], sample(b)['current_state']['proxies'])
                assert torch.equal(sample(a)['current_masks'][owner], sample(b)['seed_mask_inputs'])
                edges.append({'source_root': a, 'destination_root': b, 'owner': owner,
                              'hidden_to_proxy_bit_equal': True, 'mask_to_seed_bit_equal': True})
            cases.append({'label': label, 'propagation_edges': edges, 'windows': windows})
        findings.append({'question_id': qid, 'natural_baseline_slots': natural, 'cases': cases})
    result = {'kind': 'post hoc offline second-slot expansion and propagation audit; no additional model forwards',
              'note': 'First-slot hypotheses were predeclared; second-slot scalar audit is explicitly post hoc from already saved logits.',
              'source_digest': report['source_digest'], 'raw_report_sha256': file_hash(args.raw / 'diagnosis.json'),
              'raw_tensor_sha256': report['raw_tensor_sha256'], 'tool_sha256': file_hash(Path(__file__)),
              'helper_sha256': file_hash(ROOT / 'tools/diagnose_output_windows.py'), 'findings': findings}
    target = study / 'docs/slot_diagnosis_v1.json'
    if target.exists():
        assert read(target) == result, 'existing diagnosis differs; preserve it'
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
