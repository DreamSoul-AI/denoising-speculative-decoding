"""Verify saved fixed-window scope and summarize matched slot/state effects.

    python studies/output_space_followup/analyze_results.py --raw .tmp/followup-transfer-20261014-v1
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    study = Path(__file__).resolve().parent
    data = read(study / 'docs/execution_v1.json')
    report = read(args.raw / 'diagnosis.json')
    assert report['quality_passed'] and manifest()['digest'] == report['source_digest'] == '670c744e0535dcad'
    assert sha(study / 'docs/execution_v1.json') == report['source_binding']['matrix_sha256']
    assert sha(args.raw / 'selected_tensors.pt') == report['raw_tensor_sha256']
    tensors = torch.load(args.raw / 'selected_tensors.pt', weights_only=True)
    findings = []
    for index, prompt in enumerate(report['prompts']):
        qid, root = prompt['question_id'], prompt['intervention_root']
        if root is None:
            findings.append({'question_id': qid, 'status': prompt['status']})
            continue
        cases = []
        for c in prompt['cases']:
            label, recipient, donor = c['label'], c['recipient'], c['donor']
            sample = lambda p: tensors[f'q{qid}:{label}:{p}']
            baseline = lambda case, p: tensors[f'q{qid}:baseline:{case}:{p}']
            a, b = sample(root)['current_masks'], baseline(recipient, root)['current_masks']
            assert torch.equal(a[0, 1], baseline(donor, root)['current_masks'][0, 1])
            assert torch.equal(a[0, 0], b[0, 0]) and torch.equal(a[1:], b[1:])
            assert torch.equal(sample(root)['seed_mask_inputs'], baseline(recipient, root)['seed_mask_inputs'])
            matching = [w for w in c['windows'] if w['recipient_baseline_visited']]
            for w in matching:
                p = w['root_position']
                for state in ('anchor', 'velocity', 'proxies'):
                    assert torch.equal(sample(p)['current_state'][state], baseline(recipient, p)['current_state'][state]) == w['states'][state]['bit_equal']
                assert w['slot_rankings']['seed_logits_finite']
            if recipient == donor:
                assert c['replay']['placebo_full_semantic_equal']
            cases.append({'label': label, 'accepted': c['replay']['stats']['accepted'],
                'decode_calls': c['replay']['stats']['decode_calls'], 'matched_acceptance_changes': c['matched_acceptance_changes'],
                'anchor_velocity_bit_equal_windows': sum(all(w['states'][s]['bit_equal'] for s in ('anchor', 'velocity')) for w in matching),
                'matched_windows': len(matching), 'anchor_max_abs_delta': max(w['states']['anchor']['max_abs_delta'] for w in matching),
                'velocity_max_abs_delta': max(w['states']['velocity']['max_abs_delta'] for w in matching),
                'first_seed_changed_windows': sum(not w['first_seed_bit_equal'] for w in matching),
                'first_margin_sign_changed_roots': [w['root_position'] for w in matching if w['first_slot']['target_margin_vs_best_other'] * w['baseline_first_slot']['target_margin_vs_best_other'] < 0],
                'actual_only_roots': c['actual_only_roots'], 'baseline_only_roots': c['baseline_only_roots']})
        findings.append({'question_id': qid, 'generator_seed': 123 + index, 'intervention_root': root,
            'input_accepted': data['per_prompt'][index]['cells']['frozen_mean_chain']['accepted'],
            'zero_accepted': data['per_prompt'][index]['cells']['output_order0']['accepted'],
            'one_accepted': data['per_prompt'][index]['cells']['output_order1']['accepted'], 'cases': cases})
    result = {'kind': 'offline scope/state/slot summary; no model forward', 'source_digest': report['source_digest'],
              'raw_report_sha256': sha(args.raw / 'diagnosis.json'), 'raw_tensor_sha256': report['raw_tensor_sha256'],
              'tool_sha256': sha(Path(__file__)), 'findings': findings}
    target = study / 'docs/analysis_v1.json'
    if target.exists():
        assert read(target) == result, 'existing evidence differs; preserve it'
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
