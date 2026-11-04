"""Recompute cross-prompt propagation/state and null-hit checks; no forwards.

    python studies/output_space_small_matrix/analyze_cross_prompt.py --raw .tmp/output-cross-prompt-20261014-v1
"""

import argparse
import json
from pathlib import Path

from analyze_transfers import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest

    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    report = read(args.raw / 'diagnosis.json')
    tensor_file = args.raw / 'selected_tensors.pt'
    assert report['quality_passed'] and manifest()['digest'] == report['source_digest'] == '670c744e0535dcad'
    assert sha(tensor_file) == report['raw_tensor_sha256']
    tensors = torch.load(tensor_file, weights_only=True)
    findings = []
    for prompt in report['prompts']:
        qid, root = prompt['question_id'], prompt['intervention_root']
        baselines = {b['case']: b for b in prompt['baselines']}
        for case in prompt['cases']:
            label, recipient, donor = case['label'], case['recipient'], case['donor']
            sample = lambda r: tensors[f'q{qid}:{label}:{r}']
            baseline = lambda r: tensors[f'q{qid}:baseline:{recipient}:{r}']
            donor_baseline = tensors[f'q{qid}:baseline:{donor}:{root}']
            assert torch.equal(sample(root)['current_masks'][0, 1], donor_baseline['current_masks'][0, 1])
            assert torch.equal(sample(root)['current_masks'][0, 0], baseline(root)['current_masks'][0, 0])
            assert torch.equal(sample(root)['current_masks'][1:], baseline(root)['current_masks'][1:])
            actual_trace = case['replay']['trace']
            prior_trace = baselines[recipient]['trace']
            keys = ('root_position', 'cache_before', 'cache_after', 'accepted_nodes', 'committed_tokens', 'stopped')
            semantic_equal = [{k: t[k] for k in keys} for t in actual_trace] == [{k: t[k] for k in keys} for t in prior_trace]
            assert semantic_equal and not case['matched_acceptance_changes'] and not case['replay']['unvisited_targets']
            edges = []
            start = next(i for i, t in enumerate(actual_trace) if t['root_position'] == root)
            for source, destination in zip(actual_trace[start:start + 2], actual_trace[start + 1:start + 3]):
                a, b = source['root_position'], destination['root_position']
                owner = source['accepted_nodes'][-1] if source['accepted_nodes'] else 0
                assert torch.equal(sample(a)['current_mask_hidden'][owner], sample(b)['current_state']['proxies'])
                assert torch.equal(sample(a)['current_masks'][owner], sample(b)['seed_mask_inputs'])
                edges.append({'source_root': a, 'destination_root': b, 'owner': owner,
                              'hidden_to_proxy_bit_equal': True, 'mask_to_seed_bit_equal': True})
            changed_seeds = changed_proxies = 0
            max_margin_delta = 0.
            for window in case['windows']:
                r = window['root_position']
                a, b = sample(r), baseline(r)
                for state in ('anchor', 'velocity', 'proxies'):
                    bit_equal = torch.equal(a['current_state'][state], b['current_state'][state])
                    assert bit_equal == window['states'][state]['bit_equal']
                    if state != 'proxies':
                        assert bit_equal, 'same-position committed state differs in this measured batch'
                changed_proxies += not window['states']['proxies']['bit_equal']
                changed_seeds += not torch.equal(a['seed_mask_inputs'][0], b['seed_mask_inputs'][0])
                delta = window['first_slot']['target_margin_vs_best_other'] - window['baseline_first_slot']['target_margin_vs_best_other']
                max_margin_delta = max(max_margin_delta, abs(delta))
                assert (window['first_slot']['target_margin_vs_best_other'] > 0) == (window['baseline_first_slot']['target_margin_vs_best_other'] > 0)
            findings.append({'question_id': qid, 'label': label, 'semantic_trace_bit_equal': True,
                'matched_windows': len(case['windows']), 'committed_anchor_and_velocity_bit_equal_all_windows': True,
                'changed_first_seed_vectors': int(changed_seeds), 'changed_incoming_proxy_pairs': int(changed_proxies),
                'max_abs_target_margin_delta': max_margin_delta, 'target_margin_sign_changed_windows': 0,
                'accepted': case['replay']['stats']['accepted'], 'decode_calls': case['replay']['stats']['decode_calls'],
                'propagation_edges': edges, 'terminal_after_first_shared_hit': actual_trace[start + 1]['stopped']})
    result = {'kind': 'post hoc offline checks of fixed cross-prompt interventions; no forwards',
              'source_digest': report['source_digest'], 'raw_report_sha256': sha(args.raw / 'diagnosis.json'),
              'raw_tensor_sha256': report['raw_tensor_sha256'], 'analysis_tool_sha256': sha(Path(__file__)),
              'analysis_hash_helper_sha256': sha(Path(__file__).with_name('analyze_transfers.py')), 'findings': findings}
    target = study / 'docs/cross_prompt_path_v1.json'
    if target.exists():
        assert read(target) == result, 'existing evidence differs; preserve it'
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(findings))


if __name__ == '__main__':
    main()
