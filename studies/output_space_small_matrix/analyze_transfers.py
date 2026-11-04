"""Verify the single-slot propagation edges offline, without model forwards.

    python studies/output_space_small_matrix/analyze_transfers.py --raw .tmp/output-transfer-20261014-v1 --original .tmp/output-windows-20261014-v1
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
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest

    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    study = Path(__file__).resolve().parent
    report = read(args.raw / 'diagnosis.json')
    prior_report = read(args.original / 'diagnosis.json')
    assert report['quality_passed'] and prior_report['quality_passed']
    assert manifest()['digest'] == report['source_digest'] == '670c744e0535dcad'
    assert sha(args.raw / 'selected_tensors.pt') == report['raw_tensor_sha256']
    assert sha(args.original / 'selected_tensors.pt') == prior_report['raw_tensor_sha256'] == report['source_binding']['tensor_sha256']
    assert sha(args.original / 'diagnosis.json') == report['source_binding']['observer_sha256']
    raw = torch.load(args.raw / 'selected_tensors.pt', weights_only=True)
    old = torch.load(args.original / 'selected_tensors.pt', weights_only=True)
    placebo = 'output_order1_from_output_order1'
    assert next(c for c in report['cases'] if c['label'] == placebo)['replay']['placebo_full_semantic_equal']
    findings = []
    for case in report['cases']:
        label, recipient, donor = case['label'], case['recipient'], case['donor']
        sample = lambda root: raw[label + ':' + str(root)]
        baseline = lambda root: raw[placebo + ':' + str(root)]
        intervention = raw[label + ':intervention']
        assert torch.equal(intervention['donor_vector'], old[donor + ':83']['current_masks'][0, 1])
        assert torch.equal(sample(83)['seed_mask_inputs'], old[recipient + ':83']['seed_mask_inputs'])
        edges = []
        for source, destination, owner in [(83, 84, 0), (84, 86, 1)]:
            assert torch.equal(sample(source)['current_mask_hidden'][owner], sample(destination)['current_state']['proxies'])
            assert torch.equal(sample(source)['current_masks'][owner], sample(destination)['seed_mask_inputs'])
            assert sample(destination)['source_root'] == source and sample(destination)['source_owner'] == owner
            edges.append({'source_root': source, 'destination_root': destination, 'selected_owner': owner,
                          'hidden_to_proxy_bit_identical': True, 'mask_to_seed_bit_identical': True})
        states_equal, velocities_zero, deltas, windows = [], [], [], []
        for root in report['targets']:
            current = sample(root)['current_state']
            ref = baseline(root)['current_state']
            states_equal.append(all(torch.equal(current[k], ref[k]) for k in ('anchor', 'velocity')))
            velocities_zero.append(bool(torch.count_nonzero(current['velocity']) == 0))
            deltas.append({'root_position': root,
                           'proxy0_L2_delta_vs_order1_placebo': float((current['proxies'][0] - ref['proxies'][0]).norm()),
                           'proxy1_L2_delta_vs_order1_placebo': float((current['proxies'][1] - ref['proxies'][1]).norm())})
        for root in (94, 97):
            actual, original = sample(root), old[donor + ':' + str(root)]
            windows.append({'root_position': root,
                'first_seed_bit_identical_to_donor_order_baseline': torch.equal(actual['seed_mask_inputs'][0], original['seed_mask_inputs'][0]),
                'first_seed_logit_max_abs_delta_vs_donor_baseline': float((actual['seed_logits'][0] - original['seed_logits'][0]).abs().max())})
        if recipient == 'output_order1':
            assert all(states_equal), 'committed anchor/velocity changed at a declared root'
        else:
            assert all(velocities_zero), 'order0 acquired committed velocity'
        findings.append({'label': label, 'first_seed_at83_unchanged': True,
            'transferred_vector_norm': float(intervention['donor_vector'].norm()),
            'original_vector_norm': float(intervention['original_vector'].norm()),
            'transfer_vector_L2_delta': float((intervention['donor_vector'] - intervention['original_vector']).norm()),
            'propagation_edges': edges, 'proxy_deltas_vs_order1_placebo': deltas,
            'anchor_and_velocity_bit_identical_to_order1_placebo_at_all_declared_roots': all(states_equal),
            'committed_velocity_zero_at_all_declared_roots': all(velocities_zero),
            'later_seed_comparisons': windows})
    result = {'kind': 'post hoc offline verification of predeclared transfer measurements; no forwards',
              'source_digest': report['source_digest'], 'transfer_report_sha256': sha(args.raw / 'diagnosis.json'),
              'transfer_tensor_sha256': report['raw_tensor_sha256'],
              'original_tensor_sha256': prior_report['raw_tensor_sha256'],
              'analysis_tool_sha256': sha(Path(__file__)), 'findings': findings}
    target = study / 'docs/transfer_path_diagnosis_v1.json'
    if target.exists():
        assert read(target) == result, 'existing offline evidence differs; preserve it'
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
