"""Audit q178 repeatability, path-dependent drift and canonical state controls.

    python studies/output_space_extension/analyze_numerics.py --raw .tmp/q178-numerical-20261014-v1 --original .tmp/extension-transfer-20261014-v1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash
from diagnose_q178_numerics import tensor_equal
from analyze_results import slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    report = read(args.raw / 'diagnosis.json')
    assert report['quality_passed'] and manifest()['digest'] == report['source_digest'] == '670c744e0535dcad'
    assert file_hash(args.raw / 'selected_tensors.pt') == report['raw_tensor_sha256']
    assert file_hash(args.original / 'selected_tensors.pt') == report['source_binding']['old_transfer_tensor_sha256']
    raw = torch.load(args.raw / 'selected_tensors.pt', weights_only=True)
    old = torch.load(args.original / 'selected_tensors.pt', weights_only=True)
    study = Path(__file__).resolve().parent
    data = read(study / 'docs/execution_v1.json')
    tokens = next(r for r in data['runs'] if r['case'] == 'AR')['raw']['evidence'][3]['output_ids']
    cases = {c['label']: c for c in report['cases']}
    findings = []
    for label, case in cases.items():
        recipient = case['recipient']
        current = {p: raw[f'{label}:{p}'] for p in case['replay']['visited_targets']}
        comparison = ('canonical_one_self' if recipient == 'output_order1' else 'canonical_zero_self') if case['canonical_committed_state'] else 'baseline:' + recipient
        windows = []
        for position, sample in current.items():
            if case['canonical_committed_state'] and position > 85:
                assert all(torch.equal(sample['current_state'][k], raw['canonical_states'][recipient][position][k]) for k in ('anchor', 'velocity'))
            key = f'{comparison}:{position}'
            state_equal = None
            if key in raw:
                state_equal = all(torch.equal(sample['current_state'][k], raw[key]['current_state'][k]) for k in ('anchor', 'velocity'))
            windows.append({'root_position': position, 'source_root': sample['source_root'], 'source_owner': sample['source_owner'],
                            'anchor_velocity_equal_to_same_mode_self': state_equal, 'rankings': slots(sample, tokens, position)})
        findings.append({'label': label, 'accepted': case['replay']['stats']['accepted'], 'decode_calls': case['replay']['stats']['decode_calls'],
                         'comparison': comparison, 'windows': windows})
    for first, second in [('strict_one_from_zero_1', 'strict_one_from_zero_2'), ('strict_zero_from_one_1', 'strict_zero_from_one_2')]:
        assert cases[first]['replay'] == cases[second]['replay']
        a = {p: raw[f'{first}:{p}'] for p in cases[first]['replay']['visited_targets']}
        b = {p: raw[f'{second}:{p}'] for p in cases[second]['replay']['visited_targets']}
        assert tensor_equal(a, b)
    for strict, canonical in [('strict_one_from_zero_1', 'canonical_one_from_zero'), ('strict_zero_from_one_1', 'canonical_zero_from_one')]:
        assert cases[strict]['replay']['trace'] == cases[canonical]['replay']['trace']
    baseline = raw['baseline:output_order1:89']['current_state']
    altered = raw['strict_one_from_zero_1:89']['current_state']
    assert torch.equal(baseline['anchor'], old['q178:baseline:output_order1:89']['current_state']['anchor'])
    assert torch.equal(altered['anchor'], raw['strict_one_from_zero_1:88']['exact_root_hidden'])
    original_trace = next(r for r in data['runs'] if r['case'] == 'output_order1')['raw']['evidence'][3]['stats']['trace']
    assert next(t for t in original_trace if t['root_position'] == 86)['accepted_nodes'] == [1, 2]
    assert next(t for t in cases['strict_one_from_zero_1']['replay']['trace'] if t['root_position'] == 86)['accepted_nodes'] == [1]
    result = {'kind': 'offline fixed-control verification; no new model forward', 'source_digest': report['source_digest'],
        'raw_report_sha256': file_hash(args.raw / 'diagnosis.json'), 'raw_tensor_sha256': report['raw_tensor_sha256'],
        'tool_sha256': file_hash(Path(__file__)), 'helper_sha256': file_hash(ROOT / 'tools/diagnose_q178_numerics.py'),
        'strict_repeat_all_selected_tensors_bit_equal': True, 'strict_vs_canonical_graft_full_semantic_trace_equal': True,
        'first_observed_common_state_divergence': {'root_position': 89, 'anchor_max_abs_delta': float((altered['anchor'] - baseline['anchor']).abs().max()),
            'velocity_max_abs_delta': float((altered['velocity'] - baseline['velocity']).abs().max()),
            'baseline_anchor_origin_inferred_from_native_observation_rule': 'exact position88 from accepted row2 of block86',
            'altered_anchor_origin_verified_by_tensor': 'exact position88 from row0 of block88',
            'same_strict_baseline_anchor_as_previous_runtime': True,
            'limits': 'Block/cache packing and owner path differ; no CUDA-kernel profiler or separate proxy-rounding intervention.'},
        'findings': findings}
    target = study / 'docs/numerical_analysis_v1.json'
    if target.exists():
        assert read(target) == result, 'existing evidence differs; preserve it'
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
