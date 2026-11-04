"""Offline verification of norm/q_proj scope, effective interventions and paths.

    python studies/output_space_extension/analyze_kernels.py --norm .tmp/q178-norm-kernel-<id> --qproj .tmp/q178-qproj-<id>
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'studies' / '_shared' / 'structure'))
from window_diagnostics import file_hash
from analyze_results import slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norm', required=True, type=Path)
    parser.add_argument('--qproj', required=True, type=Path)
    args = parser.parse_args()
    import torch
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    study = Path(__file__).resolve().parent
    native = read(study / 'docs/execution_v1.json')
    ar = next(r for r in native['runs'] if r['case'] == 'AR')['raw']['evidence'][3]['output_ids']
    old_path = ROOT / '.tmp/q178-numerical-20261014-v1/selected_tensors.pt'
    old = torch.load(old_path, map_location='cpu', weights_only=True)
    result = {'analysis_sha256': file_hash(Path(__file__)), 'new_model_forwards': 0, 'groups': [], 'quality_passed': False}
    for kind, folder, count in [('norm', args.norm, 8), ('qproj', args.qproj, 6)]:
        raw = read(folder / 'diagnosis.json')
        assert raw['quality_passed'] and len(raw['cases']) == count
        assert raw['source_digest'] == native['source_digest'] == '670c744e0535dcad'
        assert file_hash(old_path) == raw['source_binding']['previous_tensor']
        tool = ROOT / 'studies' / '_shared' / 'structure' / ('diagnose_q178_norm_kernel.py' if kind == 'norm' else 'diagnose_q178_qproj.py')
        assert file_hash(tool) == raw['source_binding']['tool']
        assert file_hash(study / 'kernel_common.py') == raw['source_binding']['common']
        plan = study / 'docs/NORM_KERNEL_PLAN.md' if kind == 'norm' else ROOT / 'studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md'
        assert file_hash(plan) == raw['source_binding']['plan']
        assert file_hash(folder / 'selected_tensors.pt') == raw['raw_tensor_sha256']
        samples = torch.load(folder / 'selected_tensors.pt', map_location='cpu', weights_only=True)
        controls = {c['graft']: c for c in raw['cases'] if c['label'].startswith('native')}
        group = {'kind': kind, 'raw_report_sha256': file_hash(folder / 'diagnosis.json'),
            'raw_tensor_sha256': raw['raw_tensor_sha256'], 'cases': []}
        for case in raw['cases']:
            replay = case['replay']
            assert replay['output_equal_independent_AR'] and replay['pre_intervention_trace_equal']
            assert all(r['canonical_state_bit_equal'] and r['incoming_proxies_unchanged'] for r in case['canonical_replacements'])
            windows, max_proxy, max_hidden = [], 0., 0.
            for position in replay['visited_targets']:
                current = samples[f"{case['label']}:{position}"]
                baseline = samples[f"{controls[case['graft']]['label']}:{position}"]
                if position > 85:
                    assert all(torch.equal(current['current_state'][k], old['canonical_states']['output_order1'][position][k])
                               for k in ('anchor', 'velocity'))
                max_proxy = max(max_proxy, float((current['current_state']['proxies'] - baseline['current_state']['proxies']).abs().max()))
                max_hidden = max(max_hidden, float((current['current_mask_hidden'] - baseline['current_mask_hidden']).abs().max()))
                if position in (86, 91, 108):
                    windows.append({'root': position, 'source_root': current['source_root'], 'source_owner': current['source_owner'],
                        'both_slots': slots(current, ar, position),
                        'seed_logits_max_abs': float((current['seed_logits'] - baseline['seed_logits']).abs().max())})
            group['cases'].append({'label': case['label'], 'accepted': replay['stats']['accepted'],
                'decode_calls': replay['stats']['decode_calls'],
                'full_semantic_path_equal_same_kind_native': replay['trace'] == controls[case['graft']]['replay']['trace'],
                'proxy_max_abs_to_control': max_proxy, 'mask_hidden_max_abs_to_control': max_hidden,
                'entry_vector_changed': case.get('entry_vector_changed'),
                'q_projection_changes': case.get('q_projection_changes'), 'windows': windows})
        group['new_tokens'] = sum(c['replay']['stats']['new_tokens'] for c in raw['cases'])
        group['generation_model_forwards'] = sum(c['replay']['observer_forward_calls'] for c in raw['cases'])
        group['auxiliary_model_forwards'] = raw['auxiliary_model_forwards']
        if kind == 'norm':
            group['all_returned_mapping_values_bit_equal_to_native'] = (
                all(c['post_mapping_max_abs'] == 0 and not c['entry_vector_changed'] for c in raw['cases'])
                and all(c['mask_max_abs'] == 0 for c in raw['frozen_comparisons']))
            group['profiles_available'] = raw['native_profile']['available'] and raw['alternative_profile']['available']
            group['numerical_intervention_effective'] = not group['all_returned_mapping_values_bit_equal_to_native']
        else:
            assert len(raw['block_replays']) == 4 and all(r['logits_bit_equal'] and r['final_hidden_bit_equal']
                and r['saved_cache_unchanged'] for r in raw['block_replays'])
            group['numerical_intervention_effective'] = all(
                not change['replacement_bit_equal'] and change['other_rows_bit_equal']
                for c in raw['cases'] if c['mode'] != 'native' for change in c['q_projection_changes'])
        result['groups'].append(group)
        target = study / f'docs/{kind}_kernel_diagnosis_v1.json'
        if target.exists():
            assert file_hash(target) == file_hash(folder / 'diagnosis.json')
        else:
            target.write_bytes((folder / 'diagnosis.json').read_bytes())
    result['quality_passed'] = True
    target = study / 'docs/kernel_analysis_v1.json'
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        target.write_text(text, encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'groups': [
        {k: g[k] for k in ('kind', 'numerical_intervention_effective', 'new_tokens',
                          'generation_model_forwards', 'auxiliary_model_forwards')} for g in result['groups']]}))


if __name__ == '__main__':
    main()
