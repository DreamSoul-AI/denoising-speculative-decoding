"""Saved q178 controls only: no model loading, interventions or live profiler dispatch.

Original live orchestration remains readable from its fixed Git source member;
current entrypoints accept saved evidence only. Core q178 mechanisms stay in
studies._shared.structure.q178_diagnostics, without new execution here.
"""
import torch
from studies._shared.structure.extension_diagnostics import slots as _saved_slots
from studies._shared.structure.q178_diagnostics import tensor_equal


def slots(sample, tokens, position):
    device = sample['seed_logits'].device
    if device.type != 'cuda' or not bool(torch.isfinite(sample['seed_logits']).all()):
        raise ValueError('saved q178 numerical audit requires finite actual CUDA logits')
    result = _saved_slots(sample, tokens, position, device=device)
    return {**result, 'margin_compute_device': str(device), 'historical_CPU_margin_reference': None}


def analyze_numerics(report, raw, old, data, binding):
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
        'raw_report_sha256': binding['report'], 'raw_tensor_sha256': report['raw_tensor_sha256'],
        'tool_sha256': binding['analysis'], 'helper_sha256': binding['historical_helper'],
        'strict_repeat_all_selected_tensors_bit_equal': True, 'strict_vs_canonical_graft_full_semantic_trace_equal': True,
        'first_observed_common_state_divergence': {'root_position': 89, 'anchor_max_abs_delta': float((altered['anchor'] - baseline['anchor']).abs().max()),
            'velocity_max_abs_delta': float((altered['velocity'] - baseline['velocity']).abs().max()),
            'baseline_anchor_origin_inferred_from_native_observation_rule': 'exact position88 from accepted row2 of block86',
            'altered_anchor_origin_verified_by_tensor': 'exact position88 from row0 of block88',
            'same_strict_baseline_anchor_as_previous_runtime': True,
            'limits': 'Block/cache packing and owner path differ; no CUDA-kernel profiler or separate proxy-rounding intervention.'},
        'findings': findings}
    return result


def analyze_kernels(reports, tensors, old, native, binding):
    ar = next(r for r in native['runs'] if r['case'] == 'AR')['raw']['evidence'][3]['output_ids']
    result = {'analysis_sha256': binding['analysis'], 'new_model_forwards': 0, 'groups': [], 'quality_passed': False}
    for kind, count in [('norm', 8), ('qproj', 6)]:
        raw, samples = reports[kind], tensors[kind]
        assert len(raw['cases']) == count
        controls = {c['graft']: c for c in raw['cases'] if c['label'].startswith('native')}
        group = {'kind': kind, 'raw_report_sha256': binding[kind + '_report'],
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
    result['quality_passed'] = True
    return result


def analyze_operators(data, samples, old, native, binding):
    ar = next(r for r in native['runs'] if r['case'] == 'AR')['raw']['evidence'][3]['output_ids']
    result = {'raw_report_sha256': binding['report'], 'raw_tensor_sha256': binding['tensor'],
              'analysis_sha256': binding['analysis'], 'cases': [], 'quality_passed': False}
    bases = {c['graft']: c for c in data['cases'] if c['mode'] == 'native'}
    assert len(data['cases']) == 8 and len(bases) == 2
    total_tokens = total_calls = 0
    for case in data['cases']:
        base = bases[case['graft']]
        evidence = case['replay']
        assert evidence['output_equal_independent_AR'] and evidence['pre_intervention_trace_equal']
        assert evidence['stats']['new_tokens'] == len(ar) - 76
        total_tokens += evidence['stats']['new_tokens']
        total_calls += evidence['observer_forward_calls']
        assert all(r['canonical_state_bit_equal'] and r['incoming_proxies_unchanged']
                   for r in case['canonical_replacements'])
        windows = []
        for root in evidence['visited_targets']:
            sample = samples[f"{case['label']}:{root}"]
            if root > 85:
                assert all(torch.equal(sample['current_state'][k], old['canonical_states']['output_order1'][root][k])
                           for k in ('anchor', 'velocity'))
            if root in (86, 91, 108):
                ref = samples[f"{base['label']}:{root}"]
                windows.append({'root': root, 'source_root': sample['source_root'], 'source_owner': sample['source_owner'],
                    'both_slots': slots(sample, ar, root),
                    'seed_logits_max_abs_vs_same_kind_native': float((sample['seed_logits'] - ref['seed_logits']).abs().max()),
                    'seed_inputs_max_abs_vs_same_kind_native': float((sample['seed_mask_inputs'] - ref['seed_mask_inputs']).abs().max())})
        result['cases'].append({'label': case['label'], 'accepted': evidence['stats']['accepted'],
            'decode_calls': evidence['stats']['decode_calls'],
            'full_semantic_trace_equal_same_kind_native': evidence['trace'] == base['replay']['trace'],
            'mapping_max_abs': case['mapping_max_abs'], 'windows': windows})
    result.update(quality_passed=True, total_new_tokens=total_tokens, total_model_forwards=total_calls,
                  auxiliary_model_forwards=0, profiler=data['profiler'],
                  frozen_mapping_comparisons=data['frozen_mapping_comparisons'])
    return result
