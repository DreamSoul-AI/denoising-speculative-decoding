"""Pure fixed captured-state/path audits; current tensor arithmetic runs on explicit CUDA inputs."""
from typing import Any

from studies._shared.structure.window_diagnostics import TRACE_KEYS, SEMANTIC_KEYS
from studies._shared.structure.extension_diagnostics import choose_window
SOURCE_DIGEST = '670c744e0535dcad'


def margin(logits, target):
    import torch
    assert logits.is_cuda, 'current margin audit requires CUDA; no CPU fallback'
    row = logits.detach().float()
    top = row.topk(2)
    probs = torch.softmax(row, dim=-1)
    other = top.values[1] if int(top.indices[0]) == target else top.values[0]
    return {'top1_id': int(top.indices[0]), 'top2_id': int(top.indices[1]),
            'top1_probability': float(probs[top.indices[0]]), 'top2_probability': float(probs[top.indices[1]]),
            'target_id': int(target), 'target_probability': float(probs[target]),
            'target_margin_vs_best_other': float(row[target] - other)}


def recorded_margin(logits, previous, checks):
    current = margin(logits, previous['target_id'])
    assert all(current[key] == previous[key] for key in ('target_id', 'top1_id', 'top2_id')), 'saved margin discrete identities differ'
    checks.append({'historical_CPU_FP32_reference': previous, 'current_CUDA_FP32': current,
                   'actual_device': str(logits.device),
                   'numeric_absolute_differences': {key: abs(current[key] - previous[key]) for key in
                       ('top1_probability', 'top2_probability', 'target_probability', 'target_margin_vs_best_other')},
                   'scope': 'discrete identities passed; historical CPU probabilities/margin not revalidated across devices'})

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

def legal_second_flip(first_a, first_b, first_target, second_target, eos, nodes_a, nodes_b):
    return (second_target is not None and first_target not in eos and
            first_a == first_b == first_target and 1 in nodes_a and 1 in nodes_b and
            ((2 in nodes_a) != (2 in nodes_b)))

def check_conditional(data, logits, expected_candidates):
    import torch
    assert logits.is_cuda, "conditional audit requires actual CUDA"
    result = []
    for group in data:
        assert group['RNG_consumed'] and group['same_verifier_draws_for_all_cells']
        assert group['seeds'] == list(range(200, 264)) and group['verifier_draws'] == 64
        assert len(group['verifier_token_ids']) == 64, 'conditional verifier bank must contain all 64 saved draws'
        assert len(group['cells']) == 4 and {cell['label'] for cell in group['cells']} == {'S0P0', 'S1P0', 'S0P1', 'S1P1'}, 'conditional cells must be complete and unique'
        probs = torch.softmax(logits.float() / group['temperature'], dim=-1)
        reference64 = torch.softmax(logits.double() / group['temperature'], dim=-1)
        probability_checks = []
        for cell in group['cells']:
            assert cell['candidate'] == expected_candidates[cell['label']], 'conditional candidate differs from its captured owner protocol'
            assert cell['first_slot_hits'] == sum(t == cell['candidate'] for t in group['verifier_token_ids'])
            assert float(probs[cell['candidate']]) == cell['exact_candidate_probability']
            probability_checks.append({'label': cell['label'], 'native_cuda_FP32_mass': float(probs[cell['candidate']]),
                'CUDA_FP64_mass': float(reference64[cell['candidate']]), 'CPU_FP64_mass': None,
                'historical_CPU_FP64_reference': 'unknown; original analysis reference not supplied',
                'mass_absolute_difference': abs(float(probs[cell['candidate']]) - float(reference64[cell['candidate']]))})
        cells = {c['label']: c for c in group['cells']}
        assert cells['S0P0']['candidate'] == cells['S1P0']['candidate']
        assert cells['S0P1']['candidate'] == cells['S1P1']['candidate']
        result.append({'temperature': group['temperature'], 'draws': 64,
            'distinct_verifier_tokens': len(set(group['verifier_token_ids'])), 'cells': group['cells'],
            'probability_checks': probability_checks})
    return result

def analyze_local_extension(local, extension, a, b, prior, native, eos, binding):
    import torch
    margin_checks: list[dict[str, Any]] = []
    assert len(local['blocks']) == 24 and len(local['cases']) == 2
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in local['blocks']:
            assert record['verification_logits_bit_equal'] and record['verification_hidden_bit_equal']
            groups.setdefault(record['label'], []).append(record)
            tensor = a[record['label']]
            frame = a[f'frame:{record["root"]}']
            assert torch.equal(tensor['logits'][:, :3], frame['logits'][:, :3])
            assert torch.equal(tensor['hidden'][:, :3], frame['hidden'][:, :3])
            assert float((tensor['masks'] - frame['inputs'][:, 3:].reshape(3, 2, -1)).abs().max()) == record['mask_max_abs']
            recorded_margin(tensor['logits'][0, 3], record['first_seed'], margin_checks)
    for records in groups.values():
            assert sorted(r['repeat'] for r in records) == [0, 1, 2]
            assert all({k: v for k, v in r.items() if k != 'repeat'} ==
                       {k: v for k, v in records[0].items() if k != 'repeat'} for r in records)
    for label, old in [('r719:S0P0', 'canonical_self'), ('r719:S0P1', 'canonical_graft')]:
            assert torch.equal(a[label]['logits'][0, 3:5], prior[old + ':720']['seed_logits'])
    assert int(a['frame:720']['logits'][0, 0].argmax()) == 264
    timing = {'steps': local['steps'], 'fixed_blocks': [r[0] for r in groups.values()],
            'source719_proxy_only_reproduces_actual_graft_seed_logits_bitwise': True,
            'source719_state_only_mask_changed': groups['r719:S1P0'][0]['mask_max_abs'] > 0,
            'source719_state_only_first_candidate_unchanged':
                groups['r719:S1P0'][0]['first_seed']['top1_id'] == groups['r719:S0P0'][0]['first_seed']['top1_id'],
            'current720_verification_bit_equal_all_cells': True,
            'new_tokens': sum(c['replay']['stats']['new_tokens'] for c in local['cases']),
            'generation_forwards': sum(c['replay']['observer_forward_calls'] for c in local['cases']),
            'block_forwards': 24, 'teacher_forwards': 0}
    rows = {r['case']: r for r in native['runs']}
    summaries, depth2 = [], []
    for unit in extension['units']:
            prefix = f'q{unit["qid"]}s{unit["seed"]}:'
            cells = {c['label']: c for c in unit['cases']}
            table = []
            for suffix in ('self', 'graft'):
                counts = {mode: cells[f'{mode}_{suffix}']['replay']['stats']['accepted']
                          for mode in ('native', 'anchor', 'velocity', 'both')}
                table.append({'donor': suffix, 'accepted': counts, 'interaction': interaction(counts)})
            ar = rows['AR']['raw']['evidence'][native['question_ids'].index(unit['qid'])]['output_ids']
            def sample(label, p):
                key = prefix + ('baseline:output_order1' if label == 'native_self' else label) + f':{p}'
                return b[key] if key in b else None
            pairs = []
            for mode in ('native', 'anchor', 'velocity', 'both'):
                left, right = (cells[f'{mode}_{suffix}']['replay'] for suffix in ('self', 'graft'))
                pairs.append({'mode': mode, 'equal_accepted_counts': left['stats']['accepted'] == right['stats']['accepted'],
                              'full_semantic_trace_equal': left['trace'] == right['trace']})
                tl, tr = ({t['root_position']: t for t in replay['trace']} for replay in (left, right))
                for p in sorted(tl.keys() & tr.keys()):
                    x, y = sample(mode + '_self', p), sample(mode + '_graft', p)
                    if x is None or y is None or p + 2 >= len(ar) or ar[p + 1] in eos:
                        continue
                    if all(int(s['seed_logits'][0].argmax()) == ar[p + 1] for s in (x, y)):
                        if (len(tl[p]['accepted_nodes']) >= 2) != (len(tr[p]['accepted_nodes']) >= 2):
                            depth2.append({'qid': unit['qid'], 'seed': unit['seed'], 'mode': mode, 'root': p})
            for cell in unit['cases']:
                mode = cell['mode']
                if mode == 'native':
                    continue
                records = cell['state_replacements']
                assert all(r['incoming_proxies_unchanged'] for r in records)
                assert all(r['canonical_state_bit_equal'] if mode == 'both' else
                           r['other_state_bit_equal'] and r['selected_state_bit_equal'] for r in records)
                for p in cell['replay']['visited_targets']:
                    s = sample(cell['label'], p)
                    if p > unit['entry']:
                        for coordinate in ('anchor', 'velocity') if mode == 'both' else (mode,):
                            assert torch.equal(s['current_state'][coordinate], b[prefix + 'canonical_states'][p][coordinate])
            forwards = sum(r['replay']['observer_target_forward_calls'] for r in unit['baselines'])
            forwards += sum(c['replay']['observer_forward_calls'] for c in unit['cases'] if c['label'] != 'native_self')
            summaries.append({'qid': unit['qid'], 'seed': unit['seed'], 'entry': unit['entry'],
                'targets': unit['targets'], 'counts': table, 'self_graft_pairs': pairs,
                'generations': 9, 'new_tokens': 900, 'generation_forwards': forwards,
                'teacher_positions': unit['teacher_positions'], 'teacher_cosine_min': unit['exact_root_cosine_min']})
    seed_checks = []
    for qid in (242, 243):
            u, v = [u for u in extension['units'] if u['qid'] == qid]
            comparisons = []
            for left, right in zip(u['cases'], v['cases']):
                assert left['label'] == right['label']
                assert left['replay']['trace'] == right['replay']['trace']
                assert left['replay']['stats'] == right['replay']['stats']
                for p in left['replay']['visited_targets']:
                    x = b[f'q{qid}s{u["seed"]}:' + ('baseline:output_order1' if left['label'] == 'native_self' else left['label']) + f':{p}']
                    y = b[f'q{qid}s{v["seed"]}:' + ('baseline:output_order1' if right['label'] == 'native_self' else right['label']) + f':{p}']
                    equal = all(torch.equal(x[k], y[k]) for k in ('seed_logits', 'current_masks'))
                    comparisons.append({'label': left['label'], 'root': p, 'seed_logits_masks_bit_equal': equal})
            seed_checks.append({'qid': qid, 'seeds': [u['seed'], v['seed']], 'semantic_traces_stats_equal': True,
                'captured_seed_logits_masks_all_bit_equal': all(c['seed_logits_masks_bit_equal'] for c in comparisons)})
    result = {'analysis_sha256': binding['analysis'], 'local_raw_report_sha256': binding['local_report'],
            'extension_raw_report_sha256': binding['extension_report'],
            'local_tensor_sha256': local['raw_tensor_sha256'], 'extension_tensor_sha256': extension['raw_tensor_sha256'],
            'quality_passed': True, 'local_timing': timing, 'extension': summaries, 'seed_checks': seed_checks,
            'H5_universal_nonzero_interaction_in_fixed_extension': 'falsified' if any(t['interaction'] == 0 for u in summaries for t in u['counts']) else 'not_falsified',
            'legal_depth2_witnesses_in_six_fixed_targets': depth2, 'teacher_forwards': 4,
            'extension_generations': 36, 'extension_new_tokens': sum(u['new_tokens'] for u in summaries),
            'extension_generation_forwards': sum(u['generation_forwards'] for u in summaries),
            'scope': 'Two additional prompts; greedy seeds are replay controls, not independent stochastic samples.'}
    result['legacy_margin_comparisons'] = margin_checks
    return result
def analyze_state_factors(previous, new, op, a, b, native, eos, binding):
    import torch
    margin_checks: list[dict[str, Any]] = []
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
    ar = next(r for r in native['runs'] if r['case'] == 'AR')['raw']['evidence'][0]['output_ids']
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
    result['legacy_margin_comparisons'] = margin_checks
    return result


def analyze_sampling_owner(sampling, owner, old, prior, tensors, selection_hashes, binding):
    import torch
    margin_checks: list[dict[str, Any]] = []
    for label, previous in [('r717:S0P0', 'canonical_self'), ('r717:S0P1', 'canonical_graft')]:
            assert torch.equal(tensors[label]['logits'][0, 5:7], prior[previous + ':719']['seed_logits'])
    assert owner['owner'] == 1 and len(owner['blocks']) == 12
    for block in owner['blocks']:
            assert block['root'] == 717 and block['verification_logits_bit_equal']
            value = tensors[block['label']]
            recorded_margin(value['logits'][0, 5], block['first_seed'], margin_checks)
            assert torch.equal(value['logits'][:, :3], tensors['frame:717']['logits'][:, :3])
    summaries = []
    generation_tokens = generation_forwards = 0
    for unit in sampling['units']:
            assert unit['temperature'] == .8 and unit['seed'] in (123, 124)
            native = {r['case']: r for r in unit['native']}
            assert all(r['output_equal_independent_sampled_AR'] and r['output_ids'] == unit['AR']['output_ids'] for r in native.values())
            assert choose_window(native['output_order0']['stats']['trace'], native['output_order1']['stats']['trace']) == unit['selection']
            assert selection_hashes[unit['seed']] == unit['selection_sha256']
            tokens = unit['AR']['stats']['ar_new_tokens'] + sum(r['stats']['new_tokens'] for r in native.values())
            forwards = unit['AR']['stats']['ar_calls'] + sum(1 + r['stats']['decode_calls'] for r in native.values())
            item = {'seed': unit['seed'], 'temperature': .8, 'entry': unit['selection']['intervention_root'],
                'AR_new_tokens': unit['AR']['stats']['ar_new_tokens'],
                'native_zero_accepted': native['output_order0']['stats']['accepted'],
                'native_one_accepted': native['output_order1']['stats']['accepted']}
            if item['entry'] is None:
                assert not unit['cases']
                item['status'] = 'infeasible_not_a_null_graft_effect'
            else:
                cells = {c['label']: c['replay'] for c in unit['cases']}
                assert set(cells) == {'self', 'graft'}
                assert all(c['output_equal_independent_AR'] and c['pre_intervention_trace_equal'] for c in cells.values())
                reference = native['output_order1']['stats']
                assert cells['self']['trace'] == [{k: t[k] for k in TRACE_KEYS} for t in reference['trace']]
                assert all(cells['self']['stats'][k] == reference[k] for k in SEMANTIC_KEYS)
                item.update(self_accepted=cells['self']['stats']['accepted'], graft_accepted=cells['graft']['stats']['accepted'],
                    accepted_delta=cells['graft']['stats']['accepted'] - cells['self']['stats']['accepted'],
                    semantic_trace_changed=cells['self']['trace'] != cells['graft']['trace'], targets=unit['targets'])
                tokens += sum(native[o['case']]['stats']['new_tokens'] for o in unit['observed'])
                tokens += sum(c['stats']['new_tokens'] for c in cells.values())
                forwards += sum(o['replay']['observer_target_forward_calls'] for o in unit['observed'])
                forwards += sum(c['observer_forward_calls'] for c in cells.values())
            item.update(actual_new_tokens=tokens, generation_forwards=forwards)
            summaries.append(item)
            generation_tokens += tokens
            generation_forwards += forwards
    result = {'analysis_sha256': binding['analysis'], 'quality_passed': True,
            'sampling_report_sha256': binding['sampling_report'],
            'owner_report_sha256': binding['owner_report'],
            'sampling_tensor_sha256': sampling['raw_tensor_sha256'], 'owner_tensor_sha256': owner['raw_tensor_sha256'],
            'source719_owner0_conditional': check_conditional(sampling['conditional'], old['frame:720']['logits'][0, 0],
                {label: int(old['r719:' + label]['logits'][0, 3].argmax()) for label in ('S0P0', 'S1P0', 'S0P1', 'S1P1')}),
            'source717_owner1_conditional': check_conditional(owner['conditional'], old['frame:719']['logits'][0, 0],
                {label: int(tensors['r717:' + label]['logits'][0, 5].argmax()) for label in ('S0P0', 'S1P0', 'S0P1', 'S1P1')}),
            'owner1_proxy_only_reproduces_actual_graft_seed_logits_bitwise': True,
            'owner1_first_seed_rows': [5, 6], 'owner1_next_seed_root': 719,
            'owner1_greedy_correct_target': owner['blocks'][0]['first_seed']['target_id'],
            'full_sampled_paths': summaries,
            'AR_paths_differ_between_seeds': sampling['units'][0]['AR']['output_ids'] != sampling['units'][1]['AR']['output_ids'],
            'sampled_generation_new_tokens': generation_tokens, 'sampled_generation_forwards': generation_forwards,
            'owner_control_new_tokens': sum(c['replay']['stats']['new_tokens'] for c in owner['cases']),
            'owner_control_generation_forwards': sum(c['replay']['observer_forward_calls'] for c in owner['cases']),
            'owner_fixed_block_forwards': 12, 'teacher_forwards': 0, 'conditional_draws': 256,
            'scope': 'Conditional first-slot acceptance on two old contexts is separate from two newly sampled full paths.'}
    result['legacy_margin_comparisons'] = margin_checks
    return result


def analyze_sampled_depth(data, tensors, eos, binding):
    import torch
    margin_checks: list[dict[str, Any]] = []
    records, units = [], []
    for unit in data['units']:
            prefix = f'q{unit["qid"]}s{unit["seed"]}'
            cases = {c['label']:c['replay'] for c in unit['cases']}
            if not cases:
                units.append({'qid':unit['qid'],'seed':unit['seed'],'infeasible':True})
                continue
            traces = {c:{t['root_position']:t for t in cases[c]['trace']} for c in ('self','graft')}
            common, exclusive = [], []
            output = unit['AR']['output_ids']
            for root in unit['targets']:
                keys = [f'{prefix}:{c}:{root}' for c in ('self','graft')]
                if not all(k in tensors for k in keys):
                    exclusive.append({'root':root, 'self_captured':keys[0] in tensors,'graft_captured':keys[1] in tensors})
                    continue
                a,b = (tensors[k] for k in keys)
                assert a['seed_positions'] == b['seed_positions'] == [root,root+1]
                ca,cb = (s['seed_logits'].argmax(-1).tolist() for s in (a,b))
                first,second = output[root+1] if root+1 < len(output) else None, output[root+2] if root+2 < len(output) else None
                na,nb = traces['self'][root]['accepted_nodes'], traces['graft'][root]['accepted_nodes']
                legal = legal_second_flip(ca[0],cb[0],first,second,eos,na,nb)
                state = {k:{'bit_equal':torch.equal(a['current_state'][k],b['current_state'][k]),
                    'max_abs_delta':float((a['current_state'][k]-b['current_state'][k]).abs().max())} for k in ('anchor','velocity','proxies')}
                row = {'qid':unit['qid'],'seed':unit['seed'],'root':root,'source_roots':[a['source_root'],b['source_root']],
                    'source_owners':[a['source_owner'],b['source_owner']], 'self_candidates':ca,'graft_candidates':cb,
                    'first_target':first,'second_target':second,'self_nodes':na,'graft_nodes':nb,
                    'legal_second_depth_flip':legal,'states':state,
                    'equal_current_state_counterexample':legal and state['anchor']['bit_equal'] and state['velocity']['bit_equal'],
                    'exact_root_hidden_bit_equal':torch.equal(a['exact_root_hidden'],b['exact_root_hidden'])}
                records.append(row)
                common.append(root)
            units.append({'qid':unit['qid'],'seed':unit['seed'],'common_captured':common,'exclusive_or_unvisited':exclusive})
    counters = [r for r in records if r['equal_current_state_counterexample']]
    result = {'quality_passed':True,'source_binding':binding,'source_digest':SOURCE_DIGEST,
            'H10': 'falsified_in_fixed_capture' if counters else 'not_falsified_in_fixed_capture',
            'legal_events':[r for r in records if r['legal_second_depth_flip']], 'equal_state_counterexamples':counters,
            'units':units,'all_captured_shared_records':records,'new_model_forwards':0,'new_draws':0,
            'scope':'Current state equality is not full canonical or equal-KV causal isolation; finite six-root capture only.'}
    result['legacy_margin_comparisons'] = margin_checks
    return result
