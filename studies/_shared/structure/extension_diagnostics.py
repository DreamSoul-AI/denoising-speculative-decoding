"""Library helpers folded from one-shot structure scripts."""
def choose_window(zero, one):
    a, b = ({t['root_position']: t for t in trace} for trace in (zero, one))
    common = sorted(set(a) & set(b))
    root, reason = (None, 'no continuing common root')
    for i, hit in enumerate(common):
        if i and a[hit]['accepted_nodes'] and b[hit]['accepted_nodes']:
            previous = common[i - 1]
            if not a[previous]['accepted_nodes'] and (not b[previous]['accepted_nodes']):
                root, reason = (previous, 'before earliest eligible shared hit')
                break
    if root is None:
        for position in common:
            if not a[position]['stopped'] and (not b[position]['stopped']) and (not a[position]['accepted_nodes']) and (not b[position]['accepted_nodes']):
                root, reason = (position, 'earliest continuing common rejection')
                break
    if root is None:
        for position in common:
            if not a[position]['stopped'] and (not b[position]['stopped']):
                root, reason = (position, 'unused-owner control')
                break
    return {'intervention_root': root, 'selection_reason': reason, 'targets': [r for r in sorted(set(a) | set(b)) if root is not None and r >= root]}

def frozen_targets(selection):
    entry = selection['intervention_root']
    later = [p for p in selection['targets'] if p > entry]
    return sorted(set([entry, *later[:3], *later[-2:]]))


def slots(sample, tokens, position):
    import torch
    from studies._shared.structure.window_diagnostics import margin
    first_target = tokens[position + 1]
    first = margin(sample['seed_logits'][0], first_target)
    valid = int(sample['seed_logits'][0].argmax()) == first_target and position + 2 < len(tokens)
    second = margin(sample['seed_logits'][1], tokens[position + 2]) if position + 2 < len(tokens) else None
    return {'first': first, 'second': second, 'second_AR_path_valid': valid,
            'seed_logits_finite': bool(torch.isfinite(sample['seed_logits']).all())}


def analyze_saved_transfers(report, data, raw):
    """Audit preserved observations without model forwards or output writes."""
    import torch
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
    return findings


def analyze_saved_followup(report, data, tensors):
    """Verify fixed-window transfer scope and summarize saved observations only."""
    import torch
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
    return findings


def analyze_small_matrix_transfers(report, raw, old):
    """Verify saved single-slot propagation and state comparisons without forwards."""
    if not __debug__:
        raise RuntimeError('saved tensor audit requires Python assertions enabled')
    import torch
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
    return findings


def analyze_small_matrix_cross_prompt(report, tensors):
    """Recheck preserved cross-prompt traces, propagation and null-hit conditions."""
    if not __debug__:
        raise RuntimeError('saved tensor audit requires Python assertions enabled')
    import torch
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
    return findings
