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
