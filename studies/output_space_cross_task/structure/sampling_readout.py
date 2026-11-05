"""Pure saved sampling/trace audits; no model calls, source lookup or artifact writes."""
import json
from studies._shared.structure.window_diagnostics import TRACE_KEYS, SEMANTIC_KEYS
from studies._shared.structure.extension_diagnostics import choose_window
from studies._shared.structure.sampling_diagnostics import detectability, event_summary

SOURCE_DIGEST = '670c744e0535dcad'



def path_accounting(left, right, left_accepted, right_accepted):
    a, b = ([{k: t[k] for k in TRACE_KEYS} for t in trace] for trace in (left, right))
    assert sum(len(t['accepted_nodes']) for t in a) == left_accepted
    assert sum(len(t['accepted_nodes']) for t in b) == right_accepted
    am, bm = ({t['root_position']: t for t in trace} for trace in (a, b))
    assert len(am) == len(a) and len(bm) == len(b)
    assert [t['root_position'] for t in a] == sorted(am) and [t['root_position'] for t in b] == sorted(bm)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None:
        assert a == b and left_accepted == right_accepted
        first_root, first_delta = None, 0
    else:
        assert a[first]['root_position'] == b[first]['root_position']
        first_root = a[first]['root_position']
        first_delta = len(b[first]['accepted_nodes']) - len(a[first]['accepted_nodes'])
    common = sorted(am.keys() & bm.keys())
    changes = [{'root': p, 'delta': len(bm[p]['accepted_nodes']) - len(am[p]['accepted_nodes'])}
               for p in common if len(bm[p]['accepted_nodes']) != len(am[p]['accepted_nodes'])]
    a_only, b_only = sorted(am.keys() - bm.keys()), sorted(bm.keys() - am.keys())
    common_delta = sum(c['delta'] for c in changes)
    exclusive_delta = sum(len(bm[p]['accepted_nodes']) for p in b_only) - sum(len(am[p]['accepted_nodes']) for p in a_only)
    net = right_accepted - left_accepted
    assert common_delta + exclusive_delta == net
    sign = lambda v: (v > 0) - (v < 0)
    return {'first_divergence_root': first_root, 'first_accepted_delta': first_delta, 'final_accepted_delta': net,
        'direction_prediction_preserved': sign(first_delta) == sign(net), 'common_root_changes': changes,
        'common_root_delta': common_delta, 'exclusive_root_delta': exclusive_delta,
        'self_only_roots': a_only, 'graft_only_roots': b_only,
        'first_self': None if first is None else a[first], 'first_graft': None if first is None else b[first]}


def summarize_unit(unit, selection_sha256, development):
    native = {r['case']: r for r in unit['native']}
    assert unit['temperature'] == .8
    assert all(r['output_ids'] == unit['AR']['output_ids'] and r['output_equal_independent_sampled_AR'] for r in native.values())
    assert choose_window(native['output_order0']['stats']['trace'], native['output_order1']['stats']['trace']) == unit['selection']
    assert selection_sha256 == unit['selection_sha256']
    tokens = unit['AR']['stats']['ar_new_tokens'] + sum(r['stats']['new_tokens'] for r in native.values())
    forwards = unit['AR']['stats']['ar_calls'] + sum(1 + r['stats']['decode_calls'] for r in native.values())
    row = {'qid': unit['qid'], 'seed': unit['seed'], 'development_old_unit': development,
        'entry': unit['selection']['intervention_root'], 'AR_new_tokens': unit['AR']['stats']['ar_new_tokens'],
        'native_zero_accepted': native['output_order0']['stats']['accepted'],
        'native_one_accepted': native['output_order1']['stats']['accepted']}
    if row['entry'] is None:
        assert not unit['cases']
        row['status'] = 'infeasible_not_a_null_effect'
    else:
        cases = {c['label']: c['replay'] for c in unit['cases']}
        assert set(cases) == {'self', 'graft'}
        assert all(c['output_equal_independent_AR'] and c['pre_intervention_trace_equal'] for c in cases.values())
        assert cases['self']['trace'] == [{k: t[k] for k in TRACE_KEYS} for t in native['output_order1']['stats']['trace']]
        assert all(cases['self']['stats'][k] == native['output_order1']['stats'][k] for k in SEMANTIC_KEYS)
        row.update(self_accepted=cases['self']['stats']['accepted'], graft_accepted=cases['graft']['stats']['accepted'],
            accepted_delta=cases['graft']['stats']['accepted'] - cases['self']['stats']['accepted'],
            trace_changed=cases['graft']['trace'] != cases['self']['trace'])
        tokens += sum(native[o['case']]['stats']['new_tokens'] for o in unit['observed'])
        tokens += sum(c['stats']['new_tokens'] for c in cases.values())
        forwards += sum(o['replay']['observer_target_forward_calls'] for o in unit['observed'])
        forwards += sum(c['observer_forward_calls'] for c in cases.values())
    row.update(generated_tokens=tokens, model_forwards=forwards)
    return row


def bounded_sign_prediction(rows):
    by_unit = {(r['qid'], r['seed']): r for r in rows}
    expected = {(242, 126): 1, (244, 126): -1}
    results = []
    for unit, sign in expected.items():
        row = by_unit[unit]
        delta = row.get('accepted_delta')
        results.append({'qid': unit[0], 'seed': unit[1], 'expected_sign': sign,
            'actual_delta': delta, 'verdict': 'infeasible_not_scored' if delta is None else
                ('preserved' if delta * sign > 0 else 'falsified')})
    return results


def analyze_first_delta(new, old, summary, binding):
    assert new['quality_passed'] and old['quality_passed'] and summary['quality_passed']
    rows = []
    for development, report in ((False, new), (True, old)):
        for unit in report['units']:
            if unit['selection']['intervention_root'] is None:
                rows.append({'qid': unit['qid'], 'seed': unit['seed'], 'development_old_unit': development, 'status': 'infeasible_not_scored'})
                continue
            cases = {c['label']: c['replay'] for c in unit['cases']}
            self, graft = (cases['self'], cases['graft'])
            facts = path_accounting(self['trace'], graft['trace'], self['stats']['accepted'], graft['stats']['accepted'])
            rows.append({'qid': unit['qid'], 'seed': unit['seed'], 'development_old_unit': development, **facts})
    rows.sort(key=lambda r: (r['qid'], r['seed']))
    counters = [r for r in rows if not r['development_old_unit'] and r.get('direction_prediction_preserved') is False]
    result = {'source_binding': binding, 'quality_passed': True, 'H7_first_local_sign_predicts_final_sign_in_new_units': 'falsified' if counters else 'not_falsified', 'new_counterexamples': [{'qid': r['qid'], 'seed': r['seed'], 'first_root': r['first_divergence_root'], 'first_delta': r['first_accepted_delta'], 'final_delta': r['final_accepted_delta']} for r in counters], 'rows': rows, 'new_model_forwards': 0, 'new_sampling_draws': 0, 'scope': 'Explanatory trace accounting on fixed completed units, not a tensor causal decomposition.'}
    return result


def analyze_expansion(expansion, rare, old, data, contexts, selection_hashes, bindings, device):
    import torch
    if torch.device(device).type != 'cuda':
        raise ValueError('saved tensor audit requires CUDA')
    assert len(expansion['units']) == 10 and len(old['units']) == 2
    rows = [summarize_unit(u, selection_hashes[u['qid'], u['seed']], True) for u in old['units']]
    rows += [summarize_unit(u, selection_hashes[u['qid'], u['seed']], False) for u in expansion['units']]
    rows.sort(key=lambda r: (r['qid'], r['seed']))
    assert {(r['qid'], r['seed']) for r in rows} == {(q, s) for q in (241, 242, 243, 244) for s in (123, 124, 125)}
    assert len(rows) == 12
    for unit in expansion['units']:
        if unit['selection']['intervention_root'] is None:
            continue
        entry = unit['selection']['intervention_root']
        baseline = data[f"q{unit['qid']}s{unit['seed']}:baseline:output_order1:{entry}"]['current_masks']
        donor = data[f"q{unit['qid']}s{unit['seed']}:baseline:output_order0:{entry}"]['current_masks'][0, 1]
        actual = data[f"q{unit['qid']}s{unit['seed']}:graft:{entry}"]['current_masks']
        assert torch.equal(actual[0, 1], donor) and torch.equal(actual[0, 0], baseline[0, 0]) and torch.equal(actual[1:], baseline[1:])
    rare_checks = []
    for group in rare['groups']:
        assert group['seeds'] == list(range(300, 4396)) and group['RNG_consumed'] and (group['verifier_draws'] == 4096)
        probs = torch.softmax(contexts['frame:719']['logits'][0, 0].to(device).float() / group['temperature'], dim=-1)
        for cell in group['cells']:
            assert cell['first_slot_hits'] == sum((x == cell['candidate'] for x in group['verifier_token_ids']))
            p = cell['exact_candidate_probability']
            assert float(probs[cell['candidate']]) == p
            expected = detectability(p, 4096)
            assert all((cell[k] == v for k, v in expected.items()))
        rare_checks.append({'temperature': group['temperature'], 'draws': 4096, 'cells': group['cells'], 'distinct_verifier_tokens': len(set(group['verifier_token_ids']))})
    signs = {str(q): [r['accepted_delta'] for r in rows if r['qid'] == q and 'accepted_delta' in r] for q in (241, 242, 243, 244)}
    ar_outputs = {(u['qid'], u['seed']): tuple(u['AR']['output_ids']) for u in [*old['units'], *expansion['units']]}
    result = {'analysis_sha256': bindings['analysis'], 'expansion_report_sha256': bindings['expansion_report'], 'rare_report_sha256': bindings['rare_report'], 'old_report_sha256': bindings['old_report'], 'raw_tensor_sha256': expansion['raw_tensor_sha256'], 'quality_passed': True, 'rows': rows, 'accepted_deltas_by_prompt': signs, 'positive_units': sum((r.get('accepted_delta', 0) > 0 for r in rows)), 'negative_units': sum((r.get('accepted_delta', 0) < 0 for r in rows)), 'zero_units': sum(('accepted_delta' in r and r['accepted_delta'] == 0 for r in rows)), 'infeasible_units': sum((r['entry'] is None for r in rows)), 'distinct_AR_paths_per_prompt': {str(q): len({ar_outputs[q, s] for s in (123, 124, 125)}) for q in (241, 242, 243, 244)}, 'rare_checks': rare_checks, 'new_generated_tokens': sum((r['generated_tokens'] for r in rows if not r['development_old_unit'])), 'new_model_forwards': sum((r['model_forwards'] for r in rows if not r['development_old_unit'])), 'rare_native_verifier_draws': 8192, 'rare_model_forwards': 0, 'teacher_forwards': 0, 'scope': 'Fixed four-prompt grid and ideal Bernoulli detectability, no population generalization or native tail reachability proof.'}
    return result


def analyze_continuation(expansion, rare, data, context, selection_hashes, bindings, device):
    import torch
    if torch.device(device).type != 'cuda':
        raise ValueError('saved tensor audit requires CUDA')
    expected = [(242, 126), (244, 126), (245, 126), (245, 127), (246, 126), (246, 127)]
    assert [(u['qid'], u['seed']) for u in expansion['units']] == expected
    rows = [summarize_unit(u, selection_hashes[u['qid'], u['seed']], False) for u in expansion['units']]
    semantic = []
    for unit, row in zip(expansion['units'], rows):
        entry = unit['selection']['intervention_root']
        if entry is None:
            continue
        prefix = f"q{unit['qid']}s{unit['seed']}"
        original = data[f'{prefix}:baseline:output_order1:{entry}']['current_masks']
        donor = data[f'{prefix}:baseline:output_order0:{entry}']['current_masks'][0, 1]
        actual = data[f'{prefix}:graft:{entry}']['current_masks']
        assert torch.equal(actual[0, 1], donor) and torch.equal(actual[0, 0], original[0, 0]) and torch.equal(actual[1:], original[1:])
        cases = {c['label']: c['replay'] for c in unit['cases']}
        assert all((c['only_selected_vector_replaced'] and c['replacement_equal_donor'] and (c['modified_provider_calls'] == 1) for c in cases.values()))
        row['path_accounting'] = path_accounting(cases['self']['trace'], cases['graft']['trace'], row['self_accepted'], row['graft_accepted'])
        semantic.append({'qid': unit['qid'], 'seed': unit['seed'], 'AR': unit['AR'], 'native': [{'case': r['case'], 'stats': {k: v for k, v in r['stats'].items() if k != 'trace'}, 'trace': [{k: t[k] for k in TRACE_KEYS} for t in r['stats']['trace']]} for r in unit['native']], 'observed': unit['observed'], 'cases': unit['cases']})
    groups = []
    assert len(rare['groups']) == 3
    for group, (temperature, count) in zip(rare['groups'], ((0.8, 16384), (1.0, 16384), (2.0, 131072))):
        assert group['temperature'] == temperature and group['verifier_draws'] == count
        assert group['seeds'] == list(range(10000, 10000 + count))
        assert group['RNG_consumed'] and group['same_verifier_draws_for_all_cells']
        assert group['role'] == ('changed_temperature_stress' if temperature == 2.0 else 'original_temperature_control')
        probs = torch.softmax(context['frame:719']['logits'][0, 0].to(device).float() / temperature, dim=-1)
        for cell in group['cells']:
            p = cell['exact_candidate_probability']
            assert float(probs[cell['candidate']]) == p
            expected_cell = event_summary(group['verifier_token_ids'], group['seeds'], cell['candidate'], p)
            assert all((cell[k] == v for k, v in expected_cell.items()))
            assert cell['hits'] == cell['first_slot_hits']
        groups.append({k: v for k, v in group.items() if k not in ('seeds', 'verifier_token_ids')})
        groups[-1]['seed_range_inclusive'] = [10000, 9999 + count]
        groups[-1]['distinct_verifier_tokens'] = len(set(group['verifier_token_ids']))
        groups[-1]['draw_bank_sha256'] = __import__('hashlib').sha256(json.dumps(group['verifier_token_ids']).encode()).hexdigest()
    result = {'quality_passed': True, 'source_digest': SOURCE_DIGEST, 'model_digest': expansion['model_digest'], 'runtime': expansion['runtime'], 'analysis_sha256': bindings['analysis'], 'plan_sha256': bindings['plan'], 'expansion_report_sha256': bindings['expansion_report'], 'rare_report_sha256': bindings['rare_report'], 'tensor_sha256': expansion['raw_tensor_sha256'], 'source_binding': {'expansion': expansion['source_binding'], 'rare': rare['source_binding']}, 'rows': rows, 'full_semantic_paths': semantic, 'H8': bounded_sign_prediction(rows), 'H9_changed_temperature_each_candidate_observed': all((c['hits'] > 0 for c in groups[-1]['cells'])), 'rare_groups': groups, 'new_generated_tokens': sum((r['generated_tokens'] for r in rows)), 'new_model_forwards': sum((r['model_forwards'] for r in rows)), 'rare_native_verifier_draws': sum((g['verifier_draws'] for g in groups)), 'teacher_forwards': 0, 'scope': 'Six fixed new units; conditional T=2 stress is not original-temperature tail reachability or full-path evidence.'}
    return result
