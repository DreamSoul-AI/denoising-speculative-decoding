"""Pure native cross-task evidence contracts; no generation or current-source re-signing."""
import copy
from typing import Any
from studies._shared.structure.window_diagnostics import TRACE_KEYS
from studies._shared.structure.extension_diagnostics import choose_window

if not __debug__:
    raise RuntimeError('native saved readout requires Python assertions enabled')

QIDS = [241, 242, 243, 244]

def semantic(trace):
    return [{k: row[k] for k in TRACE_KEYS} for row in trace]


def verdict(native, diagnostic, eos):
    assert native['study_quality_passed'] and diagnostic['quality_passed']
    assert native['question_ids'] == QIDS == [p['question_id'] for p in diagnostic['prompts']]
    rows = {r['case']: r for r in native['runs']}
    assert set(rows) == {'AR', 'frozen_mean_chain', 'output_order0', 'output_order1'}
    findings, witnesses, natural_any, transfer_any, infeasible = [], [], False, False, []
    for index, prompt in enumerate(diagnostic['prompts']):
        qid = QIDS[index]
        ar = rows['AR']['raw']['evidence'][index]['output_ids']
        refs = {case: rows[case]['raw']['evidence'][index] for case in ('output_order0', 'output_order1')}
        assert all(ref['output_ids'] == ar and not ref['stats']['hit_context_limit'] for ref in refs.values())
        different = semantic(refs['output_order0']['stats']['trace']) != semantic(refs['output_order1']['stats']['trace'])
        natural_any |= different
        unit: dict[str, Any] = {'question_id': qid, 'natural_semantic_difference': different, 'transfer_semantic_differences': []}
        by_root = {case: {t['root_position']: t for t in ref['stats']['trace']} for case, ref in refs.items()}
        for pair in prompt.get('natural_baseline_margin_pairs', []):
            root = pair['root_position']
            zero, one = pair['zero_slots'], pair['one_slots']
            if root not in by_root['output_order0'] or root not in by_root['output_order1']:
                continue
            a = by_root['output_order0'][root]['accepted_nodes']
            b = by_root['output_order1'][root]['accepted_nodes']
            legal = (zero['second_AR_path_valid'] and one['second_AR_path_valid']
                     and root + 2 < len(ar) and ar[root + 1] not in eos
                     and zero['first']['top1_id'] == one['first']['top1_id'] == ar[root + 1])
            if legal and min(len(a), len(b)) >= 1 and ((len(a) >= 2) != (len(b) >= 2)):
                witnesses.append({'question_id': qid, 'label': 'natural_output0_vs_output1', 'root': root,
                    'first_AR_id': ar[root + 1], 'baseline_accepted_nodes': a, 'actual_accepted_nodes': b})
        if prompt['intervention_root'] is None:
            assert not prompt['cases']
            infeasible.append(qid)
            unit['status'] = 'infeasible fixed window; not a null transfer'
            findings.append(unit)
            continue
        assert len(prompt['baselines']) == 2
        assert {b['case'] for b in prompt['baselines']} == {'output_order0', 'output_order1'}
        assert all(semantic(b['trace']) == semantic(refs[b['case']]['stats']['trace']) for b in prompt['baselines'])
        assert all(b['replay']['replayed_output_equal'] and b['replay']['semantic_trace_equal'] for b in prompt['baselines'])
        assert len(prompt['cases']) == 3
        assert {c['label'] for c in prompt['cases']} == {'output_order1_from_output_order0', 'output_order0_from_output_order1', 'output_order1_from_output_order1'}
        for case in prompt['cases']:
            replay, recipient = case['replay'], case['recipient']
            assert replay['output_equal_independent_AR'] and replay['pre_intervention_trace_equal']
            if recipient == case['donor']:
                assert replay['placebo_full_semantic_equal']
                assert semantic(replay['trace']) == semantic(refs[recipient]['stats']['trace'])
                continue
            base = refs[recipient]['stats']['trace']
            changed = semantic(replay['trace']) != semantic(base)
            transfer_any |= changed
            unit['transfer_semantic_differences'].append({'label': case['label'], 'changed': changed})
            prior = {t['root_position']: t for t in base}
            actual = {t['root_position']: t for t in replay['trace']}
            for window in case['windows']:
                root = window['root_position']
                if not window.get('recipient_baseline_visited') or root not in prior or root not in actual:
                    continue
                a, b = prior[root]['accepted_nodes'], actual[root]['accepted_nodes']
                rank = window['slot_rankings']
                first, before = rank['first'], window['baseline_first_slot']
                legal = (rank['second_AR_path_valid'] and root + 2 < len(ar) and ar[root + 1] not in eos
                         and first['top1_id'] == before['top1_id'] == first['target_id'] == ar[root + 1])
                if legal and min(len(a), len(b)) >= 1 and ((len(a) >= 2) != (len(b) >= 2)):
                    witnesses.append({'question_id': qid, 'label': case['label'], 'root': root,
                        'first_AR_id': ar[root + 1], 'baseline_accepted_nodes': a, 'actual_accepted_nodes': b})
        unit['status'] = 'all fixed comparisons checked'
        findings.append(unit)
    h1 = 'observed' if natural_any or transfer_any else (
        'inconclusive_infeasible_windows' if infeasible else 'falsified_bounded_occurrence')
    return {'H1': h1, 'H3': 'falsified_by_legal_depth2_witness' if witnesses else 'not_falsified_in_captured_native_or_transfer_windows',
        'H2': 'not_tested_requires_new_frozen_canonical_protocol',
        'H4': 'separate_operator_experiment_not_cross_task_evidence',
        'infeasible_question_ids': infeasible, 'findings': findings, 'legal_depth2_witnesses': witnesses}


def fixture():
    trace = [{'root_position': 1, 'cache_before': 1, 'cache_after': 3, 'accepted_nodes': [1],
              'committed_tokens': [10, 11], 'stopped': False}]
    ref = {'output_ids': [0, 10, 11, 12], 'stats': {'trace': trace, 'hit_context_limit': False}}
    native = {'question_ids': QIDS, 'study_quality_passed': True, 'runs': [
        {'case': c, 'raw': {'evidence': [copy.deepcopy(ref) for _ in QIDS]}}
        for c in ('AR', 'frozen_mean_chain', 'output_order0', 'output_order1')]}
    diagnostic: dict[str, Any] = {'quality_passed': True, 'prompts': []}
    for qid in QIDS:
        cases = [{'label': a + '_from_' + b, 'recipient': a, 'donor': b,
            'replay': {'trace': copy.deepcopy(trace), 'output_equal_independent_AR': True,
                      'pre_intervention_trace_equal': True, 'placebo_full_semantic_equal': a == b}, 'windows': []}
            for a, b in [('output_order1', 'output_order0'), ('output_order0', 'output_order1'), ('output_order1', 'output_order1')]]
        diagnostic['prompts'].append({'question_id': qid, 'intervention_root': 1, 'cases': cases,
            'baselines': [{'case': case, 'trace': copy.deepcopy(trace), 'replay': {'replayed_output_equal': True, 'semantic_trace_equal': True}} for case in ('output_order0', 'output_order1')]})
    return native, diagnostic



def analyze_cross(native, diagnostic, samples, bindings):
    import torch
    rows = {r['case']: r for r in native['runs']}
    findings = []
    for i, prompt in enumerate(diagnostic['prompts']):
        qid, entry = prompt['question_id'], prompt['intervention_root']
        chosen = choose_window(rows['output_order0']['raw']['evidence'][i]['stats']['trace'],
                               rows['output_order1']['raw']['evidence'][i]['stats']['trace'])
        assert chosen == {k: prompt[k] for k in chosen}
        cases = []
        for c in prompt['cases']:
            label = c['label']
            a = samples[f'q{qid}:{label}:{entry}']['current_masks']
            b = samples[f'q{qid}:baseline:{c["recipient"]}:{entry}']['current_masks']
            donor = samples[f'q{qid}:baseline:{c["donor"]}:{entry}']['current_masks'][0, 1]
            assert torch.equal(a[0, 1], donor) and torch.equal(a[0, 0], b[0, 0]) and torch.equal(a[1:], b[1:])
            matching = [w for w in c['windows'] if w['recipient_baseline_visited']]
            for w in matching:
                p = w['root_position']
                for state in ('anchor', 'velocity', 'proxies'):
                    x = samples[f'q{qid}:{label}:{p}']['current_state'][state]
                    y = samples[f'q{qid}:baseline:{c["recipient"]}:{p}']['current_state'][state]
                    assert torch.equal(x, y) == w['states'][state]['bit_equal']
            cases.append({'label': label, 'accepted': c['replay']['stats']['accepted'],
                'decode_calls': c['replay']['stats']['decode_calls'],
                'matched_windows': len(matching), 'anchor_velocity_bit_equal_windows': sum(
                    all(w['states'][s]['bit_equal'] for s in ('anchor', 'velocity')) for w in matching),
                'anchor_max_abs_delta': max((w['states']['anchor']['max_abs_delta'] for w in matching), default=0),
                'velocity_max_abs_delta': max((w['states']['velocity']['max_abs_delta'] for w in matching), default=0),
                'matched_acceptance_changes': c['matched_acceptance_changes'],
                'actual_only_roots': c['actual_only_roots'], 'baseline_only_roots': c['baseline_only_roots']})
        findings.append({'question_id': qid, 'entry': entry,
            'native': {case: rows[case]['raw']['evidence'][i]['stats']['accepted']
                       for case in ('frozen_mean_chain', 'output_order0', 'output_order1')}, 'cases': cases})
    tokens = sum(c['replay']['stats']['new_tokens'] for p in diagnostic['prompts'] for c in p['cases'])
    tokens += sum(rows[b['case']]['raw']['evidence'][i]['stats']['new_tokens']
                  for i, p in enumerate(diagnostic['prompts']) for b in p['baselines'])
    forwards = sum(c['replay']['observer_forward_calls'] for p in diagnostic['prompts'] for c in p['cases'])
    forwards += sum(b['replay']['observer_target_forward_calls'] for p in diagnostic['prompts'] for b in p['baselines'])
    cross = {'source_digest': native['source_digest'], 'tool_sha256': bindings['analysis'],
        'raw_report_sha256': bindings['cross_report'],
        'raw_tensor_sha256': diagnostic['raw_tensor_sha256'], 'findings': findings,
        'generations': 20, 'new_tokens': tokens, 'model_forwards': forwards, 'quality_passed': True}
    return cross


def analyze_phi(phi, data, bindings):
    controls = {c['graft']: c for c in phi['cases'] if c['stage'] == 'native'}
    summary = []
    for c in phi['cases']:
        base = controls[c['graft']]
        diffs = []
        for p in c['replay']['visited_targets']:
            if f'{base["label"]}:{p}' not in data:
                continue
            x, y = data[f'{c["label"]}:{p}'], data[f'{base["label"]}:{p}']
            diffs.append(float((x['current_state']['proxies'] - y['current_state']['proxies']).abs().max()))
        summary.append({'label': c['label'], 'accepted': c['replay']['stats']['accepted'],
            'decode_calls': c['replay']['stats']['decode_calls'], 'entry_vector_changed': c['entry_vector_changed'],
            'norm_changes': c['norm_changes'], 'proxy_max_abs_to_control': max(diffs, default=0),
            'full_semantic_trace_equal_control': c['replay']['trace'] == base['replay']['trace']})
    result = {'tool_sha256': bindings['analysis'], 'raw_report_sha256': bindings['phi_report'],
        'raw_tensor_sha256': phi['raw_tensor_sha256'], 'cases': summary,
        'new_tokens': sum(c['replay']['stats']['new_tokens'] for c in phi['cases']),
        'model_forwards': sum(c['replay']['observer_forward_calls'] for c in phi['cases']), 'quality_passed': True}
    return result


def analyze_canonical(report, samples, bindings):
    import torch
    assert report['quality_passed']
    cases = {c['label']: c for c in report['cases']}
    assert set(cases) == {'native_self', 'native_graft', 'canonical_self', 'canonical_graft'}
    for name in ('canonical_self', 'canonical_graft'):
        c = cases[name]
        assert c['replay']['output_equal_independent_AR'] and c['replay']['pre_intervention_trace_equal']
        assert all(r['canonical_state_bit_equal'] and r['incoming_proxies_unchanged'] for r in c['canonical_replacements'])
        for p in c['replay']['visited_targets']:
            if p > 716:
                state = samples[f'{name}:{p}']['current_state']
                assert all(torch.equal(state[k], samples['canonical_states'][p][k]) for k in ('anchor', 'velocity'))
    counts = {name: c['replay']['stats']['accepted'] for name, c in cases.items()}
    native_delta = counts['native_graft'] - counts['native_self']
    canonical_delta = counts['canonical_graft'] - counts['canonical_self']
    result = {'analysis_sha256': bindings['analysis'], 'raw_report_sha256': bindings['report'],
        'raw_tensor_sha256': report['raw_tensor_sha256'], 'question_id': 241, 'entry': 716,
        'accepted': counts, 'native_delta': native_delta, 'canonical_delta': canonical_delta,
        'canonical_self_changed': counts['canonical_self'] != counts['native_self'],
        'canonical_self_graft_full_semantic_trace_equal':
            cases['canonical_self']['replay']['trace'] == cases['canonical_graft']['replay']['trace'],
        'H2_bounded_count_persistence_prediction': 'not_preserved' if canonical_delta != native_delta else 'preserved',
        'pure_committed_microdelta_necessity': 'not_established; synchronization changes self and subsequent proxies',
        'all_output_AR_equal': True, 'canonical_matched_state_gates_passed': True,
        'new_tokens': sum(c['replay']['stats']['new_tokens'] for c in cases.values()),
        'generation_forwards': sum(c['replay']['observer_forward_calls'] for c in cases.values()),
        'teacher_forwards': 1, 'teacher_positions': 792, 'quality_passed': True}
    return result
