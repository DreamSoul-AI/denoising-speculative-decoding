"""Verify fixed sampled expansion and rare-event power evidence without new model forwards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, SEMANTIC_KEYS, TRACE_KEYS
from diagnose_extension import choose_window
from diagnose_rare_events import detectability


def summarize_unit(unit, directory, development):
    native = {r['case']: r for r in unit['native']}
    assert unit['temperature'] == .8
    assert all(r['output_ids'] == unit['AR']['output_ids'] and r['output_equal_independent_sampled_AR'] for r in native.values())
    assert choose_window(native['output_order0']['stats']['trace'], native['output_order1']['stats']['trace']) == unit['selection']
    selection = directory / (f'selection-seed{unit["seed"]}.json' if development else
                             f'selection-q{unit["qid"]}-seed{unit["seed"]}.json')
    assert file_hash(selection) == unit['selection_sha256']
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion', required=True, type=Path)
    parser.add_argument('--rare', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    expansion, rare = (read(p / 'diagnosis.json') for p in (args.expansion, args.rare))
    old_dir = ROOT / '.tmp/sampling-paths-20261015-v1'
    old = read(old_dir / 'diagnosis.json')
    assert all(r['quality_passed'] for r in (expansion, rare, old))
    assert manifest()['digest'] == expansion['source_digest'] == rare['source_digest'] == '670c744e0535dcad'
    assert file_hash(args.expansion / 'selected_tensors.pt') == expansion['raw_tensor_sha256']
    for data, tool, plan in [(expansion, 'diagnose_sampling_expansion.py', 'SAMPLING_EXPANSION_PLAN.md'),
                             (rare, 'diagnose_rare_events.py', 'RARE_EVENT_PLAN.md')]:
        assert file_hash(ROOT / 'tools' / tool) == data['source_binding']['tool']
        assert file_hash(study / 'docs' / plan) == data['source_binding']['plan']
    assert len(expansion['units']) == 10 and len(old['units']) == 2
    rows = [summarize_unit(u, old_dir, True) for u in old['units']]
    rows += [summarize_unit(u, args.expansion, False) for u in expansion['units']]
    rows.sort(key=lambda r: (r['qid'], r['seed']))
    assert {(r['qid'], r['seed']) for r in rows} == {(q, s) for q in (241, 242, 243, 244) for s in (123, 124, 125)}
    assert len(rows) == 12
    data = torch.load(args.expansion / 'selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
    for unit in expansion['units']:
        if unit['selection']['intervention_root'] is None:
            continue
        entry = unit['selection']['intervention_root']
        baseline = data[f'q{unit["qid"]}s{unit["seed"]}:baseline:output_order1:{entry}']['current_masks']
        donor = data[f'q{unit["qid"]}s{unit["seed"]}:baseline:output_order0:{entry}']['current_masks'][0, 1]
        actual = data[f'q{unit["qid"]}s{unit["seed"]}:graft:{entry}']['current_masks']
        assert torch.equal(actual[0, 1], donor) and torch.equal(actual[0, 0], baseline[0, 0]) and torch.equal(actual[1:], baseline[1:])
    contexts = torch.load(ROOT / '.tmp/q241-local-factors-20261015-v1/selected_tensors.pt',
                          weights_only=True, map_location='cpu', mmap=True)
    rare_checks = []
    for group in rare['groups']:
        assert group['seeds'] == list(range(300, 4396)) and group['RNG_consumed'] and group['verifier_draws'] == 4096
        probs = torch.softmax(contexts['frame:719']['logits'][0, 0].cuda().float() / group['temperature'], dim=-1).cpu()
        for cell in group['cells']:
            assert cell['first_slot_hits'] == sum(x == cell['candidate'] for x in group['verifier_token_ids'])
            p = cell['exact_candidate_probability']
            assert float(probs[cell['candidate']]) == p
            expected = detectability(p, 4096)
            assert all(cell[k] == v for k, v in expected.items())
        rare_checks.append({'temperature': group['temperature'], 'draws': 4096, 'cells': group['cells'],
            'distinct_verifier_tokens': len(set(group['verifier_token_ids']))})
    signs = {str(q): [r['accepted_delta'] for r in rows if r['qid'] == q and 'accepted_delta' in r] for q in (241, 242, 243, 244)}
    ar_outputs = {(u['qid'], u['seed']): tuple(u['AR']['output_ids']) for u in [*old['units'], *expansion['units']]}
    result = {'analysis_sha256': file_hash(Path(__file__)), 'expansion_report_sha256': file_hash(args.expansion / 'diagnosis.json'),
        'rare_report_sha256': file_hash(args.rare / 'diagnosis.json'), 'old_report_sha256': file_hash(old_dir / 'diagnosis.json'),
        'raw_tensor_sha256': expansion['raw_tensor_sha256'], 'quality_passed': True, 'rows': rows,
        'accepted_deltas_by_prompt': signs, 'positive_units': sum(r.get('accepted_delta', 0) > 0 for r in rows),
        'negative_units': sum(r.get('accepted_delta', 0) < 0 for r in rows),
        'zero_units': sum('accepted_delta' in r and r['accepted_delta'] == 0 for r in rows),
        'infeasible_units': sum(r['entry'] is None for r in rows),
        'distinct_AR_paths_per_prompt': {str(q): len({ar_outputs[q, s] for s in (123, 124, 125)}) for q in (241, 242, 243, 244)},
        'rare_checks': rare_checks, 'new_generated_tokens': sum(r['generated_tokens'] for r in rows if not r['development_old_unit']),
        'new_model_forwards': sum(r['model_forwards'] for r in rows if not r['development_old_unit']),
        'rare_native_verifier_draws': 8192, 'rare_model_forwards': 0, 'teacher_forwards': 0,
        'scope': 'Fixed four-prompt grid and ideal Bernoulli detectability, no population generalization or native tail reachability proof.'}
    def save(path, value):
        text = json.dumps(value, indent=2, allow_nan=False) + '\n'
        if path.exists():
            assert path.read_text(encoding='utf-8') == text
        else:
            path.write_text(text, encoding='utf-8')
    save(study / 'docs/sampling_expansion_analysis_v1.json', result)
    save(study / 'docs/sampling_expansion_diagnosis_v1.json', expansion)
    save(study / 'docs/rare_event_diagnosis_v1.json', rare)
    print(json.dumps({k: result[k] for k in ('accepted_deltas_by_prompt', 'positive_units', 'negative_units',
                    'zero_units', 'infeasible_units', 'new_generated_tokens', 'new_model_forwards')}))


if __name__ == '__main__':
    main()
