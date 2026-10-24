"""Export native evidence and evaluate fixed q241-q244 H1/H3; H2 stays untested.

    python studies/output_space_cross_task/check_hypotheses.py export
    python studies/output_space_cross_task/check_hypotheses.py check --diagnostic .tmp/cross-task-<id>
    python studies/output_space_cross_task/check_hypotheses.py mre
"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import TRACE_KEYS, file_hash

QIDS = [241, 242, 243, 244]
STUDY = Path(__file__).resolve().parent


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
        unit = {'question_id': qid, 'natural_semantic_difference': different, 'transfer_semantic_differences': []}
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


def export():
    import torch  # noqa: F401 - Windows scientific imports require torch first.
    from rpipe.structure.artifact.config import load_config
    from rpipe.structure.artifact.provenance import check_frozen as check_plan
    from rpipe.structure.make.expand import load_study_yaml
    from studies._shared.structure.sources import manifest, check_frozen
    from studies._shared.structure.validation import verify_evidence
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    declaration = load_study_yaml(STUDY)
    source = manifest()
    check_plan(STUDY, declaration)
    check_frozen(STUDY, declaration, source)
    process = read(STUDY / 'process.json')
    assert process['complete'] and process['spec_mtp']['passed']
    rendered = read(STUDY / 'docs/prompts_manifest.json')['rendered_prompts']
    assert [p['question_id'] for p in rendered] == QIDS
    runs = []
    for experiment in read(STUDY / 'index.json')['experiments']:
        assert len(experiment['runs']) == 1
        run = experiment['runs'][0]
        folder = STUDY / 'runs' / run['id']
        config = load_config(folder / 'config.yaml')
        verify_evidence(config, folder / 'assets')
        result = read(folder / 'result.json')
        assert result['status'] == 'succeeded' and result['spec_mtp']['passed']
        inputs = read(folder / 'assets/decode_inputs.json')
        raw_path = folder / 'assets/decode_observations.json'
        raw = read(raw_path)['out']
        assert len(raw['evidence']) == len(inputs['prompts']) == 4
        assert all(a['input_ids'] == b['input_ids'] == c['input_ids']
                   for a, b, c in zip(rendered, inputs['prompts'], raw['evidence']))
        settings = config['algorithm']['config']
        spec = settings.get('spec', {})
        case = 'AR' if settings['task'] == 'ar_baseline' else (
            'frozen_mean_chain' if spec['method'] == 'esp' else 'output_order' + str(spec['output']['order']))
        runs.append({'case': case, 'run_id': run['id'], 'spec': spec, 'raw': raw,
                     'raw_sha256': file_hash(raw_path), 'runtime': inputs['runtime'], 'model_digest': inputs['model_digest']})
    rows = {r['case']: r for r in runs}
    assert len(runs) == len(rows) == 4 and len({r['model_digest'] for r in runs}) == 1
    for index in range(4):
        ar = rows['AR']['raw']['evidence'][index]['output_ids']
        assert all(r['raw']['evidence'][index]['output_ids'] == ar for r in runs)
    body = {'version': config['algorithm']['config']['version'], 'study_quality_passed': True,
        'execution_completed': True, 'question_ids': QIDS, 'source_digest': source['digest'],
        'model_digest': runs[0]['model_digest'], 'runs': runs, 'tool_sha256': file_hash(Path(__file__))}
    target = STUDY / 'docs/execution_v1.json'
    write_new_or_equal(target, body)
    from diagnose_extension import choose_window
    selection = {'matrix_sha256': file_hash(target), 'runner_sha256': file_hash(ROOT / 'tools/diagnose_study_windows.py'),
        'prompts': [{'question_id': qid, **choose_window(rows['output_order0']['raw']['evidence'][i]['stats']['trace'],
            rows['output_order1']['raw']['evidence'][i]['stats']['trace'])} for i, qid in enumerate(QIDS)]}
    write_new_or_equal(STUDY / 'docs/window_selection_v1.json', selection)


def write_new_or_equal(path, body):
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'preserve different prior result'
    else:
        path.write_text(text, encoding='utf-8')


def fixture():
    trace = [{'root_position': 1, 'cache_before': 1, 'cache_after': 3, 'accepted_nodes': [1],
              'committed_tokens': [10, 11], 'stopped': False}]
    ref = {'output_ids': [0, 10, 11, 12], 'stats': {'trace': trace, 'hit_context_limit': False}}
    native = {'question_ids': QIDS, 'study_quality_passed': True, 'runs': [
        {'case': c, 'raw': {'evidence': [copy.deepcopy(ref) for _ in QIDS]}}
        for c in ('AR', 'frozen_mean_chain', 'output_order0', 'output_order1')]}
    diagnostic = {'quality_passed': True, 'prompts': []}
    for qid in QIDS:
        cases = [{'label': a + '_from_' + b, 'recipient': a, 'donor': b,
            'replay': {'trace': copy.deepcopy(trace), 'output_equal_independent_AR': True,
                      'pre_intervention_trace_equal': True, 'placebo_full_semantic_equal': a == b}, 'windows': []}
            for a, b in [('output_order1', 'output_order0'), ('output_order0', 'output_order1'), ('output_order1', 'output_order1')]]
        diagnostic['prompts'].append({'question_id': qid, 'intervention_root': 1, 'cases': cases,
            'baselines': [{'case': case, 'trace': copy.deepcopy(trace), 'replay': {'replayed_output_equal': True, 'semantic_trace_equal': True}} for case in ('output_order0', 'output_order1')]})
    return native, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('export', 'check', 'mre'))
    parser.add_argument('--diagnostic', type=Path)
    args = parser.parse_args()
    if args.stage == 'export':
        export()
    elif args.stage == 'mre':
        n, d = fixture()
        result = verdict(n, d, [])
        assert result['H1'] == 'falsified_bounded_occurrence'
        print(json.dumps({'synthetic_only_no_model_evidence': True, 'verdict': result}))
    else:
        assert args.diagnostic is not None
        import torch  # noqa: F401 - Windows scientific imports require torch first.
        read = lambda p: json.loads(p.read_text(encoding='utf-8'))
        export()  # Recheck source/PLAN/raw gates; rejects different prior export.
        native = read(STUDY / 'docs/execution_v1.json')
        diagnostic = read(args.diagnostic / 'diagnosis.json')
        assert file_hash(STUDY / 'docs/execution_v1.json') == diagnostic['source_binding']['matrix_sha256']
        assert file_hash(args.diagnostic / 'selected_tensors.pt') == diagnostic['raw_tensor_sha256']
        folder = STUDY / 'runs' / next(r['run_id'] for r in native['runs'] if r['case'] == 'AR')
        eos = read(folder / 'assets/decode_inputs.json')['eos_ids']
        result = verdict(native, diagnostic, eos)
        result.update(native_sha256=file_hash(STUDY / 'docs/execution_v1.json'),
            diagnostic_sha256=file_hash(args.diagnostic / 'diagnosis.json'), checker_sha256=file_hash(Path(__file__)))
        write_new_or_equal(STUDY / 'docs/hypotheses_v1.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
