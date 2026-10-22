"""Verify completed cross-task and Phi/RMS evidence without model forwards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash
from diagnose_extension import choose_window


def save(path, body):
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'preserve different prior evidence'
    else:
        path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cross', type=Path, required=True)
    parser.add_argument('--phi', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    native = read(study / 'docs/execution_v1.json')
    diagnostic = read(args.cross / 'diagnosis.json')
    assert native['study_quality_passed'] and diagnostic['quality_passed']
    assert manifest()['digest'] == native['source_digest'] == diagnostic['source_digest'] == '670c744e0535dcad'
    assert file_hash(study / 'docs/execution_v1.json') == diagnostic['source_binding']['matrix_sha256']
    assert file_hash(args.cross / 'selected_tensors.pt') == diagnostic['raw_tensor_sha256']
    samples = torch.load(args.cross / 'selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
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
    cross = {'source_digest': native['source_digest'], 'tool_sha256': file_hash(Path(__file__)),
        'raw_report_sha256': file_hash(args.cross / 'diagnosis.json'),
        'raw_tensor_sha256': diagnostic['raw_tensor_sha256'], 'findings': findings,
        'generations': 20, 'new_tokens': tokens, 'model_forwards': forwards, 'quality_passed': True}
    save(study / 'docs/analysis_v1.json', cross)
    target = study / 'docs/transfer_diagnosis_v1.json'
    if target.exists():
        assert file_hash(target) == file_hash(args.cross / 'diagnosis.json')
    else:
        target.write_bytes((args.cross / 'diagnosis.json').read_bytes())

    phi = read(args.phi / 'diagnosis.json')
    assert phi['quality_passed'] and file_hash(args.phi / 'selected_tensors.pt') == phi['raw_tensor_sha256']
    assert file_hash(ROOT / 'tools/diagnose_q178_phi_rms.py') == phi['source_binding']['tool']
    assert file_hash(ROOT / 'studies/output_space_extension/docs/PHI_RMS_PLAN.md') == phi['source_binding']['plan']
    assert file_hash(ROOT / 'studies/output_space_extension/kernel_common.py') == phi['source_binding']['common']
    data = torch.load(args.phi / 'selected_tensors.pt', weights_only=True, mmap=True, map_location='cpu')
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
    result = {'tool_sha256': file_hash(Path(__file__)), 'raw_report_sha256': file_hash(args.phi / 'diagnosis.json'),
        'raw_tensor_sha256': phi['raw_tensor_sha256'], 'cases': summary,
        'new_tokens': sum(c['replay']['stats']['new_tokens'] for c in phi['cases']),
        'model_forwards': sum(c['replay']['observer_forward_calls'] for c in phi['cases']), 'quality_passed': True}
    extension = ROOT / 'studies/output_space_extension/docs'
    save(extension / 'phi_rms_analysis_v1.json', result)
    target = extension / 'phi_rms_diagnosis_v1.json'
    if target.exists():
        assert file_hash(target) == file_hash(args.phi / 'diagnosis.json')
    else:
        target.write_bytes((args.phi / 'diagnosis.json').read_bytes())
    print(json.dumps({'cross_generations': 20, 'cross_tokens': tokens, 'cross_forwards': forwards,
                      'phi_tokens': result['new_tokens'], 'phi_forwards': result['model_forwards']}))


if __name__ == '__main__':
    main()
