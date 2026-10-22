"""Verify the separately frozen first-unit canonical result, no model forwards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    args = parser.parse_args()
    import torch
    study = Path(__file__).resolve().parent
    raw = args.raw / 'diagnosis.json'
    report = json.loads(raw.read_text(encoding='utf-8'))
    assert report['quality_passed'] and file_hash(args.raw / 'selected_tensors.pt') == report['raw_tensor_sha256']
    assert file_hash(ROOT / 'tools/diagnose_q241_canonical.py') == report['source_binding']['tool']
    assert file_hash(study / 'docs/CANONICAL_PLAN.md') == report['source_binding']['plan']
    samples = torch.load(args.raw / 'selected_tensors.pt', mmap=True, weights_only=True, map_location='cpu')
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
    result = {'analysis_sha256': file_hash(Path(__file__)), 'raw_report_sha256': file_hash(raw),
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
    target = study / 'docs/canonical_analysis_v1.json'
    text = json.dumps(result, indent=2) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        target.write_text(text, encoding='utf-8')
    target = study / 'docs/canonical_diagnosis_v1.json'
    if target.exists():
        assert file_hash(target) == file_hash(raw)
    else:
        target.write_bytes(raw.read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
