"""Recheck measured operator controls and compare legal two-slot margins offline."""

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'studies' / '_shared' / 'structure'))
from diagnose_output_windows import file_hash
from analyze_results import slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    args = parser.parse_args()
    import torch
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    raw = args.raw / 'diagnosis.json'
    data = read(raw)
    tensor = args.raw / 'selected_tensors.pt'
    assert data['quality_passed'] and file_hash(tensor) == data['raw_tensor_sha256']
    frozen_tool = Path(__file__).parent / 'docs/operator_tool_v1.py.txt'
    assert file_hash(frozen_tool) == data['source_binding']['tool']
    measured = ast.parse(frozen_tool.read_text(encoding='utf-8'))
    current = ast.parse((ROOT / 'studies/_shared/structure/diagnose_q178_operators.py').read_text(encoding='utf-8'))
    assert ast.dump(ast.Module(body=measured.body[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=current.body[1:], type_ignores=[])), 'CLI execution logic differs from measured snapshot'
    assert file_hash(Path(__file__).parent / 'docs/OPERATOR_PLAN.md') == data['source_binding']['plan']
    samples = torch.load(tensor, map_location='cpu', weights_only=True)
    native = read(Path(__file__).parent / 'docs/execution_v1.json')
    assert file_hash(Path(__file__).parent / 'docs/execution_v1.json') == data['source_binding']['native']
    ar = next(r for r in native['runs'] if r['case'] == 'AR')['raw']['evidence'][3]['output_ids']
    old = torch.load(ROOT / '.tmp/q178-numerical-20261014-v1/selected_tensors.pt',
                     map_location='cpu', weights_only=True)
    result = {'raw_report_sha256': file_hash(raw), 'raw_tensor_sha256': file_hash(tensor),
              'analysis_sha256': file_hash(Path(__file__)), 'cases': [], 'quality_passed': False}
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
    target = Path(__file__).parent / 'docs/operator_analysis_v2.json'
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == text, 'preserve mismatched prior analysis'
    else:
        target.write_text(text, encoding='utf-8')
    report = Path(__file__).parent / 'docs/operator_diagnosis_v1.json'
    if report.exists():
        assert file_hash(report) == file_hash(raw)
    else:
        report.write_bytes(raw.read_bytes())
    print(json.dumps({'quality_passed': True, 'tokens': total_tokens, 'forwards': total_calls,
                      'all_semantic_paths_retained': all(c['full_semantic_trace_equal_same_kind_native'] for c in result['cases'])}))


if __name__ == '__main__':
    main()
