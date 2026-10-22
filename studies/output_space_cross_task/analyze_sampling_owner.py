"""Verify stochastic path coupling and actual owner1 conditional controls, no model forwards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, SEMANTIC_KEYS, TRACE_KEYS, margin
from diagnose_extension import choose_window


def check_conditional(data, logits):
    import torch
    result = []
    for group in data:
        assert group['RNG_consumed'] and group['same_verifier_draws_for_all_cells']
        assert group['seeds'] == list(range(200, 264)) and group['verifier_draws'] == 64
        probs = torch.softmax(logits.cuda().float() / group['temperature'], dim=-1).cpu()
        reference64 = torch.softmax(logits.double() / group['temperature'], dim=-1)
        probability_checks = []
        for cell in group['cells']:
            assert cell['first_slot_hits'] == sum(t == cell['candidate'] for t in group['verifier_token_ids'])
            assert float(probs[cell['candidate']]) == cell['exact_candidate_probability']
            probability_checks.append({'label': cell['label'], 'native_cuda_FP32_mass': float(probs[cell['candidate']]),
                'CPU_FP64_mass': float(reference64[cell['candidate']]),
                'mass_absolute_difference': abs(float(probs[cell['candidate']]) - float(reference64[cell['candidate']]))})
        cells = {c['label']: c for c in group['cells']}
        assert cells['S0P0']['candidate'] == cells['S1P0']['candidate']
        assert cells['S0P1']['candidate'] == cells['S1P1']['candidate']
        result.append({'temperature': group['temperature'], 'draws': 64,
            'distinct_verifier_tokens': len(set(group['verifier_token_ids'])), 'cells': group['cells'],
            'probability_checks': probability_checks})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampling', required=True, type=Path)
    parser.add_argument('--owner', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sampling, owner = (read(p / 'diagnosis.json') for p in (args.sampling, args.owner))
    assert sampling['quality_passed'] and owner['quality_passed']
    assert manifest()['digest'] == sampling['source_digest'] == owner['source_digest'] == '670c744e0535dcad'
    for data, folder, tool, plan in [(sampling, args.sampling, 'diagnose_sampling_paths.py', 'SAMPLING_PLAN.md'),
                                     (owner, args.owner, 'diagnose_q241_owner1.py', 'OWNER_PLAN.md')]:
        assert file_hash(folder / 'selected_tensors.pt') == data['raw_tensor_sha256']
        assert file_hash(ROOT / 'tools' / tool) == data['source_binding']['tool']
        assert file_hash(study / 'docs' / plan) == data['source_binding']['plan']
    old = torch.load(ROOT / '.tmp/q241-local-factors-20261015-v1/selected_tensors.pt',
                     weights_only=True, map_location='cpu', mmap=True)
    prior = torch.load(ROOT / '.tmp/q241-canonical-20261014-v1/selected_tensors.pt',
                       weights_only=True, map_location='cpu', mmap=True)
    tensors = torch.load(args.owner / 'selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
    for label, previous in [('r717:S0P0', 'canonical_self'), ('r717:S0P1', 'canonical_graft')]:
        assert torch.equal(tensors[label]['logits'][0, 5:7], prior[previous + ':719']['seed_logits'])
    assert owner['owner'] == 1 and len(owner['blocks']) == 12
    for block in owner['blocks']:
        assert block['root'] == 717 and block['verification_logits_bit_equal']
        value = tensors[block['label']]
        assert margin(value['logits'][0, 5], block['first_seed']['target_id']) == block['first_seed']
        assert torch.equal(value['logits'][:, :3], tensors['frame:717']['logits'][:, :3])
    summaries = []
    generation_tokens = generation_forwards = 0
    for unit in sampling['units']:
        assert unit['temperature'] == .8 and unit['seed'] in (123, 124)
        native = {r['case']: r for r in unit['native']}
        assert all(r['output_equal_independent_sampled_AR'] and r['output_ids'] == unit['AR']['output_ids'] for r in native.values())
        assert choose_window(native['output_order0']['stats']['trace'], native['output_order1']['stats']['trace']) == unit['selection']
        path = args.sampling / f'selection-seed{unit["seed"]}.json'
        assert file_hash(path) == unit['selection_sha256']
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
    result = {'analysis_sha256': file_hash(Path(__file__)), 'quality_passed': True,
        'sampling_report_sha256': file_hash(args.sampling / 'diagnosis.json'),
        'owner_report_sha256': file_hash(args.owner / 'diagnosis.json'),
        'sampling_tensor_sha256': sampling['raw_tensor_sha256'], 'owner_tensor_sha256': owner['raw_tensor_sha256'],
        'source719_owner0_conditional': check_conditional(sampling['conditional'], old['frame:720']['logits'][0, 0]),
        'source717_owner1_conditional': check_conditional(owner['conditional'], old['frame:719']['logits'][0, 0]),
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
    def save(path, data):
        text = json.dumps(data, indent=2, allow_nan=False) + '\n'
        if path.exists():
            assert path.read_text(encoding='utf-8') == text
        else:
            path.write_text(text, encoding='utf-8')
    save(study / 'docs/sampling_owner_analysis_v1.json', result)
    save(study / 'docs/sampling_diagnosis_v1.json', sampling)
    save(study / 'docs/owner1_diagnosis_v1.json', owner)
    print(json.dumps({k: result[k] for k in ('full_sampled_paths', 'AR_paths_differ_between_seeds',
        'sampled_generation_new_tokens', 'sampled_generation_forwards', 'owner_control_new_tokens')}))


if __name__ == '__main__':
    main()
