"""Recheck fixed-block timing and the four-unit state extension without new forwards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, margin
from analyze_state_factors import interaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', required=True, type=Path)
    parser.add_argument('--extension', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from studies._shared.structure.sources import manifest
    study = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    local, extension = (read(p / 'diagnosis.json') for p in (args.local, args.extension))
    assert local['quality_passed'] and extension['quality_passed']
    assert manifest()['digest'] == local['source_digest'] == extension['source_digest'] == '670c744e0535dcad'
    for data, folder, tool, plan in [(local, args.local, 'diagnose_q241_local_factors.py', 'LOCAL_FACTOR_PLAN.md'),
                                     (extension, args.extension, 'diagnose_state_extension.py', 'STATE_EXTENSION_PLAN.md')]:
        assert file_hash(folder / 'selected_tensors.pt') == data['raw_tensor_sha256']
        assert file_hash(ROOT / 'tools' / tool) == data['source_binding']['tool']
        assert file_hash(study / 'docs' / plan) == data['source_binding']['plan']
    a = torch.load(args.local / 'selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
    b = torch.load(args.extension / 'selected_tensors.pt', weights_only=True, map_location='cpu', mmap=True)
    prior = torch.load(ROOT / '.tmp/q241-canonical-20261014-v1/selected_tensors.pt',
                       weights_only=True, map_location='cpu', mmap=True)
    assert len(local['blocks']) == 24 and len(local['cases']) == 2
    groups = {}
    for record in local['blocks']:
        assert record['verification_logits_bit_equal'] and record['verification_hidden_bit_equal']
        groups.setdefault(record['label'], []).append(record)
        tensor = a[record['label']]
        frame = a[f'frame:{record["root"]}']
        assert torch.equal(tensor['logits'][:, :3], frame['logits'][:, :3])
        assert torch.equal(tensor['hidden'][:, :3], frame['hidden'][:, :3])
        assert float((tensor['masks'] - frame['inputs'][:, 3:].reshape(3, 2, -1)).abs().max()) == record['mask_max_abs']
        assert margin(tensor['logits'][0, 3], record['first_seed']['target_id']) == record['first_seed']
    for records in groups.values():
        assert sorted(r['repeat'] for r in records) == [0, 1, 2]
        assert all({k: v for k, v in r.items() if k != 'repeat'} ==
                   {k: v for k, v in records[0].items() if k != 'repeat'} for r in records)
    for label, old in [('r719:S0P0', 'canonical_self'), ('r719:S0P1', 'canonical_graft')]:
        assert torch.equal(a[label]['logits'][0, 3:5], prior[old + ':720']['seed_logits'])
    assert int(a['frame:720']['logits'][0, 0].argmax()) == 264
    timing = {'steps': local['steps'], 'fixed_blocks': [r[0] for r in groups.values()],
        'source719_proxy_only_reproduces_actual_graft_seed_logits_bitwise': True,
        'source719_state_only_mask_changed': groups['r719:S1P0'][0]['mask_max_abs'] > 0,
        'source719_state_only_first_candidate_unchanged':
            groups['r719:S1P0'][0]['first_seed']['top1_id'] == groups['r719:S0P0'][0]['first_seed']['top1_id'],
        'current720_verification_bit_equal_all_cells': True,
        'new_tokens': sum(c['replay']['stats']['new_tokens'] for c in local['cases']),
        'generation_forwards': sum(c['replay']['observer_forward_calls'] for c in local['cases']),
        'block_forwards': 24, 'teacher_forwards': 0}
    native = read(study / 'docs/execution_v1.json')
    rows = {r['case']: r for r in native['runs']}
    eos = read(study / 'runs' / rows['AR']['run_id'] / 'assets/decode_inputs.json')['eos_ids']
    summaries, depth2 = [], []
    for unit in extension['units']:
        prefix = f'q{unit["qid"]}s{unit["seed"]}:'
        cells = {c['label']: c for c in unit['cases']}
        table = []
        for suffix in ('self', 'graft'):
            counts = {mode: cells[f'{mode}_{suffix}']['replay']['stats']['accepted']
                      for mode in ('native', 'anchor', 'velocity', 'both')}
            table.append({'donor': suffix, 'accepted': counts, 'interaction': interaction(counts)})
        ar = rows['AR']['raw']['evidence'][native['question_ids'].index(unit['qid'])]['output_ids']
        def sample(label, p):
            key = prefix + ('baseline:output_order1' if label == 'native_self' else label) + f':{p}'
            return b[key] if key in b else None
        pairs = []
        for mode in ('native', 'anchor', 'velocity', 'both'):
            left, right = (cells[f'{mode}_{suffix}']['replay'] for suffix in ('self', 'graft'))
            pairs.append({'mode': mode, 'equal_accepted_counts': left['stats']['accepted'] == right['stats']['accepted'],
                          'full_semantic_trace_equal': left['trace'] == right['trace']})
            tl, tr = ({t['root_position']: t for t in replay['trace']} for replay in (left, right))
            for p in sorted(tl.keys() & tr.keys()):
                x, y = sample(mode + '_self', p), sample(mode + '_graft', p)
                if x is None or y is None or p + 2 >= len(ar) or ar[p + 1] in eos:
                    continue
                if all(int(s['seed_logits'][0].argmax()) == ar[p + 1] for s in (x, y)):
                    if (len(tl[p]['accepted_nodes']) >= 2) != (len(tr[p]['accepted_nodes']) >= 2):
                        depth2.append({'qid': unit['qid'], 'seed': unit['seed'], 'mode': mode, 'root': p})
        for cell in unit['cases']:
            mode = cell['mode']
            if mode == 'native':
                continue
            records = cell['state_replacements']
            assert all(r['incoming_proxies_unchanged'] for r in records)
            assert all(r['canonical_state_bit_equal'] if mode == 'both' else
                       r['other_state_bit_equal'] and r['selected_state_bit_equal'] for r in records)
            for p in cell['replay']['visited_targets']:
                s = sample(cell['label'], p)
                if p > unit['entry']:
                    for coordinate in ('anchor', 'velocity') if mode == 'both' else (mode,):
                        assert torch.equal(s['current_state'][coordinate], b[prefix + 'canonical_states'][p][coordinate])
        forwards = sum(r['replay']['observer_target_forward_calls'] for r in unit['baselines'])
        forwards += sum(c['replay']['observer_forward_calls'] for c in unit['cases'] if c['label'] != 'native_self')
        summaries.append({'qid': unit['qid'], 'seed': unit['seed'], 'entry': unit['entry'],
            'targets': unit['targets'], 'counts': table, 'self_graft_pairs': pairs,
            'generations': 9, 'new_tokens': 900, 'generation_forwards': forwards,
            'teacher_positions': unit['teacher_positions'], 'teacher_cosine_min': unit['exact_root_cosine_min']})
    seed_checks = []
    for qid in (242, 243):
        u, v = [u for u in extension['units'] if u['qid'] == qid]
        comparisons = []
        for left, right in zip(u['cases'], v['cases']):
            assert left['label'] == right['label']
            assert left['replay']['trace'] == right['replay']['trace']
            assert left['replay']['stats'] == right['replay']['stats']
            for p in left['replay']['visited_targets']:
                x = b[f'q{qid}s{u["seed"]}:' + ('baseline:output_order1' if left['label'] == 'native_self' else left['label']) + f':{p}']
                y = b[f'q{qid}s{v["seed"]}:' + ('baseline:output_order1' if right['label'] == 'native_self' else right['label']) + f':{p}']
                equal = all(torch.equal(x[k], y[k]) for k in ('seed_logits', 'current_masks'))
                comparisons.append({'label': left['label'], 'root': p, 'seed_logits_masks_bit_equal': equal})
        seed_checks.append({'qid': qid, 'seeds': [u['seed'], v['seed']], 'semantic_traces_stats_equal': True,
            'captured_seed_logits_masks_all_bit_equal': all(c['seed_logits_masks_bit_equal'] for c in comparisons)})
    result = {'analysis_sha256': file_hash(Path(__file__)), 'local_raw_report_sha256': file_hash(args.local / 'diagnosis.json'),
        'extension_raw_report_sha256': file_hash(args.extension / 'diagnosis.json'),
        'local_tensor_sha256': local['raw_tensor_sha256'], 'extension_tensor_sha256': extension['raw_tensor_sha256'],
        'quality_passed': True, 'local_timing': timing, 'extension': summaries, 'seed_checks': seed_checks,
        'H5_universal_nonzero_interaction_in_fixed_extension': 'falsified' if any(t['interaction'] == 0 for u in summaries for t in u['counts']) else 'not_falsified',
        'legal_depth2_witnesses_in_six_fixed_targets': depth2, 'teacher_forwards': 4,
        'extension_generations': 36, 'extension_new_tokens': sum(u['new_tokens'] for u in summaries),
        'extension_generation_forwards': sum(u['generation_forwards'] for u in summaries),
        'scope': 'Two additional prompts; greedy seeds are replay controls, not independent stochastic samples.'}
    def save(path, data):
        text = json.dumps(data, indent=2, allow_nan=False) + '\n'
        if path.exists():
            assert path.read_text(encoding='utf-8') == text
        else:
            path.write_text(text, encoding='utf-8')
    save(study / 'docs/local_extension_analysis_v1.json', result)
    save(study / 'docs/local_factor_diagnosis_v1.json', local)
    save(study / 'docs/state_extension_diagnosis_v1.json', extension)
    print(json.dumps({k: result[k] for k in ('H5_universal_nonzero_interaction_in_fixed_extension',
                      'seed_checks', 'extension_new_tokens', 'extension_generation_forwards')}))


if __name__ == '__main__':
    main()
