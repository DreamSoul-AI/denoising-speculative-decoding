"""Fixed captured-window depth analysis, no new model forwards or selection."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash


def legal_second_flip(first_a, first_b, first_target, second_target, eos, nodes_a, nodes_b):
    return (second_target is not None and first_target not in eos and
            first_a == first_b == first_target and 1 in nodes_a and 1 in nodes_b and
            ((2 in nodes_a) != (2 in nodes_b)))


def main():
    import torch
    from studies._shared.structure.sources import manifest
    docs = Path(__file__).resolve().parent / 'docs'
    directory = ROOT / '.tmp/sampling-continuation-20261015-v1'
    paths = {'tool': Path(__file__), 'plan': docs / 'SAMPLED_DEPTH_PLAN.md',
        'report': directory / 'diagnosis.json', 'tensor': directory / 'selected_tensors.pt'}
    binding = {k:file_hash(p) for k,p in paths.items()}
    data = json.loads(paths['report'].read_text(encoding='utf-8'))
    assert data['quality_passed'] and data['raw_tensor_sha256'] == binding['tensor']
    assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
    tensors = torch.load(paths['tensor'], weights_only=True, map_location='cpu', mmap=True)
    native = json.loads((docs / 'execution_v1.json').read_text(encoding='utf-8'))
    one = next(r for r in native['runs'] if r['case'] == 'output_order1')
    inputs = json.loads((ROOT / 'studies/output_space_cross_task/runs' / one['run_id'] / 'assets/decode_inputs.json').read_text(encoding='utf-8'))
    eos = inputs['eos_ids']
    records, units = [], []
    for unit in data['units']:
        prefix = f'q{unit["qid"]}s{unit["seed"]}'
        cases = {c['label']:c['replay'] for c in unit['cases']}
        if not cases:
            units.append({'qid':unit['qid'],'seed':unit['seed'],'infeasible':True})
            continue
        traces = {c:{t['root_position']:t for t in cases[c]['trace']} for c in ('self','graft')}
        common, exclusive = [], []
        output = unit['AR']['output_ids']
        for root in unit['targets']:
            keys = [f'{prefix}:{c}:{root}' for c in ('self','graft')]
            if not all(k in tensors for k in keys):
                exclusive.append({'root':root, 'self_captured':keys[0] in tensors,'graft_captured':keys[1] in tensors})
                continue
            a,b = (tensors[k] for k in keys)
            assert a['seed_positions'] == b['seed_positions'] == [root,root+1]
            ca,cb = (s['seed_logits'].argmax(-1).tolist() for s in (a,b))
            first,second = output[root+1] if root+1 < len(output) else None, output[root+2] if root+2 < len(output) else None
            na,nb = traces['self'][root]['accepted_nodes'], traces['graft'][root]['accepted_nodes']
            legal = legal_second_flip(ca[0],cb[0],first,second,eos,na,nb)
            state = {k:{'bit_equal':torch.equal(a['current_state'][k],b['current_state'][k]),
                'max_abs_delta':float((a['current_state'][k]-b['current_state'][k]).abs().max())} for k in ('anchor','velocity','proxies')}
            row = {'qid':unit['qid'],'seed':unit['seed'],'root':root,'source_roots':[a['source_root'],b['source_root']],
                'source_owners':[a['source_owner'],b['source_owner']], 'self_candidates':ca,'graft_candidates':cb,
                'first_target':first,'second_target':second,'self_nodes':na,'graft_nodes':nb,
                'legal_second_depth_flip':legal,'states':state,
                'equal_current_state_counterexample':legal and state['anchor']['bit_equal'] and state['velocity']['bit_equal'],
                'exact_root_hidden_bit_equal':torch.equal(a['exact_root_hidden'],b['exact_root_hidden'])}
            records.append(row)
            common.append(root)
        units.append({'qid':unit['qid'],'seed':unit['seed'],'common_captured':common,'exclusive_or_unvisited':exclusive})
    counters = [r for r in records if r['equal_current_state_counterexample']]
    result = {'quality_passed':True,'source_binding':binding,'source_digest':manifest()['digest'],
        'H10': 'falsified_in_fixed_capture' if counters else 'not_falsified_in_fixed_capture',
        'legal_events':[r for r in records if r['legal_second_depth_flip']], 'equal_state_counterexamples':counters,
        'units':units,'all_captured_shared_records':records,'new_model_forwards':0,'new_draws':0,
        'scope':'Current state equality is not full canonical or equal-KV causal isolation; finite six-root capture only.'}
    assert all(file_hash(p) == binding[k] for k,p in paths.items())
    path = docs / 'sampled_depth_analysis_v1.json'
    text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text
    else:
        path.write_text(text,encoding='utf-8')
    print(json.dumps({'H10':result['H10'],'shared_records':len(records),'legal_events':len(result['legal_events']),
        'equal_state_counterexamples':[(r['qid'],r['seed'],r['root']) for r in counters]}))


if __name__ == '__main__':
    main()
