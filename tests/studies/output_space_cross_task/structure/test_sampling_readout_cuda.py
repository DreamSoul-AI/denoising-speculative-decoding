"""Tiny saved-tensor graft and exact probability audit on actual CUDA; no model."""

import copy

import torch

from studies._shared.structure.extension_diagnostics import choose_window
from studies._shared.structure.sampling_diagnostics import detectability
from studies.output_space_cross_task.structure import sampling_readout as protocol


def test_saved_expansion_audits_grafted_masks_and_rare_probabilities_on_cuda(record_property):
    assert torch.cuda.is_available(), 'saved tensor audit requires real CUDA; no CPU fallback'
    device = torch.device('cuda:0')
    keys = [(qid, seed) for qid in (241, 242, 243, 244) for seed in (123, 124, 125)]
    def unit(qid, seed):
        native = [{'case': name, 'output_equal_independent_sampled_AR': True, 'output_ids': [1, 2, 3],
            'stats': {'trace': [], 'new_tokens': 3, 'decode_calls': 2, 'accepted': 0, 'committed': 2, 'hit_context_limit': False,
                      'attempt_by_depth': {1: 2}, 'accept_by_depth': {1: 0}}}
            for name in ('output_order0', 'output_order1')]
        return {'qid': qid, 'seed': seed, 'temperature': .8, 'selection': choose_window([], []),
            'selection_sha256': 'b' * 64, 'native': native, 'AR': {'output_ids': [1, 2, 3],
            'stats': {'ar_new_tokens': 3, 'ar_calls': 3}}, 'cases': [], 'observed': []}
    units = [unit(qid, seed) for qid, seed in keys]
    hit = units[2]
    trace = {'root_position': 10, 'cache_before': 10, 'cache_after': 11, 'accepted_nodes': [],
             'committed_tokens': [1], 'stopped': False}
    for native in hit['native']:
        native['stats']['trace'] = [dict(trace)]
    hit['selection'] = choose_window([trace], [trace])
    for label, accepted in [('self', 0), ('graft', 1)]:
        replay_trace = dict(trace)
        if accepted:
            replay_trace.update(accepted_nodes=[1], cache_after=12, committed_tokens=[1, 2])
        hit['cases'].append({'label': label, 'replay': {'output_equal_independent_AR': True,
            'pre_intervention_trace_equal': True, 'trace': [replay_trace], 'observer_forward_calls': 1,
            'stats': {**hit['native'][0]['stats'], 'accepted': accepted}}})
    original = torch.zeros((2, 2, 3), device=device)
    donor = torch.ones_like(original)
    actual = original.clone()
    actual[0, 1] = donor[0, 1]
    prefix = f"q{hit['qid']}s{hit['seed']}"
    tensors = {f'{prefix}:baseline:output_order1:10': {'current_masks': original},
               f'{prefix}:baseline:output_order0:10': {'current_masks': donor}, f'{prefix}:graft:10': {'current_masks': actual}}
    logits = torch.tensor([[[2., 1., 0.]]], device=device)
    groups = []
    for temperature in (.8, 1.):
        probabilities = torch.softmax(logits[0, 0] / temperature, dim=-1)
        cells = [{'candidate': token, 'exact_candidate_probability': float(probabilities[token]),
                  'first_slot_hits': 4096 if token == 0 else 0, **detectability(float(probabilities[token]), 4096)} for token in (0, 2)]
        groups.append({'temperature': temperature, 'seeds': list(range(300, 4396)), 'RNG_consumed': True,
            'verifier_draws': 4096, 'verifier_token_ids': [0] * 4096, 'cells': cells})
    result = protocol.analyze_expansion({'units': units[2:], 'raw_tensor_sha256': 'fixture'}, {'groups': groups},
        {'units': units[:2]}, tensors, {'frame:719': {'logits': logits}}, dict.fromkeys(keys, 'b' * 64),
        {'analysis': 'current', 'expansion_report': 'runtime', 'rare_report': 'rare', 'old_report': 'old'}, device)
    assert result['positive_units'] == 1 and result['infeasible_units'] == 11
    assert result['rare_native_verifier_draws'] == 8192 and len(result['rare_checks']) == 2
    assert torch.equal(original, torch.zeros_like(original))
    damaged = copy.deepcopy(tensors)
    damaged[f'{prefix}:graft:10']['current_masks'][0, 0, 0] = 9
    try:
        protocol.analyze_expansion({'units': units[2:], 'raw_tensor_sha256': 'fixture'}, {'groups': groups},
            {'units': units[:2]}, damaged, {'frame:719': {'logits': logits}}, dict.fromkeys(keys, 'b' * 64),
            {'analysis': 'current', 'expansion_report': 'runtime', 'rare_report': 'rare', 'old_report': 'old'}, device)
    except AssertionError:
        pass
    else:
        raise AssertionError('modified unselected mask must fail the saved graft audit')
    record_property('cuda_device', torch.cuda.get_device_name(device))
