"""Bounded saved CUDA state/depth and conditional probability contracts."""
from copy import deepcopy
import pytest

from studies.output_space_cross_task.structure import state_readout as protocol


def test_cuda_saved_depth_preserves_legality_and_equal_state_counterexample(record_property):
    import torch
    assert torch.cuda.is_available(), 'state correctness requires actual CUDA; no CPU fallback'
    device = torch.device('cuda:0')
    sample = {'seed_positions': [0, 1], 'seed_logits': torch.tensor([[0., 3., 1.], [0., 1., 3.]], device=device),
        'current_state': {key: torch.zeros(2, device=device) for key in ('anchor', 'velocity', 'proxies')},
        'source_root': None, 'source_owner': 0, 'exact_root_hidden': torch.zeros(2, device=device)}
    tensors = {'q241s123:self:0': deepcopy(sample), 'q241s123:graft:0': deepcopy(sample)}
    data = {'units': [{'qid': 241, 'seed': 123, 'AR': {'output_ids': [0, 1, 2]}, 'targets': [0],
        'cases': [{'label': label, 'replay': {'trace': [{'root_position': 0, 'accepted_nodes': nodes}]}}
                  for label, nodes in (('self', [1]), ('graft', [1, 2]))]}]}
    before = tensors['q241s123:self:0']['seed_logits'].clone()
    result = protocol.analyze_sampled_depth(data, tensors, [], {})
    assert result['H10'] == 'falsified_in_fixed_capture' and len(result['equal_state_counterexamples']) == 1
    tensors['q241s123:graft:0']['seed_logits'][0] = torch.tensor([0., 1., 3.], device=device)
    assert protocol.analyze_sampled_depth(data, tensors, [], {})['legal_events'] == []
    assert torch.equal(before, tensors['q241s123:self:0']['seed_logits'])
    torch.cuda.synchronize(device)
    record_property('cuda_device', torch.cuda.get_device_name(device))


@pytest.mark.parametrize('fault', ['none', 'truncated_bank', 'duplicate_cell', 'candidate'])
def test_cuda_conditional_labels_current_fp64_without_inventing_cpu_reference(record_property, fault):
    import torch
    assert torch.cuda.is_available(), 'conditional audit requires actual CUDA'
    logits = torch.tensor([3., 1., 0.], device='cuda:0')
    probabilities = torch.softmax(logits, -1)
    tokens = [0] * 40 + [1] * 24
    groups = [{'RNG_consumed': True, 'same_verifier_draws_for_all_cells': True, 'seeds': list(range(200, 264)),
        'verifier_draws': 64, 'temperature': 1., 'verifier_token_ids': tokens,
        'cells': [{'label': label, 'candidate': candidate, 'first_slot_hits': tokens.count(candidate),
                   'exact_candidate_probability': float(probabilities[candidate])}
                  for label, candidate in (('S0P0', 0), ('S1P0', 0), ('S0P1', 1), ('S1P1', 1))]}]
    expected = {cell['label']: cell['candidate'] for cell in groups[0]['cells']}
    result, = protocol.check_conditional(groups, logits, expected)
    assert all(row['CPU_FP64_mass'] is None and row['CUDA_FP64_mass'] > 0 for row in result['probability_checks'])
    saved = protocol.margin(logits, 0)
    checks = []
    protocol.recorded_margin(logits, saved, checks)
    assert checks[0]['numeric_absolute_differences']['target_probability'] == 0
    saved['top1_id'] = 2
    with pytest.raises(AssertionError, match='discrete identities'):
        protocol.recorded_margin(logits, saved, [])
    if fault != 'none':
        damaged = deepcopy(groups)
        group = damaged[0]
        if fault == 'truncated_bank':
            group['verifier_token_ids'].pop()
            for cell in group['cells']:
                cell['first_slot_hits'] = group['verifier_token_ids'].count(cell['candidate'])
        elif fault == 'duplicate_cell':
            group['cells'][0]['label'] = 'S1P0'
        else:
            for cell in group['cells']:
                if cell['label'] in ('S0P0', 'S1P0'):
                    cell.update(candidate=2, first_slot_hits=0, exact_candidate_probability=float(probabilities[2]))
        with pytest.raises(AssertionError):
            protocol.check_conditional(damaged, logits, expected)
    record_property('cuda_device', torch.cuda.get_device_name(0))
