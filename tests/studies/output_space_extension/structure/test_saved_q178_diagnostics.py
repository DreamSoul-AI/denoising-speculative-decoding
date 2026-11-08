"""Tiny saved CUDA tensors exercise the three audits; no model or profiler runs."""
from copy import deepcopy

import pytest

from studies.output_space_extension.structure import saved_diagnostics as protocol


@pytest.mark.parametrize('kind', ['numerics', 'kernels', 'operators'])
def test_saved_q178_audits_use_real_cuda_state_and_reject_corrupt_captures(kind):
    import torch
    assert torch.cuda.is_available(), 'saved q178 correctness requires actual CUDA'
    device = torch.device('cuda:0')
    print(f'saved q178 {kind} tensor.device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    state = {'anchor': torch.zeros(2, device=device), 'velocity': torch.zeros(2, device=device),
             'proxies': torch.zeros(2, 2, device=device)}
    sample = {'source_root': 85, 'source_owner': 0, 'seed_logits': torch.tensor([[0., 4., 1.], [0., 4., 1.]], device=device),
        'current_state': deepcopy(state), 'current_mask_hidden': torch.zeros(3, 2, 2, device=device),
        'seed_mask_inputs': torch.zeros(2, 2, device=device), 'exact_root_hidden': torch.ones(2, device=device)}
    tokens = [1] * 112
    native = {'runs': [{'case': 'AR', 'raw': {'evidence': [{'output_ids': tokens}] * 4}},
        {'case': 'output_order1', 'raw': {'evidence': [{'stats': {'trace': [
            {'root_position': 86, 'accepted_nodes': [1, 2]}]}}] * 4}}]}
    binding = {'analysis': 'current tool', 'report': 'original report', 'tensor': 'original tensor',
               'historical_helper': 'original helper', 'norm_report': 'norm report', 'qproj_report': 'qproj report'}
    if kind == 'numerics':
        labels = ['strict_one_from_zero_1', 'strict_one_from_zero_2', 'strict_zero_from_one_1',
                  'strict_zero_from_one_2', 'canonical_one_from_zero', 'canonical_zero_from_one']
        trace = [{'root_position': 86, 'accepted_nodes': [1]}]
        cases, tensors = [], {}
        for label in labels:
            cases.append({'label': label, 'recipient': 'output_order1' if 'one_from' in label else 'output_order0',
                'canonical_committed_state': False, 'replay': {'visited_targets': [89], 'trace': trace,
                                                             'stats': {'accepted': 1, 'decode_calls': 1}}})
            value = deepcopy(sample); value['current_state']['anchor'].fill_(1)
            tensors[label + ':89'] = value
        tensors['baseline:output_order1:89'] = deepcopy(sample)
        tensors['strict_one_from_zero_1:88'] = deepcopy(sample)
        old = {'q178:baseline:output_order1:89': deepcopy(sample)}
        report = {'source_digest': '670c744e0535dcad', 'raw_tensor_sha256': 'original tensor', 'cases': cases}
        before = tensors['strict_one_from_zero_1:89']['current_state']['anchor'].clone()
        result = protocol.analyze_numerics(report, tensors, old, native, binding)
        assert result['strict_repeat_all_selected_tensors_bit_equal'] and len(result['findings']) == 6
        assert result['first_observed_common_state_divergence']['anchor_max_abs_delta'] == 1
        assert torch.equal(before, tensors['strict_one_from_zero_1:89']['current_state']['anchor'])
        tensors['strict_one_from_zero_2:89']['seed_logits'][0, 0] += 1
        with pytest.raises(AssertionError):
            protocol.analyze_numerics(report, tensors, old, native, binding)
    else:
        old = {'canonical_states': {'output_order1': {86: deepcopy(state)}}}
        def group(count):
            cases, tensors = [], {}
            for index in range(count):
                label = 'native' + str(index) if index < 2 else 'alternate' + str(index)
                case = {'label': label, 'graft': bool(index % 2), 'mode': 'native' if index < 2 else 'changed',
                    'replay': {'visited_targets': [86], 'trace': [{'root_position': 86, 'accepted_nodes': [1]}],
                        'output_equal_independent_AR': True, 'pre_intervention_trace_equal': True,
                        'stats': {'accepted': 1, 'decode_calls': 1, 'new_tokens': len(tokens) - 76}, 'observer_forward_calls': 1},
                    'canonical_replacements': [{'canonical_state_bit_equal': True, 'incoming_proxies_unchanged': True}],
                    'mapping_max_abs': 0., 'post_mapping_max_abs': 0., 'entry_vector_changed': False,
                    'q_projection_changes': [{'replacement_bit_equal': False, 'other_rows_bit_equal': True}]}
                cases.append(case); tensors[label + ':86'] = deepcopy(sample)
            return {'cases': cases, 'raw_tensor_sha256': 'original tensor', 'auxiliary_model_forwards': 0,
                'frozen_comparisons': [{'mask_max_abs': 0}], 'native_profile': {'available': False},
                'alternative_profile': {'available': False}, 'block_replays': [
                    {'logits_bit_equal': True, 'final_hidden_bit_equal': True, 'saved_cache_unchanged': True}] * 4,
                'profiler': {'available': False}, 'frozen_mapping_comparisons': []}, tensors
        report, tensors = group(8)
        if kind == 'operators':
            result = protocol.analyze_operators(report, tensors, old, native, binding)
            assert result['quality_passed'] and len(result['cases']) == 8
            assert result['total_new_tokens'] == 8 * (len(tokens) - 76)
            report['cases'][0]['replay']['pre_intervention_trace_equal'] = False
            with pytest.raises(AssertionError):
                protocol.analyze_operators(report, tensors, old, native, binding)
        else:
            qproj, q_tensors = group(6)
            result = protocol.analyze_kernels({'norm': report, 'qproj': qproj},
                {'norm': tensors, 'qproj': q_tensors}, old, native, binding)
            assert result['quality_passed'] and [len(value['cases']) for value in result['groups']] == [8, 6]
            assert result['groups'][0]['numerical_intervention_effective'] is False
            assert result['groups'][1]['numerical_intervention_effective'] is True
            old['canonical_states']['output_order1'][86]['anchor'][0] = 1
            with pytest.raises(AssertionError):
                protocol.analyze_kernels({'norm': report, 'qproj': qproj},
                    {'norm': tensors, 'qproj': q_tensors}, old, native, binding)
    torch.cuda.synchronize(device)


def test_margin_default_cpu_dispatch_is_unchanged_and_explicit_device_is_forwarded(monkeypatch):
    import torch
    from types import SimpleNamespace
    from studies._shared.structure.window_diagnostics import margin
    targets = []
    class MetadataRow(list):
        def detach(self): return self
        def float(self): return self
        def to(self, device): targets.append(str(device)); return self
        def topk(self, count): return SimpleNamespace(values=[3., 1.], indices=[0, 1])
    monkeypatch.setattr(torch, 'softmax', lambda *a, **kw: [.8, .1, .1])
    row = MetadataRow([3., 1., 0.])
    assert margin(row, 0)['target_margin_vs_best_other'] == 2
    assert margin(row, 0, device='cuda:0')['target_probability'] == .8
    assert targets == ['cpu', 'cuda:0']
