"""CUDA correctness of the saved canonical-state gate, without a model."""
import pytest

from studies.output_space_cross_task.structure.native_readout import analyze_canonical


@pytest.mark.parametrize('changed', [False, True])
def test_canonical_saved_state_gate_uses_cuda_and_rejects_a_changed_anchor(changed):
    import torch
    assert torch.cuda.is_available(), 'CUDA is required; no CPU substitute'
    device = torch.device('cuda:0')
    print(f'canonical saved-state correctness device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    anchor = torch.tensor([1., 2.], device=device)
    velocity = torch.tensor([.5, -.25], device=device)
    cases = []
    samples = {'canonical_states': {717: {'anchor': anchor, 'velocity': velocity}}}
    for i, name in enumerate(('native_self', 'native_graft', 'canonical_self', 'canonical_graft')):
        cases.append({'label': name, 'replay': {'output_equal_independent_AR': True,
            'pre_intervention_trace_equal': True, 'visited_targets': [717], 'trace': [],
            'stats': {'accepted': i+1, 'new_tokens': 3}, 'observer_forward_calls': 2},
            'canonical_replacements': [{'canonical_state_bit_equal': True, 'incoming_proxies_unchanged': True}]})
        samples[f'{name}:717'] = {'current_state': {'anchor': anchor.clone(), 'velocity': velocity.clone()}}
    if changed:
        samples['canonical_graft:717']['current_state']['anchor'][0] += 1
        with pytest.raises(AssertionError):
            analyze_canonical({'quality_passed': True, 'raw_tensor_sha256': 'synthetic', 'cases': cases},
                              samples, {'analysis': 'synthetic', 'report': 'synthetic'})
    else:
        result = analyze_canonical({'quality_passed': True, 'raw_tensor_sha256': 'synthetic', 'cases': cases},
                                   samples, {'analysis': 'synthetic', 'report': 'synthetic'})
        assert result['native_delta'] == result['canonical_delta'] == 1
        assert result['H2_bounded_count_persistence_prediction'] == 'preserved'
