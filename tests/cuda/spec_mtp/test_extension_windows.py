"""Fixed window fallback must retain null prompts and both baseline paths."""

import pytest

import torch


@pytest.mark.parametrize('first_correct,has_second', [(True, True), (False, True), (True, False)])
def test_second_slot_ar_path_requires_correct_first_and_available_target(first_correct, has_second, record_property):
    from studies._shared.structure import extension_diagnostics as analysis
    assert torch.cuda.is_available(), 'second-slot correctness requires actual CUDA'
    device = torch.device('cuda:0')
    print(f'second-slot tensor.device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    record_property('cuda_device', torch.cuda.get_device_name(device))
    tokens = [0, 1, 2, 0] if has_second else [0, 1, 2]
    first = [0., 1., 3.] if first_correct else [3., 1., 0.]
    sample = {'seed_logits': torch.tensor([first, [3., 1., 0.]], device=device)}
    result = analysis.slots(sample, tokens, 1, device=device)
    assert result['second_AR_path_valid'] == (first_correct and has_second)
    assert (result['second'] is not None) == has_second
