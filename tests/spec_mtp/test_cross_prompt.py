"""Fixed selection and state comparisons must not hide adverse/null outcomes."""

import torch

from test_window_observer import ROOT


def test_window_selection_uses_common_root_not_forward_index(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_cross_prompt import select_window
    zero = [{'root_position': r, 'accepted_nodes': [1] if r in (5, 7) else []} for r in (3, 5, 6, 7)]
    one = [{'root_position': r, 'accepted_nodes': [1] if r == 7 else []} for r in (3, 4, 6, 7)]
    assert select_window(zero, one) == (6, 7, [6, 7])


def test_state_comparison_retains_changed_velocity_and_negative_margin(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_cross_prompt import compare_window
    baseline = {'seed_positions': [8, 9], 'source_root': 7, 'source_owner': 0,
        'seed_logits': torch.tensor([[1., 3., 2.], [0., 1., 2.]]),
        'seed_mask_inputs': torch.tensor([[1., 0.], [0., 1.]]),
        'current_state': {'anchor': torch.tensor([1., 2.]), 'velocity': torch.zeros(2),
                          'proxies': torch.tensor([[1., 0.], [0., 1.]])}}
    sample = {**baseline, 'current_state': {**baseline['current_state'], 'velocity': torch.tensor([0., 2.])}}
    result = compare_window(sample, baseline, 2)
    assert result['first_slot']['target_margin_vs_best_other'] == -1.
    assert result['states']['anchor']['bit_equal'] and result['first_seed_bit_equal']
    assert not result['states']['velocity']['bit_equal']
    assert result['states']['velocity']['max_abs_delta'] == 2.
