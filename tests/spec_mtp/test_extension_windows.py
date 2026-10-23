"""Fixed window fallback must retain null prompts and both baseline paths."""

import importlib.util

import pytest
import torch

from test_window_observer import ROOT


def row(root, accepted=(), stopped=False):
    return {'root_position': root, 'accepted_nodes': list(accepted), 'stopped': stopped}


def test_earliest_eligible_hit_and_full_union_including_trailing_roots(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_extension import choose_window
    zero = [row(3), row(4, [1]), row(6, stopped=True)]
    one = [row(3), row(4, [1]), row(6), row(7, stopped=True)]
    assert choose_window(zero, one) == {'intervention_root': 3,
        'selection_reason': 'before earliest eligible shared hit', 'targets': [3, 4, 6, 7]}


def test_no_shared_hit_uses_predeclared_rejection_control(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_extension import choose_window
    selected = choose_window([row(3), row(4, stopped=True)], [row(3), row(4, stopped=True)])
    assert selected['intervention_root'] == 3
    assert selected['selection_reason'] == 'earliest continuing common rejection'


def test_unused_owner_and_no_continuing_window_are_explicit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_extension import choose_window
    selected = choose_window([row(3, [1]), row(5, stopped=True)], [row(3, [1]), row(5, stopped=True)])
    assert selected['intervention_root'] == 3 and selected['selection_reason'] == 'unused-owner control'
    selected = choose_window([row(3, stopped=True)], [row(3, stopped=True)])
    assert selected['intervention_root'] is None and selected['targets'] == []


@pytest.mark.parametrize('first_correct,has_second', [(True, True), (False, True), (True, False)])
def test_second_slot_ar_path_requires_correct_first_and_available_target(first_correct, has_second):
    definition = importlib.util.spec_from_file_location('extension_analysis', ROOT / 'studies/output_space_extension/analyze_results.py')
    analysis = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(analysis)
    tokens = [0, 1, 2, 0] if has_second else [0, 1, 2]
    first = [0., 1., 3.] if first_correct else [3., 1., 0.]
    sample = {'seed_logits': torch.tensor([first, [3., 1., 0.]])}
    result = analysis.slots(sample, tokens, 1)
    assert result['second_AR_path_valid'] == (first_correct and has_second)
    assert (result['second'] is not None) == has_second
