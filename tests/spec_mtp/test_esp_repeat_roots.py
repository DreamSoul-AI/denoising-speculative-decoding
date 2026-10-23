"""Observed repeat exclusions are distinct from full-path pruning-off gains."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('esp_repeats', Path(__file__).resolve().parents[2] / 'tools/analyze_esp_repeat_roots.py')
repeats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repeats)


def test_terminal_repeat_is_not_an_extra_nonterminal_opportunity():
    observation = {'output_ids': [7, 7, 8], 'stats': {'trace': [
        dict(root_position=0, committed_tokens=[7], accepted_nodes=[], stopped=False),
        dict(root_position=1, committed_tokens=[8], accepted_nodes=[1], stopped=False),
        dict(root_position=2, committed_tokens=[8], accepted_nodes=[], stopped=True)]}}
    counts = repeats.count_repeats([observation])
    assert counts['calls'] == 3 and counts['accepted'] == 1
    assert counts['repeat_root_fallbacks'] == 2
    assert counts['nonterminal_repeat_root_fallbacks'] == 1
    assert counts['prompts_with_repeat_root_fallback'] == 1
    assert counts['optimistic_extra_one_same_calls'] == 1 / 3


def test_parent_repeat_acceptance_violates_declared_pruning_contract():
    observation = {'output_ids': [7], 'stats': {'trace': [
        dict(root_position=0, committed_tokens=[7], accepted_nodes=[1], stopped=False)]}}
    with pytest.raises(AssertionError, match='parent repeat'):
        repeats.count_repeats([observation])
