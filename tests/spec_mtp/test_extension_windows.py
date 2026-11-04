"""Fixed window fallback must retain null prompts and both baseline paths."""

import importlib.util

import pytest
import torch

from test_window_observer import ROOT


def row(root, accepted=(), stopped=False):
    return {'root_position': root, 'accepted_nodes': list(accepted), 'stopped': stopped}


def test_earliest_eligible_hit_and_full_union_including_trailing_roots(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'studies' / '_shared' / 'structure'))
    from studies._shared.structure.extension_diagnostics import choose_window
    zero = [row(3), row(4, [1]), row(6, stopped=True)]
    one = [row(3), row(4, [1]), row(6), row(7, stopped=True)]
    assert choose_window(zero, one) == {'intervention_root': 3,
        'selection_reason': 'before earliest eligible shared hit', 'targets': [3, 4, 6, 7]}


def test_no_shared_hit_uses_predeclared_rejection_control(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'studies' / '_shared' / 'structure'))
    from studies._shared.structure.extension_diagnostics import choose_window
    selected = choose_window([row(3), row(4, stopped=True)], [row(3), row(4, stopped=True)])
    assert selected['intervention_root'] == 3
    assert selected['selection_reason'] == 'earliest continuing common rejection'


def test_unused_owner_and_no_continuing_window_are_explicit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'studies' / '_shared' / 'structure'))
    from studies._shared.structure.extension_diagnostics import choose_window
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


@pytest.mark.parametrize('entry', [
    'studies/output_space_4b_holdout/analyze_results.py',
    'studies/output_space_cross_task/analyze_results.py',
])
def test_legacy_analysis_help_uses_canonical_helpers_without_editable_paths(tmp_path, entry):
    import os
    import subprocess
    import sys
    import sysconfig
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(root / 'src'), sysconfig.get_path('purelib'))),
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-S', str(root / entry), '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()
