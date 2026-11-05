"""Fixed window fallback must retain null prompts and both baseline paths."""

import pytest


from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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

@pytest.mark.parametrize('entry', [
    ['-m', 'studies.output_space_4b_holdout.flow.process.run'],
    ['-m', 'studies.output_space_cross_task.flow.process.run'],
])
def test_legacy_analysis_help_uses_canonical_helpers_without_editable_paths(tmp_path, entry):
    import os
    import subprocess
    import sys
    import sysconfig
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(root), str(root / 'src'), sysconfig.get_path('purelib'))),
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'CUDA_VISIBLE_DEVICES': ''}
    command = entry if entry[0] == '-m' else [str(root / entry[0])]
    result = subprocess.run([sys.executable, '-S', *command, '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()
