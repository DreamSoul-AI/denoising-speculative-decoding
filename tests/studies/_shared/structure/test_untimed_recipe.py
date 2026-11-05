"""Four current recipes share the same adapters and register only after native admission."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from spec_mtp.structure.make.recipe import load_recipe
from spec_mtp.structure.algorithm.factory import AlgorithmRegistry
from spec_mtp.structure.model.factory import ModelRegistry
from studies._shared.structure import algorithm, model, recipe

ROOT = Path(__file__).resolve().parents[4]
STUDIES = ('output_space_4b_holdout', 'output_space_4b_recheck', 'output_space_4b_recheck_v2', 'output_space_4b_input_control')


@pytest.mark.parametrize('name', STUDIES)
@pytest.mark.parametrize('native_failure', [False, True])
def test_untimed_recipe_import_is_inert_and_native_gate_precedes_opt_in(monkeypatch, name, native_failure):
    events = []
    study = ROOT/'studies'/name
    def native(ctx):
        assert ctx.study_dir == study
        events.append('native')
        if native_failure:
            raise RuntimeError('original source gate failed')
    monkeypatch.setattr(recipe, 'register', native)
    monkeypatch.setattr(recipe, 'register_sources', lambda: pytest.fail('recipe import must not register'))
    monkeypatch.setattr(recipe, 'register_untimed', lambda root: events.append(('untimed', root)))
    loaded = load_recipe(study, {'recipe': 'recipe.py'})
    assert events == []
    if native_failure:
        with pytest.raises(RuntimeError, match='source gate'):
            loaded.register(SimpleNamespace(study_dir=study))
        assert events == ['native']
    else:
        loaded.register(SimpleNamespace(study_dir=study))
        assert events == ['native', ('untimed', study)]


def test_untimed_registration_uses_shared_real_handlers_and_explicit_root(monkeypatch, tmp_path):
    calls = []
    study = tmp_path/'study'; study.mkdir(); (study/'study.yaml').write_text('study: synthetic\n')
    monkeypatch.setattr(ModelRegistry, 'register', lambda *args: calls.append(args))
    monkeypatch.setattr(AlgorithmRegistry, 'register', lambda *args: calls.append(args))
    recipe.register_untimed(study)
    assert calls == [('hf_causal', 'spec_mtp', model.build_cuda), ('eval', 'spec_mtp', algorithm.UntimedEval)]


@pytest.mark.parametrize('fault', ['missing', 'no_declaration', 'file'])
def test_untimed_registration_rejects_missing_study_root_without_registry_mutation(monkeypatch, tmp_path, fault):
    target = tmp_path/'study'
    if fault == 'no_declaration':
        target.mkdir()
    elif fault == 'file':
        target.write_text('not a Study')
    monkeypatch.setattr(ModelRegistry, 'register', lambda *args: pytest.fail('invalid root cannot register'))
    monkeypatch.setattr(AlgorithmRegistry, 'register', lambda *args: pytest.fail('invalid root cannot register'))
    with pytest.raises((ValueError, FileNotFoundError)):
        recipe.register_untimed(target)
