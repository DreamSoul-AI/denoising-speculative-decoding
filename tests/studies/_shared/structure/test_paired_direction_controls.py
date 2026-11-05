"""cpu pure IO/spies: source-bound control registration and admission ownership."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from spec_mtp.structure.artifact.layout import materialize_study
from spec_mtp.structure.algorithm.factory import AlgorithmRegistry
from spec_mtp.structure.model.factory import ModelRegistry
from studies._shared.structure import recipe, validation


@pytest.mark.parametrize('fault', ['empty', 'malformed', 'escape', 'mismatch'])
def test_direction_source_refs_reject_bad_identity_before_execution(tmp_path, fault):
    execution = tmp_path/'checkout'; execution.mkdir()
    study = execution/'studies/direction'; (study/'docs').mkdir(parents=True)
    source = execution/'input.py'; source.write_bytes(b'original source')
    outside = tmp_path/'outside.py'; outside.write_bytes(b'outside')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    refs = {'input.py': expected}
    if fault == 'empty': refs = {}
    elif fault == 'malformed': refs = {'input.py': 'not-sha256'}
    elif fault == 'escape': refs = {'../outside.py': hashlib.sha256(outside.read_bytes()).hexdigest()}
    else: refs = {'input.py': '0'*64}
    (study/'docs/source_refs.json').write_text(json.dumps(refs), encoding='utf-8')
    with pytest.raises((ValueError, AssertionError)):
        validation.check_direction_refs(study, execution)


@pytest.mark.parametrize('fault', ['none', 'source_drift', 'bad_marker'])
def test_control_registration_uses_snapshot_source_and_keeps_roots_separate(tmp_path, monkeypatch, fault):
    checkout = tmp_path/'checkout'
    pairing = checkout/'studies/qwen3_4b_bf16_five_methods'
    direction = checkout/'studies/direction'
    for root in (pairing, direction):
        root.mkdir(parents=True)
        (root/'study.yaml').write_text('study: synthetic\nrecipe: recipe.py\n', encoding='utf-8')
        (root/'recipe.py').write_text('pass\n', encoding='utf-8')
    runtime = materialize_study(direction, tmp_path/'arbitrary-external-name')
    paired_runtime = materialize_study(pairing, tmp_path/'another-unrelated-name')
    classes, builders = [], []
    monkeypatch.setattr(AlgorithmRegistry, 'register', lambda mode, source, cls: classes.append(cls))
    monkeypatch.setattr(ModelRegistry, 'register', lambda name, source, builder: builders.append(builder))
    if fault == 'source_drift':
        (direction/'recipe.py').write_text('changed source\n', encoding='utf-8')
    elif fault == 'bad_marker':
        (runtime/'study_source.json').write_text('{}', encoding='utf-8')
    if fault != 'none':
        with pytest.raises(ValueError):
            recipe.register_direction(runtime, admission_assets=runtime/'runs/a/assets')
        assert classes == builders == []
    else:
        recipe.register_pairing(paired_runtime)
        assert classes[-1].pairing_root == pairing
        recipe.register_direction(runtime, admission_assets=runtime/'runs/a/assets')
        first = classes[-1]
        assert first.study_root == runtime and first.execution_root == checkout and first.pairing_root == pairing
        assert builders[-1].keywords['admission_assets'] == runtime/'runs/a/assets'
        other = materialize_study(direction, tmp_path/'third-arbitrary-name')
        recipe.register_direction(other, admission_assets=other/'runs/b/assets', recovery=True)
        assert first.study_root == runtime and first.recovery is False
        assert classes[-1].study_root == other and classes[-1].recovery is True


@pytest.mark.parametrize('failure', [False, True])
def test_direction_source_gate_precedes_admission_and_model_with_distinct_asset_roots(tmp_path, monkeypatch, failure):
    from studies._shared.structure import resource_admission, model
    events = []
    run_assets, model_assets = tmp_path/'run-assets', tmp_path/'shared-model'
    def refs(*args):
        events.append('refs')
        if failure:
            raise RuntimeError('source mismatch')
    monkeypatch.setattr(validation, 'check_direction_refs', refs)
    monkeypatch.setattr(resource_admission, 'direction_admission', lambda assets, **kw: events.append(('admit', assets)) or {'passed': True})
    monkeypatch.setattr(model, 'build_cuda', lambda config, assets, meta: events.append(('model', assets)) or SimpleNamespace(meta={}))
    if failure:
        with pytest.raises(RuntimeError, match='source mismatch'):
            recipe._build_direction_cuda(None, model_assets, study_root=tmp_path, execution_root=tmp_path,
                                         admission_assets=run_assets, recovery=False)
        assert events == ['refs']
    else:
        result = recipe._build_direction_cuda(None, model_assets, study_root=tmp_path, execution_root=tmp_path,
                                             admission_assets=run_assets, recovery=True)
        assert result.meta['direction_admission']['passed']
        assert events == ['refs', ('admit', run_assets), 'refs', ('model', model_assets)]


@pytest.mark.parametrize('passed', [False, True])
def test_ordinary_direction_admission_is_one_observation_without_wait_output(tmp_path, monkeypatch, passed):
    from studies._shared.structure import resource_admission
    calls = []
    monkeypatch.setattr(resource_admission.time, 'sleep', lambda seconds: pytest.fail('ordinary policy never waits'))
    def observe(initial):
        calls.append(initial)
        return {'passed': passed}
    if passed:
        assert resource_admission.direction_admission(tmp_path, observer=observe)['passed']
    else:
        with pytest.raises(AssertionError, match='deferred'):
            resource_admission.direction_admission(tmp_path, observer=observe)
    assert calls == [True] and not (tmp_path/'direction_resource_wait.json').exists()


@pytest.mark.parametrize('name', ['qwen3_4b_bf16_five_methods', 'qwen3_4b_bf16_directions', 'qwen3_4b_bf16_directions_recovery'])
@pytest.mark.parametrize('failure', [False, True])
def test_paired_direction_recipes_keep_native_gate_first_and_import_inert(tmp_path, monkeypatch, name, failure):
    from pathlib import Path
    from spec_mtp.structure.make.recipe import load_recipe
    source = Path(__file__).resolve().parents[4]/'studies'/name
    events = []
    def native(ctx):
        events.append('native')
        if failure:
            raise RuntimeError('source gate failure')
    monkeypatch.setattr(recipe, 'register', native)
    monkeypatch.setattr(recipe, 'register_pairing', lambda root: events.append(('pairing', root)))
    monkeypatch.setattr(recipe, 'register_direction', lambda root, **kwargs: events.append(('direction', root, kwargs)))
    loaded = load_recipe(source, {'recipe': 'recipe.py'})
    assert events == []
    ctx = SimpleNamespace(study_dir=source, assets_dir=tmp_path/'run-assets')
    if failure:
        with pytest.raises(RuntimeError, match='source gate failure'):
            loaded.register(ctx)
        assert events == ['native']
    else:
        loaded.register(ctx)
        if name.endswith('five_methods'):
            assert events == ['native', ('pairing', source)]
        else:
            assert events == ['native', ('direction', source, {'admission_assets': ctx.assets_dir,
                'recovery': name.endswith('recovery')})]
