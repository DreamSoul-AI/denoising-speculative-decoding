"""Recipe contract: no registration at import, idempotent registration on RPipe,
construction checks before Data / Model are built, SpecMTP source freeze."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from rpipe.structure.algorithm.factory import AlgorithmRegistry
from rpipe.structure.data.factory import DataRegistry
from rpipe.structure.make.recipe import RecipeContext
from rpipe.structure.model.factory import ModelRegistry

from studies._shared.structure import recipe
from studies._shared.structure import sources


def _config(**algorithm_config):
    base = yaml.safe_load((Path(__file__).resolve().parents[3]
                           / 'studies/smoke/experiment_config.yaml').read_text(encoding='utf-8'))
    base['algorithm']['config'].update(algorithm_config)
    return base


def test_import_registers_nothing():
    code = ('import spec_mtp, studies._shared.structure.algorithm;'
            'from rpipe.structure.data.factory import DataRegistry;'
            'from rpipe.structure.model.factory import ModelRegistry;'
            'from rpipe.structure.algorithm.factory import AlgorithmRegistry;'
            'assert not [k for k in DataRegistry.list() if k[1]=="spec_mtp"];'
            'assert not [k for k in ModelRegistry.list() if k[1]=="spec_mtp"];'
            'assert ("eval","spec_mtp") not in AlgorithmRegistry.list()')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_register_sources_is_idempotent_and_keeps_rpipe_defaults():
    recipe.register_sources()
    recipe.register_sources()
    assert ('eval', 'spec_mtp') in AlgorithmRegistry.list()
    assert ('eval', 'custom_torch') in AlgorithmRegistry.list()
    assert ('train', 'custom_torch') in AlgorithmRegistry.list()
    data = {name for name, source in DataRegistry.list() if source == 'spec_mtp'}
    assert {'specbench', 'local_prompts', 'dolly'} <= data
    models = {name for name, source in ModelRegistry.list() if source == 'spec_mtp'}
    assert models == {'hf_causal', 'tiny_llama'}


@pytest.mark.parametrize('variant', ['esp_static', 'esp_dynamic', 'ema_velocity'])
def test_validate_accepts_smoke_variants(variant):
    info = recipe.validate(_config(variant=variant))
    assert info['task'] == 'esp_mtp'


@pytest.mark.parametrize('mutate, message', [
    (lambda c: c['algorithm'].update(mode='train'), 'eval Algorithm'),
    (lambda c: c['algorithm'].update(source='custom_torch'), 'algorithm.source'),
    (lambda c: c['algorithm']['config'].update(task='nope'), 'unknown spec_mtp task'),
    (lambda c: c['algorithm']['config'].update(variant='nope'), 'unknown spec variant'),
    (lambda c: c['algorithm']['config'].update(mask='nope'), 'unknown mask setting'),
    (lambda c: c['model']['config'].update(attn_implementation='flash_attention_2'), 'eager or sdpa'),
    (lambda c: c['data'].update(name='nope'), 'unknown SpecMTP data name'),
    (lambda c: c['model'].update(name='nope'), 'unknown SpecMTP model name'),
    (lambda c: c['algorithm']['config'].update(
        variant=None, spec={'impl': 'efficient', 'tree': {'mode': 'dynamic'}}), 'requires spec.tree.mode=static'),
    (lambda c: c['algorithm']['config'].update(variant=None, spec={'impl': 'fast'}), 'Unsupported spec.impl'),
])
def test_validate_rejects_unsupported_configs(mutate, message):
    config = _config(variant='esp_static')
    mutate(config)
    with pytest.raises(ValueError, match=message):
        recipe.validate(config)


def test_alignment_probe_and_ar_baseline_skip_tree_checks():
    assert recipe.validate(_config(task='alignment_probe'))['task'] == 'alignment_probe'
    assert recipe.validate(_config(task='ar_baseline'))['task'] == 'ar_baseline'


def _ctx(study_dir, config):
    return RecipeContext(study_dir=study_dir, run_id='r', seed=123, config=config,
                         shared_data_dir=study_dir / 'shared/data', shared_model_dir=study_dir / 'shared/model',
                         assets_dir=study_dir / 'runs/r/assets')


def test_register_writes_source_manifest(study_copy):
    study_dir = study_copy()
    recipe.register(_ctx(study_dir, _config(variant='esp_static')))
    body = json.loads((study_dir / 'runs/r/assets' / sources.RUN_ASSET_NAME).read_text(encoding='utf-8'))
    assert body['digest'] == sources.manifest()['digest']
    assert 'studies/_shared/structure/recipe.py' in body['files']


def test_freeze_without_lock_fails(study_copy):
    study_dir = study_copy(lambda s: s.update(freeze=True))
    with pytest.raises(sources.SpecMtpSourceChangedError, match='missing'):
        recipe.register(_ctx(study_dir, _config(variant='esp_static')))


def test_freeze_detects_changed_spec_mtp_source(study_copy):
    study_dir = study_copy(lambda s: s.update(freeze=True))
    lock = sources.write_lock(study_dir)
    recipe.register(_ctx(study_dir, _config(variant='esp_static')))  # unchanged: passes
    body = json.loads(lock.read_text(encoding='utf-8'))
    body['files']['spec_mtp/decode.py'] = '0' * 64
    lock.write_text(json.dumps(body), encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='decode.py'):
        recipe.register(_ctx(study_dir, _config(variant='esp_static')))


def test_freeze_detects_changed_rpipe_commit(study_copy):
    study_dir = study_copy(lambda s: s.update(freeze=True))
    lock = sources.write_lock(study_dir)
    body = json.loads(lock.read_text(encoding='utf-8'))
    body['rpipe'] = {**body.get('rpipe', {}), 'commit': 'f' * 40}
    lock.write_text(json.dumps(body), encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='rpipe commit'):
        recipe.register(_ctx(study_dir, _config(variant='esp_static')))


@pytest.mark.parametrize('study', ['_template', 'smoke', 'input_mask_baseline', 'output_space_prerequisite', 'output_space_small_matrix', 'output_space_extension', 'output_space_followup', 'output_space_cross_task', 'output_space_4b_recheck_v2', 'output_space_4b_holdout', 'output_space_4b_input_control'])
def test_repository_studies_expand_and_validate(tmp_path, study):
    """make-level expansion through RPipe, then the recipe checks on every Run config (no weights)."""
    import shutil

    from rpipe.structure.make import expand_study
    from rpipe.structure.make.expand import load_study_yaml

    src = Path(__file__).resolve().parents[3] / 'studies' / study
    dest = tmp_path / study
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
        'runs', 'shared', 'scripts', 'index.json', 'process.json', 'provenance.json',
        'activity.json', 'spec_mtp_sources.json', 'spec_mtp_study_check.json', '__pycache__'))
    declared = load_study_yaml(dest)
    assert declared['recipe'] == 'recipe.py'
    assert declared.get('flow', {}).get('study_phases') is True
    assert declared.get('flow', {}).get('prepare_shared') is False
    missing = [pattern for pattern in declared.get('provenance', {}).get('include', [])
               if not list(dest.glob(pattern))]
    if missing:
        with pytest.raises(FileNotFoundError, match='provenance.include matched nothing'):
            expand_study(dest)
        return
    out = expand_study(dest)
    assert out['configs']
    for path in out['configs']:
        config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        recipe.validate(config)
        assert '{' not in config['description'], config['description']


def test_layer_must_be_a_mapping():
    config = _config()
    config['data'] = ['not', 'a', 'mapping']
    with pytest.raises(recipe.SpecMtpConfigError, match='data must be a mapping'):
        recipe.validate(config)


def test_legacy_cfg_takes_device_from_system_and_seed_from_run():
    from types import SimpleNamespace

    from rpipe.structure.control.layers import AlgorithmConfig

    from studies._shared.structure.config import legacy_cfg

    algo = AlgorithmConfig.from_mapping({'mode': 'eval', 'source': 'spec_mtp',
                                         'config': {'eval': {'device': 'cuda', 'max_new_tokens': 8},
                                                    'spec': {'impl': 'naive'}}})
    cfg = legacy_cfg(SimpleNamespace(meta={'data_cfg': {'name': 'x'}}), SimpleNamespace(meta={}),
                     SimpleNamespace(device='cpu'), algo, '7')
    assert cfg.eval.device == 'cpu' and cfg.eval.max_new_tokens == 8
    assert cfg.run.seed == 7 and cfg.spec.impl == 'naive' and cfg.data.name == 'x' and dict(cfg.model) == {}


@pytest.mark.parametrize('mutate, message', [
    (lambda c: c['algorithm']['config'].update(variant='esp_statik'), "did you mean 'esp_static'"),
    (lambda c: c['algorithm']['config'].update(variant='llama3/esp_bc6'), "'llama3/esp_bc60'"),
    (lambda c: c['algorithm']['config'].update(variant='qwen3/zz'), 'paper_variants.py'),
    (lambda c: c['algorithm']['config'].update(task='esp_mpt'), "did you mean 'esp_mtp'"),
    (lambda c: c['model'].update(name='tiny_lama'), "did you mean 'tiny_llama'"),
    (lambda c: c['data'].update(name='specbnch'), "did you mean 'specbench'"),
    (lambda c: c['data'].update(name='qqqq'), 'known:'),
])
def test_config_errors_suggest_the_fix(mutate, message):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        recipe.validate(config)


def test_register_error_names_the_study(study_copy):
    study_dir = study_copy()
    config = _config(task='nope')
    with pytest.raises(recipe.SpecMtpConfigError, match=r'study\.yaml; checked before any weights load'):
        recipe.register(_ctx(study_dir, config))


def test_freeze_rejects_corrupt_lock(study_copy):
    study_dir = study_copy(lambda s: s.update(freeze=True))
    (study_dir / sources.LOCK_NAME).write_text('{not json', encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='not valid JSON'):
        recipe.register(_ctx(study_dir, _config(variant='esp_static')))


@pytest.mark.parametrize('kind', ['added', 'deleted', 'modified'])
def test_source_lock_covers_all_file_changes(study_copy, kind):
    study = study_copy()
    sources.write_lock(study)
    current = sources.manifest()
    key = next(iter(current['files']))
    if kind == 'added':
        current['files']['spec_mtp/new.py'] = 'new'
    elif kind == 'deleted':
        del current['files'][key]
    else:
        current['files'][key] = 'changed'
    with pytest.raises(sources.SpecMtpSourceChangedError):
        sources.check_frozen(study, {}, current)


def test_remake_requires_new_version_after_source_change(study_copy):
    from rpipe.structure.make import expand_study
    study = study_copy()
    sources.write_lock(study)
    expanded = expand_study(study)
    run_dir = Path(expanded['configs'][0]).parent
    (run_dir / 'result.json').write_text('{"status":"succeeded"}')
    lock = study / sources.LOCK_NAME
    saved = json.loads(lock.read_text())
    saved['files']['spec_mtp/new.py'] = 'old'
    lock.write_text(json.dumps(saved))
    with pytest.raises(sources.SpecMtpSourceChangedError, match='new.*version'):
        sources.write_lock(study)
    declaration = yaml.safe_load((study / 'study.yaml').read_text())
    declaration['fixed']['algorithm']['config']['version'] = 'new-source'
    (study / 'study.yaml').write_text(yaml.safe_dump(declaration))
    sources.write_lock(study)


def test_prompt_source_drift_rejected_before_construction(study_copy):
    study = study_copy()
    prompt = study / 'prompt.jsonl'
    prompt.write_text('first')
    declaration = yaml.safe_load((study / 'study.yaml').read_text())
    declaration['fixed']['data']['config']['path'] = str(prompt)
    (study / 'study.yaml').write_text(yaml.safe_dump(declaration))
    sources.write_lock(study)
    prompt.write_text('changed')
    with pytest.raises(RuntimeError, match='input changed'):
        sources.check_frozen(study, declaration, sources.manifest())
