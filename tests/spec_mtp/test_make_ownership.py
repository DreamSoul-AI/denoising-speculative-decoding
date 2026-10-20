"""Make publishes only while holding ownership of all expanded Runs."""

from contextlib import nullcontext

import pytest

from rpipe.structure.artifact.execution import RunLockedError, run_lock
from rpipe.structure.make import expand_patches, expand_study, load_study_yaml, write_run_configs


@pytest.fixture
def make_study(tmp_path):
    (tmp_path / 'study.yaml').write_text(
        'study: make-ownership\nseeds: [1, 2]\nflow: {prepare_shared: false}\n', encoding='utf-8')
    (tmp_path / 'experiment_config.yaml').write_text(
        'version: make-ownership-v1\ndescription: before\ndata: {name: Toy, source: stub}\n'
        'model: {name: linear}\nalgorithm: {mode: train, num_steps: 1}\nsystem: {device: cpu}\n',
        encoding='utf-8')
    configs = expand_study(tmp_path)['configs']
    return tmp_path, configs


@pytest.mark.parametrize('entry', ['expand', 'rpipe', 'configs'])
@pytest.mark.parametrize('busy', [True, False], ids=['busy', 'idle'])
def test_make_preserves_all_configs_when_later_run_is_owned(make_study, entry, busy):
    study, configs = make_study
    with (study / 'study.yaml').open('a', encoding='utf-8') as output:
        output.write('run_description: refreshed seed={seed}\n')
    for config in configs:
        with run_lock(config.parent):
            pass
    with run_lock(configs[-1].parent) if busy else nullcontext():
        before = {p: p.read_bytes() for p in study.rglob('*')
                  if p.is_file() and p.name not in ('.rpipe.lock', 'activity.json')}
        refused = False
        try:
            if entry == 'expand':
                expand_study(study)
            elif entry == 'configs':
                write_run_configs(study, expand_patches(load_study_yaml(study)))
            else:
                from importlib import import_module
                import_module(entry + '.flow.cli').main(['make', str(study), '--num-gpus', '0'])
        except RunLockedError:
            refused = True
        if busy:
            assert {p: p.read_bytes() for p in study.rglob('*')
                    if p.is_file() and p.name not in ('.rpipe.lock', 'activity.json')} == before
            assert refused
        else:
            assert not refused
            assert all(b'refreshed' in config.read_bytes() for config in configs)
    for config in configs:
        with run_lock(config.parent):
            pass


@pytest.mark.parametrize('entry', ['expand', 'configs'])
@pytest.mark.parametrize('writer_failure', [False, True])
def test_make_excludes_runner_until_all_writers_finish(make_study, monkeypatch, entry, writer_failure):
    from rpipe.flow import FlowContext, FlowRunner
    from rpipe.structure.artifact import artifact_layout
    import rpipe.structure.make as make
    import rpipe.structure.make.write as write

    study, configs = make_study
    observed = []
    primary = PermissionError('controlled make publication failure')
    def guard(name, original):
        def publish(*args, **kwargs):
            for config in configs:
                with pytest.raises(RunLockedError):
                    FlowRunner(['process']).run(FlowContext(study, artifact_layout(study, config.parent.name), {}))
                assert not (config.parent / 'execution.json').exists()
            observed.append(name)
            if writer_failure and name == ('provenance' if entry == 'expand' else 'config'):
                raise primary
            return original(*args, **kwargs)
        return publish
    for module, attribute, name in [(write, 'write_config', 'config'),
                                    (make, 'write_study_index', 'index'),
                                    (make, 'write_provenance', 'provenance')]:
        monkeypatch.setattr(module, attribute, guard(name, getattr(module, attribute)))
    def publish():
        return expand_study(study) if entry == 'expand' else write_run_configs(
            study, expand_patches(load_study_yaml(study)))
    if writer_failure:
        with pytest.raises(PermissionError) as caught:
            publish()
        assert caught.value is primary
    else:
        publish()
    assert observed[-1] == ('provenance' if entry == 'expand' else 'config')
    for config in configs:
        with run_lock(config.parent):
            pass
