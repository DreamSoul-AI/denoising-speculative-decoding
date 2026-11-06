"""Scheduler admission and owned subprocess cleanup require no real workers."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import spec_mtp

from spec_mtp.structure.make import schedule


@pytest.fixture
def saved_plan(tmp_path):
    study = tmp_path / 'runtime'
    study.mkdir()
    (study / 'study.yaml').write_text('study: scheduler-fixture\n', encoding='utf-8')
    jobs, rows = [], []
    for run_id in ('first', 'second'):
        config = study / 'runs' / run_id / 'config.yaml'
        config.parent.mkdir(parents=True)
        config.write_text(f'id: {run_id}\nseed: 7\nalgorithm:\n  mode: train\n'
                          'system:\n  config:\n    device: cpu\n', encoding='utf-8')
        jobs.append({'run_id': run_id, 'config': str(config), 'mode': 'train', 'device': 'cpu'})
        rows.append({'id': run_id, 'run_dir': run_id, 'seed': 7,
                     'config': config.relative_to(study).as_posix()})
    index = {'study': 'scheduler-fixture', 'experiments': [{'factors': {}, 'runs': rows}]}
    (study / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    path = study / 'scripts' / 'jobs.json'
    path.parent.mkdir()
    payload = {'study_dir': str(study), 'round': 2, 'init_gpu': 0, 'num_gpus': 0, 'jobs': jobs}
    path.write_text(json.dumps(payload), encoding='utf-8')
    return SimpleNamespace(study=study, jobs=jobs, index=index, path=path, payload=payload)


def _load(case):
    return schedule.load_launch_plan(case.study, init_gpu=0, num_gpus=0, round_size=0)


def test_worker_environment_pins_actual_package_and_repository_before_inherited_paths(tmp_path, monkeypatch):
    inherited = tmp_path / 'different-installation'
    monkeypatch.setenv('PYTHONPATH', str(inherited))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'inherited-device-mask')
    package_source = Path(spec_mtp.__file__).resolve().parent.parent
    repository = package_source.parent
    environment = schedule.launch_job_env({'device': 'cpu'})
    prefixes = environment['PYTHONPATH'].split(os.pathsep)
    assert [Path(value).resolve() for value in prefixes[:2]] == [package_source, repository]
    assert str(inherited) in prefixes[2:]
    assert environment['CUDA_VISIBLE_DEVICES'] == 'inherited-device-mask'
    assert os.environ['PYTHONPATH'] == str(inherited)


@pytest.mark.parametrize('blocked_run', ['first', 'second'])
def test_dispatch_checks_all_budgeted_runs_before_creating_any_worker(saved_plan, monkeypatch, blocked_run):
    from spec_mtp.structure.artifact.asset import tree

    primary = ValueError('selected Run lacks its fixed reservation')
    visits = []

    def check(study, run_id=None):
        assert study == saved_plan.study
        visits.append(run_id)
        if run_id == blocked_run:
            raise primary

    def forbidden(*args, **kwargs):
        pytest.fail('all selected reservations must be admitted before the first Popen')

    monkeypatch.setattr(tree, 'check_diagnostic_budget', check, raising=False)
    monkeypatch.setattr(schedule.subprocess, 'Popen', forbidden)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in saved_plan.study.rglob('*') if p.is_file()}
    with pytest.raises(ValueError) as caught:
        schedule.launch_jobs(saved_plan.study, saved_plan.jobs, console='shared')
    assert caught.value is primary and visits == [None, 'first', *(['second'] if blocked_run == 'second' else [])]
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in saved_plan.study.rglob('*') if p.is_file()} == before


def test_launch_uses_shared_worker_once_and_preserves_shared_console(saved_plan, monkeypatch):
    calls = []

    class FailedWorker:
        def wait(self, timeout=None):
            return 3

        def poll(self):
            return 3

    def popen(command, **options):
        calls.append((command, options))
        return FailedWorker()

    monkeypatch.setattr(schedule.subprocess, 'Popen', popen)
    codes = schedule.launch_jobs(saved_plan.study, saved_plan.jobs[:1], console='shared')
    assert codes == [3]
    assert len(calls) == 1, 'failed workers require an explicit retry request'
    command, options = calls[0]
    module = command.index('-m')
    assert command[module + 1:module + 3] == ['studies._shared.flow.cli', 'run-one']
    assert command[-2:] == [str(saved_plan.study), 'first']
    assert Path(options['cwd']).resolve() == schedule.repo_root_from(saved_plan.study)
    assert not options.get('creationflags', 0) & getattr(schedule.subprocess, 'CREATE_NEW_CONSOLE', 0x10)


@pytest.mark.parametrize('cleanup_error', [False, True])
def test_dispatch_failure_cleans_only_owned_children_and_keeps_the_first_error(saved_plan, monkeypatch, cleanup_error):
    events = []

    class Worker:
        def __init__(self, label):
            self.label = label
            self.running = True

        def poll(self):
            return None if self.running else -15

        def terminate(self):
            events.append((self.label, 'terminate'))
            self.running = False
            if cleanup_error:
                raise RuntimeError('secondary cleanup failure')

        def wait(self, timeout=None):
            events.append((self.label, 'wait'))
            self.running = False
            return -15

        def kill(self):
            events.append((self.label, 'kill'))
            self.running = False

    owned, unrelated = Worker('owned'), Worker('unrelated')
    first_error = OSError('injected second dispatch failure')
    dispatched = []

    def popen(command, **options):
        dispatched.append(command)
        if len(dispatched) == 1:
            return owned
        raise first_error

    monkeypatch.setattr(schedule.subprocess, 'Popen', popen)
    with pytest.raises(OSError, match='second dispatch') as failed:
        schedule.launch_jobs(saved_plan.study, saved_plan.jobs, round_size=2, console='shared')
    assert failed.value is first_error
    assert len(dispatched) == 2
    assert ('owned', 'terminate') in events and ('owned', 'wait') in events
    assert unrelated.running and not any(label == 'unrelated' for label, action in events)


@pytest.mark.parametrize('fault', ['source', 'plan'])
def test_admission_drift_precedes_dispatch_and_previous_result_rewrites(saved_plan, monkeypatch, fault):
    result = saved_plan.study / 'runs' / 'first' / 'result.json'
    result.write_bytes(b'{"status":"succeeded","saved":"prior execution"}\n')
    before = result.read_bytes()

    def reject(*args, **kwargs):
        raise RuntimeError('injected admission drift')

    def no_side_effect(*args, **kwargs):
        raise AssertionError('admission rejection already dispatched or rewrote a Run')

    monkeypatch.setattr(schedule, 'validate_study_root' if fault == 'source' else 'check_frozen', reject)
    monkeypatch.setattr(schedule.subprocess, 'Popen', no_side_effect)
    monkeypatch.setattr(schedule, '_fail_result', no_side_effect, raising=False)
    with pytest.raises(RuntimeError, match='admission drift'):
        schedule.launch_jobs(saved_plan.study, saved_plan.jobs[:1], console='shared')
    assert result.read_bytes() == before


@pytest.mark.parametrize('fault', ['unknown_run', 'missing_config', 'noncanonical_config',
    'index_config', 'non_mapping_job', 'missing_run_id', 'non_list_jobs', 'invalid_json', 'cpu_gpu', 'cuda_gpu'])
def test_saved_launch_plan_rejects_unbound_or_malformed_jobs(saved_plan, tmp_path, fault):
    case = saved_plan
    if fault == 'unknown_run':
        case.payload['jobs'][0]['run_id'] = 'unpublished'
    elif fault == 'missing_config':
        Path(case.jobs[0]['config']).unlink()
    elif fault == 'noncanonical_config':
        other = tmp_path / 'same-content-config.yaml'
        other.write_bytes(Path(case.jobs[0]['config']).read_bytes())
        case.payload['jobs'][0]['config'] = str(other)
    elif fault == 'index_config':
        case.index['experiments'][0]['runs'][0]['config'] = 'runs/second/config.yaml'
        (case.study / 'index.json').write_text(json.dumps(case.index), encoding='utf-8')
    elif fault == 'non_mapping_job':
        case.payload['jobs'][0] = 'first'
    elif fault == 'missing_run_id':
        del case.payload['jobs'][0]['run_id']
    elif fault == 'non_list_jobs':
        case.payload['jobs'] = {}
    elif fault == 'cpu_gpu':
        case.payload['jobs'][0]['gpu'] = 0
    elif fault == 'cuda_gpu':
        config = Path(case.payload['jobs'][0]['config'])
        config.write_text(config.read_text(encoding='utf-8').replace('device: cpu', 'device: cuda'), encoding='utf-8')
        case.payload['jobs'][0].update(device='cuda', gpu=99)
    case.path.write_text('{' if fault == 'invalid_json' else json.dumps(case.payload), encoding='utf-8')
    previous = case.path.read_bytes()
    with pytest.raises(ValueError):
        _load(case)
    assert case.path.read_bytes() == previous


def test_missing_launch_plan_returns_none_without_creating_artifacts(tmp_path):
    study = tmp_path / 'unmade-study'
    assert schedule.load_launch_plan(study, init_gpu=0, num_gpus=0, round_size=0) is None
    assert not study.exists()


def test_generated_powershell_stays_shared_and_propagates_worker_exit_status(saved_plan):
    paths = schedule.write_launch_scripts(saved_plan.study, saved_plan.jobs, round_size=2, num_gpus=0)
    text = paths['ps1'].read_text(encoding='utf-8')
    assert '--console shared' in text
    assert 'exit $LASTEXITCODE' in text
    assert '--init-gpu 0 --num-gpus 0 --round 2' in text
    assert paths['ps1'].read_bytes().startswith(b'\xef\xbb\xbf')


def test_worker_admission_failure_does_not_rewrite_a_previous_scientific_success(saved_plan, monkeypatch):
    result = saved_plan.study / 'runs/first/result.json'
    before = b'{"status":"succeeded","saved":"previous scientific evidence"}\n'
    result.write_bytes(before)
    monkeypatch.setattr(schedule.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(
        wait=lambda timeout=None: 1, poll=lambda: 1))
    assert schedule.launch_jobs(saved_plan.study, saved_plan.jobs[:1], console='shared') == [1]
    assert result.read_bytes() == before
