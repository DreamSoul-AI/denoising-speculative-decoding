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


@pytest.mark.parametrize('primary_type,operation,cleanup_type,broken_output', [
    (KeyboardInterrupt, 'poll', SystemExit, False),
    (SystemExit, 'terminate', KeyboardInterrupt, False),
    (OSError, 'wait', SystemExit, False),
    (KeyboardInterrupt, 'kill', OSError, False),
    (SystemExit, 'kill_wait', KeyboardInterrupt, False),
    (OSError, 'terminate', RuntimeError, True),
    (KeyboardInterrupt, 'wait', OSError, True),
    (SystemExit, 'poll', KeyboardInterrupt, True),
    (OSError, 'kill', SystemExit, True),
    (KeyboardInterrupt, 'kill_wait', SystemExit, True),
])
def test_cancellation_cleanup_preserves_primary_and_attempts_all_owned_workers(
    saved_plan, monkeypatch, primary_type, operation, cleanup_type, broken_output,
):
    import builtins

    case = saved_plan
    config = case.study / 'runs/later/config.yaml'
    config.parent.mkdir()
    config.write_bytes(Path(case.jobs[0]['config']).read_bytes().replace(b'id: first', b'id: later'))
    later = {**case.jobs[0], 'run_id': 'later', 'config': str(config)}
    case.index['experiments'][0]['runs'].append({
        'id': 'later', 'run_dir': 'later', 'seed': 7, 'config': 'runs/later/config.yaml',
    })
    (case.study / 'index.json').write_text(json.dumps(case.index), encoding='utf-8')
    result = case.study / 'runs/first/result.json'
    result.write_bytes(b'{"status":"succeeded","saved":"prior result"}\n')
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in case.study.rglob('*') if path.is_file()}
    primary = primary_type('original worker wait interrupted')
    prior = LookupError('an existing cleanup attachment')
    primary.cleanup_errors = (prior,)
    cleanup = cleanup_type('injected first worker cleanup failure')
    tail = RuntimeError('second worker cleanup failure')
    expected = [prior]
    events, dispatched, cleanup_output = [], [], []

    def fail(error):
        expected.append(error)
        raise error

    class Worker:
        def __init__(self, label):
            self.label = label
            self.killed = False

        def poll(self):
            events.append((self.label, 'poll'))
            if self.label == 'first' and operation == 'poll':
                fail(cleanup)
            return None

        def terminate(self):
            events.append((self.label, 'terminate'))
            if self.label == 'second':
                fail(tail)
            if operation == 'terminate':
                fail(cleanup)

        def wait(self, timeout=None):
            events.append((self.label, 'wait', timeout))
            if timeout is None:
                assert self.label == 'first'
                raise primary
            assert timeout == 5  # Preserve the existing escalation boundary.
            if self.label == 'first':
                if operation == 'wait':
                    fail(cleanup)
                if operation in ('kill', 'kill_wait'):
                    if not self.killed:
                        raise schedule.subprocess.TimeoutExpired('owned-worker', timeout)
                    if operation == 'kill_wait':
                        fail(cleanup)
            return -15

        def kill(self):
            events.append((self.label, 'kill'))
            self.killed = True
            if operation == 'kill':
                fail(cleanup)

    workers = {name: Worker(name) for name in ('first', 'second', 'unrelated')}

    def popen(command, **kwargs):
        run_id = command[-1]
        dispatched.append(run_id)
        assert run_id in ('first', 'second'), 'cancellation dispatched a later group or retry'
        return workers[run_id]

    original_print = builtins.print

    def printed(*values, **kwargs):
        if values and str(values[0]).startswith('worker cleanup '):
            cleanup_output.append(values[0])
            if broken_output:
                raise BrokenPipeError('cleanup reporting must not replace the original interruption')
        return original_print(*values, **kwargs)

    monkeypatch.setattr(schedule.subprocess, 'Popen', popen)
    monkeypatch.setattr(builtins, 'print', printed)
    with pytest.raises(BaseException) as caught:
        schedule.launch_jobs(case.study, [*case.jobs, later], round_size=2,
                             retry_failed=True, console='shared')
    assert caught.value is primary
    assert dispatched == ['first', 'second']
    assert ('first', 'wait', 5) in events and ('second', 'wait', 5) in events
    assert ('second', 'terminate') in events
    assert cleanup in expected and tail in expected
    assert tuple(primary.cleanup_errors) == tuple(expected)
    assert not cleanup_output, 'secondary failures must remain attached without requiring a writable console'
    assert not any(event[0] == 'unrelated' for event in events)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    receipt = Path(primary.cleanup_receipt_path)
    assert receipt.name == 'receipt.json' and receipt.parent.parent == case.study / 'scripts/launch-attempts'
    assert len(receipt.parent.name) == 32 and all(char in '0123456789abcdef' for char in receipt.parent.name)
    assert {path for path in case.study.rglob('*') if path.is_file()} == {*before, receipt}
    body = json.loads(receipt.read_bytes())
    assert body['schema'] == 1 and body['scope'] == 'owned_current_group_only'
    assert [row['run_id'] for row in body['jobs']] == ['first', 'second']
    assert [row['run_id'] for row in body['workers']] == ['first', 'second']


@pytest.fixture
def interruption_case(saved_plan, monkeypatch):
    import hashlib
    from datetime import datetime, timezone

    case = saved_plan
    state = SimpleNamespace(primary=KeyboardInterrupt('original wait interrupted'), mode='wait_returned',
                            spawn_failure=False, on_spawn=None, events=[], spawned=[])

    class Worker:
        def __init__(self, run_id, pid):
            self.run_id, self.pid, self.killed = run_id, pid, False

        def poll(self):
            state.events.append((self.run_id, 'poll'))
            return None

        def terminate(self):
            state.events.append((self.run_id, 'terminate'))

        def kill(self):
            state.events.append((self.run_id, 'kill'))
            self.killed = True

        def wait(self, timeout=None):
            state.events.append((self.run_id, 'wait', timeout))
            if timeout is None:
                if state.mode == 'normal':
                    return 0
                raise state.primary
            assert timeout == 5
            if self.run_id == 'first' and state.mode != 'wait_returned':
                if not self.killed or state.mode == 'kill_wait_timeout':
                    raise schedule.subprocess.TimeoutExpired('owned-worker', timeout)
            return -9 if self.killed else -15

    workers = {'first': Worker('first', 1001), 'second': Worker('second', 1002)}

    def popen(command, **kwargs):
        run_id = command[-1]
        state.spawned.append(run_id)
        if state.on_spawn is not None:
            state.on_spawn(run_id)
        if state.spawn_failure:
            raise state.primary
        return workers[run_id]

    def invoke():
        with pytest.raises(BaseException) as caught:
            schedule.launch_jobs(case.study, case.jobs, round_size=2, retry_failed=True, console='shared')
        assert caught.value is state.primary
        return getattr(state.primary, 'cleanup_receipt_path', None)

    def body():
        return {'schema': 1, 'scope': 'owned_current_group_only', 'study': str(case.study.resolve()),
                'observed_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'launcher_pid': os.getpid(), 'primary_exception_type': 'KeyboardInterrupt',
                'index_sha256': hashlib.sha256((case.study / 'index.json').read_bytes()).hexdigest(),
                'jobs': [{'run_id': job['run_id'], 'config_sha256': hashlib.sha256(Path(job['config']).read_bytes()).hexdigest()}
                         for job in case.jobs], 'workers': [], 'descendants': 'unknown',
                'writer_stop': 'unknown', 'unreturned_popen': 'unknown', 'cleanup_wait_seconds': 5}

    monkeypatch.setattr(schedule.subprocess, 'Popen', popen)
    state.study, state.jobs, state.invoke, state.body = case.study, case.jobs, invoke, body
    return state


@pytest.mark.parametrize('mode', ['wait_returned', 'kill_wait_returned', 'kill_wait_timeout'])
def test_interruption_receipt_separates_requests_from_confirmed_direct_exit(interruption_case, mode):
    from datetime import datetime, timezone

    case = interruption_case
    case.mode = mode
    path = Path(case.invoke())
    body = json.loads(path.read_bytes())
    assert body['schema'] == 1 and body['scope'] == 'owned_current_group_only'
    assert body['study'] == str(case.study.resolve()) and body['launcher_pid'] == os.getpid()
    observed = datetime.fromisoformat(body['observed_at'].replace('Z', '+00:00'))
    assert observed.tzinfo is not None and observed.utcoffset() == timezone.utc.utcoffset(observed)
    assert body['primary_exception_type'] == 'KeyboardInterrupt' and body['cleanup_wait_seconds'] == 5
    assert body['descendants'] == body['writer_stop'] == body['unreturned_popen'] == 'unknown'
    first, second = body['workers']
    assert (first['run_id'], first['pid'], second['run_id'], second['pid']) == ('first', 1001, 'second', 1002)
    assert first['terminate_requested'] is second['terminate_requested'] is True
    assert first['kill_requested'] is (mode != 'wait_returned') and second['kill_requested'] is False
    assert second['direct_exit_confirmed'] is True and type(second['returncode']) is int and second['returncode'] == -15
    if mode == 'kill_wait_timeout':
        assert first['direct_exit_confirmed'] is False and first.get('returncode') is None
        assert 'TimeoutExpired' in first['cleanup_error_types']
    else:
        assert first['direct_exit_confirmed'] is True and type(first['returncode']) is int
        assert first['returncode'] == (-9 if mode == 'kill_wait_returned' else -15)
        assert first['cleanup_error_types'] == []
    assert not path.with_name('partial.json').exists()


def test_interruption_receipt_keeps_unreturned_spawn_ownership_unknown(interruption_case):
    case = interruption_case
    case.spawn_failure = True
    body = json.loads(Path(case.invoke()).read_bytes())
    assert case.spawned == ['first'] and body['workers'] == []
    assert [row['run_id'] for row in body['jobs']] == ['first', 'second']
    assert body['unreturned_popen'] == body['writer_stop'] == body['descendants'] == 'unknown'


@pytest.mark.parametrize('failure_type', [OSError, KeyboardInterrupt, SystemExit])
def test_interruption_publication_failure_is_after_cleanup_and_preserves_primary(interruption_case, monkeypatch, failure_type):
    case = interruption_case
    prior = LookupError('old attachment')
    case.primary.cleanup_errors = (prior,)
    case.primary.cleanup_receipt_path = 'old receipt pointer'
    failure = failure_type('private payload; token=never_echo_this')
    calls = []

    def publish(study, body):
        calls.append(body)
        assert study == case.study
        assert ('first', 'wait', 5) in case.events and ('second', 'wait', 5) in case.events
        raise failure

    monkeypatch.setattr(schedule, '_write_interruption_receipt', publish, raising=False)
    assert case.invoke() is None
    assert len(calls) == 1 and tuple(case.primary.cleanup_errors) == (prior, failure)
    assert not hasattr(case.primary, 'cleanup_receipt_path')


@pytest.mark.parametrize('member', ['index', 'config'])
def test_interruption_receipt_binds_original_inputs_before_any_spawn(interruption_case, member):
    import hashlib

    case = interruption_case
    original = case.body()
    path = case.study / 'index.json' if member == 'index' else Path(case.jobs[1]['config'])

    def change_after_admission(run_id):
        if run_id == 'first':
            path.write_bytes(path.read_bytes() + b' ')

    case.on_spawn = change_after_admission
    body = json.loads(Path(case.invoke()).read_bytes())
    assert body['index_sha256'] == original['index_sha256'] and body['jobs'] == original['jobs']
    if member == 'index':
        assert body['index_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        assert body['jobs'][1]['config_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['namespace_alias', 'attempt_alias', 'collision', 'open', 'fsync', 'rename', 'oversized'])
def test_interruption_publisher_refuses_alias_collision_and_failed_publication(interruption_case, tmp_path, monkeypatch, fault):
    case = interruption_case
    body = case.body()
    namespace = case.study / 'scripts/launch-attempts'
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in case.study.rglob('*') if path.is_file()}
    failure = OSError('controlled publication failure')
    resolve, mkdir, opened = Path.resolve, Path.mkdir, Path.open
    collided = []

    def redirected(path, *args, **kwargs):
        if ((fault == 'namespace_alias' and path.is_relative_to(namespace))
                or (fault == 'attempt_alias' and path.parent == namespace and len(path.name) == 32)):
            return resolve(tmp_path / 'outside' / path.relative_to(namespace), *args, **kwargs)
        return resolve(path, *args, **kwargs)

    def directory(path, *args, **kwargs):
        if fault == 'collision' and path.parent == namespace:
            assert not collided, 'a collision must not silently retry a different attempt ID'
            mkdir(path, parents=True)
            sentinel = path / 'receipt.json'
            sentinel.write_bytes(b'old immutable receipt\n')
            collided.append(sentinel)
            raise FileExistsError('attempt already exists')
        return mkdir(path, *args, **kwargs)

    def stream(path, *args, **kwargs):
        if path.name == 'partial.json' and path.is_relative_to(namespace):
            raise failure
        return opened(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if fault in ('namespace_alias', 'attempt_alias'):
            patch.setattr(Path, 'resolve', redirected)
        elif fault == 'collision':
            patch.setattr(Path, 'mkdir', directory)
        elif fault == 'open':
            patch.setattr(Path, 'open', stream)
        elif fault == 'fsync':
            patch.setattr(os, 'fsync', lambda fd: (_ for _ in ()).throw(failure))
        elif fault == 'rename':
            patch.setattr(os, 'rename', lambda *args, **kwargs: (_ for _ in ()).throw(failure))
        else:
            body['primary_exception_type'] = 'X' * 65536
        with pytest.raises((ValueError, OSError)):
            schedule._write_interruption_receipt(case.study, body)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    if collided:
        assert len(collided) == 1 and collided[0].read_bytes() == b'old immutable receipt\n'
    else:
        assert not list(namespace.glob('*/receipt.json'))


def test_repeated_interruptions_never_overwrite_an_earlier_attempt(interruption_case):
    case = interruption_case
    first = Path(case.invoke())
    original = first.read_bytes(), first.stat().st_mtime_ns
    case.primary = KeyboardInterrupt('second attempt interrupted')
    second = Path(case.invoke())
    assert first != second and second.parent.parent == first.parent.parent
    assert (first.read_bytes(), first.stat().st_mtime_ns) == original
    assert set(first.parent.parent.glob('*/receipt.json')) == {first, second}


def test_normal_worker_completion_does_not_create_interruption_receipts(interruption_case):
    case = interruption_case
    case.mode = 'normal'
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in case.study.rglob('*') if path.is_file()}
    directories = sorted(path.relative_to(case.study).as_posix()
                         for path in case.study.rglob('*') if path.is_dir())
    assert schedule.launch_jobs(case.study, case.jobs, round_size=2, console='shared') == [0, 0]
    assert case.spawned == ['first', 'second']
    assert not (case.study / 'scripts/launch-attempts').exists()
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in case.study.rglob('*') if path.is_file()} == before
    assert sorted(path.relative_to(case.study).as_posix()
                  for path in case.study.rglob('*') if path.is_dir()) == directories
