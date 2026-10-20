"""CPU-only hang controls for the local CLI and workflow execution boundaries."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import conftest
from tools import ci_local


@pytest.mark.parametrize('platform,proc_available,waitid_available', [
    ('darwin', False, False), ('linux', False, True), ('linux', True, False)])
def test_unsupported_ownership_platform_refuses_before_start(monkeypatch, platform, proc_available, waitid_available):
    from types import SimpleNamespace
    from tools import _process_tree as ownership

    environment = SimpleNamespace(name='posix')
    if waitid_available:
        environment.waitid = lambda: None
        environment.WNOWAIT = 0
    monkeypatch.setattr(ownership, 'os', environment)
    monkeypatch.setattr(ownership, 'sys', SimpleNamespace(platform=platform))
    monkeypatch.setattr(ownership, 'Path', lambda path: SimpleNamespace(is_dir=lambda: proc_available))
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported ownership must not start a command')
    monkeypatch.setattr(ownership.subprocess, 'Popen', forbidden)
    tree = ownership.ProcessTree()
    with pytest.raises(NotImplementedError, match='Bounded launch requires'):
        tree.start(['not-executed'])
    assert tree.proc is None


def test_cli_timeout_preserves_partial_output(tmp_path, monkeypatch):
    (tmp_path / 'hung_cli.py').write_text(
        'import sys, time\nprint("started", flush=True)\n'
        'print("stderr-before-hang", file=sys.stderr, flush=True)\ntime.sleep(30)\n', encoding='utf-8')
    monkeypatch.setattr(conftest, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        conftest.cli(entry='hung_cli', timeout=1)
    assert time.monotonic() - start < 10
    assert 'started' in caught.value.stdout
    evidence, = (tmp_path / '.tmp/cli-timeouts').iterdir()
    assert json.loads((evidence / 'command.json').read_text())['timeout'] == 1
    assert (evidence / 'stdout.log').read_text() == 'started\n'
    assert (evidence / 'stderr.log').read_text() == 'stderr-before-hang\n'
    assert json.loads((evidence / 'command.json').read_text())['cleanup_confirmed']


def test_cli_keeps_normal_failure_contract(tmp_path, monkeypatch):
    (tmp_path / 'failed_cli.py').write_text(
        'import sys\nprint("expected failure")\nsys.exit(7)\n', encoding='utf-8')
    monkeypatch.setattr(conftest, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    assert conftest.cli(entry='failed_cli', timeout=5).returncode == 7
    with pytest.raises(AssertionError, match='expected failure'):
        conftest.cli(entry='failed_cli', timeout=5, check=True)


def test_cli_cleanup_failure_is_retained_without_replacing_timeout(tmp_path, monkeypatch):
    (tmp_path / 'hung_cli.py').write_text(
        'import time\nprint("partial", flush=True)\ntime.sleep(30)\n', encoding='utf-8')
    monkeypatch.setattr(conftest, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))

    def denied(*args, **kwargs):
        raise PermissionError('controlled cleanup access denied')

    monkeypatch.setattr(ci_local.ProcessTree, 'terminate', denied)
    with pytest.raises(subprocess.TimeoutExpired):
        conftest.cli(entry='hung_cli', timeout=1)
    evidence, = (tmp_path / '.tmp/cli-timeouts').iterdir()
    body = json.loads((evidence / 'command.json').read_text())
    assert not body['cleanup_confirmed'] and body['direct_process_returncode'] is not None
    assert body['cleanup_errors'][0]['phase'] == 'process-tree'
    assert 'controlled cleanup access denied' in body['cleanup_errors'][0]['error']
    assert (evidence / 'stdout.log').read_text() == 'partial\n'


def test_timeout_stops_owned_child(tmp_path):
    marker = tmp_path / 'child-survived.txt'
    child = f'import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text("alive")'
    parent = ('import subprocess, sys, time; '
              f'subprocess.Popen([sys.executable, "-c", {child!r}]); '
              'print("child-started", flush=True); time.sleep(30)')
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        ci_local.sh([sys.executable, '-u', '-c', parent], timeout=1, capture_output=True, text=True)
    assert 'child-started' in caught.value.stdout
    time.sleep(2.2)
    assert not marker.exists(), 'the deadline left an owned child running'


def test_expired_budget_does_not_start_a_command(monkeypatch):
    monkeypatch.setattr(ci_local.subprocess, 'Popen', lambda *a, **k: pytest.fail('budget already exhausted'))
    with pytest.raises(subprocess.TimeoutExpired):
        ci_local.sh(['unused'], timeout=0)


def run_workflow(tmp_path, monkeypatch, steps, minutes, *, fail_cleanup=False):
    """Replace only git/setup/shell selection; the step process and deadline are real."""
    workflow = tmp_path / 'workflow.yml'
    workflow.write_text(yaml.safe_dump({'jobs': {'cpu-checks': {
        'timeout-minutes': minutes, 'steps': steps}}}), encoding='utf-8')
    monkeypatch.setattr(ci_local, 'ROOT', tmp_path)
    monkeypatch.setattr(ci_local, 'fcntl', None)
    monkeypatch.setattr(sys, 'argv', ['ci_local.py', '--workflow', str(workflow)])
    original = ci_local.sh
    commands = []

    def invoke(cmd, **kwargs):
        commands.append((cmd, kwargs['timeout']))
        if cmd[:3] == ['git', 'worktree', 'add']:
            Path(cmd[-2]).mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[:3] == ['git', 'worktree', 'remove']:
            if fail_cleanup:
                raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[:3] == ['bash', '-e', '-c']
        return original([sys.executable, '-u', '-c', cmd[-1]], **kwargs)

    monkeypatch.setattr(ci_local, 'sh', invoke)
    return commands


def test_local_ci_shares_the_job_deadline_and_keeps_timeout_evidence(tmp_path, monkeypatch):
    commands = run_workflow(tmp_path, monkeypatch, [
        {'name': 'first', 'run': 'import time; time.sleep(0.3)'},
        {'name': 'hung', 'run': 'import time; print("step-started", flush=True); time.sleep(30)'},
        {'name': 'must-not-start', 'run': 'raise AssertionError("budget expired")'},
    ], minutes=0.025)
    start = time.monotonic()
    assert ci_local.main() == 1
    assert time.monotonic() - start < 10
    shell = [timeout for cmd, timeout in commands if cmd[0] == 'bash']
    assert len(shell) == 2 and 0 < shell[1] < shell[0] <= 1.5
    run, = (tmp_path / '.tmp/ci_local').glob('*-*')
    assert [r[2] for r in json.loads((run / 'summary.json').read_text())] == ['ok', 'TIMEOUT']
    log = (run / 'cpu-checks/step-01.log').read_text().splitlines()
    assert log[0] == 'step-started' and json.loads(log[-1])['cleanup_errors'] == []
    assert (run / 'cpu-checks/repo').is_dir()


def test_local_ci_step_deadline_caps_job_budget(tmp_path, monkeypatch):
    run_workflow(tmp_path, monkeypatch, [
        {'name': 'hung', 'timeout-minutes': 0.01,
         'run': 'import time; print("step-started", flush=True); time.sleep(30)'},
    ], minutes=1)
    start = time.monotonic()
    assert ci_local.main() == 1
    assert time.monotonic() - start < 10
    run, = (tmp_path / '.tmp/ci_local').glob('*-*')
    assert json.loads((run / 'summary.json').read_text())[0][2] == 'TIMEOUT'


def test_local_ci_failure_keeps_output_without_keep_flag(tmp_path, monkeypatch):
    run_workflow(tmp_path, monkeypatch, [
        {'name': 'failed', 'run': 'import sys; print("failed-step"); sys.exit(7)'},
    ], minutes=1)
    assert ci_local.main() == 1
    run, = (tmp_path / '.tmp/ci_local').glob('*-*')
    assert json.loads((run / 'summary.json').read_text())[0][2] == 'FAILED (7)'
    assert (run / 'cpu-checks/step-00.log').read_text() == 'failed-step\n'


def test_local_ci_success_still_cleans_its_worktree(tmp_path, monkeypatch):
    commands = run_workflow(tmp_path, monkeypatch, [{'run': 'print("ok")'}], minutes=1)
    assert ci_local.main() == 0
    assert any(cmd[:3] == ['git', 'worktree', 'remove'] for cmd, _ in commands)
    assert not list((tmp_path / '.tmp/ci_local').glob('*-*'))


def test_local_ci_cleanup_failure_keeps_evidence(tmp_path, monkeypatch):
    run_workflow(tmp_path, monkeypatch, [{'run': 'print("ok")'}], minutes=1, fail_cleanup=True)
    with pytest.raises(subprocess.TimeoutExpired):
        ci_local.main()
    run, = (tmp_path / '.tmp/ci_local').glob('*-*')
    assert 'TimeoutExpired' in json.loads((run / 'summary.json').read_text())[-1][2]


def test_local_ci_artifact_step_keeps_reports_after_successful_cleanup(tmp_path, monkeypatch):
    commands = run_workflow(tmp_path, monkeypatch, [
        {'run': 'from pathlib import Path; p=Path(".tmp/test-reports/run"); '
                'p.mkdir(parents=True); (p/"events.jsonl").write_text("phase evidence")'},
        {'name': 'reports', 'uses': 'actions/upload-artifact@v4',
         'with': {'name': 'test-reports', 'path': '.tmp/test-reports/'}},
    ], minutes=1)
    assert ci_local.main() == 0
    assert any(cmd[:3] == ['git', 'worktree', 'remove'] for cmd, _ in commands)
    run, = (tmp_path / '.tmp/ci_local').glob('*-*')
    assert (run / 'cpu-checks/artifacts/test-reports/run/events.jsonl').read_text() == 'phase evidence'


@pytest.mark.parametrize('phase', ['process-tree', 'direct-process', 'output-drain', 'ownership-handle'])
@pytest.mark.parametrize('interrupt', [False, True])
def test_cleanup_failure_preserves_the_primary_error(monkeypatch, phase, interrupt):
    primary = KeyboardInterrupt('controller interrupted') if interrupt else subprocess.TimeoutExpired(['owned'], 1)

    class ControlledProcess:
        pid = 123456
        returncode = None

        def poll(self):
            return self.returncode

        def kill(self):
            if phase == 'direct-process':
                raise PermissionError('controlled direct-process failure')
            self.returncode = -9

        def wait(self, timeout):
            return self.returncode

    proc = ControlledProcess()

    class Tree:
        job = None

        def __init__(self):
            self.proc = proc

        def start(self, *args, **kwargs):
            kwargs['stdout'].write(b'partial output')
            return self.proc

        def wait(self, timeout):
            raise primary

        def terminate(self, timeout):
            if phase == 'process-tree':
                raise PermissionError('controlled process-tree failure')
            return 0

        def close(self):
            if phase == 'ownership-handle':
                raise PermissionError('controlled ownership-handle failure')

    native = ci_local.tempfile.TemporaryFile

    class Capture:
        def __init__(self):
            self.handle = native()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, *args):
            if phase == 'output-drain':
                raise PermissionError('controlled output-drain failure')
            return self.handle.read(*args)

    monkeypatch.setattr(ci_local, 'ProcessTree', Tree)
    monkeypatch.setattr(ci_local.tempfile, 'TemporaryFile', Capture)
    with pytest.raises(type(primary)) as caught:
        ci_local.sh(['owned'], timeout=1, capture_output=True, text=True)
    assert caught.value is primary
    assert primary.cleanup_errors[0]['phase'] == phase
    assert not primary.cleanup_confirmed and primary.process_id == 123456
    assert primary.direct_process_returncode == (None if phase == 'direct-process' else -9)


def test_normal_exit_stops_worker_without_waiting_for_inherited_output(tmp_path):
    worker = 'import time; time.sleep(30); print("WORKER_SURVIVED",flush=True)'
    parent = ('import subprocess,sys; '
              f'subprocess.Popen([sys.executable,"-u","-c",{worker!r}]); '
              'print("direct-exiting",flush=True)')
    start = time.monotonic()
    result = ci_local.sh([sys.executable, '-u', '-c', parent], timeout=2,
                         capture_output=True, text=True)
    (tmp_path / 'owned-exit.json').write_text(json.dumps({
        'argv': result.args, 'direct_returncode': result.returncode, 'stdout': result.stdout,
        'cleanup_confirmed': result.cleanup_confirmed, 'cleanup_errors': result.cleanup_errors,
        'owned_processes_before_cleanup': result.owned_processes_before_cleanup,
    }), encoding='utf-8')
    assert result.returncode == 0 and result.stdout == 'direct-exiting\n'
    assert time.monotonic() - start < 5
    assert result.cleanup_confirmed and not result.cleanup_errors
    assert result.owned_processes_before_cleanup >= 1


@pytest.mark.parametrize('text', [False, True])
def test_capture_does_not_block_large_stdout_and_stderr(text):
    size = 1024 * 1024
    result = ci_local.sh([sys.executable, '-c', f'import os; os.write(1,b"x"*{size}); os.write(2,b"y"*{size})'],
                         timeout=5, capture_output=True, text=text)
    assert result.stdout == ('x' if text else b'x') * size
    assert result.stderr == ('y' if text else b'y') * size
    assert result.cleanup_confirmed


def test_empty_piped_stdin_still_delivers_eof():
    result = ci_local.sh([sys.executable, '-c', 'import sys; print(repr(sys.stdin.read()))'],
                         timeout=3, stdin=subprocess.PIPE, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "''\n"


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_budget_does_not_start_a_command(monkeypatch, value):
    monkeypatch.setattr(ci_local.subprocess, 'Popen', lambda *a, **k: pytest.fail('invalid budget'))
    with pytest.raises(subprocess.TimeoutExpired):
        ci_local.sh(['unused'], timeout=value)


@pytest.mark.parametrize('failure', ['persistence', 'diagnostic'])
def test_cli_evidence_failure_preserves_original_timeout(tmp_path, monkeypatch, failure):
    primary = subprocess.TimeoutExpired(['owned'], 1, output='partial output', stderr='original stderr')
    monkeypatch.setattr(conftest, 'ROOT', tmp_path)

    def timeout(*args, **kwargs):
        raise primary

    monkeypatch.setattr(conftest, 'sh', timeout)
    if failure == 'persistence':
        def denied(*args, **kwargs):
            raise PermissionError('controlled evidence write failure')
        monkeypatch.setattr(Path, 'write_text', denied)
    else:
        class ClosedStderr:
            def write(self, value):
                raise BrokenPipeError('controlled diagnostic failure')

            def flush(self):
                pass
        monkeypatch.setattr(sys, 'stderr', ClosedStderr())
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        conftest.cli(entry='owned', timeout=1)
    assert caught.value is primary
    if failure == 'persistence':
        assert 'controlled evidence write failure' in primary.evidence_errors[0]
    else:
        assert 'BrokenPipeError' in primary.diagnostic_errors[0]


@pytest.mark.parametrize('primary_failure', [False, True], ids=['normal', 'setup-timeout'])
@pytest.mark.parametrize('failure', ['none', 'unlock', 'close', 'summary', 'diagnostic'])
def test_local_ci_final_io_preserves_primary_and_attempts_evidence(tmp_path, monkeypatch, primary_failure, failure):
    _final_io_case(tmp_path, monkeypatch, primary_failure, failure)


def _final_io_case(tmp_path, monkeypatch, primary_failure, failure):
    from types import SimpleNamespace

    native_sh = ci_local.sh
    native_open = Path.open
    native_write = Path.write_text
    native_print = print
    run_workflow(tmp_path, monkeypatch, [{'run': 'print("controlled step")'}], minutes=1)
    sys.argv.append('--keep')
    selected = ci_local.sh
    primary = []
    operations = []
    handles = []
    cleanup_error = PermissionError('controlled final ' + failure + ' failure')

    def invoke(command, **kwargs):
        if primary_failure and command[:3] == ['git', 'worktree', 'add']:
            try:
                return native_sh([sys.executable, '-u', '-c',
                                  'import time; print("setup started", flush=True); time.sleep(30)'],
                                 timeout=0.2, capture_output=True, text=True)
            except subprocess.TimeoutExpired as error:
                primary.append(error)
                raise
        return selected(command, **kwargs)

    def flock(fd, operation):
        if operation == 2:
            operations.append('unlock')
            if failure == 'unlock':
                raise cleanup_error

    class LockFile:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def close(self):
            operations.append('close')
            self.handle.close()
            if failure == 'close':
                raise cleanup_error

    def open_path(path, *args, **kwargs):
        handle = native_open(path, *args, **kwargs)
        if path.name == 'ci_local.lock':
            handle = LockFile(handle)
            handles.append(handle)
        return handle

    def write_path(path, *args, **kwargs):
        if path.name == 'summary.json':
            operations.append('summary')
            if failure == 'summary':
                raise cleanup_error
        return native_write(path, *args, **kwargs)

    def diagnostic(*args, **kwargs):
        if args and str(args[0]).startswith('kept local CI evidence:'):
            operations.append('diagnostic')
            if failure == 'diagnostic':
                raise cleanup_error
        return native_print(*args, **kwargs)

    monkeypatch.setattr(ci_local, 'sh', invoke)
    monkeypatch.setattr(ci_local, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_UN=2, flock=flock))
    monkeypatch.setattr(Path, 'open', open_path)
    monkeypatch.setattr(Path, 'write_text', write_path)
    monkeypatch.setattr(ci_local, 'print', diagnostic, raising=False)
    caught = None
    returned = None
    try:
        returned = ci_local.main()
    except BaseException as error:
        caught = error
    run, = (tmp_path / '.tmp/ci_local').glob('*-*')
    summary = run / 'summary.json'
    native_write(tmp_path / 'final-io-observation.json', json.dumps({
        'primary_failure': primary_failure, 'failure': failure, 'returned': returned,
        'raised': repr(caught), 'same_primary': bool(primary and caught is primary[0]),
        'operations': operations, 'lock_closed': bool(handles and handles[0].handle.closed),
        'summary_exists': summary.is_file(), 'summary': summary.read_text() if summary.is_file() else None,
        'notes': getattr(caught, '__notes__', []),
    }, indent=2), encoding='utf-8')
    if primary_failure:
        assert len(primary) == 1 and primary[0].cleanup_confirmed
        assert caught is primary[0], 'final I/O replaced the real owned setup timeout'
    elif failure == 'none':
        assert caught is None and returned == 0
    else:
        assert caught is cleanup_error
    assert operations == ['unlock', 'close', 'summary', 'diagnostic']
    assert handles[0].handle.closed
    assert summary.is_file() == (failure != 'summary')
    if summary.is_file():
        rows = json.loads(summary.read_text())
        assert rows and ('TimeoutExpired' in rows[-1][2] if primary_failure else rows[-1][2] == 'ok')
    if primary_failure and failure != 'none':
        assert any('controlled final ' + failure in note for note in primary[0].__notes__)


@pytest.mark.parametrize('failure', ['none', 'unlock', 'close', 'summary', 'diagnostic'])
def test_local_ci_final_io_does_not_treat_callers_exception_as_its_primary(tmp_path, monkeypatch, failure):
    ambient = RuntimeError('caller is handling this exception')
    try:
        raise ambient
    except RuntimeError:
        _final_io_case(tmp_path, monkeypatch, False, failure)
    assert not getattr(ambient, '__notes__', [])
