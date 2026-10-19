"""Shared helpers: copy the smoke Study into a temp dir and run it through RPipe."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tools.ci_local import sh

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / 'studies' / 'smoke'


def rebuild_test_report(directory):
    """Rebuild categorical test outcomes from the final collection and phase events."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    required = manifest['tests']
    if len(required) != len(set(required)):
        raise ValueError('Duplicate required test IDs')
    attempts = {test_id: {} for test_id in required}
    incomplete_log = False
    exit_status = None
    lines = (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            incomplete_log = True
            break
        if event.get('event') == 'session_finish':
            exit_status = event['exit_status']
            continue
        if (event['test_id'] not in attempts or event['attempt'] < 1
                or event['phase'] not in {'setup', 'call', 'teardown'}
                or event['outcome'] not in {'passed', 'failed', 'skipped'}):
            raise ValueError(f'Invalid test phase event: {event}')
        phases = attempts[event['test_id']].setdefault(event['attempt'], {})
        if event['phase'] in phases:
            raise ValueError('Duplicate test phase in one attempt')
        phases[event['phase']] = event
    cases = []
    for test_id, records in attempts.items():
        rows = []
        for attempt, phases in sorted(records.items()):
            outcomes = {event['outcome'] for event in phases.values()}
            status = ('failed' if 'failed' in outcomes else 'skipped' if 'skipped' in outcomes
                      else 'passed' if set(phases) == {'setup', 'call', 'teardown'} else 'incomplete')
            rows.append({'attempt': attempt, 'status': status, 'phases': phases})
        cases.append({'test_id': test_id, 'attempts': rows, 'status': rows[-1]['status'] if rows else 'missing'})
    body = {'complete': exit_status == 0 and manifest['collection_complete'] and bool(required) and not incomplete_log
                       and all(case['status'] == 'passed' for case in cases),
            'cases': cases, 'missing': [case['test_id'] for case in cases if case['status'] == 'missing'],
            'retry': [case['test_id'] for case in cases if case['status'] != 'passed'],
            'incomplete_event_log': incomplete_log, 'exit_status': exit_status}
    (directory / 'report.json').write_text(json.dumps(body, indent=2) + '\n', encoding='utf-8')
    return body


class TestReport:
    def __init__(self, directory):
        self.directory = directory
        self.attempts = {}

    def pytest_collection_finish(self, session):
        (self.directory / 'manifest.json').write_text(
            json.dumps({'tests': [item.nodeid for item in session.items],
                        'collection_complete': not bool(session.testsfailed)}) + '\n', encoding='utf-8')

    def pytest_runtest_logreport(self, report):
        if report.when == 'setup':
            self.attempts[report.nodeid] = self.attempts.get(report.nodeid, 0) + 1
        event = {'test_id': report.nodeid, 'attempt': self.attempts[report.nodeid],
                 'phase': report.when, 'outcome': report.outcome, 'duration_seconds': report.duration}
        if report.failed or report.skipped:
            event['detail'] = str(report.longrepr)
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(event) + '\n')

    def pytest_sessionfinish(self, session, exitstatus):
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps({'event': 'session_finish', 'exit_status': int(exitstatus)}) + '\n')
        rebuild_test_report(self.directory)


def pytest_addoption(parser):
    parser.addoption('--test-report-dir', help='new directory for the selected tests and phase events')
    parser.addoption('--test-report-only', help='rebuild report.json from an existing report directory, then exit')


def pytest_configure(config):
    previous = config.getoption('--test-report-only')
    if previous:
        rebuild_test_report(previous)
        pytest.exit(f'Rebuilt test report: {previous}', returncode=0)
    directory = Path(config.getoption('--test-report-dir') or
                     ROOT / '.tmp/test-reports' / f'{time.time_ns()}-{os.getpid()}')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'manifest.json').write_text('{"tests": [], "collection_complete": false}\n', encoding='utf-8')
    (directory / 'events.jsonl').touch()
    config.pluginmanager.register(TestReport(directory), 'spec-mtp-test-report')


def copy_study(tmp_path, patch=None, name='smoke_copy'):
    """Copy declarations, recipe and flow (no runs) and apply a study.yaml patch."""
    dest = tmp_path / name
    shutil.copytree(SMOKE, dest, ignore=shutil.ignore_patterns(
        'runs', 'shared', 'scripts', 'index.json', 'process.json', 'provenance.json',
        'spec_mtp_*.json', 'activity.json', '__pycache__', 'docs'))
    study = yaml.safe_load((dest / 'study.yaml').read_text(encoding='utf-8'))
    study['study'] = name
    study['fixed']['algorithm']['config']['version'] = name
    if patch:
        patch(study)
    (dest / 'study.yaml').write_text(yaml.safe_dump(study, sort_keys=False), encoding='utf-8')
    return dest


def cli(*args, entry='spec_mtp', check=False, timeout=300):
    env = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg'}
    cmd = [sys.executable, '-m', entry, *map(str, args)]
    try:
        proc = sh(cmd, check=False, capture_output=True, text=True, env=env, cwd=ROOT, timeout=timeout)
    except subprocess.SubprocessError as error:
        evidence = ROOT / '.tmp' / 'cli-timeouts' / f'{time.time_ns()}-{os.getpid()}'
        evidence.mkdir(parents=True)
        (evidence / 'command.json').write_text(json.dumps({
            'argv': cmd, 'timeout': timeout, 'error': str(error),
            'cleanup_errors': getattr(error, 'cleanup_errors', []),
            'cleanup_confirmed': getattr(error, 'cleanup_confirmed', False),
            'process_id': getattr(error, 'process_id', None),
            'direct_process_returncode': getattr(error, 'direct_process_returncode', None),
        }) + '\n', encoding='utf-8')
        for name, value in [('stdout', error.stdout), ('stderr', error.stderr)]:
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            (evidence / f'{name}.log').write_text(value or '', encoding='utf-8')
        print(f'CLI timeout evidence: {evidence}', file=sys.stderr)
        raise
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout[-4000:] + proc.stderr[-4000:])
    return proc


@pytest.fixture
def study_copy(tmp_path):
    return lambda patch=None, name='smoke_copy': copy_study(tmp_path, patch, name)
