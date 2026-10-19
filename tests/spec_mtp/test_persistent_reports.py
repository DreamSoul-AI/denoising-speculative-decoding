"""Real pytest phase failures and hard interruption retain recoverable evidence."""

import json
import os
import sys
from pathlib import Path

import pytest

import conftest
from tools.ci_local import sh

ROOT = Path(__file__).resolve().parents[2]


def child_pytest(source, tmp_path, *, relative_report=False):
    test = tmp_path / 'test_controlled.py'
    test.write_text(source, encoding='utf-8')
    reports = tmp_path / 'reports'
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'tests/spec_mtp'),
           'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONUTF8': '1'}
    cmd = [sys.executable, '-m', 'pytest', '-p', 'conftest', str(test),
           '--test-report-dir', os.path.relpath(reports, ROOT) if relative_report else str(reports),
           '--basetemp', str(tmp_path / 'pytest-temp')]
    result = sh(cmd, timeout=20, check=False, capture_output=True, text=True, cwd=ROOT, env=env)
    (tmp_path / 'child.stdout.log').write_text(result.stdout, encoding='utf-8')
    (tmp_path / 'child.stderr.log').write_text(result.stderr, encoding='utf-8')
    return result, reports


def test_relative_report_directory_survives_test_changing_cwd(tmp_path):
    result, reports = child_pytest('''
import os
def test_changes_cwd(tmp_path):
    os.chdir(tmp_path)
''', tmp_path, relative_report=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((reports / 'report.json').read_text())
    assert report['complete'] and not report['missing'] and not report['retry']


def test_real_pytest_phase_failures_and_skip_are_persisted(tmp_path):
    result, reports = child_pytest('''
import pytest
@pytest.fixture
def setup_error():
    raise RuntimeError('controlled setup error')
@pytest.fixture
def teardown_error():
    yield
    raise RuntimeError('controlled teardown error')
def test_ok():
    pass
def test_setup(setup_error):
    pass
def test_teardown(teardown_error):
    pass
def test_call_failure():
    assert False, 'controlled call failure'
@pytest.mark.skip(reason='controlled skip reason')
def test_skip():
    pass
''', tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    body = json.loads((reports / 'report.json').read_text())
    cases = {case['test_id'].rsplit('::', 1)[-1]: case for case in body['cases']}
    assert len(json.loads((reports / 'manifest.json').read_text())['tests']) == 5
    assert {name: row['status'] for name, row in cases.items()} == {
        'test_ok': 'passed', 'test_setup': 'failed', 'test_teardown': 'failed',
        'test_call_failure': 'failed', 'test_skip': 'skipped'}
    teardown = cases['test_teardown']['attempts'][0]['phases']
    assert teardown['call']['outcome'] == 'passed' and teardown['teardown']['outcome'] == 'failed'
    assert 'controlled teardown error' in teardown['teardown']['detail']
    assert 'controlled skip reason' in cases['test_skip']['attempts'][0]['phases']['setup']['detail']
    assert all(event['duration_seconds'] >= 0 for case in cases.values()
               for event in case['attempts'][0]['phases'].values())
    assert not body['complete'] and not body['missing'] and len(body['retry']) == 4
    before = [(reports / name).read_bytes() for name in ('manifest.json', 'events.jsonl')]
    rebuilt = conftest.rebuild_test_report(reports)
    assert rebuilt == body
    assert before == [(reports / name).read_bytes() for name in ('manifest.json', 'events.jsonl')]


def test_hard_exit_after_call_pass_keeps_incomplete_and_missing_items(tmp_path):
    result, reports = child_pytest('''
import os
import pytest
@pytest.fixture
def interrupted():
    yield
    os._exit(23)
def test_a_done():
    pass
def test_b_interrupted(interrupted):
    pass
def test_c_missing():
    pass
''', tmp_path)
    assert result.returncode == 23, result.stdout + result.stderr
    assert not (reports / 'report.json').exists()
    result = sh([sys.executable, '-m', 'pytest', 'tests/spec_mtp', '--test-report-only', str(reports)],
                timeout=20, check=False, capture_output=True, text=True, cwd=ROOT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONUTF8': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    body = json.loads((reports / 'report.json').read_text())
    assert [case['status'] for case in body['cases']] == ['passed', 'incomplete', 'missing']
    assert len(body['missing']) == 1 and len(body['retry']) == 2 and not body['complete']
    assert body['exit_status'] is None


def test_rebuild_preserves_attempt_history_and_rejects_duplicate_phases(tmp_path):
    manifest = {'tests': ['one'], 'collection_complete': True}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    events = [{'test_id': 'one', 'attempt': attempt, 'phase': phase, 'outcome': outcome,
               'duration_seconds': 0.01}
              for attempt in (1, 2) for phase in ('setup', 'call', 'teardown')
              for outcome in ['failed' if attempt == 1 and phase == 'teardown' else 'passed']]
    path = tmp_path / 'events.jsonl'
    path.write_text(''.join(json.dumps(event) + '\n' for event in events)
                    + json.dumps({'event': 'session_finish', 'exit_status': 0}) + '\n', encoding='utf-8')
    body = conftest.rebuild_test_report(tmp_path)
    assert [row['status'] for row in body['cases'][0]['attempts']] == ['failed', 'passed']
    assert body['complete'] and not body['retry']
    with path.open('a', encoding='utf-8') as output:
        output.write(json.dumps(events[-1]) + '\n')
    with pytest.raises(ValueError, match='Duplicate test phase'):
        conftest.rebuild_test_report(tmp_path)


def test_truncated_last_event_cannot_be_reported_complete(tmp_path):
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'tests': ['one'], 'collection_complete': True}), encoding='utf-8')
    (tmp_path / 'events.jsonl').write_text('{"test_id":', encoding='utf-8')
    body = conftest.rebuild_test_report(tmp_path)
    assert body['incomplete_event_log'] and not body['complete']
    assert body['missing'] == body['retry'] == ['one']


@pytest.mark.parametrize('raw', [b'{broken}\n', b'{broken}\n{"event":"session_finish","exit_status":0}\n'])
def test_rebuild_rejects_terminated_or_middle_corruption(tmp_path, raw):
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'tests': ['one'], 'collection_complete': True}), encoding='utf-8')
    path = tmp_path / 'events.jsonl'
    path.write_bytes(raw)
    with pytest.raises(json.JSONDecodeError):
        conftest.rebuild_test_report(tmp_path)
    assert path.read_bytes() == raw


def test_unterminated_unicode_tail_is_kept_as_incomplete_raw_evidence(tmp_path):
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'tests': ['one'], 'collection_complete': True}), encoding='utf-8')
    raw = b'{"detail":"\xf0\x9f'
    path = tmp_path / 'events.jsonl'
    path.write_bytes(raw)
    body = conftest.rebuild_test_report(tmp_path)
    assert body['incomplete_event_log'] and not body['complete']
    assert body['truncated_final_event'] == repr(raw) and path.read_bytes() == raw


def test_manifest_hard_exit_during_update_preserves_previous_complete_bytes(tmp_path):
    path = tmp_path / 'manifest.json'
    original = b'{"tests":["original"],"collection_complete":true}\n'
    path.write_bytes(original)
    driver = (
        'import sys,os\nfrom pathlib import Path\n'
        f'sys.path.insert(0,{str(ROOT / "tests/spec_mtp")!r})\nimport conftest\n'
        'native=Path.write_text\n'
        'def interrupted(path,*args,**kwargs):\n'
        '    native(path,"partial update",encoding="utf-8")\n    os._exit(33)\n'
        'Path.write_text=interrupted\n'
        f'conftest._write_manifest(Path({str(path)!r}),{{"tests":["new"],"collection_complete":True}})\n')
    result = sh([sys.executable, '-c', driver], timeout=10, capture_output=True, check=False, text=True,
                cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'})
    assert result.returncode == 33, result.stdout + result.stderr
    assert path.read_bytes() == original
    partial, = tmp_path.glob('.manifest.json-*.tmp')
    assert partial.read_text() == 'partial update'


@pytest.mark.parametrize('exit_status', [None, 2])
def test_completed_cases_do_not_replace_missing_or_failed_session_exit(tmp_path, exit_status):
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'tests': ['one'], 'collection_complete': True}), encoding='utf-8')
    events = [{'test_id': 'one', 'attempt': 1, 'phase': phase, 'outcome': 'passed', 'duration_seconds': 0.01}
              for phase in ('setup', 'call', 'teardown')]
    if exit_status is not None:
        events.append({'event': 'session_finish', 'exit_status': exit_status})
    (tmp_path / 'events.jsonl').write_text(
        ''.join(json.dumps(event) + '\n' for event in events), encoding='utf-8')
    body = conftest.rebuild_test_report(tmp_path)
    assert body['cases'][0]['status'] == 'passed' and not body['complete']
    assert body['exit_status'] == exit_status
