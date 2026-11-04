"""Diagnostic records and legacy read-only views share no scientific state."""
import json
from pathlib import Path

import pytest

from spec_mtp.structure.system.logger import Logger
from spec_mtp.structure.artifact.readout.logs import format_logs
from spec_mtp.structure.artifact.readout.status import diagnostic_lines, log_path, _note_from_log


@pytest.mark.parametrize('separator', ['\n', '\u2028', '\u2029'])
def test_logger_writes_real_identity_and_escapes_payload(tmp_path, capsys, separator):
    logger = Logger(tmp_path / 'runs' / 'r1' / 'assets')
    logger.info('hello' + separator + 'FORGED\x1b[31m password=hidden-secret')
    raw = logger.path.read_text(encoding='utf-8')
    assert len(raw.splitlines()) == 1 and logger.path.name == 'events.jsonl'
    record = json.loads(raw)
    assert record['timestamp'].endswith('Z') and record['level'] == 'INFO'
    assert record['event'] == 'task.progress' and record['task_id'] == 'r1'
    assert record['app_name'] == record['logger'] == 'spec_mtp'
    assert record['pid'] > 0 and record['runtime_id'] and record['process_name'] and record['app_version']
    assert 'hidden-secret' not in raw + capsys.readouterr().out
    assert separator not in record['message'] and '\x1b' not in record['message']


def test_exception_is_one_record_without_exception_payload(tmp_path):
    logger = Logger(tmp_path)
    try:
        raise ValueError('prompt=private-prompt-secret')
    except ValueError as error:
        logger.exception('execute failed', error)
    rows = logger.path.read_text(encoding='utf-8').splitlines()
    assert len(rows) == 1 and 'private-prompt-secret' not in rows[0]
    record = json.loads(rows[0])
    assert record['exception_type'] == 'ValueError'
    assert 'test_exception_is_one_record' in record['traceback']


def test_log_write_failure_is_reported_to_the_caller(tmp_path, monkeypatch):
    logger = Logger(tmp_path)
    def deny(*args, **kwargs):
        raise PermissionError('blocked')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', deny)
        with pytest.raises(PermissionError):
            logger.info('cannot persist')


def test_new_and_legacy_logs_remain_readable_without_rewrite(tmp_path):
    logger = Logger(tmp_path / 'runs' / 'new' / 'assets')
    logger.error('execute failed')
    old = tmp_path / 'runs' / 'old' / 'assets' / 'logs' / 'run.log'
    old.parent.mkdir(parents=True)
    content = '2026-01-01T00:00:00Z INFO old [flow] historical\n'
    old.write_text(content, encoding='utf-8')
    index = {'study': 's', 'experiments': [{'runs': [{'id': 'new'}, {'id': 'old'}]}]}
    (tmp_path / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    assert log_path(tmp_path, {}, 'old') == old
    assert _note_from_log(logger.path, errors_only=True) == '[error] execute failed'
    assert 'historical' in format_logs(tmp_path) and 'execute failed' in format_logs(tmp_path)
    assert old.read_text(encoding='utf-8') == content


def test_truncated_diagnostic_is_not_rendered_as_complete(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text('{"event":"task.failed"}', encoding='utf-8')
    with pytest.raises(ValueError, match='truncated'):
        diagnostic_lines(path)
    assert _note_from_log(path) == '[error] diagnostic log unavailable'


def test_log_order_compares_utc_instants_across_legacy_offsets(tmp_path):
    logger = Logger(tmp_path / 'runs' / 'new' / 'assets')
    logger.info('new record')
    row = json.loads(logger.path.read_text(encoding='utf-8'))
    row['timestamp'] = '2026-01-01T00:00:00Z'
    logger.path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    old = tmp_path / 'runs' / 'old' / 'assets' / 'logs' / 'run.log'
    old.parent.mkdir(parents=True)
    old.write_text('2026-01-01T07:00:00+08:00 INFO old [flow] earlier\n', encoding='utf-8')
    index = {'experiments': [{'runs': [{'id': 'new'}, {'id': 'old'}]}]}
    (tmp_path / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    lines = format_logs(tmp_path).splitlines()
    assert 'earlier' in lines[0] and 'new record' in lines[1]


@pytest.mark.parametrize('legacy,explicit', [(True, False), (False, False), (True, True), (False, True)])
def test_report_links_follow_actual_new_or_legacy_log(tmp_path, legacy, explicit):
    from spec_mtp.structure.artifact.readout.report import _log_links
    name = 'run.log' if legacy else 'events.jsonl'
    path = tmp_path / 'runs' / 'r1' / 'assets' / 'logs' / name
    path.parent.mkdir(parents=True)
    path.write_text('', encoding='utf-8')
    run = {'id': 'r1'}
    if explicit:
        opposite = 'events.jsonl' if legacy else 'run.log'
        run['log'] = 'runs/r1/assets/logs/' + opposite
    index = {'experiments': [{'runs': [run]}]}
    original = json.dumps(index)
    assert _log_links(index, tmp_path) == {'r1': '../runs/r1/assets/logs/' + name}
    assert json.dumps(index) == original


def test_malformed_diagnostic_stops_the_logs_command(tmp_path):
    logger = Logger(tmp_path / 'runs' / 'r1' / 'assets')
    logger.path.write_text('{broken}\n', encoding='utf-8')
    index = {'experiments': [{'runs': [{'id': 'r1'}]}]}
    (tmp_path / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    with pytest.raises(ValueError):
        format_logs(tmp_path)


def test_report_rejects_a_log_link_outside_the_study(tmp_path):
    from spec_mtp.structure.artifact.readout.report import _log_links
    index = {'experiments': [{'runs': [{'id': 'r1', 'log': '../outside/events.jsonl'}]}]}
    with pytest.raises(ValueError):
        _log_links(index, tmp_path)


def test_loggers_in_one_process_share_runtime_but_keep_run_identity(tmp_path):
    first = Logger(tmp_path / 'runs' / 'r1' / 'assets')
    second = Logger(tmp_path / 'runs' / 'r2' / 'assets')
    assert first.runtime_id == second.runtime_id
    assert first.run_id == 'r1' and second.run_id == 'r2'


@pytest.mark.parametrize('escape', ['relative', 'absolute', 'resolved_link', 'fallback_link'])
def test_all_log_readers_reject_resolved_paths_outside_study(tmp_path, monkeypatch, escape):
    study = tmp_path / 'study'
    study.mkdir()
    outside = tmp_path / 'private.log'
    outside.write_text('private content', encoding='utf-8')
    if escape in ('relative', 'absolute'):
        raw = '../private.log' if escape == 'relative' else str(outside)
        with pytest.raises(ValueError):
            log_path(study, {'log': raw}, 'r1')
    else:
        candidate = study / 'runs/r1/assets/logs' / ('run.log' if escape == 'fallback_link' else 'events.jsonl')
        resolve = Path.resolve
        def resolved(path, *args, **kwargs):
            return outside if path == candidate else resolve(path, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'resolve', resolved)
            with pytest.raises(ValueError):
                log_path(study, {}, 'r1')
