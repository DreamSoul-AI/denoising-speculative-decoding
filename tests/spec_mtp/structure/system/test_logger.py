"""Diagnostic records and legacy read-only views share no scientific state."""
import json
from datetime import datetime, timedelta, timezone
import os
import sys
from types import SimpleNamespace
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

from tests.conftest import saved_diagnostic

def test_log_segments_preserve_closed_bytes_and_share_one_thread_safe_sequence(tmp_path, monkeypatch):
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from spec_mtp.structure.system import logger as implementation
    from spec_mtp.structure.artifact.asset.tree import diagnostic_files

    limit = 1024
    monkeypatch.setattr(implementation, '_SEGMENT_BYTES', limit)
    assets = tmp_path / 'runs/r1/assets'
    anchor = assets / 'logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    seed = saved_diagnostic('s' * (limit - len(saved_diagnostic('')))).encode()
    assert len(seed) == limit
    anchor.write_bytes(seed)
    stamp = anchor.stat().st_mtime_ns
    first, second = Logger(assets), Logger(assets)
    assert anchor.read_bytes() == seed and anchor.stat().st_mtime_ns == stamp
    for index in range(8):
        first.info(f'warm-{index}')
    frozen = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in diagnostic_files(anchor)[:-1]}
    assert len(frozen) >= 2
    third = Logger(assets)
    instances = (first, second, third)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: instances[index % 3].info(f'case-{index}'), range(24)))
    parts = diagnostic_files(anchor)
    assert [p.name for p in parts] == ['events.jsonl'] + [f'events.{i:06d}.jsonl' for i in range(1, len(parts))]
    assert all(p.stat().st_size <= limit for p in parts)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in frozen.items())
    records = [json.loads(line) for p in parts for line in p.read_text().splitlines()]
    messages = [row['message'] for row in records[1:]]
    assert len(records) == 33
    assert sorted(messages) == sorted([f'warm-{i}' for i in range(8)] + [f'case-{i}' for i in range(24)])
    assert all(instance.path == anchor and instance.state_dict() == {'path': str(anchor)} for instance in instances)


@pytest.mark.parametrize('fault', ['below_limit', 'exact_limit', 'oversized', 'segment_cap',
                                  'anchor_tail', 'part_tail', 'create', 'flush'])
def test_log_segment_write_admission_preserves_previous_bytes(tmp_path, monkeypatch, capsys, fault):
    from spec_mtp.structure.system import logger as implementation

    logger = Logger(tmp_path / 'assets')
    anchor = logger.path
    seed = saved_diagnostic('seed').encode()
    if fault in ('create', 'flush'):
        monkeypatch.setattr(implementation, '_SEGMENT_BYTES', 1024)
        seed = saved_diagnostic('s' * (1024 - len(saved_diagnostic('')))).encode()
    anchor.write_bytes(seed[:-1] if fault == 'anchor_tail' else seed)
    if fault == 'part_tail':
        anchor.with_name('events.000001.jsonl').write_bytes(seed[:-1])
    if fault == 'segment_cap':
        monkeypatch.setattr(implementation, '_SEGMENT_BYTES', 64)
    target = {'below_limit': 65535, 'exact_limit': 65536, 'oversized': 65537}.get(fault)
    if target is not None:
        dumps = json.dumps
        def sized(record, **kwargs):
            record['padding'] = ''
            record['padding'] = 'x' * (target - 1 - len(dumps(record, **kwargs)))
            return dumps(record, **kwargs)
        from types import SimpleNamespace
        monkeypatch.setattr(implementation, 'json', SimpleNamespace(dumps=sized))
    primary = OSError('injected segment persistence failure')
    if fault in ('create', 'flush'):
        original_open = Path.open
        class FailedFlush:
            def __init__(self, handle): self.handle = handle
            def __getattr__(self, name): return getattr(self.handle, name)
            def __enter__(self): self.handle.__enter__(); return self
            def __exit__(self, *args): return self.handle.__exit__(*args)
            def flush(self): raise primary
        def opened(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            if path.name == 'events.000001.jsonl' and mode == 'x':
                if fault == 'create': raise primary
                return FailedFlush(original_open(path, *args, **kwargs))
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', opened)
    before = {p: p.read_bytes() for p in anchor.parent.iterdir() if p.is_file()}
    if fault in ('below_limit', 'exact_limit'):
        logger.info('sized')
        line = anchor.read_bytes().splitlines(keepends=True)[-1]
        assert len(line) == target and json.loads(line)['message'] == 'sized'
        assert anchor.read_bytes().startswith(seed)
    else:
        with pytest.raises(OSError if fault in ('create', 'flush') else ValueError) as caught:
            logger.info('must not succeed')
        if fault in ('create', 'flush'): assert caught.value is primary
        assert all(p.read_bytes() == body for p, body in before.items())
        if fault != 'flush': assert set(anchor.parent.iterdir()) == set(before)
        assert not capsys.readouterr().out
    assert logger.path == anchor and logger.state_dict()['path'] == str(anchor)


@pytest.mark.parametrize('alias', ['resolved_legacy', 'hardlinked_legacy'])
def test_log_anchor_rejects_aliases_without_appending_to_legacy_bytes(tmp_path, monkeypatch, alias):
    import os

    assets = tmp_path / 'assets'
    parent = assets / 'logs'
    parent.mkdir(parents=True)
    legacy = parent / 'run.log'
    legacy.write_bytes(b'2026-01-01T00:00:00Z INFO saved [flow] old bytes\n')
    anchor = parent / 'events.jsonl'
    if alias == 'hardlinked_legacy':
        os.link(legacy, anchor)
        assert anchor.stat().st_nlink > 1
    else:
        anchor.write_bytes(b'placeholder\n')
        resolve = Path.resolve
        monkeypatch.setattr(Path, 'resolve', lambda path, *a, **k: legacy if path == anchor else resolve(path, *a, **k))
    prior = legacy.read_bytes()
    anchor_prior = anchor.read_bytes()
    with pytest.raises(ValueError): Logger(assets)
    assert legacy.read_bytes() == prior and anchor.read_bytes() == anchor_prior


@pytest.mark.parametrize('fault', ['constructor', 'sanitize', 'serialize', 'open', 'flush', 'stdout'])
def test_managed_logger_isolates_its_own_diagnostic_failures(tmp_path, monkeypatch, fault):
    from spec_mtp.structure.system import logger as implementation
    assets = tmp_path / 'runs/r1/assets'
    path = assets / 'logs/events.jsonl'
    primary = OSError('prompt=private diagnostic failure')
    original_open = Path.open
    def fail(*args, **kwargs): raise primary
    class FailedFlush:
        def __init__(self, handle): self.handle = handle
        def __getattr__(self, name): return getattr(self.handle, name)
        def __enter__(self): self.handle.__enter__(); return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def flush(self): raise primary
    def opened(candidate, *args, **kwargs):
        if candidate == path:
            if fault in ('constructor', 'open'): raise primary
            handle = original_open(candidate, *args, **kwargs)
            mode = args[0] if args else kwargs.get('mode', 'r')
            return FailedFlush(handle) if fault == 'flush' and mode in ('a', 'x') else handle
        return original_open(candidate, *args, **kwargs)
    logger = None if fault == 'constructor' else Logger(assets, managed=True)
    with monkeypatch.context() as patch:
        if fault in ('constructor', 'open', 'flush'): patch.setattr(Path, 'open', opened)
        elif fault == 'sanitize': patch.setattr(implementation, '_text', fail)
        elif fault == 'serialize': patch.setattr(implementation, 'json', SimpleNamespace(dumps=fail))
        else: patch.setattr(sys, 'stdout', SimpleNamespace(write=fail, flush=fail))
        if logger is None: logger = Logger(assets, managed=True)
        else: logger.info('safe message')
    assert logger.failure_count == 1 and logger.last_error_type == 'OSError'
    assert logger.state_dict() == {'path': str(path)} and not hasattr(logger, 'dropped_count')
    if fault == 'stdout':
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(records) == 1 and records[0]['message'] == 'safe message'


@pytest.mark.parametrize('stderr_fails', [False, True])
def test_managed_logger_reports_first_failure_once_and_keeps_direct_api_strict(tmp_path, monkeypatch, stderr_fails):
    managed, strict = Logger(tmp_path / 'managed', managed=True), Logger(tmp_path / 'strict')
    primary = PermissionError('prompt=private path=D:/private/secret')
    attempted = []
    class Emergency:
        def write(self, text):
            attempted.append(text)
            if stderr_fails: raise OSError('emergency unavailable')
            return len(text)
        def flush(self): pass
    original_open = Path.open
    def opened(path, *args, **kwargs):
        if path in (managed.path, strict.path): raise primary
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', opened)
        patch.setattr(sys, 'stderr', Emergency())
        managed.info('first'); managed.warning('second')
        with pytest.raises(PermissionError) as caught: strict.info('strict')
        assert caught.value is primary
    text = ''.join(attempted)
    assert text.count('unknown') == 1 and 'PermissionError' in text
    assert 'private' not in text and 'D:/' not in text
    assert managed.failure_count == 2 and managed.last_error_type == 'PermissionError'
    assert strict.state_dict() == {'path': str(strict.path)}


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_logging_never_swallows_process_control_exceptions(tmp_path, monkeypatch, managed, error_type):
    from spec_mtp.structure.system import logger as implementation
    logger = Logger(tmp_path, managed=managed)
    primary = error_type('requested interruption')
    def interrupted(*args, **kwargs): raise primary
    with monkeypatch.context() as patch:
        patch.setattr(implementation, '_text', interrupted)
        with pytest.raises(error_type) as caught: logger.info('interrupted')
    assert caught.value is primary and logger.path.read_bytes() == b''
    assert logger.failure_count == 0


@pytest.mark.parametrize('operation', ['report', 'exception'])
def test_managed_logger_covers_report_and_exception_formatting(tmp_path, monkeypatch, operation):
    from spec_mtp.structure.system import logger as implementation
    logger = Logger(tmp_path, managed=True)
    primary = ValueError('prompt=private formatting payload')
    def failed(*args, **kwargs): raise primary
    with monkeypatch.context() as patch:
        if operation == 'exception':
            patch.setattr(implementation.traceback, 'extract_tb', failed)
            logger.exception('safe exception', RuntimeError('original business exception'))
        else: logger.report(SimpleNamespace(display_mean=failed, last=lambda split: {}), 'test')
    assert logger.failure_count == 1 and logger.last_error_type == 'ValueError'
    assert logger.path.read_bytes() == b''


@pytest.fixture
def frozen_diagnostic_clock(monkeypatch):
    from spec_mtp.structure.system import logger as implementation

    clock = {'now': datetime(2026, 11, 6, 12, tzinfo=timezone.utc)}
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock['now'].astimezone(tz) if tz is not None else clock['now'].replace(tzinfo=None)
    monkeypatch.setattr(implementation, 'datetime', Clock)
    return clock


def _diagnostic_policy_bytes(created, cap=4096):
    return (json.dumps({'schema': 1, 'owner': 'run_operator', 'max_bytes': cap,
                        'created_at': created.isoformat().replace('+00:00', 'Z'),
                        'write_until': (created + timedelta(days=7)).isoformat().replace('+00:00', 'Z')},
                       separators=(',', ':')) + '\n').encode('ascii')


def test_managed_capture_policy_is_frozen_across_instances_and_mtime(tmp_path, frozen_diagnostic_clock):
    assets = tmp_path / 'runs/r1/assets'
    first = Logger(assets, managed=True)
    path = first.path.with_name('events.policy.json')
    raw = path.read_bytes()
    policy = json.loads(raw)
    assert len(raw) <= 4096
    assert set(policy) == {'schema', 'owner', 'max_bytes', 'created_at', 'write_until'}
    assert policy['schema'] == 1 and type(policy['schema']) is int
    assert policy['owner'] == 'run_operator' and policy['max_bytes'] == 64 * 1024 * 1024
    created = datetime.fromisoformat(policy['created_at'].replace('Z', '+00:00'))
    deadline = datetime.fromisoformat(policy['write_until'].replace('Z', '+00:00'))
    assert created == frozen_diagnostic_clock['now'] and deadline - created == timedelta(days=7)
    first.info('first')
    os.utime(path, (1, 1))
    frozen_diagnostic_clock['now'] += timedelta(days=1)
    second = Logger(assets, managed=True)
    second.info('second')
    strict = Logger(assets)
    strict.info('strict shares the frozen policy')
    assert path.read_bytes() == raw and path.stat().st_mtime_ns == 1_000_000_000
    assert [json.loads(row)['message'] for row in first.path.read_text().splitlines()] == [
        'first', 'second', 'strict shares the frozen policy']
    assert first.suppressed_records == second.suppressed_records == 0
    assert first.failure_count == second.failure_count == 0
    assert first.state_dict() == second.state_dict() == {'path': str(first.path)}


@pytest.mark.parametrize('managed', [False, True], ids=['strict', 'managed'])
@pytest.mark.parametrize('offset', [-1, 0, 1], ids=['below_cap', 'exact_cap', 'above_cap'])
def test_capture_budget_counts_policy_all_parts_and_next_record(
        tmp_path, capsys, frozen_diagnostic_clock, managed, offset):
    # The probe obtains the real wire length without replacing json.dumps or sizing a fake record.
    probe = Logger(tmp_path / 'probe/runs/r1/assets')
    probe.info('bounded record')
    line_size = probe.path.stat().st_size
    capsys.readouterr()
    assets = tmp_path / 'actual/runs/r1/assets'
    anchor = assets / 'logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    cap = 4096
    policy_path = anchor.with_name('events.policy.json')
    policy_bytes = _diagnostic_policy_bytes(frozen_diagnostic_clock['now'], cap)
    policy_path.write_bytes(policy_bytes)
    saved_size = cap - len(policy_bytes) - line_size + offset
    # Two actual parts make an implementation that counts only its last file fail.
    seed_size = saved_size // 2
    for path, size in ((anchor, seed_size), (anchor.with_name('events.000001.jsonl'), saved_size - seed_size)):
        payload = saved_diagnostic('')
        assert size >= len(payload)
        path.write_bytes(saved_diagnostic('x' * (size - len(payload))).encode('ascii'))
    before = {path: path.read_bytes() for path in anchor.parent.iterdir()}
    logger = Logger(assets, managed=managed)
    if offset <= 0:
        logger.info('bounded record')
        assert sum(path.stat().st_size for path in anchor.parent.iterdir()) == cap + offset
        assert all(path.read_bytes().startswith(raw) for path, raw in before.items())
        if managed:
            assert logger.suppressed_records == logger.failure_count == 0
    else:
        if managed:
            logger.info('bounded record')
            assert logger.suppressed_records == 1 and logger.stop_reason == 'capacity'
            assert logger.failure_count == 0
        else:
            with pytest.raises(ValueError):
                logger.info('bounded record')
        assert {path: path.read_bytes() for path in anchor.parent.iterdir()} == before
        assert not capsys.readouterr().out
    assert policy_path.read_bytes() == policy_bytes


@pytest.mark.parametrize('remaining', [1, 0, -1], ids=['before_deadline', 'at_deadline', 'after_deadline'])
def test_capture_deadline_never_renews_from_reopen_or_file_mtime(
        tmp_path, capsys, frozen_diagnostic_clock, remaining):
    now = frozen_diagnostic_clock['now']
    assets = tmp_path / 'assets'
    anchor = assets / 'logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    created = now - timedelta(days=7) + timedelta(seconds=remaining)
    policy_path = anchor.with_name('events.policy.json')
    policy_path.write_bytes(_diagnostic_policy_bytes(created))
    anchor.write_bytes(saved_diagnostic('saved').encode())
    os.utime(anchor, (now.timestamp(), now.timestamp()))
    os.utime(policy_path, (now.timestamp(), now.timestamp()))
    before = {p: p.read_bytes() for p in anchor.parent.iterdir()}
    managed = Logger(assets, managed=True)
    assert managed.suppressed_records == 0  # construction is not a lost log record
    if remaining > 0:
        managed.info('still authorized')
        assert managed.suppressed_records == managed.failure_count == 0
        assert anchor.read_bytes().startswith(before[anchor])
    else:
        managed.info('must stop')
        managed.warning('must remain stopped')
        reopened = Logger(assets, managed=True)
        reopened.info('reopening is not renewal')
        assert managed.suppressed_records == 2 and reopened.suppressed_records == 1
        assert managed.stop_reason == reopened.stop_reason == 'expired'
        assert managed.failure_count == reopened.failure_count == 0
        with pytest.raises(ValueError):
            Logger(assets).info('strict also respects expiry')
        assert {p: p.read_bytes() for p in anchor.parent.iterdir()} == before
        captured = capsys.readouterr()
        assert not captured.out and captured.err.count('\n') <= 2
    assert policy_path.read_bytes() == before[policy_path]


@pytest.mark.parametrize('damage', ['partial', 'oversized', 'extra_field', 'bool_cap',
                                  'over_cap', 'future_created', 'wrong_window', 'wrong_owner'])
def test_untrusted_capture_policy_is_not_repaired_or_classified_as_capacity(
        tmp_path, frozen_diagnostic_clock, damage):
    assets = tmp_path / 'assets'
    anchor = assets / 'logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(saved_diagnostic('saved').encode())
    policy = json.loads(_diagnostic_policy_bytes(frozen_diagnostic_clock['now']))
    if damage == 'extra_field': policy['extra'] = 'not authorized'
    if damage == 'bool_cap': policy['max_bytes'] = True
    if damage == 'over_cap': policy['max_bytes'] = 64 * 1024 * 1024 + 1
    if damage == 'future_created':
        policy = json.loads(_diagnostic_policy_bytes(frozen_diagnostic_clock['now'] + timedelta(seconds=1)))
    if damage == 'wrong_window':
        policy['write_until'] = (frozen_diagnostic_clock['now'] + timedelta(days=8)).isoformat()
    if damage == 'wrong_owner': policy['owner'] = 'guessed_owner'
    raw = (b'{"schema":1' if damage == 'partial' else b' ' * 4097 if damage == 'oversized'
           else json.dumps(policy).encode())
    policy_path = anchor.with_name('events.policy.json')
    policy_path.write_bytes(raw)
    before = {p: p.read_bytes() for p in anchor.parent.iterdir()}
    managed = Logger(assets, managed=True)
    managed.info('must refuse unknown source')
    assert managed.failure_count >= 1
    assert managed.suppressed_records == 0
    assert managed.stop_reason not in ('capacity', 'expired', 'unbound')
    with pytest.raises(ValueError):
        Logger(assets).info('strict invalid policy')
    assert {p: p.read_bytes() for p in anchor.parent.iterdir()} == before


@pytest.mark.parametrize('name', ['events.jsonl', 'run.log'], ids=['old_jsonl', 'legacy_text'])
def test_managed_capture_never_backfills_a_policy_for_nonempty_old_logs(tmp_path, capsys, name):
    assets = tmp_path / 'assets'
    parent = assets / 'logs'
    parent.mkdir(parents=True)
    old = parent / name
    old.write_bytes(saved_diagnostic('original').encode() if name.endswith('jsonl') else b'old bytes\n')
    before = {p: p.read_bytes() for p in parent.iterdir()}
    managed = Logger(assets, managed=True)
    assert managed.suppressed_records == 0
    managed.info('not authorized to bind old bytes')
    managed.info('not authorized on retry')
    assert managed.stop_reason == 'unbound' and managed.suppressed_records == 2
    assert managed.failure_count == 0
    assert {p: p.read_bytes() for p in parent.iterdir()} == before
    assert not (parent / 'events.policy.json').exists()
    captured = capsys.readouterr()
    assert not captured.out and captured.err.count('\n') <= 1
    assert str(old) not in captured.err and 'original' not in captured.err
    # Strict no-policy API keeps its pre-existing explicit append behavior.
    strict = Logger(assets)
    strict.info('explicit strict append')
    assert not (parent / 'events.policy.json').exists()
    if name == 'run.log':
        assert old.read_bytes() == before[old]
    else:
        assert old.read_bytes().startswith(before[old])


@pytest.mark.parametrize('alias', ['resolved_alias', 'hardlink'])
def test_capture_policy_aliases_never_authorize_writing_existing_bytes(
        tmp_path, monkeypatch, frozen_diagnostic_clock, alias):
    assets = tmp_path / 'assets'
    anchor = assets / 'logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(saved_diagnostic('saved').encode())
    policy_path = anchor.with_name('events.policy.json')
    original = anchor.parent / 'original-policy.json'
    raw = _diagnostic_policy_bytes(frozen_diagnostic_clock['now'])
    original.write_bytes(raw)
    if alias == 'hardlink':
        os.link(original, policy_path)
        assert policy_path.stat().st_nlink > 1
    else:
        policy_path.write_bytes(raw)
        resolve = Path.resolve
        monkeypatch.setattr(Path, 'resolve', lambda p, *a, **k: original if p == policy_path else resolve(p, *a, **k))
    before = {p: p.read_bytes() for p in anchor.parent.iterdir()}
    managed = Logger(assets, managed=True)
    managed.info('must not trust linked policy')
    assert managed.failure_count >= 1 and managed.suppressed_records == 0
    with pytest.raises(ValueError):
        Logger(assets).info('strict also refuses')
    assert {p: p.read_bytes() for p in anchor.parent.iterdir()} == before


def test_capture_policy_bytes_alone_can_exhaust_the_frozen_budget(tmp_path, frozen_diagnostic_clock):
    assets = tmp_path / 'assets'
    anchor = assets / 'logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b'')
    policy_path = anchor.with_name('events.policy.json')
    policy_path.write_bytes(_diagnostic_policy_bytes(frozen_diagnostic_clock['now'], cap=1))
    before = {p: p.read_bytes() for p in anchor.parent.iterdir()}
    managed = Logger(assets, managed=True)
    assert managed.suppressed_records == 0
    managed.info('no record fits')
    assert managed.suppressed_records == 1 and managed.stop_reason == 'capacity'
    assert managed.failure_count == 0
    assert {p: p.read_bytes() for p in anchor.parent.iterdir()} == before


@pytest.mark.parametrize('problem', ['new_policy_exceeds_budget', 'bound_policy_missing_anchor'])
def test_initial_policy_failure_never_creates_authority_or_repairs_an_anchor(
        tmp_path, monkeypatch, frozen_diagnostic_clock, problem):
    from spec_mtp.structure.system import logger as implementation

    assets = tmp_path / 'assets'
    anchor = assets / 'logs/events.jsonl'
    policy = anchor.with_name('events.policy.json')
    if problem == 'new_policy_exceeds_budget':
        monkeypatch.setattr(implementation, '_TOTAL_BYTES', 1)
    else:
        policy.parent.mkdir(parents=True)
        policy.write_bytes(_diagnostic_policy_bytes(frozen_diagnostic_clock['now']))
    before = policy.read_bytes() if policy.exists() else None
    logger = Logger(assets, managed=True)
    assert logger.failure_count == 1 and logger.suppressed_records == 0
    if problem == 'new_policy_exceeds_budget':
        assert not policy.exists() and anchor.read_bytes() == b''
    else:
        assert policy.read_bytes() == before and not anchor.exists()
        with pytest.raises(ValueError):
            Logger(assets)
