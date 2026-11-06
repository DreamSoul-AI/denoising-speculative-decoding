"""Diagnostic part layout admission reads metadata only."""
from pathlib import Path
import pytest
from spec_mtp.structure.artifact.asset.tree import diagnostic_files
from tests.conftest import saved_diagnostic
from datetime import datetime, timedelta, timezone
import json
import os
from spec_mtp.structure.artifact.asset import tree as asset_tree


@pytest.mark.parametrize('layout', ['gap', 'duplicate', 'missing_anchor', 'nonfile',
                                   'escape', 'duplicate_identity', 'inbounds_alias'])
def test_diagnostic_segments_validate_named_layout_without_reading_bodies(tmp_path, monkeypatch, layout):
    parent = tmp_path / 'logs'
    parent.mkdir()
    anchor = parent / 'events.jsonl'
    if layout != 'missing_anchor': anchor.write_text(saved_diagnostic())
    part = parent / ('events.000002.jsonl' if layout == 'gap' else 'events.000001.jsonl')
    if layout == 'nonfile': part.mkdir()
    else: part.write_text(saved_diagnostic('part'))
    if layout == 'duplicate': (parent / 'events.1.jsonl').write_text(saved_diagnostic('duplicate'))
    outside = tmp_path / 'outside.jsonl'
    outside.write_text(saved_diagnostic('outside'))
    history = parent / 'historical.jsonl'
    history.write_text(saved_diagnostic('history'))
    resolve, stat = Path.resolve, Path.stat
    destination = outside if layout == 'escape' else history
    def resolved(path, *args, **kwargs):
        return destination if path == part and layout in ('escape', 'inbounds_alias') else resolve(path, *args, **kwargs)
    original_identity = anchor.stat() if anchor.exists() else None
    history_identity = history.stat()
    def stated(path, *args, **kwargs):
        if path == part and layout == 'duplicate_identity': return original_identity
        if path == part and layout == 'inbounds_alias': return history_identity
        return stat(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'resolve', resolved)
        patch.setattr(Path, 'stat', stated)
        patch.setattr(Path, 'open', lambda *a, **k: pytest.fail('enumeration must not read byte bodies'))
        with pytest.raises(ValueError): diagnostic_files(anchor)




def _inventory_policy(created, maximum=4096):
    return (json.dumps({'schema': 1, 'owner': 'run_operator', 'max_bytes': maximum,
                        'created_at': created.isoformat().replace('+00:00', 'Z'),
                        'write_until': (created + timedelta(days=7)).isoformat().replace('+00:00', 'Z')},
                       separators=(',', ':')) + '\n').encode('ascii')


@pytest.mark.parametrize('layout', ['empty', 'anchor', 'segmented_and_legacy', 'legacy_only'])
def test_diagnostic_inventory_counts_only_known_files_without_reading_bodies(tmp_path, monkeypatch, layout):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
    anchor = tmp_path / 'logs/events.jsonl'
    anchor.parent.mkdir()
    if layout in ('anchor', 'segmented_and_legacy'):
        anchor.write_bytes(b'\xffnot-json\n')  # Inventory is not a JSONL content validator.
    if layout == 'segmented_and_legacy':
        anchor.with_name('events.000001.jsonl').write_bytes(b'part\n')
    if layout in ('segmented_and_legacy', 'legacy_only'):
        anchor.with_name('run.log').write_bytes(b'original legacy\n')
    anchor.with_name('unrelated.bin').write_bytes(b'not part of diagnostic inventory')
    before = {p.name: p.read_bytes() for p in anchor.parent.iterdir()}
    known = {name: len(raw) for name, raw in before.items() if name != 'unrelated.bin'}
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', lambda *a, **k: pytest.fail('metadata inventory read a byte body'))
        body = asset_tree.diagnostic_inventory(anchor, now)
    assert {row['name']: row['bytes'] for row in body['files']} == known
    assert body['bytes'] == sum(known.values())
    assert body['capture_bytes'] == sum(size for name, size in known.items() if name != 'run.log')
    assert body['capture'] == {'state': 'unbound', 'max_bytes': None,
                               'created_at': None, 'write_until': None}
    assert body['disposition'] == 'retain'
    assert {p.name: p.read_bytes() for p in anchor.parent.iterdir()} == before


@pytest.mark.parametrize('boundary', ['within_limits', 'at_deadline', 'after_deadline', 'at_capacity', 'over_capacity'])
def test_diagnostic_inventory_reports_frozen_limits_without_cleanup_or_renewal(tmp_path, boundary):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
    created = now - timedelta(days=7) + timedelta(seconds=1)
    if boundary == 'at_deadline': created = now - timedelta(days=7)
    if boundary == 'after_deadline': created = now - timedelta(days=7, seconds=1)
    anchor = tmp_path / 'logs/events.jsonl'
    anchor.parent.mkdir()
    policy = anchor.with_name('events.policy.json')
    raw = _inventory_policy(created)
    policy.write_bytes(raw)
    size = 4096 - len(raw) + (boundary == 'over_capacity') if boundary in ('at_capacity', 'over_capacity') else 10
    anchor.write_bytes(b'x' * size)
    anchor.with_name('run.log').write_bytes(b'legacy' * 1000)  # Not charged to the JSONL capture budget.
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in anchor.parent.iterdir()}
    assert asset_tree.read_capture_policy(anchor, now) == (4096, created, created + timedelta(days=7), len(raw))
    body = asset_tree.diagnostic_inventory(anchor, now)
    assert body['capture_bytes'] == size + len(raw)
    assert body['bytes'] == body['capture_bytes'] + 6000
    expected = 'expired' if boundary in ('at_deadline', 'after_deadline') else 'capacity' if boundary in ('at_capacity', 'over_capacity') else 'within_limits'
    assert body['capture']['state'] == expected
    assert body['capture']['max_bytes'] == 4096
    assert datetime.fromisoformat(body['capture']['created_at'].replace('Z', '+00:00')) == created
    assert datetime.fromisoformat(body['capture']['write_until'].replace('Z', '+00:00')) == created + timedelta(days=7)
    assert body['disposition'] == 'retain'
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in anchor.parent.iterdir()} == before


@pytest.mark.parametrize('fault', ['partial', 'duplicate_field', 'boolean_maximum', 'oversized', 'future_created', 'wrong_window'])
def test_diagnostic_inventory_refuses_invalid_policy_without_repair(tmp_path, fault):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
    anchor = tmp_path / 'logs/events.jsonl'
    anchor.parent.mkdir()
    anchor.write_bytes(b'old diagnostic bytes\n')
    policy = anchor.with_name('events.policy.json')
    policy.write_bytes(_inventory_policy(now))
    assert asset_tree.read_capture_policy(anchor, now)[0] == 4096  # Valid baseline precedes the one fault.
    body = json.loads(policy.read_bytes())
    if fault == 'boolean_maximum': body['max_bytes'] = True
    if fault == 'future_created': body = json.loads(_inventory_policy(now + timedelta(seconds=1)))
    if fault == 'wrong_window': body['write_until'] = (now + timedelta(days=8)).isoformat()
    raw = (b'{"schema":1' if fault == 'partial' else b' ' * 4097 if fault == 'oversized'
           else _inventory_policy(now).replace(b'"schema":1', b'"schema":1,"schema":1') if fault == 'duplicate_field'
           else json.dumps(body).encode('ascii'))
    policy.write_bytes(raw)
    before = {p.name: p.read_bytes() for p in anchor.parent.iterdir()}
    with pytest.raises(ValueError): asset_tree.read_capture_policy(anchor, now)
    with pytest.raises(ValueError): asset_tree.diagnostic_inventory(anchor, now)
    assert {p.name: p.read_bytes() for p in anchor.parent.iterdir()} == before


@pytest.mark.parametrize('fault', ['missing_anchor', 'gap', 'policy_alias', 'legacy_alias', 'policy_hardlink', 'legacy_hardlink'])
def test_diagnostic_inventory_refuses_unsafe_file_families_before_body_reads(tmp_path, monkeypatch, fault):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
    anchor = tmp_path / 'logs/events.jsonl'
    anchor.parent.mkdir()
    if fault != 'missing_anchor': anchor.write_bytes(b'anchor\n')
    if fault in ('missing_anchor', 'gap'):
        anchor.with_name('events.000001.jsonl' if fault == 'missing_anchor' else 'events.000002.jsonl').write_bytes(b'part\n')
    selected = anchor.with_name('events.policy.json' if fault.startswith('policy') else 'run.log')
    outside = tmp_path / 'original.bin'
    outside.write_bytes(_inventory_policy(now) if fault.startswith('policy') else b'legacy\n')
    if fault.endswith('alias'): selected.write_bytes(outside.read_bytes())
    if fault.endswith('hardlink'): os.link(outside, selected)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    resolve = Path.resolve
    with monkeypatch.context() as patch:
        if fault.endswith('alias'):
            patch.setattr(Path, 'resolve', lambda p, *a, **k: outside if p == selected else resolve(p, *a, **k))
        patch.setattr(Path, 'open', lambda *a, **k: pytest.fail('unsafe family was opened before metadata admission'))
        with pytest.raises(ValueError): asset_tree.diagnostic_inventory(anchor, now)
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('mutation', ['size_changed', 'part_added'])
def test_diagnostic_inventory_refuses_a_family_changed_during_policy_read(tmp_path, monkeypatch, mutation):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
    anchor = tmp_path / 'logs/events.jsonl'
    anchor.parent.mkdir()
    anchor.write_bytes(b'original\n')
    anchor.with_name('events.policy.json').write_bytes(_inventory_policy(now))
    actual = asset_tree.read_capture_policy
    observed = {}
    def read_then_change(path, instant):
        result = actual(path, instant)
        if mutation == 'size_changed': anchor.write_bytes(b'changed and longer\n')
        else: anchor.with_name('events.000001.jsonl').write_bytes(b'new part\n')
        observed.update({p.name: p.read_bytes() for p in anchor.parent.iterdir()})
        return result
    with monkeypatch.context() as patch:
        patch.setattr(asset_tree, 'read_capture_policy', read_then_change)
        with pytest.raises(ValueError): asset_tree.diagnostic_inventory(anchor, now)
    assert observed  # The failure reached the deliberately changing valid-policy boundary.
    assert {p.name: p.read_bytes() for p in anchor.parent.iterdir()} == observed


@pytest.mark.parametrize('fault', ['enumeration', 'policy_stat'])
def test_diagnostic_inventory_io_failure_is_not_an_empty_or_zero_usage_report(tmp_path, monkeypatch, fault):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
    anchor = tmp_path / 'logs/events.jsonl'
    anchor.parent.mkdir()
    anchor.write_bytes(b'saved diagnostic bytes\n')
    policy = anchor.with_name('events.policy.json')
    policy.write_bytes(_inventory_policy(now))
    before = {p: p.read_bytes() for p in anchor.parent.iterdir()}
    iterate, stated = Path.iterdir, Path.stat
    failure = PermissionError('injected inventory metadata denial')
    def list_directory(path):
        if fault == 'enumeration' and path == anchor.parent:
            raise failure
        return iterate(path)
    def stat_file(path, *args, **options):
        if fault == 'policy_stat' and path == policy:
            raise failure
        return stated(path, *args, **options)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'iterdir', list_directory)
        patch.setattr(Path, 'stat', stat_file)
        with pytest.raises(PermissionError) as caught:
            asset_tree.diagnostic_inventory(anchor, now)
        assert caught.value is failure
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('writer', ['old_managed', 'new_managed', 'new_strict'])
def test_closed_capture_marker_fences_old_and_new_loggers_before_any_write(tmp_path, writer):
    from spec_mtp.structure.system.logger import Logger

    assets = tmp_path / 'runs/owned/assets'
    assets.mkdir(parents=True)
    logger = Logger(assets, managed=True) if writer == 'old_managed' else None
    marker = assets / '.diagnostic-capture.json'
    marker.write_text(json.dumps({'schema': 1, 'state': 'closed', 'assets': str(assets.resolve())}),
                      encoding='utf-8', newline='\n')
    files = {p.relative_to(assets).as_posix(): p.read_bytes() for p in assets.rglob('*') if p.is_file()}
    directories = sorted(p.relative_to(assets).as_posix() for p in assets.rglob('*') if p.is_dir())
    if writer == 'new_strict':
        with pytest.raises(PermissionError):
            Logger(assets)
    else:
        if logger is None:
            logger = Logger(assets, managed=True)
        logger.info('closed capture must not append')
        assert logger.failure_count + logger.suppressed_records >= 1
    assert {p.relative_to(assets).as_posix(): p.read_bytes() for p in assets.rglob('*') if p.is_file()} == files
    assert sorted(p.relative_to(assets).as_posix() for p in assets.rglob('*') if p.is_dir()) == directories



def _capture_state(root):
    files = {}
    for path in root.rglob('*'):
        if path.is_file():
            if path.name == '.diagnostic-capture.lock':
                assert path.stat().st_size == 0  # Windows rejects byte reads while this native lease is held.
                raw = b''
            else:
                raw = path.read_bytes()
            files[path.relative_to(root).as_posix()] = raw
    return files, sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_dir())


@pytest.fixture
def prepared_capture(tmp_path):
    from spec_mtp.structure.system.logger import Logger
    anchor = tmp_path / 'runs/owned/assets/logs/events.jsonl'
    assert asset_tree.prepare_diagnostic_capture(anchor)['state'] == 'ready'
    logger = Logger(anchor.parent.parent, managed=True)
    logger.info('saved diagnostic fixture')
    return anchor, logger


@pytest.mark.parametrize('case', ['default_unchanged', 'new', 'existing_logs', 'legacy',
                                  'pending', 'closed', 'assets_alias', 'control_hardlink',
                                  'ready_missing_lock', 'closed_missing_lock'])
def test_capture_prepare_preflights_ownership_without_changing_existing_files(tmp_path, monkeypatch, case):
    from spec_mtp.structure.system.logger import Logger
    assets = tmp_path / 'runs/owned/assets'
    assets.mkdir(parents=True)
    anchor = assets / 'logs/events.jsonl'
    marker = assets / '.diagnostic-capture.json'
    if case in ('ready_missing_lock', 'closed_missing_lock'):
        asset_tree.prepare_diagnostic_capture(anchor)
        logger = Logger(assets, managed=True)
        logger.info('original captured bytes')
        seal_sha = asset_tree.seal_diagnostic_capture(anchor) if case == 'closed_missing_lock' else None
        (assets / '.diagnostic-capture.lock').unlink()  # External damage must never grant a fresh lease.
        before = _capture_state(assets)
        with pytest.raises((FileNotFoundError, PermissionError, ValueError)):
            if seal_sha is None:
                asset_tree.seal_diagnostic_capture(anchor)
            else:
                asset_tree.purge_sealed_diagnostics(anchor, seal_sha)
        logger.info('missing permanent lease must not reopen')
        reopened = Logger(assets, managed=True)
        reopened.info('new writer cannot recreate a deleted lock')
        assert _capture_state(assets) == before and not (assets / '.diagnostic-capture.lock').exists()
        return
    if case == 'default_unchanged':
        logger = Logger(assets, managed=True)
        logger.info('ordinary existing default')
        assert anchor.is_file() and anchor.with_name('events.policy.json').is_file()
        assert list(assets.glob('.diagnostic-capture*')) == []
        return
    if case == 'new':
        assert asset_tree.prepare_diagnostic_capture(anchor) == {'state': 'ready'}
        assert json.loads(marker.read_bytes()) == {'schema': 1, 'state': 'ready', 'assets': str(assets.resolve())}
        assert anchor.parent.is_dir() and list(anchor.parent.iterdir()) == []
        assert (assets / '.diagnostic-capture.lock').stat().st_size == 0
        return
    if case in ('existing_logs', 'legacy'):
        anchor.parent.mkdir()
        if case == 'legacy': anchor.with_name('run.log').write_bytes(b'original legacy bytes\n')
    if case in ('pending', 'closed'):
        marker.write_text(json.dumps({'schema': 1, 'state': case, 'assets': str(assets.resolve())}), encoding='utf-8')
    outside = tmp_path / 'outside'
    if case == 'assets_alias': outside.mkdir()
    if case == 'control_hardlink':
        outside.write_text('{}', encoding='utf-8')
        os.link(outside, marker)
    before = _capture_state(tmp_path)
    resolve = Path.resolve
    with monkeypatch.context() as patch:
        if case == 'assets_alias':
            patch.setattr(Path, 'resolve', lambda p, *a, **k: outside if p == assets else resolve(p, *a, **k))
        with pytest.raises((ValueError, FileExistsError)):
            asset_tree.prepare_diagnostic_capture(anchor)
    assert _capture_state(tmp_path) == before


@pytest.mark.parametrize('managed', [False, True], ids=['optional_policy', 'managed_policy'])
def test_capture_seal_and_purge_only_the_exact_rotated_family(tmp_path, monkeypatch, managed):
    import hashlib
    from spec_mtp.structure.system import logger as logger_module
    assets = tmp_path / 'runs/owned/assets'
    anchor = assets / 'logs/events.jsonl'
    assets.mkdir(parents=True)
    preserved = [assets / 'model.pt', assets / 'Result.json', assets / 'checkpoints/latest.pt',
                 assets.parent / 'result.json', assets / 'outside_logs/sentinel.bin']
    for path in preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('original ' + path.name).encode())
    originals = {path: path.read_bytes() for path in preserved}
    asset_tree.prepare_diagnostic_capture(anchor)
    monkeypatch.setattr(logger_module, '_SEGMENT_BYTES', 1024)
    logger = logger_module.Logger(assets, managed=managed)
    for _ in range(4): logger.info('rotation fixture ' + 'x' * 300)
    assert len(diagnostic_files(anchor)) >= 2
    members = {path.name: path.read_bytes() for path in anchor.parent.iterdir()}
    seal_sha = asset_tree.seal_diagnostic_capture(anchor)
    seal_path = assets / '.diagnostic-capture.seal.json'
    seal_raw = seal_path.read_bytes()
    assert seal_sha == hashlib.sha256(seal_raw).hexdigest()
    seal = json.loads(seal_raw)
    assert {row['name']: (row['bytes'], row['sha256']) for row in seal['files']} == {
        name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in members.items()}
    if managed:
        logger.info('closed must not append')
        assert logger.failure_count + logger.suppressed_records >= 1
    else:
        with pytest.raises(PermissionError): logger.info('closed must not append')
    assert {path.name: path.read_bytes() for path in anchor.parent.iterdir()} == members
    result = asset_tree.purge_sealed_diagnostics(anchor, seal_sha)
    assert result['state'] == 'purged' and json.loads((assets / '.diagnostic-capture.purge.json').read_bytes()) == result
    assert json.loads((assets / '.diagnostic-capture.json').read_bytes())['state'] == 'closed'
    assert (assets / '.diagnostic-capture.lock').stat().st_size == 0
    assert seal_path.read_bytes() == seal_raw and anchor.parent.is_dir() and list(anchor.parent.iterdir()) == []
    assert {path: path.read_bytes() for path in preserved} == originals
    with pytest.raises(FileExistsError): asset_tree.prepare_diagnostic_capture(anchor)


@pytest.mark.parametrize('fault', ['legacy', 'unknown', 'alias', 'hardlink'])
def test_seal_refuses_untrusted_members_and_keeps_the_capture_closed(prepared_capture, tmp_path, monkeypatch, fault):
    anchor, _ = prepared_capture
    assets = anchor.parent.parent
    outside = tmp_path / 'outside.jsonl'
    outside.write_bytes(anchor.read_bytes())
    if fault in ('legacy', 'unknown'):
        anchor.with_name('run.log' if fault == 'legacy' else 'unknown.bin').write_bytes(b'preserve unknown original')
    if fault == 'hardlink':
        anchor.unlink()
        os.link(outside, anchor)
    before = {path: path.read_bytes() for path in anchor.parent.iterdir()}
    resolve = Path.resolve
    with monkeypatch.context() as patch:
        if fault == 'alias':
            patch.setattr(Path, 'resolve', lambda p, *a, **k: outside if p == anchor else resolve(p, *a, **k))
        with pytest.raises(ValueError): asset_tree.seal_diagnostic_capture(anchor)
    assert json.loads((assets / '.diagnostic-capture.json').read_bytes())['state'] == 'closed'
    assert not (assets / '.diagnostic-capture.seal.json').exists()
    assert {path: path.read_bytes() for path in anchor.parent.iterdir()} == before


@pytest.mark.parametrize('fault', ['wrong_sha', 'member_added', 'bytes_changed', 'alias', 'hardlink', 'marker_alias', 'seal_same_read_digest'])
def test_purge_rejects_changed_binding_before_any_unlink(prepared_capture, tmp_path, monkeypatch, fault):
    anchor, _ = prepared_capture
    assets = anchor.parent.parent
    seal_sha = asset_tree.seal_diagnostic_capture(anchor)
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(anchor.read_bytes())
    if fault == 'member_added': anchor.with_name('events.000001.jsonl').write_bytes(b'new unsealed part\n')
    if fault == 'bytes_changed': anchor.write_bytes(anchor.read_bytes() + b'changed bytes\n')
    if fault == 'hardlink':
        anchor.unlink()
        os.link(outside, anchor)
    if fault == 'marker_alias': outside.write_bytes((assets / '.diagnostic-capture.json').read_bytes())
    before = _capture_state(tmp_path)
    resolve = Path.resolve
    alias = assets / '.diagnostic-capture.json' if fault == 'marker_alias' else anchor
    expected_sha = '0' * 64 if fault == 'wrong_sha' else seal_sha
    actual_read = asset_tree._read_capture_control
    replacement_raw = None
    if fault == 'seal_same_read_digest':
        import hashlib
        replacement_body = json.loads((assets / '.diagnostic-capture.seal.json').read_bytes())
        replacement_body['files'] = []
        replacement_raw = (json.dumps(replacement_body, sort_keys=True) + '\n').encode()
        expected_sha = hashlib.sha256(replacement_raw).hexdigest()
    def read_then_swap(path, *args, **kwargs):
        body = actual_read(path, *args, **kwargs)
        if path == assets / '.diagnostic-capture.seal.json':
            path.write_bytes(replacement_raw)
        return body
    with monkeypatch.context() as patch:
        if fault in ('alias', 'marker_alias'):
            patch.setattr(Path, 'resolve', lambda p, *a, **k: outside if p == alias else resolve(p, *a, **k))
        if fault == 'seal_same_read_digest': patch.setattr(asset_tree, '_read_capture_control', read_then_swap)
        patch.setattr(Path, 'unlink', lambda *a, **k: pytest.fail('purge unlinked before all binding checks'))
        with pytest.raises(ValueError):
            asset_tree.purge_sealed_diagnostics(anchor, expected_sha)
    assert _capture_state(tmp_path) == before
    assert not (assets / '.diagnostic-capture.purge.json').exists()


@pytest.mark.parametrize('error_type', [OSError, KeyboardInterrupt, SystemExit, 'journal_member_drift', 'journal_parent_drift'])
def test_purge_second_unlink_failure_stays_pending_and_cannot_retry(prepared_capture, monkeypatch, error_type):
    anchor, logger = prepared_capture
    assets = anchor.parent.parent
    seal_sha = asset_tree.seal_diagnostic_capture(anchor)
    seal = json.loads((assets / '.diagnostic-capture.seal.json').read_bytes())
    names = {row['name'] for row in seal['files']}
    assert len(names) >= 2
    original = {path.name: path.read_bytes() for path in anchor.parent.iterdir()}
    if isinstance(error_type, str):
        actual_write = asset_tree._write_capture_control
        observed = {}
        def write_then_replace(path, body, *, exclusive=False):
            published = actual_write(path, body, exclusive=exclusive)
            if path == assets / '.diagnostic-capture.purge.json' and body['state'] == 'pending':
                if error_type == 'journal_member_drift':
                    replacement = assets / 'external-replacement.jsonl'
                    replacement.write_bytes(anchor.read_bytes())
                    os.replace(replacement, anchor)
                else:
                    retained = assets / 'retained-original-logs'
                    anchor.parent.resolve().relative_to(assets.resolve())
                    retained.absolute().relative_to(assets.absolute())
                    anchor.parent.rename(retained)
                    anchor.parent.mkdir()
                    for member in retained.iterdir():
                        anchor.with_name(member.name).write_bytes(member.read_bytes())
                observed['state'] = _capture_state(assets)
            return published
        with monkeypatch.context() as patch:
            patch.setattr(asset_tree, '_write_capture_control', write_then_replace)
            patch.setattr(Path, 'unlink', lambda *a, **k: pytest.fail('journal-time drift reached the first unlink'))
            with pytest.raises(ValueError): asset_tree.purge_sealed_diagnostics(anchor, seal_sha)
        assert observed and _capture_state(assets) == observed['state']
        assert json.loads((assets / '.diagnostic-capture.purge.json').read_bytes())['state'] == 'pending'
        assert json.loads((assets / '.diagnostic-capture.json').read_bytes())['state'] == 'closed'
        assert {p.name for p in anchor.parent.iterdir()} == names
        return
    actual = Path.unlink
    failure = error_type('second owned unlink interrupted')
    calls = []
    def interrupted(path, *args, **kwargs):
        if path.parent == anchor.parent and path.name in names:
            calls.append(path.name)
            if len(calls) == 2: raise failure
        return actual(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupted)
        with pytest.raises(error_type) as caught:
            asset_tree.purge_sealed_diagnostics(anchor, seal_sha)
        assert caught.value is failure
    assert len(calls) == 2
    journal = json.loads((assets / '.diagnostic-capture.purge.json').read_bytes())
    assert journal['state'] == 'pending' and set(journal['files']) == names
    assert json.loads((assets / '.diagnostic-capture.json').read_bytes())['state'] == 'closed'
    remaining = {path.name: path.read_bytes() for path in anchor.parent.iterdir()}
    assert len(remaining) == len(original) - 1 and all(raw == original[name] for name, raw in remaining.items())
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', lambda *a, **k: pytest.fail('pending purge was retried'))
        with pytest.raises(FileExistsError): asset_tree.purge_sealed_diagnostics(anchor, seal_sha)
    logger.info('closed partial purge cannot recreate anchor')
    assert {path.name: path.read_bytes() for path in anchor.parent.iterdir()} == remaining


def test_native_capture_lease_busy_then_owned_process_exit_releases_it(prepared_capture):
    import queue
    import subprocess
    import sys
    import threading
    anchor, _ = prepared_capture
    repository = Path(__file__).resolve().parents[5]
    holder = "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[2]);from spec_mtp.structure.artifact.asset import tree;\nwith tree._diagnostic_lease(Path(sys.argv[1])):\n print('held',flush=True);sys.stdin.read(1)"
    probe = "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[2]);from spec_mtp.structure.artifact.asset import tree;\ntry: tree.seal_diagnostic_capture(Path(sys.argv[1]))\nexcept BlockingIOError: print('busy',flush=True)\nelse: raise SystemExit('seal incorrectly acquired an occupied lease')"
    child = subprocess.Popen([sys.executable, '-c', holder, str(anchor), str(repository / 'src')],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=repository)
    try:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
        assert ready.get(timeout=10).strip() == 'held'
        before = _capture_state(anchor.parent.parent)
        check = subprocess.run([sys.executable, '-c', probe, str(anchor), str(repository / 'src')],
                               capture_output=True, text=True, cwd=repository, timeout=10)
        assert check.returncode == 0 and check.stdout.strip() == 'busy', check.stderr
        assert _capture_state(anchor.parent.parent) == before
    finally:
        if child.poll() is None: child.kill()
        child.communicate(timeout=10)
    assert child.poll() is not None
    assert len(asset_tree.seal_diagnostic_capture(anchor)) == 64


@pytest.mark.parametrize('boundary', ['prepare_mkdir_failure', 'marker_after_mkdir', 'marker_after_policy_read'])
def test_capture_bootstrap_races_leave_a_fence_and_never_publish_extra_bytes(tmp_path, monkeypatch, boundary):
    from spec_mtp.structure.system import logger as logger_module
    assets = tmp_path / 'runs/owned/assets'
    assets.mkdir(parents=True)
    anchor = assets / 'logs/events.jsonl'
    marker = assets / '.diagnostic-capture.json'
    pending = {'schema': 1, 'state': 'pending', 'assets': str(assets.resolve())}
    if boundary == 'prepare_mkdir_failure':
        mkdir = Path.mkdir
        failure = OSError('mkdir failed after durable pending fence')
        def fail_logs(path, *args, **kwargs):
            if path == anchor.parent: raise failure
            return mkdir(path, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'mkdir', fail_logs)
            with pytest.raises(OSError) as caught: asset_tree.prepare_diagnostic_capture(anchor)
            assert caught.value is failure
        assert json.loads(marker.read_bytes()) == pending and not anchor.parent.exists()
        before = _capture_state(assets)
        logger = logger_module.Logger(assets, managed=True)
        logger.info('pending must not bootstrap')
        with pytest.raises(FileExistsError): asset_tree.prepare_diagnostic_capture(anchor)
        assert _capture_state(assets) == before
        return
    observed = {}
    if boundary == 'marker_after_mkdir':
        mkdir = Path.mkdir
        def raced_mkdir(path, *args, **kwargs):
            result = mkdir(path, *args, **kwargs)
            if path == anchor.parent:
                asset_tree._write_capture_control(marker, pending, exclusive=True)
                observed['state'] = _capture_state(assets)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'mkdir', raced_mkdir)
            logger = logger_module.Logger(assets, managed=True)
    else:
        logger = logger_module.Logger(assets, managed=True)
        actual = logger_module.read_capture_policy
        def raced_read(path, now):
            result = actual(path, now)
            asset_tree._write_capture_control(marker, pending, exclusive=True)
            observed['state'] = _capture_state(assets)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(logger_module, 'read_capture_policy', raced_read)
            logger.info('marker wins the final write admission race')
    assert observed and logger.failure_count + logger.suppressed_records >= 1
    assert _capture_state(assets) == observed['state']
