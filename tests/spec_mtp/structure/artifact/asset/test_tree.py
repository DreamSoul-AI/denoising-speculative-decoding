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
