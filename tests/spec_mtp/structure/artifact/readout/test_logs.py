"""Study event aggregation rejects partial Run reads."""
import json

from tests.conftest import saved_diagnostic as _saved_diagnostic
from spec_mtp.structure.artifact.readout.status import diagnostic_lines, _note_from_log
from spec_mtp.structure.artifact.readout import status as status_readout
from spec_mtp.structure.artifact.readout import logs as logs_readout
from spec_mtp.structure.artifact.readout.logs import format_logs
from datetime import datetime, timezone
from pathlib import Path
import pytest


def test_midstream_io_failure_discards_that_runs_partial_events(tmp_path, monkeypatch):
    paths = {}
    for run_id in ('first', 'broken', 'last'):
        path = tmp_path / f'runs/{run_id}/assets/logs/events.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(_saved_diagnostic(run_id), encoding='utf-8')
        paths[run_id] = path
    (tmp_path / 'index.json').write_text(json.dumps({
        'experiments': [{'runs': [{'id': run_id} for run_id in paths]}],
    }))
    actual = getattr(status_readout, 'iter_diagnostic_lines', lambda path: iter(diagnostic_lines(path)))

    def interrupted(path):
        yield from actual(path)
        if path == paths['broken']:
            raise OSError('injected after one valid event')

    monkeypatch.setattr(status_readout, 'iter_diagnostic_lines', interrupted, raising=False)
    monkeypatch.setattr(logs_readout, 'iter_diagnostic_lines', interrupted, raising=False)
    assert _note_from_log(paths['broken']) == '[error] diagnostic log unavailable'
    lines = format_logs(tmp_path).splitlines()
    assert len(lines) == 2 and '[flow] first' in lines[0] and '[flow] last' in lines[1]




@pytest.mark.parametrize('binding', ['own_run', 'cross_run_log', 'run_alias', 'parent_run_dir'])
def test_inventory_logs_binds_each_run_to_its_own_canonical_diagnostic_family(tmp_path, monkeypatch, binding):
    paths = {}
    for run_id in ('r1', 'r2'):
        path = tmp_path / f'runs/{run_id}/assets/logs/events.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(_saved_diagnostic(run_id), encoding='utf-8', newline='\n')
        paths[run_id] = path
    run = {'id': 'r1', 'run_dir': '../r2' if binding == 'parent_run_dir' else 'r1',
           'log': 'runs/r2/assets/logs/events.jsonl' if binding == 'cross_run_log' else 'runs/r1/assets/logs/events.jsonl'}
    (tmp_path / 'index.json').write_text(json.dumps({'experiments': [{'runs': [run]}]}), encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    resolve = Path.resolve
    run_root = tmp_path / 'runs/r1'
    def redirected(path, *args, **kwargs):
        if path.is_relative_to(run_root): return resolve(tmp_path / 'runs/r2' / path.relative_to(run_root), *args, **kwargs)
        return resolve(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        if binding == 'run_alias': patch.setattr(Path, 'resolve', redirected)
        if binding == 'own_run':
            body = logs_readout.inventory_logs(tmp_path)
            assert body['scope'] == 'run_diagnostics_only'
            assert len(body['runs']) == 1 and body['runs'][0]['run_id'] == 'r1'
            assert body['runs'][0]['bytes'] == len(before[paths['r1']])
            assert body['runs'][0]['disposition'] == 'retain'
        else:
            with pytest.raises(ValueError): logs_readout.inventory_logs(tmp_path)
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('populated', [False, True], ids=['absent_log_directories', 'saved_runs'])
def test_inventory_logs_preserves_index_order_and_absent_roots_without_creating_files(tmp_path, populated):
    index = {'experiments': [{'runs': [{'id': 'r2'}]}, {'runs': [{'id': 'r1'}]}]}
    (tmp_path / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    if populated:
        for run_id in ('r2', 'r1'):
            path = tmp_path / f'runs/{run_id}/assets/logs/events.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(_saved_diagnostic(run_id), encoding='utf-8', newline='\n')
    before_files = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    before_dirs = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_dir())
    body = logs_readout.inventory_logs(tmp_path)
    assert body['scope'] == 'run_diagnostics_only'
    observed = datetime.fromisoformat(body['observed_at'].replace('Z', '+00:00'))
    assert observed.tzinfo is not None and observed.utcoffset() == timezone.utc.utcoffset(observed)
    assert [row['run_id'] for row in body['runs']] == ['r2', 'r1']
    for row in body['runs']:
        assert row['disposition'] == 'retain' and row['capture']['state'] == 'unbound'
        assert row['capture']['max_bytes'] is None and row['capture']['write_until'] is None
        assert row['bytes'] > 0 if populated else row['bytes'] == row['capture_bytes'] == 0
        if not populated: assert row['files'] == []
    assert {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before_files
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_dir()) == before_dirs


@pytest.mark.parametrize('duplicate', ['same_id_same_dir', 'same_id_different_dir', 'different_id_same_dir'])
def test_inventory_logs_rejects_duplicate_run_id_or_directory_across_experiments(tmp_path, duplicate):
    first = {'id': 'id1', 'run_dir': 'r1'}
    second = {'id': 'id2', 'run_dir': 'r2'}
    for run_dir in ('r1', 'r2'):
        path = tmp_path / f'runs/{run_dir}/assets/logs/events.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(_saved_diagnostic(run_dir), encoding='utf-8', newline='\n')
    index_path = tmp_path / 'index.json'
    valid = {'experiments': [{'runs': [first]}, {'runs': [second]}]}
    index_path.write_text(json.dumps(valid), encoding='utf-8')
    baseline = logs_readout.inventory_logs(tmp_path)
    assert [row['run_id'] for row in baseline['runs']] == ['id1', 'id2']
    assert all(row['disposition'] == 'retain' for row in baseline['runs'])
    assert [row['bytes'] for row in baseline['runs']] == [
        (tmp_path / f'runs/{run_dir}/assets/logs/events.jsonl').stat().st_size
        for run_dir in ('r1', 'r2')]

    changed = dict(second)
    if duplicate in ('same_id_same_dir', 'same_id_different_dir'):
        changed['id'] = first['id']
    if duplicate in ('same_id_same_dir', 'different_id_same_dir'):
        changed['run_dir'] = first['run_dir']
    # Duplication crosses experiment boundaries; no missing-file or path fault is introduced.
    index_path.write_text(json.dumps({'experiments': [{'runs': [first]}, {'runs': [changed]}]}), encoding='utf-8')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    directories = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_dir())
    with pytest.raises(ValueError, match='duplicate inventory Run'):
        logs_readout.inventory_logs(tmp_path)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_dir()) == directories


@pytest.mark.parametrize('action', ['prepare', 'seal', 'purge'])
def test_retention_logs_dispatches_only_the_selected_registered_run(tmp_path, monkeypatch, action):
    from spec_mtp.structure.artifact.asset import tree as asset_tree

    runs = [{'id': 'id1', 'run_dir': 'r1'}, {'id': 'id2', 'run_dir': 'r2'}]
    (tmp_path / 'index.json').write_text(json.dumps({'experiments': [{'runs': runs}]}), encoding='utf-8')
    for run in runs:
        (tmp_path / 'runs' / run['run_dir']).mkdir(parents=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    calls = []
    seal_sha = 'a' * 64

    def prepare(anchor):
        calls.append(('prepare', anchor))
        return {'state': 'ready'}

    def seal(anchor):
        calls.append(('seal', anchor))
        return seal_sha

    def purge(anchor, expected):
        calls.append(('purge', anchor, expected))
        return {'state': 'purged', 'deleted_files': ['events.jsonl']}

    monkeypatch.setattr(asset_tree, 'prepare_diagnostic_capture', prepare, raising=False)
    monkeypatch.setattr(asset_tree, 'seal_diagnostic_capture', seal, raising=False)
    monkeypatch.setattr(asset_tree, 'purge_sealed_diagnostics', purge, raising=False)
    body = logs_readout.retention_logs(tmp_path, 'id2', action,
                                       seal_sha256=seal_sha if action == 'purge' else None)
    anchor = tmp_path / 'runs/r2/assets/logs/events.jsonl'
    assert calls == [(action, anchor, seal_sha) if action == 'purge' else (action, anchor)]
    assert body['scope'] == 'run_diagnostics_only' and body['run_id'] == 'id2' and body['action'] == action
    assert body['state'] == {'prepare': 'ready', 'seal': 'sealed', 'purge': 'purged'}[action]
    if action == 'seal':
        assert body['seal_sha256'] == seal_sha
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    assert not (tmp_path / 'runs/r1/assets').exists() and not (tmp_path / 'runs/r2/assets').exists()


@pytest.mark.parametrize('fault', [
    'unregistered', 'duplicate_id', 'duplicate_root', 'cross_run_log', 'parent_run_dir',
    'run_alias', 'missing_run', 'existing_result', 'wrong_action', 'missing_seal', 'extra_seal',
])
def test_retention_logs_rejects_invalid_run_or_action_before_mutation(tmp_path, monkeypatch, fault):
    from spec_mtp.structure.artifact.asset import tree as asset_tree

    runs = [{'id': 'id1', 'run_dir': 'r1'}, {'id': 'id2', 'run_dir': 'r2'}]
    for run in runs:
        (tmp_path / 'runs' / run['run_dir']).mkdir(parents=True)
    action, run_id, seal_sha = 'prepare', 'id1', None
    if fault == 'unregistered':
        run_id = 'unknown'
    elif fault == 'duplicate_id':
        runs[1]['id'] = 'id1'
    elif fault == 'duplicate_root':
        runs[1]['run_dir'] = 'r1'
    elif fault == 'cross_run_log':
        runs[0]['log'] = 'runs/r2/assets/logs/events.jsonl'
    elif fault == 'parent_run_dir':
        runs[0]['run_dir'] = '../r2'
    elif fault == 'missing_run':
        (tmp_path / 'runs/r1').rmdir()
    elif fault == 'existing_result':
        (tmp_path / 'runs/r1/result.json').write_bytes(b'{"status":"succeeded"}\n')
    elif fault == 'wrong_action':
        action = 'delete_everything'
    elif fault == 'missing_seal':
        action = 'purge'
    elif fault == 'extra_seal':
        seal_sha = 'a' * 64
    (tmp_path / 'index.json').write_text(json.dumps({'experiments': [{'runs': [run]} for run in runs]}), encoding='utf-8')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    directories = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_dir())

    def forbidden(*args, **kwargs):
        pytest.fail('an invalid selection must not reach a mutating core operation')

    for name in ('prepare_diagnostic_capture', 'seal_diagnostic_capture', 'purge_sealed_diagnostics'):
        monkeypatch.setattr(asset_tree, name, forbidden, raising=False)
    resolve = Path.resolve
    run_root = tmp_path / 'runs/r1'

    def redirected(path, *args, **kwargs):
        if path.is_relative_to(run_root):
            return resolve(tmp_path / 'runs/r2' / path.relative_to(run_root), *args, **kwargs)
        return resolve(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if fault == 'run_alias':
            patch.setattr(Path, 'resolve', redirected)
        with pytest.raises((ValueError, FileNotFoundError)):
            logs_readout.retention_logs(tmp_path, run_id, action, seal_sha256=seal_sha)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_dir()) == directories
