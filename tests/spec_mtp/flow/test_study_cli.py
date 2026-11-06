"""External Study selection happens before any source lock or runtime writes."""

import json

import pytest

from tests.conftest import saved_diagnostic as _saved_diagnostic
from spec_mtp.structure.artifact.readout.status import _note_from_log, diagnostic_lines, log_path
from spec_mtp.structure.artifact.readout.report import _log_links
from spec_mtp.structure.artifact.readout.logs import format_logs

from spec_mtp.flow.study_cli import prepare_study_argv


@pytest.mark.parametrize('command', [['make'], ['run'], ['launch'], ['study', 'run']])
def test_run_root_selection_preserves_source_and_selects_external_study(tmp_path, command):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.yaml').write_bytes(b'study: cli-fixture\n')
    output = tmp_path / 'runtime'
    assert prepare_study_argv([*command, str(source), '--run-root', str(output)]) == [*command, str(output)]
    assert (output / 'study_source.json').is_file()
    assert list(source.iterdir()) == [source / 'study.yaml']


@pytest.mark.parametrize('suffix', [['--unknown'], ['--help']])
def test_invalid_arguments_and_help_do_not_create_a_runtime(tmp_path, suffix):
    from studies._shared.flow.cli import main

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.yaml').write_bytes(b'study: cli-fixture\n')
    output = tmp_path / 'runtime'
    with pytest.raises(SystemExit) as stopped:
        main(['make', str(source), '--run-root', str(output), *suffix])
    assert stopped.value.code == (0 if suffix == ['--help'] else 2)
    assert not output.exists()
    assert list(source.iterdir()) == [source / 'study.yaml']


@pytest.mark.parametrize('empty_queue,complete,codes', [(True, False, []), (False, False, []), (False, True, [1])])
def test_launch_cannot_report_success_for_an_incomplete_saved_study(tmp_path, monkeypatch, empty_queue, complete, codes):
    from argparse import Namespace
    from spec_mtp.flow import study_cli

    monkeypatch.setattr(study_cli, 'load_launch_plan', lambda *a, **k: {
        'job_list': [] if empty_queue else [{'run_id': 'finished'}],
        'jobs_json': tmp_path / 'jobs.json', 'n_jobs': 0 if empty_queue else 1,
        'round': 1, 'batches': None, 'expand': {'study_dir': tmp_path}})
    monkeypatch.setattr(study_cli, '_provenance_gate', lambda root: True)
    monkeypatch.setattr(study_cli, 'launch_jobs', lambda *a, **k: codes)
    monkeypatch.setattr(study_cli, 'run_succeeded', lambda *a: True)
    monkeypatch.setattr(study_cli, 'process_study', lambda root: {'complete': complete})
    monkeypatch.setattr(study_cli, '_print_launch_counts', lambda root: None)
    args = Namespace(study_dir=tmp_path, remake=False, init_gpu=0, num_gpus=0,
                     round=1, include_done=False, modes=None, console='shared')
    assert study_cli._execute_launch(args) == 1


@pytest.mark.parametrize('fault', ['malformed', 'truncated', 'oversized'])
def test_late_diagnostic_damage_invalidates_notes_and_cli_output(tmp_path, capsys, fault):
    from spec_mtp.flow.study_cli import main

    path = tmp_path / 'runs/r1/assets/logs/events.jsonl'
    path.parent.mkdir(parents=True)
    bad = {'malformed': '{broken}\n', 'truncated': _saved_diagnostic()[:-1],
           'oversized': _saved_diagnostic('x' * 65536)}[fault]
    raw = _saved_diagnostic('earlier success') + bad
    path.write_text(raw, encoding='utf-8', newline='\n')
    (tmp_path / 'index.json').write_text(json.dumps({'experiments': [{'runs': [{'id': 'r1'}]}]}))
    assert _note_from_log(path) == '[error] diagnostic log unavailable'
    with pytest.raises(ValueError):
        format_logs(tmp_path)
    assert main(['logs', str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert path.read_bytes() == raw.encode('utf-8')


@pytest.mark.parametrize('tail', ['valid', 'malformed', 'truncated'])
def test_logs_command_reads_all_segments_or_rejects_late_damage(tmp_path, capsys, tail):
    from spec_mtp.flow.study_cli import main

    anchor = tmp_path / 'runs/r1/assets/logs/events.jsonl'
    anchor.parent.mkdir(parents=True)
    anchor.write_text(_saved_diagnostic('first'))
    last = anchor.with_name('events.000001.jsonl')
    raw = _saved_diagnostic('last')
    last.write_text(raw if tail == 'valid' else '{broken}\n' if tail == 'malformed' else raw[:-1])
    index = {'experiments': [{'runs': [{'id': 'r1'}]}]}
    (tmp_path / 'index.json').write_text(json.dumps(index))
    before = {p: p.read_bytes() for p in (anchor, last)}
    assert _log_links(index, tmp_path) == {'r1': '../runs/r1/assets/logs/events.000001.jsonl'}
    if tail == 'valid':
        assert len(diagnostic_lines(anchor)) == 2
        assert _note_from_log(anchor) == '[flow] last'
        text = format_logs(tmp_path)
        assert 'first' in text and 'last' in text and text.index('first') < text.index('last')
        assert main(['logs', str(tmp_path)]) == 0
        assert 'last' in capsys.readouterr().out
    else:
        with pytest.raises(ValueError): diagnostic_lines(anchor)
        assert _note_from_log(anchor) == '[error] diagnostic log unavailable'
        with pytest.raises(ValueError): format_logs(tmp_path)
        assert main(['logs', str(tmp_path)]) == 2
        captured = capsys.readouterr()
        assert not captured.out and captured.err
    assert all(p.read_bytes() == body for p, body in before.items())


def test_missing_diagnostic_anchor_never_falls_back_to_a_legacy_log(tmp_path, capsys):
    from spec_mtp.flow.study_cli import main

    parent = tmp_path / 'runs/r1/assets/logs'
    parent.mkdir(parents=True)
    anchor = parent / 'events.jsonl'
    (parent / 'events.000001.jsonl').write_text(_saved_diagnostic('orphan'))
    (parent / 'run.log').write_text('2026-01-01T00:00:00Z INFO r1 [flow] legacy\n')
    index = {'experiments': [{'runs': [{'id': 'r1'}]}]}
    (tmp_path / 'index.json').write_text(json.dumps(index))
    before = {p: p.read_bytes() for p in parent.iterdir()}
    with pytest.raises(ValueError): log_path(tmp_path, {'id': 'r1'}, 'r1')
    assert _note_from_log(anchor) == '[error] diagnostic log unavailable'
    with pytest.raises(ValueError): format_logs(tmp_path)
    with pytest.raises(ValueError): _log_links(index, tmp_path)
    assert main(['logs', str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert not anchor.exists() and all(p.read_bytes() == body for p, body in before.items())




@pytest.mark.parametrize('outcome', ['success', 'late_value_error', 'late_os_error', 'late_file_not_found'])
def test_logs_inventory_emits_one_complete_json_or_no_partial_output(tmp_path, monkeypatch, capsys, outcome):
    from spec_mtp.flow.study_cli import main
    from spec_mtp.structure.artifact.asset import tree as asset_tree
    from spec_mtp.structure.artifact.readout import logs as logs_readout

    (tmp_path / 'index.json').write_text(json.dumps({'experiments': [{'runs': [{'id': 'r1'}, {'id': 'r2'}]}]}), encoding='utf-8')
    for run_id in ('r1', 'r2'):
        path = tmp_path / f'runs/{run_id}/assets/logs/events.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(_saved_diagnostic(run_id), encoding='utf-8', newline='\n')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    actual = asset_tree.diagnostic_inventory
    visits = []
    def inventory_with_late_fault(anchor, now):
        visits.append(anchor)
        if len(visits) == 2 and outcome != 'success':
            error_type = {'late_value_error': ValueError, 'late_os_error': OSError,
                          'late_file_not_found': FileNotFoundError}[outcome]
            raise error_type(r'private payload sentinel; D:\private\secret-run; credential=never_echo_this')
        return actual(anchor, now)
    with monkeypatch.context() as patch:
        patch.setattr(asset_tree, 'diagnostic_inventory', inventory_with_late_fault)
        patch.setattr(logs_readout, 'diagnostic_inventory', inventory_with_late_fault, raising=False)
        code = main(['logs', str(tmp_path), '--inventory'])
    captured = capsys.readouterr()
    assert len(visits) == 2  # The late failure is after one genuinely inspected Run.
    if outcome == 'success':
        assert code == 0 and not captured.err
        body = json.loads(captured.out)  # Reject extra text or multiple JSON payloads.
        assert body['scope'] == 'run_diagnostics_only'
        assert [row['run_id'] for row in body['runs']] == ['r1', 'r2']
        assert all(row['disposition'] == 'retain' for row in body['runs'])
        expected = format_logs(tmp_path)
        assert main(['logs', str(tmp_path)]) == 0
        assert capsys.readouterr().out == expected  # No flag keeps the original event text.
    else:
        category = {'late_value_error': 'ValueError', 'late_os_error': 'OSError',
                    'late_file_not_found': 'FileNotFoundError'}[outcome]
        assert code == 2 and not captured.out
        assert captured.err == f'diagnostic inventory unavailable: {category}\n'
        assert 'private payload' not in captured.err and 'never_echo_this' not in captured.err
        assert str(tmp_path) not in captured.err and 'secret-run' not in captured.err
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('action', ['prepare', 'seal', 'purge'])
def test_logs_retention_emits_only_the_complete_core_receipt(tmp_path, monkeypatch, capsys, action):
    from spec_mtp.flow import study_cli

    calls = []
    seal_sha = 'b' * 64
    expected = {'scope': 'run_diagnostics_only', 'run_id': 'r1', 'action': action,
                'state': {'prepare': 'ready', 'seal': 'sealed', 'purge': 'purged'}[action]}

    def retained(study_dir, run_id, operation, *, seal_sha256=None):
        calls.append((study_dir, run_id, operation, seal_sha256))
        return expected

    monkeypatch.setattr(study_cli, 'retention_logs', retained, raising=False)
    argv = ['logs', str(tmp_path), '--retention', action, '--run-id', 'r1']
    if action == 'purge':
        argv += ['--seal-sha256', seal_sha]
    assert study_cli.main(argv) == 0
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out) == expected
    assert calls == [(tmp_path, 'r1', action, seal_sha if action == 'purge' else None)]
    assert list(tmp_path.iterdir()) == []  # No implicit make, source lock, or Run materialization.


@pytest.mark.parametrize('suffix', [
    ['--run-id', 'r1'], ['--seal-sha256', 'a' * 64],
    ['--inventory', '--run-id', 'r1'], ['--inventory', '--seal-sha256', 'a' * 64],
    ['--retention', 'prepare'], ['--retention', 'seal'],
    ['--retention', 'purge', '--run-id', 'r1'],
    ['--retention', 'prepare', '--run-id', 'r1', '--seal-sha256', 'a' * 64],
    ['--retention', 'seal', '--run-id', 'r1', '--seal-sha256', 'a' * 64],
    ['--inventory', '--retention', 'seal', '--run-id', 'r1'],
])
def test_logs_retention_requires_an_explicit_unambiguous_action(tmp_path, monkeypatch, capsys, suffix):
    from spec_mtp.flow import study_cli

    def forbidden(*args, **kwargs):
        pytest.fail('invalid action flags must not inspect or mutate a Study')

    monkeypatch.setattr(study_cli, 'retention_logs', forbidden, raising=False)
    monkeypatch.setattr(study_cli, 'inventory_logs', forbidden)
    monkeypatch.setattr(study_cli, 'format_logs', forbidden)
    try:
        code = study_cli.main(['logs', str(tmp_path), *suffix])
    except SystemExit as stopped:
        code = stopped.code
    captured = capsys.readouterr()
    assert code == 2 and not captured.out and captured.err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('error_type', [OSError, ValueError, FileNotFoundError, KeyboardInterrupt, SystemExit])
def test_logs_retention_failure_is_safe_and_cancellation_propagates(tmp_path, monkeypatch, capsys, error_type):
    from spec_mtp.flow import study_cli

    error = error_type(r'sensitive prompt secret; D:\private\source; token=not_public')

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(study_cli, 'retention_logs', failed, raising=False)
    argv = ['logs', str(tmp_path), '--retention', 'seal', '--run-id', 'r1']
    if error_type in (KeyboardInterrupt, SystemExit):
        with pytest.raises(error_type) as caught:
            study_cli.main(argv)
        assert caught.value is error
    else:
        assert study_cli.main(argv) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == ('' if error_type in (KeyboardInterrupt, SystemExit)
                            else f'diagnostic retention unavailable: {error_type.__name__}\n')
    assert list(tmp_path.iterdir()) == []
