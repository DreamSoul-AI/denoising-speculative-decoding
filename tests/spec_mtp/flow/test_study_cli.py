"""External Study selection happens before any source lock or runtime writes."""

import json

import pytest

from tests.conftest import saved_diagnostic as _saved_diagnostic
from spec_mtp.structure.artifact.readout.status import _note_from_log
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
