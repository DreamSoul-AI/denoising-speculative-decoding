"""External Study selection happens before any source lock or runtime writes."""

import pytest

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
