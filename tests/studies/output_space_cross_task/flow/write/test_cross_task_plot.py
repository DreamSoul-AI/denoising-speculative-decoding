"""The cross-task plot keeps native and canonical historical identities independent."""
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

writer = importlib.import_module('studies.output_space_cross_task.flow.write.run')


@pytest.fixture
def case(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance
    folder, repo = tmp_path / 'inputs', tmp_path / 'repo'
    folder.mkdir(); repo.mkdir()
    native = {'study_quality_passed': True, 'source_digest': '670c744e0535dcad', 'question_ids': [241, 242, 243, 244],
        'runs': [{'case': name, 'raw': {'evidence': [{'stats': {'accepted': j + i}} for j in range(4)]}}
                 for i, name in enumerate(('frozen_mean_chain', 'output_order0', 'output_order1'))]}
    canonical = {'quality_passed': True, 'question_id': 241,
                 'accepted': dict(zip(('native_self', 'native_graft', 'canonical_self', 'canonical_graft'), (11, 12, 14, 14)))}
    archive = {role: json.dumps(data).encode() for role, data in (('execution', native), ('canonical', canonical))}
    paths = {}
    for role, raw in archive.items():
        paths[role] = folder / (role + '.json'); paths[role].write_bytes(raw + b'\r\n')
    def read(repository, commit, member, expected):
        assert repository == repo and commit == 'e' * 40
        role = next(role for role, name in writer.MEMBERS.items() if member == name)
        if hashlib.sha256(archive[role]).hexdigest() != expected: raise ValueError('historical hash differs')
        return archive[role]
    monkeypatch.setattr(provenance, 'read_historical_bytes', read)
    return SimpleNamespace(paths=paths, archive=archive, repo=repo, output=tmp_path / 'export')


def _write(case):
    return writer.write_plot(case.paths['execution'], case.paths['canonical'], case.repo, 'e' * 40,
        hashlib.sha256(case.archive['execution']).hexdigest(), hashlib.sha256(case.archive['canonical']).hexdigest(), case.output)


def test_cross_task_plot_binds_both_inputs_and_preserves_separate_count_bars(case):
    before = {path: path.read_bytes() for path in case.paths.values()}
    image = _write(case)
    receipt = json.loads((case.output / 'plot_complete.json').read_text())
    assert image.read_bytes().startswith(b'\x89PNG') and receipt['figure']['sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert len(receipt['historical_inputs']) == 2 and receipt['rendering_sources']
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('fault', ['content', 'quality', 'source', 'existing', 'overlap'])
def test_cross_task_plot_refuses_unbound_reports_or_unsafe_output(case, fault):
    if fault == 'existing': case.output.mkdir()
    elif fault == 'overlap': case.output = case.paths['execution'].parent / 'out'
    else:
        data = json.loads(case.archive['execution'])
        if fault == 'content': data['different'] = True
        if fault == 'quality': data['study_quality_passed'] = False
        if fault == 'source': data['source_digest'] = 'unknown'
        raw = json.dumps(data).encode(); case.paths['execution'].write_bytes(raw)
        if fault != 'content': case.archive['execution'] = raw
    with pytest.raises((ValueError, FileExistsError)):
        _write(case)
    if fault != 'existing': assert not case.output.exists()


def test_cross_task_plot_drift_during_save_prevents_completion(case, monkeypatch):
    from studies.output_space_cross_task.structure import plots
    native = plots.acceptance_figure
    def figure(*args):
        result = native(*args); save = result.savefig
        def mutate(*args, **kwargs):
            save(*args, **kwargs); case.paths['canonical'].write_bytes(b'changed')
        result.savefig = mutate
        return result
    monkeypatch.setattr(plots, 'acceptance_figure', figure)
    with pytest.raises(ValueError, match='changed'):
        _write(case)
    assert (case.output / 'acceptance_v1.png').exists() and not (case.output / 'plot_complete.json').exists()


@pytest.mark.parametrize('missing', ['--execution', '--canonical', '--repository', '--commit', '--execution-git-sha256', '--canonical-git-sha256', '--output-root'])
def test_cross_task_plot_cli_requires_every_explicit_history_argument(tmp_path, monkeypatch, missing):
    values = {name: str(tmp_path / 'value') for name in ('--execution', '--canonical', '--repository', '--commit', '--execution-git-sha256', '--canonical-git-sha256', '--output-root')}
    monkeypatch.setattr(writer, 'write_plot', lambda *a: pytest.fail('invalid arguments dispatched'))
    with pytest.raises(SystemExit) as caught:
        writer.main([p for k, v in values.items() if k != missing for p in (k, v)])
    assert caught.value.code == 2


@pytest.mark.parametrize('fault', ['cleanup_only', 'body_plus_cleanup', 'cleanup_mutates_input'])
def test_output_space_cross_task_figure_cleanup_is_part_of_completion(case, monkeypatch, fault):
    from studies.output_space_cross_task.structure import plots

    before = {path: path.read_bytes() for path in list(case.paths.values())}
    fake_source = case.output.parent / 'cleanup-rendering-source.py'
    fake_source.write_bytes(b'# initial rendering source\n')
    monkeypatch.setattr(plots, '__file__', str(fake_source))
    source_before = fake_source.read_bytes()
    primary = OSError('injected Figure body failure')
    cleanup = RuntimeError('injected Figure cleanup failure')
    calls = []

    class Figure:
        canvas = object()
        def savefig(self, stream, **kwargs):
            calls.append('save')
            stream.write(b'partial PNG retained')
            if fault == 'body_plus_cleanup':
                raise primary
        def clear(self):
            calls.append('clear')
            if fault == 'cleanup_mutates_input':
                case.paths['canonical'].write_bytes(b'changed during Figure cleanup')
            else:
                raise cleanup

    monkeypatch.setattr(plots, 'acceptance_figure', lambda *args, **kwargs: Figure())
    expected = ValueError if fault == 'cleanup_mutates_input' else OSError if fault == 'body_plus_cleanup' else RuntimeError
    with pytest.raises(expected) as caught:
        _write(case)
    if fault == 'body_plus_cleanup':
        assert caught.value is primary and caught.value.figure_cleanup_error is cleanup
    elif fault == 'cleanup_only':
        assert caught.value is cleanup
    else:
        assert 'changed' in str(caught.value)
    assert calls == ['save', 'clear']
    assert (case.output / 'acceptance_v1.png').read_bytes() == b'partial PNG retained'
    assert not (case.output / 'plot_complete.json').exists()
    if fault != 'cleanup_mutates_input':
        assert {path: path.read_bytes() for path in before} == before
    if fault != 'cleanup_mutates_source':
        assert fake_source.read_bytes() == source_before
