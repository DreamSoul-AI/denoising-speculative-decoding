"""Input-mask figures bind fixed Git bytes and actual inputs without rewriting old evidence."""

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

command = importlib.import_module('studies.input_mask_baseline.flow.write.run')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def saved_report(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance

    inputs, repository = tmp_path / 'inputs', tmp_path / 'repository'
    inputs.mkdir()
    repository.mkdir()
    report = {'study_quality_passed': True, 'question_id': 171, 'source_digest': command.SOURCE_DIGEST,
        'source_binding': {'tool_sha256': 'unknown original tool identity'}, 'rows': [
            {'task': 'esp_mtp', 'mask': mask, 'raw': {'rows': [{'accepted': 19}]}}
            for mask in ('frozen_mean', 'gamma_last')]}
    report['rows'].append({'task': 'alignment_probe', 'raw': {
        'counts': {'accept': 1, 'reject': 3}, 'sums': {'accept': [.2, .4, .6], 'reject': [1.5, 1.8, 2.1]}}})
    archived = [(json.dumps(report, indent=2) + '\n').encode()]
    path = inputs / 'v4_execution.json'
    path.write_bytes(archived[0].replace(b'\n', b'\r\n'))
    calls = []
    def history(repo, commit, member, expected):
        calls.append(member)
        assert repo == repository and commit == 'd' * 40 and member == command.EXECUTION_MEMBER
        if _sha(archived[0]) != expected:
            raise ValueError('historical report SHA256 differs')
        return archived[0]
    monkeypatch.setattr(provenance, 'read_historical_bytes', history)
    return SimpleNamespace(report=report, path=path, archived=archived, repository=repository,
                           output=tmp_path / 'new-output', calls=calls)


def _render(case, **overrides):
    options = {'input_path': case.path, 'repository': case.repository, 'commit': 'd' * 40,
               'input_git_sha256': _sha(case.archived[0]), 'output_root': case.output}
    return command.write_plot(**(options | overrides))


def test_input_mask_plot_records_separate_git_runtime_and_rendering_identities(saved_report):
    case = saved_report
    before = case.path.read_bytes()
    target = _render(case)
    assert target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    result = json.loads((case.output / 'plot_complete.json').read_text())
    assert result['source_digest'] == 'ff2212ccbccb0e4e'
    assert result['historical_input']['sha256'] == _sha(case.archived[0]) != _sha(before)
    assert result['runtime_input'] == {'path': str(case.path), 'sha256': _sha(before)}
    assert result['figure'] == {'path': 'input_mask_baseline_v4.png', 'sha256': _sha(target.read_bytes())}
    assert result['original_source_binding'] == case.report['source_binding'] and result['rendering_sources']
    assert result['rendering_environment']['matplotlib'] and result['rendering_environment']['backend'] == 'Agg'
    assert result['rendering_environment']['canvas'].endswith('FigureCanvasAgg')
    assert case.path.read_bytes() == before


@pytest.mark.parametrize('fault', ['missing', 'git_sha', 'git_content', 'quality', 'source', 'question', 'nonfinite'])
def test_input_mask_plot_rejects_unbound_or_invalid_saved_reports_before_output(saved_report, fault):
    case, options = saved_report, {}
    if fault == 'missing':
        case.path.unlink()
    elif fault == 'git_sha':
        options['input_git_sha256'] = '0' * 64
    else:
        report = deepcopy(case.report)
        if fault == 'git_content': report['new'] = True
        if fault == 'quality': report['study_quality_passed'] = False
        if fault == 'source': report['source_digest'] = '670c744e0535dcad'
        if fault == 'question': report['question_id'] = 172
        if fault == 'nonfinite': report['rows'][2]['raw']['sums']['reject'][0] = float('nan')
        encoded = json.dumps(report).encode()
        case.path.write_bytes(encoded)
        if fault != 'git_content': case.archived[0] = encoded
    expected = {'source': 'source/quality', 'quality': 'source/quality', 'question': 'fixed q171',
                'nonfinite': 'cosine sums must be finite'}.get(fault)
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        _render(case, **options)
    assert not case.output.exists()


@pytest.mark.parametrize('placement', ['inputs', 'repository', 'ancestor', 'checkout', 'existing'])
def test_input_mask_plot_rejects_output_overlap_and_existing_evidence(saved_report, placement):
    case = saved_report
    output = {'inputs': case.path.parent / 'out', 'repository': case.repository / 'out',
        'ancestor': case.repository.parent, 'checkout': case.repository.parent / 'checkout/out', 'existing': case.output}[placement]
    if placement == 'checkout': (output.parent / '.git').mkdir(parents=True)
    if placement == 'existing':
        output.mkdir()
        (output / 'keep').write_bytes(b'old plot evidence')
    before = case.path.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        _render(case, output_root=output)
    assert not case.calls and case.path.read_bytes() == before
    if placement == 'existing': assert (output / 'keep').read_bytes() == b'old plot evidence'


@pytest.mark.parametrize('drift', ['input', 'source'])
def test_input_mask_plot_source_drift_keeps_partial_png_without_completion(saved_report, monkeypatch, drift):
    from studies.input_mask_baseline.structure import plots

    case = saved_report
    fake_source = case.output.parent / 'fake-plots.py'
    fake_source.write_bytes(b'# original rendering source\n')
    monkeypatch.setattr(plots, '__file__', str(fake_source))
    native = plots.baseline_figure
    def figure(data):
        result = native(data)
        save = result.savefig
        def change(*args, **kwargs):
            save(*args, **kwargs)
            (case.path if drift == 'input' else fake_source).write_bytes(b'changed while saving')
        result.savefig = change
        return result
    monkeypatch.setattr(plots, 'baseline_figure', figure)
    with pytest.raises(ValueError, match='changed'):
        _render(case)
    assert (case.output / 'input_mask_baseline_v4.png').is_file()
    assert not (case.output / 'plot_complete.json').exists()


def test_input_mask_plot_write_failure_clears_figure_without_publishing_completion(saved_report, monkeypatch):
    from studies.input_mask_baseline.structure import plots

    figure = plots.baseline_figure(saved_report.report)
    def fail(stream, **kwargs):
        stream.write(b'partial PNG')
        raise OSError('injected plot write failure')
    figure.savefig = fail
    monkeypatch.setattr(plots, 'baseline_figure', lambda data: figure)
    with pytest.raises(OSError, match='injected plot write failure'):
        _render(saved_report)
    assert not figure.axes
    assert (saved_report.output / 'input_mask_baseline_v4.png').read_bytes() == b'partial PNG'
    assert not (saved_report.output / 'plot_complete.json').exists()


def test_input_mask_plot_reads_the_resolved_input_when_its_argument_alias_changes(saved_report, monkeypatch):
    from spec_mtp.structure.artifact import provenance

    case = saved_report
    alias, other = case.output.parent / 'alias.json', case.output.parent / 'other.json'
    other.write_bytes(b'{"unbound":true}')
    selected = [case.path]
    native = Path.resolve
    def resolve(path, *args, **kwargs):
        return selected[0] if path == alias else native(path, *args, **kwargs)
    history = provenance.read_historical_bytes
    def redirect(*args):
        selected[0] = other
        return history(*args)
    monkeypatch.setattr(Path, 'resolve', resolve)
    monkeypatch.setattr(provenance, 'read_historical_bytes', redirect)
    _render(case, input_path=alias)
    result = json.loads((case.output / 'plot_complete.json').read_text())
    assert result['runtime_input']['path'] == str(case.path)
    assert other.read_bytes() == b'{"unbound":true}'


@pytest.mark.parametrize('missing', ['--input', '--repository', '--commit', '--input-git-sha256', '--output-root'])
def test_input_mask_plot_cli_requires_explicit_history_input_and_external_output(tmp_path, monkeypatch, missing):
    values = {'--input': str(tmp_path / 'report.json'), '--repository': str(tmp_path), '--commit': 'd' * 40,
              '--input-git-sha256': 'e' * 64, '--output-root': str(tmp_path / 'output')}
    monkeypatch.setattr(command, 'write_plot', lambda *args, **kwargs: pytest.fail('invalid CLI dispatched'))
    with pytest.raises(SystemExit) as stopped:
        command.main([value for key, item in values.items() if key != missing for value in (key, item)])
    assert stopped.value.code == 2 and not list(tmp_path.iterdir())
