"""Saved figures bind immutable Git content and actual runtime bytes before publishing completion."""

from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

command = importlib.import_module('studies.output_space_small_matrix.flow.write.run')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def plot_inputs(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance
    from studies.output_space_small_matrix.structure import plots

    folder, repository = tmp_path / 'inputs', tmp_path / 'repository'
    folder.mkdir()
    repository.mkdir()
    methods = plots.METHODS
    execution = {'study_quality_passed': True, 'question_ids': [172, 173, 174],
        'source_digest': command.SOURCE_DIGEST, 'per_prompt': [
            {'question_id': qid, 'cells': {case: {'accepted': i + j} for j, case in enumerate(methods)}}
            for i, qid in enumerate((172, 173, 174))]}
    window = {'quality_passed': True, 'question_id': 173, 'windows': [83, 94, 97], 'model_digest': 'model',
        'raw_tensor_sha256': 'a' * 64,
        'source_binding': {'source_digest': command.SOURCE_DIGEST, 'helper_sha256': 'unknown original helper'},
        'cases': [{'case': case, 'windows': [
            {'root_position': root, 'first_slot': {'target_margin_vs_best_other': j - i / 2}}
            for j, root in enumerate((83, 94, 97))]} for i, case in enumerate(methods)]}
    paths, archives = {}, {}
    for kind, report in (('acceptance', execution), ('windows', window)):
        paths[kind] = folder / (kind + '.json')
        data = (json.dumps(report, indent=2) + '\n').encode()
        archives[command.MEMBERS[kind]] = data
        paths[kind].write_bytes(data.replace(b'\n', b'\r\n'))
    scale = {'quality_passed': True, 'question_id': 173, 'root_position': 83, 'model_digest': 'model',
        'source_binding': {'source_digest': command.SOURCE_DIGEST, 'observer_sha256': _sha(paths['windows'].read_bytes()),
                           'tensor_sha256': window['raw_tensor_sha256']},
        'cells': [{'direction': direction, 'norm_source': norm,
                   'first_slot': {'target_margin_vs_best_other': i - j}}
                  for i, direction in enumerate(('input', 'output')) for j, norm in enumerate(('input', 'output'))]}
    paths['scale'] = folder / 'scale.json'
    archives[command.SCALE_MEMBER] = (json.dumps(scale, indent=2) + '\n').encode()
    paths['scale'].write_bytes(archives[command.SCALE_MEMBER])

    def historical(repo, commit, member, expected):
        assert repo == repository and commit == 'b' * 40
        if _sha(archives[member]) != expected:
            raise ValueError('historical plot SHA256 differs')
        return archives[member]

    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    return SimpleNamespace(paths=paths, archives=archives, reports={'acceptance': execution, 'windows': window, 'scale': scale},
        repository=repository, output=tmp_path / 'new-output')


def _render(case, kind='acceptance', **overrides):
    options = {'kind': kind, 'input_path': case.paths[kind], 'repository': case.repository, 'commit': 'b' * 40,
        'input_git_sha256': _sha(case.archives[command.MEMBERS[kind]]), 'output_root': case.output}
    if kind == 'windows':
        options.update(scale_path=case.paths['scale'], scale_git_sha256=_sha(case.archives[command.SCALE_MEMBER]))
    return command.write_plots(**(options | overrides))


@pytest.mark.parametrize('kind', ['acceptance', 'windows'])
def test_saved_plot_writes_png_and_completion_with_separate_runtime_git_and_tool_identities(plot_inputs, kind):
    case = plot_inputs
    before = {path: path.read_bytes() for path in case.paths.values()}
    target = _render(case, kind)
    assert target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    result = json.loads((case.output / 'plot_complete.json').read_text())
    assert result['figure'] == {'path': target.name, 'sha256': _sha(target.read_bytes())}
    assert result['runtime_inputs'][str(case.paths[kind])] == _sha(before[case.paths[kind]])
    assert result['historical_inputs'][0]['sha256'] != result['runtime_inputs'][str(case.paths[kind])]
    assert result['rendering_sources'] and result['rendering_environment']['canvas'].endswith('FigureCanvasAgg')
    if kind == 'windows':
        assert result['original_source_bindings'][str(case.paths['windows'])] == case.reports['windows']['source_binding']
        assert len(result['historical_inputs']) == 2
    else:
        assert result['original_source_bindings'] == {}
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('fields', [('resources',), ('cross_order_first_seed_masks',),
                                   ('resources', 'cross_order_first_seed_masks')])
def test_original_window_plot_keeps_verified_archive_annotations_separate(plot_inputs, fields):
    case = plot_inputs
    additions = {name: {'archival_only': True} for name in fields}
    case.archives[command.MEMBERS['windows']] = json.dumps(case.reports['windows'] | additions).encode()
    case.archives[command.SCALE_MEMBER] = json.dumps(case.reports['scale'] | {'resources': {'archival_only': True}}).encode()
    before = {path: path.read_bytes() for path in case.paths.values()}
    _render(case, 'windows')
    result = json.loads((case.output / 'plot_complete.json').read_text())
    assert result['archival_annotations'] == {command.MEMBERS['windows']: additions,
                                             command.SCALE_MEMBER: {'resources': {'archival_only': True}}}
    assert result['original_source_bindings'][str(case.paths['windows'])] == case.reports['windows']['source_binding']
    assert result['runtime_inputs'][str(case.paths['windows'])] == _sha(before[case.paths['windows']])
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('fault', ['unknown_archive_field', 'missing_scientific_field', 'changed_common_field',
                                  'runtime_extra_field', 'changed_common_annotation'])
def test_window_archive_annotations_do_not_relax_shared_content_or_schema(plot_inputs, fault):
    case = plot_inputs
    actual, archived = deepcopy(case.reports['windows']), deepcopy(case.reports['windows'])
    archived['resources'] = {'value': 1}
    if fault == 'unknown_archive_field':
        archived['not_an_annotation'] = True
    elif fault == 'missing_scientific_field':
        del actual['cases']
    elif fault == 'changed_common_field':
        actual['question_id'] = 174
    elif fault == 'runtime_extra_field':
        actual['runtime_extra'] = True
    else:
        actual['resources'] = {'value': 2}
    case.paths['windows'].write_text(json.dumps(actual))
    case.archives[command.MEMBERS['windows']] = json.dumps(archived).encode()
    with pytest.raises(ValueError, match='runtime plot report differs'):
        _render(case, 'windows')
    assert not case.output.exists()


@pytest.mark.parametrize('fault', ['missing', 'git_sha', 'git_content', 'quality', 'scale_quality', 'source', 'observer', 'tensor', 'model'])
def test_saved_plot_rejects_missing_unverified_or_unpaired_reports_before_output(plot_inputs, fault):
    case, options = plot_inputs, {}
    if fault == 'missing':
        case.paths['windows'].unlink()
    elif fault == 'git_sha':
        options['input_git_sha256'] = '0' * 64
    else:
        role = 'windows' if fault in ('git_content', 'quality', 'source') else 'scale'
        report = deepcopy(case.reports[role])
        if fault == 'git_content': report['new'] = True
        if fault == 'quality': report['quality_passed'] = False
        if fault == 'scale_quality': report['quality_passed'] = False
        if fault == 'source': report['source_binding']['source_digest'] = 'unknown'
        if fault == 'observer': report['source_binding']['observer_sha256'] = '0' * 64
        if fault == 'tensor': report['source_binding']['tensor_sha256'] = '0' * 64
        if fault == 'model': report['model_digest'] = 'other-model'
        encoded = json.dumps(report).encode()
        case.paths[role].write_bytes(encoded)
        if fault != 'git_content':
            member = command.MEMBERS['windows'] if role == 'windows' else command.SCALE_MEMBER
            case.archives[member] = encoded
    with pytest.raises((ValueError, FileNotFoundError), match='quality gate' if fault in ('quality', 'scale_quality') else None):
        _render(case, 'windows', **options)
    assert not case.output.exists()


@pytest.mark.parametrize('placement', ['inputs', 'repository', 'ancestor', 'checkout', 'existing'])
def test_saved_plot_refuses_overlapping_or_existing_output_without_overwrite(plot_inputs, placement):
    case = plot_inputs
    output = {'inputs': case.paths['windows'].parent / 'out', 'repository': case.repository / 'out',
              'ancestor': case.repository.parent, 'checkout': case.repository.parent / 'checkout/out',
              'existing': case.output}[placement]
    if placement == 'checkout': (output.parent / '.git').mkdir(parents=True)
    if placement == 'existing':
        output.mkdir()
        (output / 'keep').write_bytes(b'old figure')
    with pytest.raises((ValueError, FileExistsError)):
        _render(case, output_root=output)
    if placement == 'existing': assert (output / 'keep').read_bytes() == b'old figure'


@pytest.mark.parametrize('drift', ['input', 'rendering_source'])
def test_saved_plot_drift_after_rendering_keeps_partial_png_without_completion(plot_inputs, monkeypatch, drift):
    from studies.output_space_small_matrix.structure import plots

    case = plot_inputs
    fake_source = case.repository.parent / 'fake-plots.py'
    fake_source.write_bytes(b'# original rendering source\n')
    monkeypatch.setattr(plots, '__file__', str(fake_source))
    original = plots.acceptance_figure
    def figure(data):
        result = original(data)
        save = result.savefig
        def save_and_change(*args, **kwargs):
            save(*args, **kwargs)
            (case.paths['acceptance'] if drift == 'input' else fake_source).write_bytes(b'changed while rendering')
        result.savefig = save_and_change
        return result
    monkeypatch.setattr(plots, 'acceptance_figure', figure)
    with pytest.raises(ValueError, match='changed'):
        _render(case)
    assert (case.output / 'acceptance_v1.png').is_file()
    assert not (case.output / 'plot_complete.json').exists()


def test_saved_plot_uses_the_resolved_input_even_if_its_argument_alias_changes(plot_inputs, monkeypatch):
    from pathlib import Path
    from spec_mtp.structure.artifact import provenance

    case = plot_inputs
    alias = case.output.parent / 'argument-alias.json'
    alternate = case.output.parent / 'unbound.json'
    alternate.write_bytes(b'{"unbound":true}')
    actual = case.paths['acceptance']
    resolve = Path.resolve
    target = [actual]
    def resolved(path, *args, **kwargs):
        return target[0] if path == alias else resolve(path, *args, **kwargs)
    historical = provenance.read_historical_bytes
    def change_alias(*args):
        target[0] = alternate
        return historical(*args)
    monkeypatch.setattr(Path, 'resolve', resolved)
    monkeypatch.setattr(provenance, 'read_historical_bytes', change_alias)
    _render(case, input_path=alias)
    result = json.loads((case.output / 'plot_complete.json').read_text())
    assert str(actual) in result['runtime_inputs']
    assert str(alias) not in result['runtime_inputs'] and str(alternate) not in result['runtime_inputs']
    assert alternate.read_bytes() == b'{"unbound":true}'


@pytest.mark.parametrize('missing', ['--input', '--repository', '--commit', '--input-git-sha256', '--output-root'])
def test_saved_plot_cli_requires_explicit_bound_history_and_new_output(tmp_path, monkeypatch, missing):
    options = {'--input': str(tmp_path / 'input.json'), '--repository': str(tmp_path), '--commit': 'b' * 40,
               '--input-git-sha256': 'c' * 64, '--output-root': str(tmp_path / 'output')}
    monkeypatch.setattr(command, 'write_plots', lambda *args, **kwargs: pytest.fail('invalid CLI dispatched'))
    with pytest.raises(SystemExit) as stopped:
        command.main(['acceptance', *[part for key, value in options.items() if key != missing for part in (key, value)]])
    assert stopped.value.code == 2 and not list(tmp_path.iterdir())
