"""Offline transfer analysis binds immutable Git and runtime inputs separately."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    module = importlib.import_module('studies.output_space_extension.flow.process.run')
    from spec_mtp.structure.artifact import provenance
    from studies._shared.structure import extension_diagnostics
    import torch

    raw = tmp_path / 'raw'
    raw.mkdir()
    repository = tmp_path / 'repository'
    repository.mkdir()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    execution = runtime / 'execution.json'
    data = {'source_digest': module.SOURCE_DIGEST, 'runs': [{'id': 'historical-run'}]}
    git_bytes = (json.dumps(data, indent=2) + '\n').encode()
    execution.write_bytes(git_bytes.replace(b'\n', b'\r\n'))
    tensor = raw / 'selected_tensors.pt'
    tensor.write_bytes(b'fixture-saved-tensors')
    report = {'quality_passed': True, 'source_digest': module.SOURCE_DIGEST,
              'raw_tensor_sha256': _sha(tensor.read_bytes()),
              'source_binding': {'matrix_sha256': _sha(execution.read_bytes())}}
    report_path = raw / 'diagnosis.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    commit = 'a' * 40
    git_sha = _sha(git_bytes)
    calls = []

    def historical(repo, revision, member, expected_sha256):
        calls.append(('history', repo, revision, member, expected_sha256))
        if expected_sha256 != git_sha:
            raise ValueError('historical SHA256 differs')
        return git_bytes

    def load(path, *, map_location, weights_only):
        calls.append(('tensor', path, str(map_location), weights_only))
        assert path == tensor and str(map_location) == 'cuda:0' and weights_only is True
        return {'saved': 'tensor fixture'}

    def compute(saved_report, matrix, tensors, *, device):
        assert str(device) == 'cuda:0'
        assert saved_report == report and matrix == data
        assert tensors == {'saved': 'tensor fixture'}
        return {'offline': True}

    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'IO fixture; no CUDA calculation')
    monkeypatch.setattr(extension_diagnostics, 'analyze_saved_transfers', compute)
    return SimpleNamespace(module=module, raw=raw, repository=repository, execution=execution,
                           report=report, report_path=report_path, tensor=tensor, commit=commit,
                           git_sha=git_sha, calls=calls, output=tmp_path / 'export')


def _analyze(case, **overrides):
    args = {'raw_root': case.raw, 'execution': case.execution, 'repository': case.repository,
            'commit': case.commit, 'execution_git_sha256': case.git_sha, 'output_root': case.output, 'device': 'cuda:0'}
    return case.module.analyze(**(args | overrides))


def test_analysis_distinguishes_git_blob_and_runtime_execution_bytes(saved_inputs):
    from studies._shared.structure import window_diagnostics

    case = saved_inputs
    inputs = (case.report_path, case.tensor, case.execution)
    before = {path: path.read_bytes() for path in inputs}
    target = _analyze(case)
    result = json.loads(target.read_text(encoding='utf-8'))
    assert target == case.output / 'slot_diagnosis.json'
    assert result['historical_input'] == {'commit': case.commit,
        'path': case.module.EXECUTION_MEMBER, 'sha256': case.git_sha}
    assert result['runtime_execution_sha256'] == _sha(before[case.execution]) != case.git_sha
    assert result['execution_content_equal'] is True
    assert result['analysis_provenance']['window_sha256'] == _sha(Path(window_diagnostics.__file__).read_bytes())
    assert result['findings'] == {'offline': True}
    assert case.calls == [('history', case.repository, case.commit,
                           case.module.EXECUTION_MEMBER, case.git_sha),
                          ('tensor', case.tensor, 'cuda:0', True)]
    assert before == {path: path.read_bytes() for path in inputs}


def test_analysis_rejects_computation_source_drift_before_output(saved_inputs, monkeypatch):
    from studies._shared.structure import extension_diagnostics, window_diagnostics

    case = saved_inputs
    real_source = Path(window_diagnostics.__file__)
    real_bytes = real_source.read_bytes()
    fake_source = case.raw.parent / 'margin.py'
    fake_source.write_bytes(b'# margin before computation\n')
    monkeypatch.setattr(window_diagnostics, '__file__', str(fake_source))

    def drift(*args, **kwargs):
        fake_source.write_bytes(b'# margin changed during computation\n')
        return {'offline': True}

    monkeypatch.setattr(extension_diagnostics, 'analyze_saved_transfers', drift)
    with pytest.raises(ValueError, match='changed'):
        _analyze(case)
    assert not case.output.exists()
    assert real_source.read_bytes() == real_bytes


@pytest.mark.parametrize('fault', ['missing_report', 'missing_tensor', 'missing_execution',
    'runtime_hash', 'git_hash', 'git_content', 'tensor_hash', 'source_digest', 'quality'])
def test_analysis_rejects_missing_or_unbound_inputs_without_output(saved_inputs, fault):
    case = saved_inputs
    options = {}
    if fault.startswith('missing_'):
        {'missing_report': case.report_path, 'missing_tensor': case.tensor,
         'missing_execution': case.execution}[fault].unlink()
    elif fault == 'runtime_hash':
        case.execution.write_bytes(case.execution.read_bytes() + b' ')
    elif fault == 'git_hash':
        options['execution_git_sha256'] = '0' * 64
    elif fault == 'git_content':
        case.execution.write_text(json.dumps({'source_digest': case.module.SOURCE_DIGEST,
                                             'runs': [{'id': 'different-run'}]}), encoding='utf-8')
        case.report['source_binding']['matrix_sha256'] = _sha(case.execution.read_bytes())
    elif fault == 'tensor_hash':
        case.tensor.write_bytes(b'changed tensor')
    else:
        case.report[{'source_digest': 'source_digest', 'quality': 'quality_passed'}[fault]] = False
    case.report_path.parent.mkdir(exist_ok=True)
    if fault not in ('missing_report',):
        case.report_path.write_text(json.dumps(case.report), encoding='utf-8')
    inputs = [path for path in (case.report_path, case.tensor, case.execution) if path.exists()]
    before = {path: path.read_bytes() for path in inputs}
    with pytest.raises((ValueError, FileNotFoundError)):
        _analyze(case, **options)
    assert not case.output.exists()
    assert before == {path: path.read_bytes() for path in inputs}


def test_analysis_never_overwrites_an_existing_output(saved_inputs):
    case = saved_inputs
    case.output.mkdir()
    target = case.output / 'slot_diagnosis.json'
    target.write_bytes(b'previous audit\n')
    with pytest.raises(FileExistsError):
        _analyze(case)
    assert target.read_bytes() == b'previous audit\n'
    assert not case.calls


def test_transfer_analysis_requires_explicit_cuda_without_cpu_fallback(saved_inputs):
    case = saved_inputs
    with pytest.raises(ValueError, match='no CPU fallback'):
        _analyze(case, device='cpu')
    assert not case.calls and not case.output.exists()


def test_transfer_analysis_pins_resolved_input_leaves_before_loading(saved_inputs, monkeypatch):
    import torch
    case = saved_inputs
    bound = case.raw / 'bound'; bound.mkdir()
    report, tensor = bound / 'report.json', bound / 'tensor.pt'
    report.write_bytes(case.report_path.read_bytes()); tensor.write_bytes(case.tensor.read_bytes())
    original = Path.resolve
    redirected = [False]
    def resolve(path, *args, **kwargs):
        if path in (case.report_path, case.tensor):
            if redirected[0]:
                return bound / 'unrelated-later-target'
            return report if path == case.report_path else tensor
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    loaded = []
    def load(path, *, map_location, weights_only):
        assert path == tensor and str(map_location) == 'cuda:0' and weights_only
        loaded.append(path); redirected[0] = True
        return {'saved': 'tensor fixture'}
    monkeypatch.setattr(torch, 'load', load)
    result = json.loads(_analyze(case).read_bytes())
    assert loaded == [tensor] and result['findings'] == {'offline': True}
    assert result['raw_report_sha256'] == _sha(report.read_bytes())
    assert result['raw_tensor_sha256'] == _sha(tensor.read_bytes())


@pytest.mark.parametrize('placement', ['raw', 'runtime', 'repository', 'ancestor'])
def test_analysis_output_root_is_separate_from_inputs_and_source(saved_inputs, placement):
    case = saved_inputs
    root = {'raw': case.raw / 'export', 'runtime': case.execution.parent / 'export',
            'repository': case.repository / 'export', 'ancestor': case.raw.parent}[placement]
    with pytest.raises(ValueError, match='separate'):
        _analyze(case, output_root=root)
    assert not (root / 'slot_diagnosis.json').exists()


@pytest.mark.parametrize('missing', ['--raw', '--execution', '--repository', '--commit',
                                     '--execution-git-sha256', '--output-root', '--device'])
def test_analysis_cli_requires_every_input_and_output_identity(saved_inputs, missing, capsys):
    case = saved_inputs
    options = {'--raw': str(case.raw), '--execution': str(case.execution),
               '--repository': str(case.repository), '--commit': case.commit,
               '--execution-git-sha256': case.git_sha, '--output-root': str(case.output), '--device': 'cuda:0'}
    args = [item for key, value in options.items() if key != missing for item in (key, value)]
    with pytest.raises(SystemExit) as stopped:
        case.module.main(args)
    assert stopped.value.code == 2 and missing in capsys.readouterr().err
    assert not case.output.exists()


def test_saved_transfer_analysis_preserves_skipped_prompts_and_second_slot_validity(record_property):
    import copy
    import torch
    from studies._shared.structure.extension_diagnostics import analyze_saved_transfers

    assert torch.cuda.is_available(), 'saved transfer correctness requires actual CUDA'
    device = torch.device('cuda:0')
    print(f'saved transfer engineering tensor.device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    record_property('cuda_device', torch.cuda.get_device_name(device))
    report = {'prompts': [
        {'question_id': 10, 'intervention_root': None, 'status': 'no continuing common root'},
        {'question_id': 11, 'intervention_root': 0, 'cases': [], 'baselines': [
            {'case': case, 'trace': [{'root_position': 0}]}
            for case in ('output_order0', 'output_order1')]},
    ]}
    raw = {
        'q11:baseline:output_order0:0': {'seed_logits': torch.tensor([[0., 3., 1.], [0., 1., 4.]], device=device)},
        'q11:baseline:output_order1:0': {'seed_logits': torch.tensor([[4., 3., 0.], [0., 4., 1.]], device=device)},
    }
    before_report = copy.deepcopy(report)
    before_raw = {key: value['seed_logits'].clone() for key, value in raw.items()}
    for tokens in ([0, 1, 2], [0, 1]):
        data = {'runs': [{'case': 'AR', 'raw': {'evidence': [
            {'output_ids': [0]}, {'output_ids': tokens}]}}]}
        before_data = copy.deepcopy(data)
        findings = analyze_saved_transfers(report, data, raw, device=device)
        assert findings[0] == {'question_id': 10, 'status': 'no continuing common root'}
        assert findings[1]['question_id'] == 11 and findings[1]['cases'] == []
        natural, = findings[1]['natural_baseline_slots']
        assert natural['root_position'] == 0
        assert natural['zero']['second_AR_path_valid'] is (len(tokens) == 3)
        assert natural['one']['second_AR_path_valid'] is False
        assert natural['valid_second_slot_opposite_signs'] is False
        if len(tokens) == 3:
            assert natural['zero']['second']['target_margin_vs_best_other'] == 3.
            assert natural['one']['second']['target_margin_vs_best_other'] == -3.
        else:
            assert natural['zero']['second'] is natural['one']['second'] is None
        assert data == before_data and report == before_report
        assert all(torch.equal(value['seed_logits'], before_raw[key]) for key, value in raw.items())
