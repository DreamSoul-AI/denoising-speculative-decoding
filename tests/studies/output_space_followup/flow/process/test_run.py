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
    module = importlib.import_module('studies.output_space_followup.flow.process.run')
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

    def compute(saved_report, matrix, tensors):
        assert saved_report == report and matrix == data
        assert tensors == {'saved': 'tensor fixture'}
        return {'offline': True}

    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'CUDA fixture; no tensor operations')
    monkeypatch.setattr(extension_diagnostics, 'analyze_saved_followup', compute)
    return SimpleNamespace(module=module, raw=raw, repository=repository, execution=execution,
                           report=report, report_path=report_path, tensor=tensor, commit=commit,
                           git_sha=git_sha, calls=calls, output=tmp_path / 'export')


def _analyze(case, **overrides):
    args = {'raw_root': case.raw, 'execution': case.execution, 'repository': case.repository,
            'commit': case.commit, 'execution_git_sha256': case.git_sha, 'output_root': case.output}
    return case.module.analyze(**(args | overrides))


def test_analysis_distinguishes_git_blob_and_runtime_execution_bytes(saved_inputs):
    from studies._shared.structure import extension_diagnostics

    case = saved_inputs
    inputs = (case.report_path, case.tensor, case.execution)
    before = {path: path.read_bytes() for path in inputs}
    target = _analyze(case)
    result = json.loads(target.read_text(encoding='utf-8'))
    assert target == case.output / 'followup_analysis.json'
    assert result['historical_input'] == {'commit': case.commit,
        'path': case.module.EXECUTION_MEMBER, 'sha256': case.git_sha}
    assert result['runtime_execution_sha256'] == _sha(before[case.execution]) != case.git_sha
    assert result['execution_content_equal'] is True
    assert result['analysis_provenance']['helper_sha256'] == _sha(Path(extension_diagnostics.__file__).read_bytes())
    assert result['findings'] == {'offline': True}
    assert case.calls == [('history', case.repository, case.commit,
                           case.module.EXECUTION_MEMBER, case.git_sha),
                          ('tensor', case.tensor, 'cuda:0', True)]
    assert before == {path: path.read_bytes() for path in inputs}


def test_analysis_rejects_computation_source_drift_before_output(saved_inputs, monkeypatch):
    from studies._shared.structure import extension_diagnostics

    case = saved_inputs
    real_source = Path(extension_diagnostics.__file__)
    real_bytes = real_source.read_bytes()
    fake_source = case.raw.parent / 'margin.py'
    fake_source.write_bytes(b'# margin before computation\n')
    monkeypatch.setattr(extension_diagnostics, '__file__', str(fake_source))

    def drift(*args):
        fake_source.write_bytes(b'# margin changed during computation\n')
        return {'offline': True}

    monkeypatch.setattr(extension_diagnostics, 'analyze_saved_followup', drift)
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
    target = case.output / 'followup_analysis.json'
    target.write_bytes(b'previous audit\n')
    with pytest.raises(FileExistsError):
        _analyze(case)
    assert target.read_bytes() == b'previous audit\n'
    assert not case.calls


@pytest.mark.parametrize('placement', ['raw', 'runtime', 'repository', 'ancestor'])
def test_analysis_output_root_is_separate_from_inputs_and_source(saved_inputs, placement):
    case = saved_inputs
    root = {'raw': case.raw / 'export', 'runtime': case.execution.parent / 'export',
            'repository': case.repository / 'export', 'ancestor': case.raw.parent}[placement]
    with pytest.raises(ValueError, match='separate'):
        _analyze(case, output_root=root)
    assert not (root / 'followup_analysis.json').exists()


@pytest.mark.parametrize('missing', ['--raw', '--execution', '--repository', '--commit',
                                     '--execution-git-sha256', '--output-root'])
def test_analysis_cli_requires_every_input_and_output_identity(saved_inputs, missing, capsys):
    case = saved_inputs
    options = {'--raw': str(case.raw), '--execution': str(case.execution),
               '--repository': str(case.repository), '--commit': case.commit,
               '--execution-git-sha256': case.git_sha, '--output-root': str(case.output)}
    args = [item for key, value in options.items() if key != missing for item in (key, value)]
    with pytest.raises(SystemExit) as stopped:
        case.module.main(args)
    assert stopped.value.code == 2 and missing in capsys.readouterr().err
    assert not case.output.exists()


def test_analysis_refuses_cpu_fallback(saved_inputs, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is required'):
        _analyze(saved_inputs)
    assert not saved_inputs.output.exists()
    assert not any(call[0] == 'tensor' for call in saved_inputs.calls)


def test_followup_saved_scope_and_state_checks_on_cuda(capsys):
    import copy
    import torch
    from studies._shared.structure.extension_diagnostics import analyze_saved_followup

    assert torch.cuda.is_available(), 'CUDA correctness validation requires an available CUDA device'
    device = torch.device('cuda', torch.cuda.current_device())
    print(f'followup correctness device={device} name={torch.cuda.get_device_name(device)}')
    masks = torch.zeros((2, 2, 2), device=device)
    donor_masks = masks.clone()
    donor_masks[0, 1] = torch.tensor([1., 2.], device=device)
    transferred = masks.clone()
    transferred[0, 1] = donor_masks[0, 1]
    state = {key: torch.zeros(2, device=device) for key in ('anchor', 'velocity', 'proxies')}
    seed = torch.zeros((2, 2), device=device)
    recipient = {'current_masks': masks, 'seed_mask_inputs': seed, 'current_state': state}
    actual = {'current_masks': transferred, 'seed_mask_inputs': seed.clone(),
              'current_state': {key: value.clone() for key, value in state.items()}}
    tensors = {'q11:transfer:0': actual, 'q11:baseline:output_order0:0': recipient,
               'q11:baseline:output_order1:0': {'current_masks': donor_masks}}
    window = {'root_position': 0, 'recipient_baseline_visited': True,
              'states': {key: {'bit_equal': True, 'max_abs_delta': 0.} for key in state},
              'slot_rankings': {'seed_logits_finite': True}, 'first_seed_bit_equal': False,
              'first_slot': {'target_margin_vs_best_other': -1.},
              'baseline_first_slot': {'target_margin_vs_best_other': 1.}}
    case = {'label': 'transfer', 'recipient': 'output_order0', 'donor': 'output_order1',
            'windows': [window], 'replay': {'stats': {'accepted': 2, 'decode_calls': 1}},
            'matched_acceptance_changes': [0], 'actual_only_roots': [2], 'baseline_only_roots': [3]}
    report = {'prompts': [
        {'question_id': 10, 'intervention_root': None, 'status': 'no shared root'},
        {'question_id': 11, 'intervention_root': 0, 'cases': [case]}]}
    data = {'per_prompt': [{}, {'cells': {name: {'accepted': value} for name, value in
            [('frozen_mean_chain', 1), ('output_order0', 2), ('output_order1', 3)]}}]}
    before_report, before_data = copy.deepcopy(report), copy.deepcopy(data)
    findings = analyze_saved_followup(report, data, tensors)
    assert findings[0] == {'question_id': 10, 'status': 'no shared root'}
    assert findings[1]['generator_seed'] == 124 and findings[1]['one_accepted'] == 3
    summary, = findings[1]['cases']
    assert summary['matched_windows'] == summary['anchor_velocity_bit_equal_windows'] == 1
    assert summary['first_margin_sign_changed_roots'] == [0]
    assert summary['accepted'] == 2 and summary['actual_only_roots'] == [2]
    assert report == before_report and data == before_data
    actual['current_masks'][0, 1, 0] += 1
    with pytest.raises(AssertionError):
        analyze_saved_followup(report, data, tensors)
    actual['current_masks'][0, 1] = donor_masks[0, 1]
    actual['current_state']['anchor'][0] = 1
    with pytest.raises(AssertionError):
        analyze_saved_followup(report, data, tensors)
    captured = capsys.readouterr().out
    assert 'device=cuda:' in captured
    print(captured, end='')
