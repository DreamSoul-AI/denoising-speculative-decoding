"""Saved trajectory JSON admission and bounded CUDA slot-one reconstruction."""

from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

from studies._shared.structure import extension_diagnostics as diagnostics

command = importlib.import_module('studies.output_space_small_matrix.flow.process.run')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _matrix():
    runs = []
    for name, accepted in [('AR', ([], [])), ('frozen_mean_chain', ([], [0])),
                           ('output_order0', ([1], [])), ('output_order1', ([2, 3], [0]))]:
        trace = [{'root_position': root, 'cache_before': root, 'accepted_nodes': nodes,
                  'committed_tokens': [2], 'stopped': False,
                  'output_hidden': {'last_owner_diff_from_root': name == 'output_order1'}}
                 for root, nodes in zip((2, 4), accepted)]
        evidence = [{'input_ids': [0, 1], 'output_ids': [0, 1, 2, 3, 4, 5],
                     'stats': {'trace': deepcopy(trace), 'output_hidden_prefill': {'scale': .5}}}
                    for _ in range(3)]
        runs.append({'case': name, 'raw': {'evidence': evidence},
                     'spec': {'output': {'order': 1 if name == 'output_order1' else 0}}})
    return {'study_quality_passed': True, 'question_ids': [172, 173, 174],
            'source_digest': command.SOURCE_DIGEST, 'runs': runs}


@pytest.fixture
def saved_window(tmp_path, monkeypatch):
    import torch
    from spec_mtp.structure.artifact import provenance

    raw, repository = tmp_path / 'raw', tmp_path / 'repository'
    raw.mkdir()
    repository.mkdir()
    matrix = _matrix()
    matrix_git = (json.dumps(matrix, indent=2) + '\n').encode()
    matrix_path = tmp_path / 'original-execution.json'
    matrix_path.write_bytes(matrix_git.replace(b'\n', b'\r\n'))
    tensor = raw / 'selected_tensors.pt'
    tensor.write_bytes(b'preserved observer tensors IO fixture')
    report = {'quality_passed': True, 'raw_tensor_sha256': _sha(tensor.read_bytes()),
              'source_binding': {'source_digest': command.SOURCE_DIGEST,
                                 'matrix_sha256': _sha(matrix_path.read_bytes()),
                                 'tool_sha256': 'unknown original helper remains unknown'}}
    report_git = (json.dumps(report, indent=2) + '\n').encode()
    report_path = raw / 'diagnosis.json'
    report_path.write_bytes(report_git.replace(b'\n', b'\r\n'))
    archives = {command.MATRIX_MEMBER: matrix_git, command.OBSERVER_MEMBER: report_git}
    calls = []

    def history(repo, commit, member, sha):
        calls.append(('history', member))
        if sha != _sha(archives[member]):
            raise ValueError('pinned historical SHA256 mismatch')
        return archives[member]

    def load(path, *, map_location, weights_only):
        calls.append(('tensor', path, str(map_location), weights_only))
        return {'IO-only': True}

    monkeypatch.setattr(provenance, 'read_historical_bytes', history)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'IO fixture; no CUDA execution')
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(diagnostics, 'analyze_small_matrix_windows', lambda *args: [{'IO-only': True}])
    return SimpleNamespace(raw=raw, repository=repository, matrix=matrix, matrix_path=matrix_path,
        matrix_sha=_sha(matrix_git), report=report, report_path=report_path, report_sha=_sha(report_git),
        tensor=tensor, archives=archives, calls=calls, output=tmp_path / 'new-output')


def _window(case, **overrides):
    options = {'kind': 'windows', 'raw_root': case.raw, 'repository': case.repository, 'commit': 'a' * 40,
               'report_git_sha256': case.report_sha, 'output_root': case.output, 'device': 'cuda:0',
               'execution': case.matrix_path, 'execution_git_sha256': case.matrix_sha}
    return command.analyze(**(options | overrides))


def test_trajectory_reads_only_pinned_json_and_preserves_pairing_scope(saved_window, monkeypatch):
    import torch

    case = saved_window
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('JSON analysis must not query CUDA'))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('JSON analysis must not load tensors'))
    before = case.matrix_path.read_bytes()
    target = command.analyze('trajectories', case.matrix_path, case.repository, 'a' * 40, case.matrix_sha, case.output)
    result = json.loads(target.read_text())
    assert len(result['findings']['pairs']) == 9 and result['findings']['new_model_forwards_for_analysis'] == 0
    assert 'analysis_device' not in result and 'original_source_binding' not in result
    assert result['runtime_inputs'][str(case.matrix_path)] == _sha(before) != case.matrix_sha
    assert case.matrix_path.read_bytes() == before


@pytest.mark.parametrize('fault', ['none', 'output', 'duplicate_root', 'input', 'cache_root', 'question_scope', 'quality'])
def test_trajectory_original_prefix_quality_and_unique_root_checks(fault):
    data = _matrix()
    selected = data['runs'][2]['raw']['evidence'][0]
    if fault == 'output': selected['output_ids'][-1] = 9
    if fault == 'duplicate_root': selected['stats']['trace'].append(deepcopy(selected['stats']['trace'][0]))
    if fault == 'input': selected['input_ids'] = [9]
    if fault == 'cache_root': selected['stats']['trace'][0]['cache_before'] = -1
    if fault == 'question_scope': data['question_ids'] = [171, 173, 174]
    if fault == 'quality': data['study_quality_passed'] = False
    before = deepcopy(data)
    if fault != 'none':
        with pytest.raises(AssertionError): diagnostics.diagnose_small_matrix_trajectories(data)
    else:
        result = diagnostics.diagnose_small_matrix_trajectories(data)
        pair = next(row for row in result['pairs'] if row['question_id'] == 172
                    and row['first'] == 'output_order0' and row['second'] == 'output_order1')
        assert pair['common_root_positions'] == [2, 4]
        assert len(pair['changed_acceptance_length_at_common_roots']) == 2
        assert pair['output_owner_difference_stat_changed_at_common_roots'] == 2
    assert data == before


def test_fixed_execution_history_reproduces_original_trajectory_diagnosis(historical_file):
    execution = historical_file('studies/output_space_small_matrix/docs/execution_v1.json')
    expected = historical_file('studies/output_space_small_matrix/docs/trajectory_diagnosis.json')
    original_bytes = execution.read_bytes()
    diagnosis = diagnostics.diagnose_small_matrix_trajectories(json.loads(original_bytes))
    assert diagnosis == json.loads(expected.read_bytes())
    assert execution.read_bytes() == original_bytes


def test_window_reader_separates_runtime_matrix_and_git_bytes(saved_window):
    case = saved_window
    before = {path: path.read_bytes() for path in (case.report_path, case.matrix_path, case.tensor)}
    result = json.loads(_window(case).read_text())
    assert len(result['historical_inputs']) == 2
    assert result['runtime_inputs'][str(case.matrix_path)] == _sha(before[case.matrix_path]) != case.matrix_sha
    assert result['original_source_binding'] == case.report['source_binding']
    assert 'Phi(proxy[0])' in result['formula']
    assert any(row[0] == 'tensor' and row[2:] == ('cuda:0', True) for row in case.calls)
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('fault', ['missing_matrix', 'matrix_git_sha', 'matrix_content', 'matrix_runtime_sha',
    'matrix_quality', 'tensor_hash', 'observer_git_sha', 'observer_source', 'missing_observer', 'missing_tensor'])
def test_window_reader_rejects_missing_or_unbound_matrix_before_tensor_loading(saved_window, fault):
    case, changes = saved_window, {}
    if fault == 'missing_matrix': case.matrix_path.unlink()
    elif fault == 'matrix_git_sha': changes['execution_git_sha256'] = '0' * 64
    elif fault == 'matrix_content': case.matrix_path.write_text('{"changed":true}')
    elif fault == 'tensor_hash': case.tensor.write_bytes(b'changed saved tensors')
    elif fault == 'observer_git_sha': changes['report_git_sha256'] = '0' * 64
    elif fault == 'missing_observer': case.report_path.unlink()
    elif fault == 'missing_tensor': case.tensor.unlink()
    elif fault == 'matrix_quality':
        case.matrix['study_quality_passed'] = False
        encoded = json.dumps(case.matrix).encode()
        case.matrix_path.write_bytes(encoded)
        case.archives[command.MATRIX_MEMBER] = encoded
        changes['execution_git_sha256'] = _sha(encoded)
    else:
        case.report['source_binding']['matrix_sha256' if fault == 'matrix_runtime_sha' else 'source_digest'] = 'wrong'
        encoded = json.dumps(case.report).encode()
        case.report_path.write_bytes(encoded)
        case.archives[command.OBSERVER_MEMBER] = encoded
        changes['report_git_sha256'] = _sha(encoded)
    with pytest.raises((ValueError, FileNotFoundError)): _window(case, **changes)
    assert not case.output.exists() and not any(row[0] == 'tensor' for row in case.calls)


@pytest.mark.parametrize('drift', ['runtime', 'source'])
def test_window_reader_rechecks_frozen_input_and_helper_before_output(saved_window, monkeypatch, drift):
    case = saved_window
    fake_source = case.raw.parent / 'window-helper.py'
    fake_source.write_bytes(b'# before computation')
    monkeypatch.setattr(diagnostics, '__file__', str(fake_source))

    def mutate(*args):
        (case.matrix_path if drift == 'runtime' else fake_source).write_bytes(b'changed during reading')
        return []

    monkeypatch.setattr(diagnostics, 'analyze_small_matrix_windows', mutate)
    with pytest.raises(ValueError, match='changed'): _window(case)
    assert not case.output.exists()


@pytest.mark.parametrize('missing', ['--execution', '--execution-git-sha256', '--device'])
def test_window_cli_requires_matrix_and_cuda_identity(tmp_path, missing):
    values = {'--raw': str(tmp_path), '--repository': str(tmp_path), '--output-root': str(tmp_path / 'out'),
              '--commit': 'a' * 40, '--report-git-sha256': 'b' * 64, '--execution': str(tmp_path / 'matrix.json'),
              '--execution-git-sha256': 'c' * 64, '--device': 'cuda:0'}
    with pytest.raises(SystemExit) as caught:
        command.main(['windows', *[part for key, value in values.items() if key != missing for part in (key, value)]])
    assert caught.value.code == 2 and not list(tmp_path.iterdir())


@pytest.fixture
def cuda_windows():
    import torch

    assert torch.cuda.is_available(), 'saved window reconstruction requires CUDA'
    device = torch.device('cuda', torch.cuda.current_device())
    print(f'saved window reconstruction device={device} name={torch.cuda.get_device_name(device)}')
    raw = {}
    for case in ('output_order0', 'output_order1'):
        for root in (83, 94, 97):
            proxies = torch.tensor([[1., 2.], [2., 1.]], device=device)
            saved = .5 * proxies[0] / proxies[0].norm()
            raw[f'{case}:{root}'] = {
                'source_owner': 0, 'source_root': root - 1,
                'source_state': {'anchor': torch.tensor([2., 1.], device=device),
                                 'velocity': torch.tensor([.3, -.2], device=device), 'proxies': proxies},
                'seed_mask_inputs': torch.stack((saved, saved)),
            }
    return SimpleNamespace(device=device, raw=raw, matrix=_matrix(), observer={'quality_passed': True})


@pytest.mark.parametrize('fault', ['none', 'owner', 'seed', 'zero_proxy'])
def test_saved_window_cuda_reconstruction_preserves_owner_and_tolerance_checks(cuda_windows, fault):
    case = cuda_windows
    sample = case.raw['output_order1:83']
    if fault == 'owner': sample['source_owner'] = 1
    if fault == 'seed': sample['seed_mask_inputs'][0, 0] += .01
    if fault == 'zero_proxy': sample['source_state']['proxies'][0].zero_()
    if fault != 'none':
        with pytest.raises((AssertionError, ValueError)):
            diagnostics.analyze_small_matrix_windows(case.observer, case.matrix, case.raw)
    else:
        findings = diagnostics.analyze_small_matrix_windows(case.observer, case.matrix, case.raw)
        assert len(findings) == 6 and all(row['slot1_identical_with_source_velocity_zeroed'] for row in findings)
        assert all(row['saved_seed_reconstruction_max_abs_error'] < 1e-7 for row in findings)
        assert sample['source_state']['velocity'].abs().sum().item() > 0


def test_saved_window_tensor_file_runs_on_cuda_with_original_matrix_binding(cuda_windows, tmp_path, monkeypatch):
    import torch
    from spec_mtp.structure.artifact import provenance

    case = cuda_windows
    raw, repository = tmp_path / 'raw', tmp_path / 'repository'
    raw.mkdir()
    repository.mkdir()
    tensor = raw / 'selected_tensors.pt'
    torch.save(case.raw, tensor)
    matrix_bytes = (json.dumps(case.matrix, indent=2) + '\n').encode()
    matrix_path = tmp_path / 'actual-execution.json'
    matrix_path.write_bytes(matrix_bytes.replace(b'\n', b'\r\n'))
    report = {'quality_passed': True, 'raw_tensor_sha256': _sha(tensor.read_bytes()),
              'source_binding': {'source_digest': command.SOURCE_DIGEST,
                                 'matrix_sha256': _sha(matrix_path.read_bytes()), 'tool_sha256': 'unknown original'}}
    report_bytes = (json.dumps(report, indent=2) + '\n').encode()
    report_path = raw / 'diagnosis.json'
    report_path.write_bytes(report_bytes)
    archives = {command.MATRIX_MEMBER: matrix_bytes, command.OBSERVER_MEMBER: report_bytes}

    def history(repo, commit, member, expected):
        assert _sha(archives[member]) == expected
        return archives[member]

    monkeypatch.setattr(provenance, 'read_historical_bytes', history)
    before = {path: path.read_bytes() for path in (tensor, matrix_path, report_path)}
    target = command.analyze('windows', raw, repository, 'a' * 40, _sha(report_bytes), tmp_path / 'new-output',
                             device=str(case.device), execution=matrix_path, execution_git_sha256=_sha(matrix_bytes))
    result = json.loads(target.read_text())
    assert len(result['findings']) == 6 and result['analysis_device'] == str(case.device)
    assert result['original_source_binding'] == report['source_binding']
    assert before == {path: path.read_bytes() for path in before}
