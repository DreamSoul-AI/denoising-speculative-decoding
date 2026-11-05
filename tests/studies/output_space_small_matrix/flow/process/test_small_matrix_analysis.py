"""Pinned small-matrix report admission and CUDA saved-tensor calculations."""

from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

command = importlib.import_module('studies.output_space_small_matrix.flow.process.run')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    import torch
    from spec_mtp.structure.artifact import provenance
    from studies._shared.structure import extension_diagnostics

    raw, original, repository = (tmp_path / name for name in ('raw', 'original', 'repository'))
    for folder in (raw, original, repository):
        folder.mkdir()
    tensor, old_tensor = raw / 'selected_tensors.pt', original / 'selected_tensors.pt'
    tensor.write_bytes(b'saved tensor IO fixture')
    old_tensor.write_bytes(b'original observer IO fixture')
    observer = {'quality_passed': True, 'raw_tensor_sha256': _sha(old_tensor.read_bytes()),
                'source_binding': {'source_digest': command.SOURCE_DIGEST}}
    observer_bytes = (json.dumps(observer, indent=2) + '\n').encode()
    observer_path = original / 'diagnosis.json'
    observer_path.write_bytes(observer_bytes.replace(b'\n', b'\r\n'))
    report = {'quality_passed': True, 'source_digest': command.SOURCE_DIGEST,
              'raw_tensor_sha256': _sha(tensor.read_bytes()),
              'source_binding': {'observer_sha256': _sha(observer_path.read_bytes()),
                                 'tensor_sha256': observer['raw_tensor_sha256'],
                                 'helper_sha256': '0' * 64}}
    report_path = raw / 'diagnosis.json'
    report_bytes = (json.dumps(report, indent=2) + '\n').encode()
    report_path.write_bytes(report_bytes.replace(b'\n', b'\r\n'))
    archives = {name: report_bytes for name in command.REPORT_MEMBERS.values()}
    archives[command.OBSERVER_MEMBER] = observer_bytes
    calls = []

    def history(repo, commit, member, expected_sha):
        calls.append(('history', member, expected_sha))
        assert repo == repository and commit == 'a' * 40
        data = archives[member]
        if _sha(data) != expected_sha:
            raise ValueError('pinned historical SHA256 mismatch')
        return data

    def load(path, *, map_location, weights_only):
        calls.append(('tensor', path, str(map_location), weights_only))
        assert str(map_location) == 'cuda:0' and weights_only is True
        return {'io-fixture': path.name}

    monkeypatch.setattr(provenance, 'read_historical_bytes', history)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'IO fixture; no CUDA execution')
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(extension_diagnostics, 'analyze_small_matrix_transfers', lambda *args: {'io-only': True})
    monkeypatch.setattr(extension_diagnostics, 'analyze_small_matrix_cross_prompt', lambda *args: {'io-only': True})
    return SimpleNamespace(raw=raw, original=original, repository=repository, tensor=tensor, old_tensor=old_tensor,
        report=report, observer=observer, report_path=report_path, observer_path=observer_path, archives=archives,
        report_sha=_sha(report_bytes), observer_sha=_sha(observer_bytes), calls=calls, output=tmp_path / 'export')


def _analyze(case, kind='transfers', **overrides):
    options = {'kind': kind, 'raw_root': case.raw, 'repository': case.repository, 'commit': 'a' * 40,
               'report_git_sha256': case.report_sha, 'output_root': case.output, 'device': 'cuda:0'}
    if kind == 'transfers':
        options.update(original_root=case.original, observer_git_sha256=case.observer_sha)
    return command.analyze(**(options | overrides))


@pytest.mark.parametrize('kind', ['transfers', 'cross-prompt'])
def test_saved_analysis_keeps_runtime_git_and_current_source_identities_separate(saved_inputs, kind):
    case = saved_inputs
    before = {path: path.read_bytes() for path in (case.report_path, case.tensor, case.observer_path, case.old_tensor)}
    target = _analyze(case, kind)
    result = json.loads(target.read_text())
    assert result['source_digest'] == command.SOURCE_DIGEST and result['findings'] == {'io-only': True}
    assert result['original_source_binding'] == case.report['source_binding']
    assert result['runtime_inputs'][str(case.report_path)] == _sha(before[case.report_path]) != case.report_sha
    assert result['historical_inputs'][0]['sha256'] == case.report_sha
    assert len(result['historical_inputs']) == (2 if kind == 'transfers' else 1)
    assert result['analysis_provenance']['package_sources'] and result['analysis_device'] == 'cuda:0'
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('fault', ['missing_report', 'missing_tensor', 'git_sha', 'git_content', 'tensor_sha',
    'quality', 'source', 'missing_original_report', 'missing_original_tensor', 'observer_sha', 'observer_git_sha',
    'observer_content', 'original_tensor_sha'])
def test_saved_analysis_rejects_missing_or_unbound_originals_without_output(saved_inputs, fault):
    case, options = saved_inputs, {}
    missing = {'missing_report': case.report_path, 'missing_tensor': case.tensor,
               'missing_original_report': case.observer_path, 'missing_original_tensor': case.old_tensor}
    if fault in missing:
        missing[fault].unlink()
    elif fault == 'git_sha': options['report_git_sha256'] = '0' * 64
    elif fault == 'observer_git_sha': options['observer_git_sha256'] = '0' * 64
    elif fault == 'tensor_sha': case.tensor.write_bytes(b'changed tensor')
    elif fault == 'original_tensor_sha': case.old_tensor.write_bytes(b'changed observer tensor')
    elif fault in ('git_content', 'quality', 'source', 'observer_sha'):
        if fault == 'git_content': case.report['different-content'] = True
        if fault == 'quality': case.report['quality_passed'] = False
        if fault == 'source': case.report['source_digest'] = 'unverified'
        if fault == 'observer_sha': case.report['source_binding']['observer_sha256'] = '0' * 64
        case.report_path.write_text(json.dumps(case.report))
        if fault != 'git_content':
            data = json.dumps(case.report).encode()
            case.archives[command.REPORT_MEMBERS['transfers']] = data
            options['report_git_sha256'] = _sha(data)
    else:
        case.observer_path.write_bytes(b'{"changed-observer":true}')
    before = {path: path.read_bytes() for path in (case.report_path, case.tensor, case.observer_path, case.old_tensor)
              if path.exists()}
    with pytest.raises((ValueError, FileNotFoundError)):
        _analyze(case, **options)
    assert not case.output.exists() and not any(call[0] == 'tensor' for call in case.calls)
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('placement', ['raw', 'original', 'repository', 'ancestor', 'existing', 'checkout'])
def test_saved_analysis_rejects_output_overlap_or_reuse(saved_inputs, placement):
    case = saved_inputs
    target = {'raw': case.raw / 'out', 'original': case.original / 'out', 'repository': case.repository / 'out',
              'ancestor': case.raw.parent, 'existing': case.output, 'checkout': case.raw.parent / 'checkout/out'}[placement]
    if placement == 'existing':
        target.mkdir()
        (target / 'keep').write_bytes(b'prior output')
    if placement == 'checkout':
        (target.parent / '.git').mkdir(parents=True)
    with pytest.raises((ValueError, FileExistsError)):
        _analyze(case, output_root=target)
    assert not case.calls
    if placement == 'existing': assert (target / 'keep').read_bytes() == b'prior output'


@pytest.mark.parametrize('drift', ['report', 'source'])
def test_saved_analysis_rechecks_input_and_computation_sources_before_output(saved_inputs, monkeypatch, drift):
    from studies._shared.structure import extension_diagnostics

    case = saved_inputs
    fake_source = case.raw.parent / 'analysis-helper.py'
    fake_source.write_bytes(b'# original helper\n')
    monkeypatch.setattr(extension_diagnostics, '__file__', str(fake_source))

    def mutate(*args):
        (case.report_path if drift == 'report' else fake_source).write_bytes(b'changed during analysis')
        return []

    monkeypatch.setattr(extension_diagnostics, 'analyze_small_matrix_transfers', mutate)
    with pytest.raises(ValueError, match='changed'):
        _analyze(case)
    assert not case.output.exists()


def test_saved_analysis_never_falls_back_to_cpu(saved_inputs, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        _analyze(saved_inputs)
    assert not saved_inputs.output.exists() and not any(call[0] == 'tensor' for call in saved_inputs.calls)


def test_cross_prompt_does_not_invent_missing_original_helper_identity(saved_inputs):
    case = saved_inputs
    case.report.pop('source_binding')
    encoded = json.dumps(case.report).encode()
    case.report_path.write_bytes(encoded)
    case.archives[command.REPORT_MEMBERS['cross-prompt']] = encoded
    result = json.loads(_analyze(case, 'cross-prompt', report_git_sha256=_sha(encoded)).read_text())
    assert 'original_source_binding' not in result


def test_saved_analysis_rejects_optimized_python_before_reading_inputs(tmp_path):
    import os
    import subprocess
    import sys

    script = (
        "from importlib import import_module\n"
        "module = import_module('studies.output_space_small_matrix.flow.process.run')\n"
        "helpers = import_module('studies._shared.structure.extension_diagnostics')\n"
        "for call in (lambda: module.analyze('transfers', '.', '.', 'a' * 40, 'b' * 64, 'uncreated', device='cuda:0'),\n"
        "             lambda: helpers.analyze_small_matrix_transfers({}, {}, {}),\n"
        "             lambda: helpers.analyze_small_matrix_cross_prompt({}, {})):\n"
        "    try:\n"
        "        call()\n"
        "    except RuntimeError as error:\n"
        "        if 'assertions enabled' not in str(error):\n"
        "            raise\n"
        "    else:\n"
        "        raise RuntimeError('optimized audit did not reject')\n"
        "print('optimized-audit-rejected')\n"
    )
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-B', '-O', '-c', script], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'optimized-audit-rejected' in result.stdout and not list(tmp_path.iterdir())


@pytest.mark.parametrize('missing', ['--raw', '--repository', '--commit', '--report-git-sha256', '--output-root', '--device'])
def test_saved_analysis_cli_requires_explicit_history_input_device_and_output(tmp_path, missing, monkeypatch):
    values = {'--raw': str(tmp_path), '--repository': str(tmp_path), '--commit': 'a' * 40,
              '--report-git-sha256': 'b' * 64, '--output-root': str(tmp_path / 'output'), '--device': 'cuda:0'}
    monkeypatch.setattr(command, 'analyze', lambda **kwargs: pytest.fail('invalid CLI must not dispatch'))
    with pytest.raises(SystemExit) as caught:
        command.main(['cross-prompt', *[part for key, value in values.items() if key != missing for part in (key, value)]])
    assert caught.value.code == 2 and not list(tmp_path.iterdir())


@pytest.fixture
def cuda_observations():
    import torch

    assert torch.cuda.is_available(), 'saved tensor numerical validation requires CUDA'
    device = torch.device('cuda', torch.cuda.current_device())
    print(f'small matrix saved tensor device={device} name={torch.cuda.get_device_name(device)}')
    roots, label = (83, 84, 86, 94, 97), 'output_order1_from_output_order1'

    def sample(root):
        masks = torch.zeros((3, 2, 2), device=device)
        masks[0, 1] = torch.tensor([1., 2.], device=device)
        seeds = masks[0].clone() if root == 84 else torch.zeros((2, 2), device=device)
        return {'current_masks': masks, 'seed_mask_inputs': seeds, 'seed_logits': torch.zeros((2, 3), device=device),
                'current_mask_hidden': torch.zeros((3, 2, 2), device=device),
                'current_state': {'anchor': torch.zeros(2, device=device), 'velocity': torch.zeros(2, device=device),
                                  'proxies': torch.zeros((2, 2), device=device)},
                'source_root': {84: 83, 86: 84}.get(root), 'source_owner': {84: 0, 86: 1}.get(root)}

    raw = {f'{label}:{root}': sample(root) for root in roots}
    raw[label + ':intervention'] = {'donor_vector': raw[label + ':83']['current_masks'][0, 1].clone(),
                                   'original_vector': raw[label + ':83']['current_masks'][0, 1].clone()}
    original = {f'output_order1:{root}': deepcopy(raw[f'{label}:{root}']) for root in (83, 94, 97)}
    transfer = {'targets': list(roots), 'cases': [{'label': label, 'recipient': 'output_order1', 'donor': 'output_order1',
                                                 'replay': {'placebo_full_semantic_equal': True}}]}
    trace = [{'root_position': root, 'cache_before': root, 'cache_after': root + 1,
              'accepted_nodes': [1] if root == 84 else [], 'committed_tokens': [0], 'stopped': root == 86}
             for root in roots[:3]]
    windows = [{'root_position': root, 'states': {name: {'bit_equal': True} for name in ('anchor', 'velocity', 'proxies')},
                'first_slot': {'target_margin_vs_best_other': 1.},
                'baseline_first_slot': {'target_margin_vs_best_other': 1.}} for root in roots[:3]]
    cross = {'prompts': [{'question_id': 173, 'intervention_root': 83,
                         'baselines': [{'case': 'output_order1', 'trace': deepcopy(trace)}],
                         'cases': [{'label': 'swap', 'recipient': 'output_order1', 'donor': 'output_order1',
                                    'matched_acceptance_changes': [], 'windows': windows,
                                    'replay': {'trace': trace, 'unvisited_targets': [],
                                               'stats': {'accepted': 1, 'decode_calls': 3}}}]}]}
    cross_raw = {f'q173:swap:{root}': deepcopy(raw[f'{label}:{root}']) for root in roots[:3]}
    cross_raw.update({f'q173:baseline:output_order1:{root}': deepcopy(raw[f'{label}:{root}']) for root in roots[:3]})
    return SimpleNamespace(device=device, label=label, transfer=transfer, raw=raw, original=original,
                           cross=cross, cross_raw=cross_raw)


@pytest.mark.parametrize('fault', ['none', 'donor', 'propagation', 'committed_state', 'placebo'])
def test_transfer_cuda_audit_preserves_propagation_and_committed_state_guards(cuda_observations, fault):
    from studies._shared.structure.extension_diagnostics import analyze_small_matrix_transfers

    case = cuda_observations
    if fault == 'donor': case.raw[case.label + ':intervention']['donor_vector'][0] += 1
    if fault == 'propagation': case.raw[case.label + ':84']['current_state']['proxies'][0, 0] += 1
    if fault == 'committed_state':
        altered = deepcopy(case.transfer['cases'][0])
        altered['label'] = 'modified-self-transfer'
        case.transfer['cases'].append(altered)
        case.raw.update({key.replace(case.label, altered['label']): deepcopy(value) for key, value in list(case.raw.items())})
        case.raw[altered['label'] + ':84']['current_state']['anchor'][0] += 1
    if fault == 'placebo': case.transfer['cases'][0]['replay']['placebo_full_semantic_equal'] = False
    if fault != 'none':
        with pytest.raises(AssertionError):
            analyze_small_matrix_transfers(case.transfer, case.raw, case.original)
    else:
        result, = analyze_small_matrix_transfers(case.transfer, case.raw, case.original)
        assert len(result['propagation_edges']) == 2 and result['first_seed_at83_unchanged'] is True
        assert result['transferred_vector_norm'] == pytest.approx(5 ** .5)
        assert result['anchor_and_velocity_bit_identical_to_order1_placebo_at_all_declared_roots'] is True
        assert all(window['first_seed_logit_max_abs_delta_vs_donor_baseline'] == 0 for window in result['later_seed_comparisons'])


@pytest.mark.parametrize('fault', ['none', 'donor', 'propagation', 'committed_state', 'semantic', 'sign'])
def test_cross_prompt_cuda_audit_preserves_trace_state_and_margin_guards(cuda_observations, fault):
    from studies._shared.structure.extension_diagnostics import analyze_small_matrix_cross_prompt

    case = cuda_observations
    if fault == 'donor': case.cross_raw['q173:swap:83']['current_masks'][0, 1, 0] += 1
    if fault == 'propagation': case.cross_raw['q173:swap:84']['current_state']['proxies'][0, 0] += 1
    if fault == 'committed_state': case.cross_raw['q173:swap:84']['current_state']['anchor'][0] += 1
    if fault == 'semantic': case.cross['prompts'][0]['cases'][0]['replay']['trace'][0]['committed_tokens'] = [9]
    if fault == 'sign': case.cross['prompts'][0]['cases'][0]['windows'][0]['first_slot']['target_margin_vs_best_other'] = -1
    if fault != 'none':
        with pytest.raises(AssertionError):
            analyze_small_matrix_cross_prompt(case.cross, case.cross_raw)
    else:
        result, = analyze_small_matrix_cross_prompt(case.cross, case.cross_raw)
        assert result['semantic_trace_bit_equal'] is True and len(result['propagation_edges']) == 2
        assert result['changed_first_seed_vectors'] == result['changed_incoming_proxy_pairs'] == 0
        assert result['target_margin_sign_changed_windows'] == 0 and result['max_abs_target_margin_delta'] == 0


@pytest.mark.parametrize('kind', ['transfers', 'cross-prompt'])
def test_saved_tensor_files_run_on_cuda_with_pinned_report_bindings(cuda_observations, tmp_path, monkeypatch, kind):
    import torch
    from spec_mtp.structure.artifact import provenance

    case = cuda_observations
    raw, original, repository = (tmp_path / name for name in ('raw', 'original', 'repository'))
    for folder in (raw, original, repository): folder.mkdir()
    torch.save(case.raw if kind == 'transfers' else case.cross_raw, raw / 'selected_tensors.pt')
    torch.save(case.original, original / 'selected_tensors.pt')
    observer = {'quality_passed': True, 'raw_tensor_sha256': _sha((original / 'selected_tensors.pt').read_bytes()),
                'source_binding': {'source_digest': command.SOURCE_DIGEST}}
    observer_bytes = (json.dumps(observer, indent=2) + '\n').encode()
    (original / 'diagnosis.json').write_bytes(observer_bytes)
    report = deepcopy(case.transfer if kind == 'transfers' else case.cross)
    report.update(quality_passed=True, source_digest=command.SOURCE_DIGEST,
                  raw_tensor_sha256=_sha((raw / 'selected_tensors.pt').read_bytes()),
                  source_binding={'observer_sha256': _sha(observer_bytes), 'tensor_sha256': observer['raw_tensor_sha256'],
                                  'helper_sha256': 'unknown original helper; not replaced'})
    report_bytes = (json.dumps(report, indent=2) + '\n').encode()
    (raw / 'diagnosis.json').write_bytes(report_bytes)
    archived = {command.REPORT_MEMBERS[kind]: report_bytes, command.OBSERVER_MEMBER: observer_bytes}

    def history(repo, commit, member, expected):
        assert _sha(archived[member]) == expected
        return archived[member]

    monkeypatch.setattr(provenance, 'read_historical_bytes', history)
    before = {path: path.read_bytes() for root in (raw, original) for path in root.iterdir()}
    arguments = {'original_root': original, 'observer_git_sha256': _sha(observer_bytes)} if kind == 'transfers' else {}
    target = command.analyze(kind, raw, repository, 'a' * 40, _sha(report_bytes), tmp_path / 'new-output',
                             device=str(case.device), **arguments)
    result = json.loads(target.read_text())
    assert result['analysis_device'] == str(case.device) and result['findings']
    assert result['original_source_binding'] == report['source_binding']
    assert before == {path: path.read_bytes() for path in before}
