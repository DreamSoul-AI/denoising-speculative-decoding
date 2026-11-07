"""Saved q178 role/source IO contracts; all tensor computation is stubbed here."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

reader = importlib.import_module('studies.output_space_extension.flow.process.run')


@pytest.fixture
def saved_roles(tmp_path, monkeypatch):
    import torch
    from spec_mtp.structure.artifact import provenance
    from studies.output_space_extension.structure import saved_diagnostics
    folder = tmp_path / 'inputs'; folder.mkdir()
    repository = tmp_path / 'repository'; repository.mkdir()
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    blobs = {}
    def entry(member, data):
        raw = json.dumps(data).encode() if isinstance(data, dict) else data
        path = folder / (str(len(blobs)) + '-' + Path(member).name)
        path.write_bytes(raw); blobs[member] = raw
        return {'path': str(path), 'member': member, 'git_sha256': sha(raw)}
    docs = 'studies/output_space_extension/docs/'
    shared = 'studies/_shared/structure/'
    source = {'source_digest': reader.SOURCE_DIGEST, 'quality_passed': True}
    native = {**source, 'runs': []}
    native_entry = entry(reader.EXECUTION_MEMBER, native)
    numerical_sources = {claim: entry(member, ('# original ' + claim).encode()) for claim, member in (
        ('tool_sha256', shared + 'diagnose_q178_numerics.py'),
        ('plan_sha256', docs + 'EXPLICIT_NUMERICAL_PLAN.md'),
        ('observer_helper_sha256', shared + 'diagnose_output_windows.py'),
        ('transfer_helper_sha256', shared + 'diagnose_proxy_transfer.py'))}
    previous_tensor = folder / 'previous.pt'; previous_tensor.write_bytes(b'previous saved tensor')
    operator_tensor = folder / 'operator.pt'; operator_tensor.write_bytes(b'operator saved tensor')
    previous = {**source, 'raw_tensor_sha256': sha(previous_tensor.read_bytes()),
        'source_binding': {**{claim: value['git_sha256'] for claim, value in numerical_sources.items()},
                           'matrix_sha256': native_entry['git_sha256']}}
    previous_path = folder / 'previous.json'; previous_path.write_text(json.dumps(previous))
    previous_sha = sha(previous_path.read_bytes())
    previous_entry = entry(docs + 'numerical_analysis_v1.json', {'raw_report_sha256': previous_sha})
    previous_entry.update(path=str(previous_path), runtime_sha256_field='raw_report_sha256')
    operator_sources = {
        'tool': entry(docs + 'operator_tool_v1.py.txt', b'# original operator producer'),
        'plan': entry(docs + 'OPERATOR_PLAN.md', b'# original operator plan'),
        'canonical_helper': numerical_sources['tool_sha256'],
        'observer_helper': numerical_sources['observer_helper_sha256'],
        'transfer_helper': numerical_sources['transfer_helper_sha256']}
    operator = {**source, 'raw_tensor_sha256': sha(operator_tensor.read_bytes()),
        'source_binding': {**{claim: value['git_sha256'] for claim, value in operator_sources.items()},
            'native': native_entry['git_sha256'], 'old_report': previous_sha,
            'old_tensor': sha(previous_tensor.read_bytes())}}
    operator_entry = entry(docs + 'operator_diagnosis_v1.json', operator)
    index = {'schema': 1, 'kind': 'extension-operators', 'repository': str(repository), 'commit': 'a' * 40,
        'reports': {'native': native_entry, 'previous': previous_entry, 'operator': operator_entry},
        'files': {'operator_tensor': {'path': str(operator_tensor), 'sha256': sha(operator_tensor.read_bytes())},
                  'previous_tensor': {'path': str(previous_tensor), 'sha256': sha(previous_tensor.read_bytes())}},
        'producer_sources': {'previous': numerical_sources, 'operator': operator_sources}}
    path = folder / 'index.json'
    def write_index():
        path.write_text(json.dumps(index))
        return sha(path.read_bytes())
    write_index()
    def historical(repo, commit, member, expected):
        raw = blobs[member]
        if sha(raw) != expected:
            raise ValueError('Git fixture SHA differs')
        return raw
    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'IO fixture; no CUDA computation')
    loads = []
    def load(path, *, map_location, weights_only, mmap):
        assert str(map_location) == 'cuda:0' and weights_only and mmap
        loads.append(path)
        return {'IO': str(path)}
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(saved_diagnostics, 'analyze_operators', lambda *a: {'quality_passed': True, 'cases': []})
    return SimpleNamespace(index=index, path=path, output=tmp_path / 'output', sha=sha, blobs=blobs,
        loads=loads, write_index=write_index, protocol=saved_diagnostics)


def test_saved_q178_binds_original_roles_and_separates_current_tool_from_old_cpu_reference(saved_roles):
    case = saved_roles
    before = {path: path.read_bytes() for path in case.path.parent.rglob('*') if path.is_file()}
    result = reader.analyze_saved('operators', case.path, case.write_index(), case.output, device='cuda:0')
    assert result['quality_passed'] and result['readout_model_forwards'] == result['readout_profiler_calls'] == 0
    assert result['historical_CPU_margin_revalidated'] is False and len(case.loads) == 2
    assert (case.output / 'operators_saved_evidence_v1.json').is_file()
    assert (case.output / 'readout_inputs.json').is_file()
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('fault', ['extra_tensor', 'missing_tensor', 'wrong_role', 'source_bytes', 'cpu'])
def test_saved_q178_rejects_unbound_roles_sources_and_cpu_before_loading(saved_roles, fault):
    case = saved_roles
    if fault == 'extra_tensor':
        case.index['files']['extra_tensor'] = dict(case.index['files']['operator_tensor'])
    elif fault == 'missing_tensor':
        del case.index['files']['previous_tensor']
    elif fault == 'wrong_role':
        case.index['producer_sources']['operator']['tool']['member'] = 'tools/unrelated.py'
    elif fault == 'source_bytes':
        Path(case.index['producer_sources']['operator']['tool']['path']).write_bytes(b'changed old source')
    with pytest.raises((ValueError, KeyError)):
        reader.analyze_saved('operators', case.path, case.write_index(), case.output,
                             device='cpu' if fault == 'cpu' else 'cuda:0')
    assert not case.loads and not case.output.exists()


def test_saved_q178_rechecks_source_drift_before_completion(saved_roles, monkeypatch):
    case = saved_roles
    def drift(*args):
        Path(case.index['producer_sources']['operator']['tool']['path']).write_bytes(b'changed during audit')
        return {'quality_passed': True}
    monkeypatch.setattr(case.protocol, 'analyze_operators', drift)
    with pytest.raises(ValueError, match='changed'):
        reader.analyze_saved('operators', case.path, case.write_index(), case.output, device='cuda:0')
    assert not case.output.exists()


@pytest.mark.parametrize('entry', ['live', 'profile', 'kernel-load'])
def test_stopped_q178_live_entries_reject_before_dispatch(monkeypatch, entry):
    monkeypatch.setattr(reader, 'analyze_saved', lambda *a, **kw: pytest.fail('stopped live work dispatched'))
    with pytest.raises(RuntimeError, match='permanently stopped'):
        reader.main([entry])
