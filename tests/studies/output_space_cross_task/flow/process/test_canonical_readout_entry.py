"""Explicit saved canonical source identity and actual CUDA entry validation."""

import hashlib
import json
from pathlib import Path

import pytest

from studies.output_space_cross_task.flow.process.native import analyze


@pytest.fixture
def canonical_index(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance

    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    repository = inputs / 'repository'
    repository.mkdir()
    blobs = {}
    source_commit, report_commit = 'a' * 40, 'b' * 40

    def entry(name, data, member, commit):
        data = data if isinstance(data, bytes) else json.dumps(data).encode()
        path = inputs / name
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        blobs[commit, member] = data
        return {'path': str(path), 'sha256': sha, 'member': member, 'git_sha256': sha, 'commit': commit}

    tool = entry('tool.py', b'# exact original producer fixture\n', 'tools/diagnose_q241_canonical.py', source_commit)
    plan = entry('plan.md', b'# exact original plan fixture\n',
                 'studies/output_space_cross_task/docs/CANONICAL_PLAN.md', source_commit)
    tensor = inputs / 'saved.pt'
    tensor.write_bytes(b'saved tensor placeholder; rejected IO tests must never load it')
    cases = [{'label': name, 'replay': {'output_equal_independent_AR': True,
              'pre_intervention_trace_equal': True, 'visited_targets': [717], 'trace': [],
              'stats': {'accepted': i + 1, 'new_tokens': 3}, 'observer_forward_calls': 2},
              'canonical_replacements': [{'canonical_state_bit_equal': True, 'incoming_proxies_unchanged': True}]}
             for i, name in enumerate(('native_self', 'native_graft', 'canonical_self', 'canonical_graft'))]
    report = {'quality_passed': True, 'source_digest': '670c744e0535dcad',
              'source_binding': {'tool': tool['sha256'], 'plan': plan['sha256']},
              'raw_tensor_sha256': provenance.file_sha256(tensor), 'cases': cases}
    member = 'studies/output_space_cross_task/docs/canonical_diagnosis_v1.json'
    original = entry('report.json', report, member, report_commit)
    index = {'schema': 1, 'kind': 'canonical', 'repository': str(repository), 'commit': report_commit,
             'reports': {'canonical': original},
             'files': {'tensor': {'path': str(tensor), 'sha256': report['raw_tensor_sha256']}},
             'producer_sources': {'canonical': {'tool': tool, 'plan': plan}}}
    path = inputs / 'index.json'

    def save():
        path.write_text(json.dumps(index), encoding='utf-8')
        return provenance.file_sha256(path)

    calls = []

    def historical(repository, commit, member, sha):
        data = blobs[commit, member]
        assert hashlib.sha256(data).hexdigest() == sha
        calls.append((commit, member))
        return data

    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    return index, path, save, report, entry, calls, tmp_path / 'output'


@pytest.mark.parametrize('fault', ['tool_hash', 'plan_hash', 'report_member', 'tensor_binding',
                                  'index_sha', 'missing_source', 'device'])
def test_canonical_entry_rejects_unbound_saved_bytes_before_loading(canonical_index, monkeypatch, fault):
    import torch

    index, path, save, report, entry, calls, output = canonical_index
    if fault in ('tool_hash', 'plan_hash'):
        claim = fault.split('_')[0]
        Path(index['producer_sources']['canonical'][claim]['path']).write_bytes(b'changed producer bytes')
    elif fault == 'report_member':
        index['reports']['canonical'] = entry('report.json', report,
            'studies/output_space_cross_task/docs/transfer_diagnosis_v1.json', 'b' * 40)
    elif fault == 'tensor_binding':
        report['raw_tensor_sha256'] = '0' * 64
        index['reports']['canonical'] = entry('report.json', report,
            'studies/output_space_cross_task/docs/canonical_diagnosis_v1.json', 'b' * 40)
    elif fault == 'missing_source':
        index['producer_sources']['canonical'].pop('tool')
    sha = save()
    before = {str(p): p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('unbound input reached tensor loading'))
    with pytest.raises((ValueError, AssertionError)):
        analyze('canonical', path, '0' * 64 if fault == 'index_sha' else sha, output,
                device='cpu' if fault == 'device' else 'cuda:0')
    assert not output.exists()
    assert before == {str(p): p.read_bytes() for p in path.parent.iterdir() if p.is_file()}


def test_canonical_entry_reads_actual_cuda_states_with_separate_historical_source_commit(canonical_index, record_property):
    import torch
    from spec_mtp.structure.artifact.provenance import file_sha256

    assert torch.cuda.is_available(), 'actual CUDA required; no CPU fallback'
    device = torch.device('cuda:0')
    print(f'canonical saved entry device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    record_property('cuda_device', torch.cuda.get_device_name(device))
    index, path, save, report, entry, calls, output = canonical_index
    state = {'anchor': torch.tensor([1., 2.], device=device), 'velocity': torch.tensor([.5, -.25], device=device)}
    data = {'canonical_states': {717: state}}
    data.update({name + ':717': {'current_state': {key: value.clone() for key, value in state.items()}}
                 for name in ('canonical_self', 'canonical_graft')})
    tensor = Path(index['files']['tensor']['path'])
    torch.save(data, tensor)
    sha = file_sha256(tensor)
    index['files']['tensor']['sha256'] = report['raw_tensor_sha256'] = sha
    index['reports']['canonical'] = entry('report.json', report,
        'studies/output_space_cross_task/docs/canonical_diagnosis_v1.json', 'b' * 40)
    result = analyze('canonical', path, save(), output, device=str(device))
    assert result['native_delta'] == result['canonical_delta'] == 1
    assert result['H2_bounded_count_persistence_prediction'] == 'preserved'
    assert result['readout_device'] == str(device) and file_sha256(tensor) == sha
    binding = json.loads((output / 'readout_inputs.json').read_bytes())
    assert binding['output_sources']['canonical_readout.json']
    assert ('a' * 40, 'tools/diagnose_q241_canonical.py') in calls
    assert ('b' * 40, 'studies/output_space_cross_task/docs/canonical_diagnosis_v1.json') in calls
