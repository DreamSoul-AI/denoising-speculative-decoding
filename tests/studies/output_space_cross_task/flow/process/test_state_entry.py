"""State entry dispatch requires complete roles, exact members and explicit CUDA admission."""
from types import SimpleNamespace
import pytest

from studies.output_space_cross_task.flow.process import state


@pytest.mark.parametrize('kind', tuple(state.ROLES))
def test_state_entry_rejects_cpu_before_loading_any_saved_tensor(monkeypatch, kind):
    import torch
    from studies.output_space_cross_task.flow.process import run as phase
    import importlib
    coordinator = importlib.import_module('studies.output_space_cross_task.flow.process.run')
    index = {'reports': {role: {'member': state.MEMBERS[role],
        **({'runtime_sha256_field': 'expansion_report_sha256'} if role == 'data' else {})} for role in state.ROLES[kind]}}
    monkeypatch.setattr(coordinator, 'bind_inputs', lambda *a, **kw: (SimpleNamespace(), index, {}, {}, {}))
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('CPU admission reached tensor loading'))
    with pytest.raises(ValueError, match='no CPU fallback'):
        state.analyze(kind, 'fixture', 'a' * 64, 'new-root', device='cpu')
    assert callable(phase)


@pytest.mark.parametrize('kind', tuple(state.ROLES))
@pytest.mark.parametrize('missing', ['--inputs', '--inputs-sha256', '--output-root', '--device'])
def test_state_cli_requires_all_explicit_input_output_and_device_arguments(monkeypatch, kind, missing):
    values = {'--inputs': 'input.json', '--inputs-sha256': 'a' * 64, '--output-root': 'new-root', '--device': 'cuda:0'}
    monkeypatch.setattr(state, 'analyze', lambda *a, **kw: pytest.fail('invalid CLI dispatched'))
    with pytest.raises(SystemExit) as caught:
        state.main([kind, *[p for k, v in values.items() if k != missing for p in (k, v)]])
    assert caught.value.code == 2


@pytest.mark.parametrize('role', ['data', 'sampling', 'owner', 'local', 'prior'])
@pytest.mark.parametrize('wrong_role', [False, True])
def test_state_original_tools_are_accepted_only_in_their_exact_role(tmp_path, monkeypatch, role, wrong_role):
    import hashlib
    import importlib
    import json
    from spec_mtp.structure.artifact import provenance
    coordinator = importlib.import_module('studies.output_space_cross_task.flow.process.run')
    folder, repo = tmp_path / 'inputs', tmp_path / 'repo'
    folder.mkdir(); repo.mkdir()
    tool, plan = b'# historical continuation source\n', b'# frozen plan\n'
    sha = lambda data: hashlib.sha256(data).hexdigest()
    report = {'quality_passed': True, 'source_digest': '670c744e0535dcad',
              'source_binding': {'tool': sha(tool), 'plan': sha(plan)}}
    data = json.dumps(report).encode()
    for name, raw in (('report.json', data), ('tool.py', tool), ('plan.md', plan)):
        (folder / name).write_bytes(raw)
    report_member = state.MEMBERS[role]
    tool_name, plan_name = state.PRODUCERS[role]
    member = 'tools/' + ('diagnose_q241_state_factors.py' if wrong_role else tool_name)
    plan_member = state.ROOT + plan_name
    sources = {'tool': {'path': str(folder / 'tool.py'), 'member': member, 'git_sha256': sha(tool)},
               'plan': {'path': str(folder / 'plan.md'), 'member': plan_member, 'git_sha256': sha(plan)}}
    index = {'schema': 1, 'kind': 'sampled-depth', 'repository': str(repo), 'commit': 'a' * 40,
        'reports': {role: {'path': str(folder / 'report.json'), 'member': report_member, 'git_sha256': sha(data)}},
        'files': {}, 'producer_sources': {role: sources}}
    index_path = folder / 'index.json'; index_path.write_text(json.dumps(index))
    blobs = {report_member: data, member: tool, plan_member: plan}
    monkeypatch.setattr(provenance, 'read_historical_bytes', lambda repo, commit, member, expected: blobs[member])
    producers = state.producer_members('sampled-depth' if role == 'data' else 'sampling-owner')
    call = lambda: coordinator.bind_inputs('sampled-depth', index_path, sha(index_path.read_bytes()), tmp_path / 'export',
                                           roles={role}, producers={role: producers[role]})
    if not wrong_role:
        session, _, reports, _, _ = call()
        assert reports[role] == report and not session.output.exists()
    else:
        with pytest.raises(ValueError, match='original producer'):
            call()
