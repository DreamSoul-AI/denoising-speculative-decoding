"""Office command rejection must precede output creation or scientific work."""

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest


def _tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def local_office(tmp_path, monkeypatch):
    import torch
    from studies.trajectory_observation.structure import office_traj_fork

    command = importlib.import_module('studies.trajectory_observation.flow.execute.office')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').write_bytes(b'{"local":"fixture"}\n')
    manifest = tmp_path / 'model-sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256(
        (weights / 'config.json').read_bytes()).hexdigest()}), encoding='utf-8')
    dolly = tmp_path / 'dolly.jsonl'
    dolly.write_text('{"category":"creative_writing","instruction":"local fixture"}\n', encoding='utf-8')
    plan = tmp_path / 'research-plan.md'
    plan.write_bytes('本地工程入口夹具，不执行研究。\n'.encode('utf-8'))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append('scientific work')
        raise AssertionError('rejected command reached model, fitting or dataset construction')

    for module in (command.q, command.t, command.tr):
        for name in ('_load_model', '_fit', 'load_dolly_rows'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(office_traj_fork, '_load_fp32_model', forbidden)
    monkeypatch.setattr(command.q, '_identity', lambda: {'git_head': 'a' * 40, 'dirty': False})
    monkeypatch.setattr(command.t, '_identity', lambda: {'git_head': 'a' * 40, 'dirty': False})
    monkeypatch.setattr(command.tr, '_run_identity', lambda: {'git_head': 'a' * 40, 'dirty': False})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'IO admission stub')
    values = {'--dest': str(tmp_path / 'new-office-run'), '--model-root': str(weights),
              '--model-sources': str(manifest), '--dolly': str(dolly), '--plan': str(plan),
              '--plan-sha256': hashlib.sha256(plan.read_bytes()).hexdigest()}
    return SimpleNamespace(command=command, root=tmp_path, weights=weights, manifest=manifest,
                           dolly=dolly, plan=plan, values=values, calls=calls, torch=torch)


def _argv(case, name, *, values=None, extra=()):
    options = case.values if values is None else values
    return [name, *[part for key, value in options.items() for part in (key, value)],
            '--only', 'smoke', *extra]


@pytest.mark.parametrize('name', ['qwen8b', 'tree', 'traj'])
def test_office_binding_records_the_actual_protocol_and_closes_its_session(local_office, monkeypatch, name):
    case = local_office
    closed = []
    model = SimpleNamespace(device='cuda:0', dtype='IO admission stub', close=lambda: closed.append(True))
    for module in (case.command.q, case.command.t):
        monkeypatch.setattr(module, '_load_model', lambda *a: (model, object(), 'stub'))
        monkeypatch.setattr(module, '_smoke', lambda *a: None)
    monkeypatch.setattr(case.command.tr, '_smoke', lambda *a: None)
    monkeypatch.setattr(case.command.q, '_offline_caches', lambda root: None)
    monkeypatch.setattr(case.command.q, '_fit', lambda *a: ({}, {}))
    monkeypatch.setattr(case.command.t, '_registered', lambda *a: ([], [], 'prompt-sha', 'actual-tree-protocol'))
    monkeypatch.setattr(case.command.tr, 'round_config', lambda *a: {
        'protocol': 'actual-trajectory-protocol', 'select': lambda rows: ([], []),
        'check': lambda *a: 'prompt-sha'})
    assert case.command.main(_argv(case, name)) == 0
    output = case.root / 'new-office-run'
    saved = json.loads((output / 'protocol_inputs.json').read_text(encoding='utf-8'))
    assert saved['protocol'] == {'qwen8b': case.command.q.PROTOCOL,
        'tree': 'actual-tree-protocol', 'traj': 'actual-trajectory-protocol'}[name]
    assert saved['inputs'][str(case.plan)] == case.values['--plan-sha256']
    assert (output / 'protocol_complete.json').is_file() and closed == [True]
    assert not case.calls


@pytest.mark.parametrize('name', ['qwen8b', 'tree', 'traj'])
@pytest.mark.parametrize('missing', ['--dest', '--model-root', '--model-sources', '--dolly', '--plan', '--plan-sha256'])
def test_office_requires_every_local_input_plan_and_output_argument(local_office, name, missing):
    case = local_office
    options = {key: value for key, value in case.values.items() if key != missing}
    before = _tree(case.root)
    with pytest.raises(SystemExit) as stopped:
        case.command.main(_argv(case, name, values=options))
    assert stopped.value.code == 2
    assert not case.calls and _tree(case.root) == before
    assert not (case.root / 'new-office-run').exists()


@pytest.mark.parametrize('name', ['qwen8b', 'tree', 'traj'])
@pytest.mark.parametrize('fault', ['plan_sha', 'missing_plan', 'missing_model', 'model_not_directory',
    'model_hash', 'manifest_inside_model', 'missing_dolly', 'existing_output', 'output_inside_model',
    'cpu_device', 'cuda_unavailable'])
def test_office_rejects_unbound_inputs_before_scientific_work_or_output(local_office, name, fault):
    case = local_office
    options, extra = dict(case.values), []
    if fault == 'plan_sha':
        options['--plan-sha256'] = '0' * 64
    elif fault == 'missing_plan':
        case.plan.unlink()
    elif fault == 'missing_model':
        options['--model-root'] = str(case.root / 'absent-weights')
    elif fault == 'model_not_directory':
        options['--model-root'] = str(case.dolly)
    elif fault == 'model_hash':
        (case.weights / 'config.json').write_bytes(b'changed model member')
    elif fault == 'manifest_inside_model':
        nested = case.weights / 'sources.json'
        nested.write_bytes(case.manifest.read_bytes())
        options['--model-sources'] = str(nested)
    elif fault == 'missing_dolly':
        case.dolly.unlink()
    elif fault == 'existing_output':
        output = case.root / 'new-office-run'
        output.mkdir()
        (output / 'keep.bin').write_bytes(b'prior protocol evidence')
    elif fault == 'output_inside_model':
        options['--dest'] = str(case.weights / 'new-office-run')
    elif fault == 'cpu_device':
        extra = ['--device', 'cpu']
    else:
        case.torch.cuda.is_available = lambda: False
    before = _tree(case.root)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        case.command.main(_argv(case, name, values=options, extra=extra))
    assert not case.calls and _tree(case.root) == before
    if fault != 'existing_output':
        assert not (case.root / 'new-office-run').exists()
    assert not (case.weights / 'new-office-run').exists()


@pytest.mark.parametrize('name', ['qwen8b', 'tree', 'traj'])
def test_office_formal_source_drift_is_rejected_before_fit_or_output(local_office, monkeypatch, name):
    case = local_office
    dirty = lambda: {'git_head': 'a' * 40, 'dirty': True}
    monkeypatch.setattr(case.command.q, '_identity', dirty)
    monkeypatch.setattr(case.command.t, '_identity', dirty)
    monkeypatch.setattr(case.command.tr, '_run_identity', dirty)
    args = _argv(case, name)
    args[args.index('smoke')] = 'formal'
    before = _tree(case.root)
    with pytest.raises((ValueError, RuntimeError)):
        case.command.main(args)
    assert not case.calls and _tree(case.root) == before
    assert not (case.root / 'new-office-run').exists()


@pytest.mark.parametrize('protocol', ['first_fork', 'ar_determinism'])
def test_office_permanently_stopped_protocols_never_dispatch(local_office, protocol):
    case = local_office
    before = _tree(case.root)
    with pytest.raises((ValueError, RuntimeError, SystemExit)):
        case.command.main(_argv(case, 'traj', extra=['--protocol', protocol]))
    assert not case.calls and _tree(case.root) == before
    assert not (case.root / 'new-office-run').exists()


@pytest.mark.parametrize('changed', ['plan', 'manifest'])
def test_office_rejects_input_change_during_admission(local_office, monkeypatch, changed):
    case = local_office
    original = case.command.verify_model_files

    def mutate(weights, sources):
        original(weights, sources)
        target = getattr(case, changed)
        target.write_bytes(target.read_bytes() + b'\n')

    monkeypatch.setattr(case.command, 'verify_model_files', mutate)
    monkeypatch.setattr(case.command.q, '_offline_caches', lambda root: None)
    with pytest.raises(ValueError, match='changed'):
        case.command.main(_argv(case, 'qwen8b'))
    assert not case.calls and not (case.root / 'new-office-run').exists()


@pytest.mark.parametrize('name', ['qwen8b', 'tree'])
def test_office_loader_closes_session_when_tokenizer_loading_fails(tmp_path, monkeypatch, name):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.model import hf_causal
    from studies.trajectory_observation.structure import office_qwen8b as q, office_tree as t
    closed = []
    model = SimpleNamespace(hf=SimpleNamespace(to=lambda device: None),
        device='cuda:0', dtype=torch.bfloat16, close=lambda: closed.append(True))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (22000 * 1024 ** 2, 24000 * 1024 ** 2))
    monkeypatch.setattr(hf_causal, 'HFCausalLM', lambda *args, **kwargs: model)

    def fail(*args, **kwargs):
        raise RuntimeError('tokenizer fixture failure')

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', fail)
    with pytest.raises(RuntimeError, match='tokenizer fixture failure'):
        if name == 'qwen8b':
            q._load_model(tmp_path, 'cuda:0')
        else:
            t._load_model('8b', tmp_path, 'cuda:0')
    assert closed == [True]
