"""Synthetic and local-weight entries reject drift before scientific execution."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from studies.trajectory_observation.flow.execute import pilot, preforward_signal


@pytest.fixture
def local_args(tmp_path, monkeypatch):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.model import hf_causal
    from studies.trajectory_observation.structure import pilot as synthetic, preforward_signal as signals, fair_branch

    plan = tmp_path / 'plan.md'
    plan.write_bytes(b'# Explicit engineering fixture\n')
    config = json.loads(json.dumps(synthetic.DEFAULT))
    config.update(content_vocab=8, max_new_tokens=4, trees=[{'num_masks': 1, 'branches': [3]}], seeds=[7])
    config['train'].update(sequences=16, length=16, steps=1, batch=2)
    config['model'].update(hidden_size=16, intermediate_size=32, max_position_embeddings=64)
    config['prompts'] = {'fit': 2, 'holdout': 2, 'cross': 2, 'length': 8}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').write_bytes(b'{"fixture":true}')
    manifest = tmp_path / 'model-sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256((weights / 'config.json').read_bytes()).hexdigest()}))
    data = tmp_path / 'specbench.jsonl'
    data.write_text(''.join(json.dumps({'question_id': qid, 'category': 'fixture', 'turns': [f'local {qid}']}) + '\n'
                            for qid in range(81, 209)), encoding='utf-8')
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'IO stub; no CUDA work')
    clean = lambda: {'git_head': 'fixture', 'dirty': False}
    monkeypatch.setattr(synthetic, 'source_identity', clean)
    monkeypatch.setattr(fair_branch, 'identity', clean)
    def seed_run(config, seed, device):
        calls.append(('seed', seed, device))
        return {'seed': seed, 'trees': [], 'argmax_all_passed': True}
    monkeypatch.setattr(synthetic, 'run_seed', seed_run)
    monkeypatch.setattr(synthetic, 'run_oracle', seed_run)

    class Model:
        dtype = torch.bfloat16
        def __init__(self, path, **options):
            self.hf = self
            calls.append(('model', path, options))
        def to(self, device):
            self.device = device
        def forward(self, *values, **options):
            assert options['last_hidden_only'] is True
        def close(self):
            calls.append(('closed',))
    monkeypatch.setattr(hf_causal, 'HFCausalLM', Model)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: object())
    def collect(model, prompts, tokens, path):
        model.forward(None)
        calls.append(('collect', tokens, [item['qid'] for item in prompts]))
        return {'qid': [item['qid'] for item in prompts]}
    monkeypatch.setattr(signals, 'run_prompts', collect)
    monkeypatch.setattr(signals, 'fit_and_summarize', lambda *args, device: calls.append(('fit', str(device))) or {'fixture': True})
    args = SimpleNamespace(config=config_path, plan=plan, plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        output=tmp_path / 'new-run', device='cuda:0', model='4b', model_root=weights, model_sources=manifest, input=data)
    return SimpleNamespace(args=args, calls=calls, config=config, synthetic=synthetic, signals=signals, fair=fair_branch)


@pytest.mark.parametrize('entry', ['pilot', 'preforward'])
def test_entry_binds_local_protocol_inputs_and_records_device_without_old_matrix(local_args, entry):
    case = local_args
    (pilot if entry == 'pilot' else preforward_signal).run(case.args)
    header = json.loads((case.args.output / 'header.json').read_text())
    assert header['device'] == 'cuda:0' and header['plan']['sha256'] == case.args.plan_sha256
    assert (case.args.output / 'protocol_complete.json').is_file()
    if entry == 'pilot':
        assert header['protocol'] == case.synthetic.PROTOCOL
        assert case.calls == [('seed', 7, 'cuda:0')]
    else:
        assert header['protocol'] == 'preforward-signal-r2-v1'
        assert header['holdout_qids'] == list(range(81, 209))
        assert ('fit', 'cuda:0') in case.calls and case.calls[-1] == ('closed',)
        assert all(call[2]['local_files_only'] for call in case.calls if call[0] == 'model')


@pytest.mark.parametrize('entry', ['pilot', 'preforward'])
@pytest.mark.parametrize('fault', ['plan_sha', 'plan_missing', 'existing', 'checkout', 'cpu', 'dirty'])
def test_entry_rejects_unbound_inputs_without_output_or_scientific_calls(local_args, monkeypatch, entry, fault):
    case, args = local_args, local_args.args
    if fault == 'plan_sha':
        args.plan_sha256 = '0' * 64
    elif fault == 'plan_missing':
        args.plan.unlink()
    elif fault == 'existing':
        args.output.mkdir()
        (args.output / 'keep.bin').write_bytes(b'old protocol evidence')
    elif fault == 'checkout':
        checkout = args.output.parent / 'checkout'
        (checkout / '.git').mkdir(parents=True)
        args.output = checkout / 'new-run'
    elif fault == 'cpu':
        args.device = 'cpu'
    else:
        dirty = lambda: {'git_head': 'fixture', 'dirty': True}
        monkeypatch.setattr(case.synthetic, 'source_identity', dirty)
        monkeypatch.setattr(case.fair, 'identity', dirty)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        (pilot if entry == 'pilot' else preforward_signal).run(args)
    assert not case.calls
    if fault == 'existing':
        assert (args.output / 'keep.bin').read_bytes() == b'old protocol evidence'
    else:
        assert not args.output.exists()


@pytest.mark.parametrize('fault', ['missing_model_field', 'negative_steps', 'nonfinite', 'unknown_mode',
                                  'conflicting_modes', 'mode_and_variants', 'empty_variants', 'bad_tree', 'non_string_variants'])
def test_pilot_configuration_rejects_ambiguous_or_invalid_protocol_before_work(local_args, fault):
    case, config = local_args, local_args.config
    if fault == 'missing_model_field':
        config['model'].pop('hidden_size')
    elif fault == 'negative_steps':
        config['train']['steps'] = -1
    elif fault == 'nonfinite':
        config['train']['lr'] = float('nan')
    elif fault == 'unknown_mode':
        config['unknown_mode'] = True
    elif fault == 'conflicting_modes':
        config.update(kalman=True, affine=True, variants=[])
    elif fault == 'mode_and_variants':
        config['kalman'] = True
    elif fault == 'empty_variants':
        config['variants'] = []
    elif fault == 'non_string_variants':
        config['variants'] = [[[], 'none']]
    else:
        config['trees'] = [{'num_masks': 2, 'branches': [3]}]
    case.args.config.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError):
        pilot.run(case.args)
    assert not case.calls and not case.args.output.exists()


@pytest.mark.parametrize('entry', ['pilot', 'preforward'])
def test_entry_source_drift_does_not_publish_completion(local_args, monkeypatch, entry):
    case = local_args
    def changed(*args, **kwargs):
        case.args.plan.write_bytes(b'changed plan')
        return {'qid': list(range(128)), 'trees': []}
    monkeypatch.setattr(case.synthetic, 'run_seed', changed)
    monkeypatch.setattr(case.signals, 'fit_and_summarize', changed)
    with pytest.raises(ValueError, match='changed'):
        (pilot if entry == 'pilot' else preforward_signal).run(case.args)
    assert (case.args.output / 'header.json').is_file()
    assert not (case.args.output / 'protocol_complete.json').exists()
    if entry == 'preforward':
        assert case.calls[-1] == ('closed',)


@pytest.mark.parametrize('marker', [None, False])
def test_pilot_missing_or_failed_correctness_audit_cannot_publish_completion(local_args, monkeypatch, marker):
    result = {} if marker is None else {'argmax_all_passed': marker}
    monkeypatch.setattr(local_args.synthetic, 'run_seed', lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match='argmax audit'):
        pilot.run(local_args.args)
    assert not (local_args.args.output / 'protocol_complete.json').exists()


@pytest.mark.parametrize('entry,missing', [('pilot', '--config'), ('pilot', '--plan-sha256'),
    ('pilot', '--device'), ('preforward', '--input'), ('preforward', '--model-sources'), ('preforward', '--plan')])
def test_entry_requires_explicit_bindings_before_dispatch(local_args, monkeypatch, entry, missing):
    command = pilot if entry == 'pilot' else preforward_signal
    values = {'--config': str(local_args.args.config)} if entry == 'pilot' else {
        '--model': '4b', '--model-root': str(local_args.args.model_root),
        '--model-sources': str(local_args.args.model_sources), '--input': str(local_args.args.input)}
    values.update({'--plan': str(local_args.args.plan), '--plan-sha256': local_args.args.plan_sha256,
                   '--output': str(local_args.args.output), '--device': 'cuda:0'})
    monkeypatch.setattr(command, 'run', lambda args: pytest.fail('invalid arguments dispatched'))
    with pytest.raises(SystemExit) as stopped:
        command.main([item for key, value in values.items() if key != missing for item in (key, value)])
    assert stopped.value.code == 2 and not local_args.args.output.exists()


@pytest.mark.parametrize('unavailable', [False, True])
def test_pilot_source_identity_limits_git_trust_and_preserves_unknown_state(monkeypatch, unavailable):
    from pathlib import Path
    from studies.trajectory_observation.structure import pilot as protocol

    root = Path(protocol.__file__).resolve().parents[3]
    calls = []
    def git(command, **options):
        calls.append(command)
        if unavailable:
            raise OSError('Git unavailable in fixture')
        return 'a' * 40 if command[-2:] == ['rev-parse', 'HEAD'] else ''
    monkeypatch.setattr(protocol.subprocess, 'check_output', git)
    assert protocol.source_identity() == ({'git_head': None, 'dirty': None} if unavailable
                                          else {'git_head': 'a' * 40, 'dirty': False})
    assert all(command[:5] == ['git', '-c', 'safe.directory=' + str(root), '-C', str(root)] for command in calls)
