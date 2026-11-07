"""Fresh command admission uses local IO and model stubs, without tensor execution."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from studies.trajectory_observation.flow.execute import fresh_replace as command


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    import torch
    from studies.trajectory_observation.structure import fair_branch
    from spec_mtp.structure.model import hf_causal
    from transformers import AutoTokenizer

    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').write_bytes(b'{"fixture":true}')
    manifest = tmp_path / 'model-sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256((weights / 'config.json').read_bytes()).hexdigest()}))
    dolly = tmp_path / 'dolly.jsonl'
    dolly.write_text('{"category":"creative_writing","instruction":"fixture"}\n')
    plan = tmp_path / 'research-plan.md'
    plan.write_bytes(b'# Frozen fixture research plan\n')
    calls = []

    class Model:
        def __init__(self, path, **options):
            calls.append(('model', path, options))
            self.hf = self

        def to(self, device):
            self.device = device

        def close(self):
            calls.append(('closed',))

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'IO stub; no CUDA execution')
    monkeypatch.setattr(hf_causal, 'HFCausalLM', Model)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda path, **options: calls.append(('tokenizer', path, options)))
    monkeypatch.setattr(command.protocol, 'fresh_prompts', lambda rows: [{'qid': 128, 'text': 'fresh fixture'}])
    monkeypatch.setattr(command.protocol, 'assert_known_slice', lambda prompts: 'validated-fixture-slice')
    monkeypatch.setattr(command.protocol, 'TAKE', 1)
    monkeypatch.setattr(fair_branch, 'identity', lambda: {'git_head': 'fixture', 'dirty': False})
    monkeypatch.setattr(fair_branch, 'calibrate', lambda *args: {'k_mean': 1})
    monkeypatch.setattr(fair_branch, 'run_split', lambda *args: calls.append(('split', args[-1])))
    monkeypatch.setattr(fair_branch, 'read_rows', lambda path: [{'qid': 128}])
    monkeypatch.setattr(command.protocol, 'replication_summary', lambda *args: {'complete': True})
    args = SimpleNamespace(model='4b', model_root=weights, model_sources=manifest, device='cuda:0',
                           dolly=dolly, output=tmp_path / 'new-output', plan=plan,
                           plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    return SimpleNamespace(args=args, calls=calls, torch=torch)


def test_fresh_command_binds_local_inputs_and_writes_only_new_external_output(local_inputs):
    case = local_inputs
    summary = command.run(case.args)
    header = json.loads((case.args.output / 'header.json').read_text())
    assert header['device'] == 'cuda:0' and header['model_root'] == str(case.args.model_root)
    assert header['model_sources']['config.json']
    assert header['plan'] == {'path': str(case.args.plan), 'sha256': case.args.plan_sha256}
    assert header['package_sources'] and header['prompt_sha256'] == 'validated-fixture-slice'
    assert summary['complete'] is True
    assert case.calls[-1] == ('closed',)
    assert (case.args.output / '4b/replication-summary.json').is_file()
    assert all(call[2]['local_files_only'] for call in case.calls if call[0] in ('model', 'tokenizer'))


def test_fresh_entry_reads_the_resolved_input_bound_before_argument_alias_changes(local_inputs, monkeypatch):
    case = local_inputs
    original = command.prepare_local_inputs
    admitted = case.args.dolly
    replacement = admitted.with_name('rebound-alias.jsonl')
    replacement.write_bytes(b'42\n')

    def admit(args, data):
        result = original(args, data)
        args.dolly = replacement
        return result

    monkeypatch.setattr(command, 'prepare_local_inputs', admit)
    assert command.run(case.args)['complete'] is True
    header = json.loads((case.args.output / 'header.json').read_text())
    assert str(admitted) in header['input_sources'] and str(replacement) not in header['input_sources']


@pytest.mark.parametrize('fault', ['existing_output', 'inside_model', 'source_checkout', 'model_hash', 'manifest_inside_model',
                                  'cpu_device', 'missing_cuda', 'missing_device_index', 'bad_device_index',
                                  'planwrong', 'planmissing', 'plan_inside_model'])
def test_fresh_command_rejects_unadmitted_inputs_before_model_or_output(local_inputs, fault):
    case, args = local_inputs, local_inputs.args
    if fault == 'existing_output':
        args.output.mkdir()
        (args.output / 'keep.txt').write_bytes(b'prior output')
    elif fault == 'inside_model':
        args.output = args.model_root / 'output'
    elif fault == 'source_checkout':
        checkout = args.output.parent / 'foreign-checkout'
        (checkout / '.git').mkdir(parents=True)
        args.output = checkout / 'output'
    elif fault == 'model_hash':
        (args.model_root / 'config.json').write_bytes(b'changed model member')
    elif fault == 'manifest_inside_model':
        inside = args.model_root / 'sources.json'
        inside.write_bytes(args.model_sources.read_bytes())
        args.model_sources = inside
    elif fault == 'missing_cuda':
        case.torch.cuda.is_available = lambda: False
    elif fault == 'planwrong':
        args.plan_sha256 = '0' * 64
    elif fault == 'planmissing':
        args.plan.unlink()
    elif fault == 'plan_inside_model':
        inside = args.model_root / 'research-plan.md'
        inside.write_bytes(args.plan.read_bytes())
        args.plan = inside
    else:
        args.device = {'cpu_device': 'cpu', 'missing_device_index': 'cuda', 'bad_device_index': 'cuda:2'}[fault]
    with pytest.raises((ValueError, RuntimeError, FileExistsError, FileNotFoundError)):
        command.run(args)
    assert not case.calls
    if fault == 'existing_output':
        assert (args.output / 'keep.txt').read_bytes() == b'prior output'
    else:
        assert not args.output.exists()


@pytest.mark.parametrize('changed', ['dolly', 'plan'])
def test_fresh_command_source_drift_keeps_partial_output_without_complete_summary(local_inputs, monkeypatch, changed):
    from studies.trajectory_observation.structure import fair_branch

    case = local_inputs
    monkeypatch.setattr(fair_branch, 'run_split', lambda *args: getattr(case.args, changed).write_bytes(b'changed input'))
    with pytest.raises(ValueError, match='changed'):
        command.run(case.args)
    assert (case.args.output / 'header.json').is_file()
    assert not (case.args.output / '4b/replication-summary.json').exists()


@pytest.mark.parametrize('missing', ['--model-root', '--model-sources', '--device', '--dolly', '--output',
                                    '--plan', '--plan-sha256'])
def test_fresh_command_requires_explicit_local_input_device_and_output(missing, tmp_path, monkeypatch):
    values = {'--model': '4b', '--model-root': str(tmp_path), '--model-sources': str(tmp_path / 'sources.json'),
              '--device': 'cuda:0', '--dolly': str(tmp_path / 'dolly.jsonl'), '--output': str(tmp_path / 'output'),
              '--plan': str(tmp_path / 'research-plan.md'), '--plan-sha256': '0' * 64}
    monkeypatch.setattr(command, 'run', lambda args: pytest.fail('invalid CLI must not dispatch'))
    with pytest.raises(SystemExit) as caught:
        command.main([part for key, value in values.items() if key != missing for part in (key, value)])
    assert caught.value.code == 2 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('scenario', ['close_only', 'body_error', 'interrupt', 'close_mutates_input', 'ambient_close_only'])
def test_fresh_cleanup_preserves_primary_and_blocks_completion(local_inputs, monkeypatch, scenario):
    case = local_inputs
    close_error = OSError('injected model close failure')
    primary = (ValueError('body failed') if scenario == 'body_error' else
               KeyboardInterrupt('body interrupted') if scenario == 'interrupt' else None)
    body_seen = []
    from spec_mtp.structure.model import hf_causal
    from studies.trajectory_observation.structure import fair_branch
    original = fair_branch.run_split
    def close(model):
        case.calls.append(('closed',))
        if scenario == 'close_mutates_input':
            case.args.plan.write_bytes(b'changed during close')
            return
        raise close_error
    def body(*values):
        body_seen.append(True)
        if primary is not None:
            raise primary
        return original(*values)
    monkeypatch.setattr(hf_causal.HFCausalLM, 'close', close)
    monkeypatch.setattr(fair_branch, 'run_split', body)
    expected = primary if primary is not None else close_error
    ambient = LookupError('caller exception must not become this function primary')
    with pytest.raises(ValueError if scenario == 'close_mutates_input' else type(expected)) as caught:
        if scenario == 'ambient_close_only':
            try:
                raise ambient
            except LookupError:
                command.run(case.args)
        else:
            command.run(case.args)
    assert not hasattr(ambient, 'cleanup_errors')
    if scenario == 'close_mutates_input':
        assert 'changed' in str(caught.value)
    else:
        assert caught.value is expected
        assert getattr(expected, 'cleanup_errors', ()) == (() if primary is None else (close_error,))
    assert body_seen == [True]
    output = case.args.output
    assert (output / 'header.json').is_file()
    assert not (output / 'protocol_complete.json').exists()
    assert case.calls.count(('closed',)) == 1
    summary = output / '4b/replication-summary.json'
    assert summary.exists() is (primary is None)
    if primary is None:
        assert json.loads(summary.read_text())['complete'] is True
