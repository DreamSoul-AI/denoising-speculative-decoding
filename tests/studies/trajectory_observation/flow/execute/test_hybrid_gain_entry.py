"""Hybrid protocol entry boundaries, with model and scientific work stubbed."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from studies.trajectory_observation.flow.execute import hybrid_gain as command


@pytest.fixture
def local_case(tmp_path, monkeypatch):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.model import hf_causal
    from studies.trajectory_observation.structure import fair_branch

    plan = tmp_path / 'plan.md'
    plan.write_bytes(b'# Fixed local engineering fixture\n')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').write_bytes(b'{}')
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256(b'{}').hexdigest()}))
    data = tmp_path / 'input.jsonl'
    data.write_text(''.join(json.dumps({'question_id': qid, 'category': 'fixture', 'turns': [f'prompt {qid}']}) + '\n'
                            for qid in range(209, 561)), encoding='utf-8')
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'IO stub; no GPU work')
    monkeypatch.setattr(fair_branch, 'identity', lambda: {'git_head': 'fixture', 'dirty': False})
    class Model:
        tied_embeddings = True
        config = SimpleNamespace(tie_word_embeddings=True)
        dtype = 'bfloat16'
        def __init__(self, path, **options):
            self.hf = self
            calls.append(('model', path, options))
        def to(self, device):
            self.device = device
        def close(self):
            calls.append(('closed',))
    monkeypatch.setattr(hf_causal, 'HFCausalLM', Model)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: object())
    def write_rows(path, prompts):
        path.write_text(''.join(json.dumps({'qid': item['qid'], 'argmax_passed': True}) + '\n' for item in prompts))
    for module in (command.bonus, command.hidden):
        def calibrate(stored, device, module=module):
            prior = {key: [1.0] for key in ('k', 'p', 'r', 'q')}
            return {'first': prior, 'last': prior} if module is command.bonus else prior
        monkeypatch.setattr(module, 'calibrate', calibrate)
        monkeypatch.setattr(module, 'summarize', lambda *args: {'complete': True})
    def run_prompts(model, tokenizer, prompts, spec, tokens, path, *, pair_keys=()):
        calls.append(('calibrate', str(model.device)) if pair_keys else ('arm', path.name, tokens, [item['qid'] for item in prompts]))
        write_rows(path, prompts)
        return {'qid': [item['qid'] for item in prompts]}
    monkeypatch.setattr(command, 'run_prompts', run_prompts)
    args = SimpleNamespace(protocol='bonus-aligned', model='4b', model_root=weights, model_sources=manifest,
        plan=plan, plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(), input=data,
        output=tmp_path / 'new-run', device='cuda:0')
    return SimpleNamespace(args=args, calls=calls, fair=fair_branch, write_rows=write_rows)


@pytest.mark.parametrize('protocol', ['bonus-aligned', 'output-hidden'])
def test_hybrid_entry_preserves_protocol_slices_and_closes_before_completion(local_case, protocol):
    case, args = local_case, local_case.args
    args.protocol = protocol
    assert command.run(args) == {'complete': True}
    header = json.loads((args.output / 'header.json').read_text())
    expected = list(range(465, 561)) if protocol == 'bonus-aligned' else list(range(209, 337))
    assert header['split_qids']['specbench'] == expected
    assert header['calibration_qids'] == [f'template-{index}' for index in range(128)]
    assert header['plan']['sha256'] == args.plan_sha256 and header['device'] == 'cuda:0'
    assert len([call for call in case.calls if call[0] == 'arm']) == (4 if protocol == 'bonus-aligned' else 12)
    assert case.calls[-1] == ('closed',) and (args.output / 'protocol_complete.json').is_file()
    assert all(call[2]['local_files_only'] for call in case.calls if call[0] == 'model')


@pytest.mark.parametrize('fault', ['plan_sha', 'existing', 'source', 'cpu', 'model_hash', 'slice'])
def test_hybrid_entry_rejects_invalid_binding_before_output_or_model(local_case, fault, monkeypatch):
    case, args = local_case, local_case.args
    if fault == 'plan_sha':
        args.plan_sha256 = '0' * 64
    elif fault == 'existing':
        args.output.mkdir()
        (args.output / 'keep').write_bytes(b'old')
    elif fault == 'source':
        monkeypatch.setattr(case.fair, 'identity', lambda: {'git_head': None, 'dirty': None})
    elif fault == 'cpu':
        args.device = 'cpu'
    elif fault == 'model_hash':
        (args.model_root / 'config.json').write_bytes(b'changed')
    else:
        args.input.write_text('{}\n')
    with pytest.raises((ValueError, RuntimeError, FileExistsError, KeyError)):
        command.run(args)
    assert not case.calls
    assert not args.output.exists() if fault != 'existing' else (args.output / 'keep').read_bytes() == b'old'


@pytest.mark.parametrize('fault', ['drift', 'duplicate', 'audit', 'close'])
def test_hybrid_entry_partial_failure_never_publishes_completion(local_case, monkeypatch, fault):
    case, args = local_case, local_case.args
    original = command.run_prompts
    def run_prompts(*values, **options):
        result = original(*values, **options)
        path = values[-1]
        if fault == 'drift':
            args.plan.write_bytes(b'changed after admission')
        elif fault == 'duplicate':
            rows = path.read_text().splitlines()
            path.write_text('\n'.join(rows[:-1] + rows[:1]) + '\n')
        elif fault == 'audit':
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['argmax_passed'] = False
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return result
    monkeypatch.setattr(command, 'run_prompts', run_prompts)
    if fault == 'close':
        from spec_mtp.structure.model import hf_causal
        def close(self):
            case.calls.append(('closed',))
            raise RuntimeError('close failed')
        monkeypatch.setattr(hf_causal.HFCausalLM, 'close', close)
    with pytest.raises((ValueError, RuntimeError)):
        command.run(args)
    assert case.calls[-1] == ('closed',)
    assert not (args.output / 'protocol_complete.json').exists()


def test_hybrid_entry_uses_bound_resolved_data_after_argument_changes(local_case, monkeypatch):
    args = local_case.args
    original = command.prepare_local_inputs
    def prepare(*values):
        binding = original(*values)
        args.input = args.input.parent / 'unbound-missing-input'
        return binding
    monkeypatch.setattr(command, 'prepare_local_inputs', prepare)
    command.run(args)
    assert (args.output / 'protocol_complete.json').is_file()


def test_hybrid_entry_preserves_primary_failure_when_logging_and_close_also_fail(local_case, monkeypatch):
    from spec_mtp.structure.model import hf_causal
    from spec_mtp.structure.system.logger import Logger

    primary, log_error, close_error = ValueError('primary calibration failure'), OSError('log write failed'), OSError('close failed')
    def calibrate(*values):
        raise primary
    def log(*values):
        raise log_error
    def close(*values):
        raise close_error
    monkeypatch.setattr(command.bonus, 'calibrate', calibrate)
    monkeypatch.setattr(Logger, 'exception', log)
    monkeypatch.setattr(hf_causal.HFCausalLM, 'close', close)
    with pytest.raises(ValueError) as error:
        command.run(local_case.args)
    assert error.value is primary
    assert primary.cleanup_errors == (log_error, close_error)
    assert not (local_case.args.output / 'protocol_complete.json').exists()


@pytest.mark.parametrize('pair_keys', [('m', 'z', 'first', 'last'), ('h', 'z', 'y')])
def test_hybrid_collector_preserves_protocol_pair_fields_and_refuses_existing_outputs(tmp_path, monkeypatch, pair_keys):
    import torch
    from spec_mtp.structure.algorithm import mask
    from spec_mtp.structure.algorithm.decode import loop
    from spec_mtp.structure.data import prompts

    moved = []
    class Ids:
        def to(self, device):
            moved.append(device)
            return self
    class Provider:
        def calibration_rows(self):
            return {key: f'saved-{key}' for key in pair_keys}
    class Decoder:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, ids):
            return None, {'verification_argmax_audit': {'passed': True}, 'accepted': 1, 'committed': 2,
                          'decode_calls': 1, 'new_tokens': 2}
    monkeypatch.setattr(mask, 'build_mask_provider', lambda *args: Provider())
    monkeypatch.setattr(loop, 'SpecDecoder', Decoder)
    monkeypatch.setattr(prompts, 'render', lambda *args: Ids())
    path = tmp_path / 'calibration.jsonl'
    model = SimpleNamespace(embedding_table=object(), device='cuda:fixture; no GPU work')
    stored = command.run_prompts(model, object(), [{'qid': 'fixture', 'text': 'bound'}], {}, 2, path, pair_keys=pair_keys)
    assert stored == {'qid': ['fixture'], **{key: [f'saved-{key}'] for key in pair_keys}}
    assert torch.load(path.with_suffix('.pt'), weights_only=True) == stored
    assert command.read_rows(path)[0]['argmax_passed'] is True
    before = path.read_bytes(), path.with_suffix('.pt').read_bytes()
    with pytest.raises(FileExistsError):
        command.run_prompts(model, object(), [], {}, 2, path, pair_keys=pair_keys)
    assert before == (path.read_bytes(), path.with_suffix('.pt').read_bytes())
    assert moved == [model.device]


@pytest.mark.parametrize('missing', ['--model-root', '--model-sources', '--input', '--plan', '--plan-sha256', '--device', '--output'])
def test_hybrid_cli_requires_explicit_local_bindings(missing):
    values = ['bonus-aligned', '--model', '4b']
    for flag in ('--model-root', '--model-sources', '--input', '--plan', '--plan-sha256', '--device', '--output'):
        if flag != missing:
            values.extend((flag, 'fixture'))
    with pytest.raises(SystemExit) as error:
        command.main(values)
    assert error.value.code == 2
