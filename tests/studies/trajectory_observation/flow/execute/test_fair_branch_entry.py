"""The original replacement protocol uses explicit local inputs and a new run root."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from studies.trajectory_observation.flow.execute import fair_branch as command


def _rows(split):
    if split == 'specbench':
        return [{'question_id': index, 'category': 'fixture', 'turns': [f'local {index}']}
                for index in reversed(range(336, 466))]
    return [{'category': 'creative_writing', 'instruction': f'local {index}', 'context': 'context'}
            for index in range(128)]


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.model import hf_causal
    from studies.trajectory_observation.structure import fair_branch as protocol

    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').write_bytes(b'{"fixture":true}')
    manifest = tmp_path / 'model-sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256((weights / 'config.json').read_bytes()).hexdigest()}))
    data = tmp_path / 'local-input.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in _rows('dolly')), encoding='utf-8')
    plan = tmp_path / 'research-plan.md'
    plan.write_bytes(b'# Explicit local engineering fixture\n')
    calls = []

    class Model:
        dtype = torch.bfloat16

        def __init__(self, path, **options):
            self.hf = self
            calls.append(('model', path, options))

        def to(self, device):
            self.device = device

        def close(self):
            calls.append(('closed',))

    def split(model, tokenizer, prompts, prior, output, name):
        calls.append(('split', name, [item['qid'] for item in prompts]))
        return {'complete': True, 'protocol': protocol.PROTOCOL, 'argmax_all_passed': True, 'passed': False}

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'IO stub; no CUDA computation')
    monkeypatch.setattr(hf_causal, 'HFCausalLM', Model)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda path, **options: calls.append(('tokenizer', path, options)))
    monkeypatch.setattr(protocol, 'identity', lambda: {'git_head': 'fixture', 'dirty': False})
    monkeypatch.setattr(protocol, 'calibrate', lambda *args: {'k_mean': 1})
    monkeypatch.setattr(protocol, 'run_split', split)
    args = SimpleNamespace(model='4b', split='dolly', model_root=weights, model_sources=manifest,
        device='cuda:0', input=data, plan=plan, plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        output=tmp_path / 'new-output')
    return SimpleNamespace(args=args, calls=calls, torch=torch, protocol=protocol, split=split)


@pytest.mark.parametrize('split', ['specbench', 'dolly'])
def test_fair_entry_preserves_protocol_and_local_slice_and_closes_model(local_inputs, split):
    case, args = local_inputs, local_inputs.args
    args.split = split
    args.input.write_text(''.join(json.dumps(row) + '\n' for row in _rows(split)), encoding='utf-8')
    before = {path: path.read_bytes() for path in (args.input, args.plan, args.model_sources)}
    summary = command.run(args)
    header = json.loads((args.output / 'header.json').read_text(encoding='utf-8'))
    assert header['protocol'] == 'extra-branch-replace-v1' and header['split'] == split
    expected = list(range(337, 465)) if split == 'specbench' else list(range(128))
    assert header['qids'] == expected and ('split', split, expected) in case.calls
    assert header['plan'] == {'path': str(args.plan), 'sha256': args.plan_sha256}
    assert header['device'] == 'cuda:0' and header['model_sources'] and header['package_sources']
    assert header['input_sources'][str(args.input)] == hashlib.sha256(before[args.input]).hexdigest()
    assert summary['complete'] is True and summary['passed'] is False
    assert case.calls[-1] == ('closed',)
    assert all(call[2]['local_files_only'] for call in case.calls if call[0] in ('model', 'tokenizer'))
    complete = json.loads((args.output / 'protocol_complete.json').read_text())
    assert complete['protocol'] == header['protocol'] and complete['inputs_verified'] is True
    assert before == {path: path.read_bytes() for path in before}


def test_fair_entry_reads_the_resolved_input_bound_before_argument_alias_changes(local_inputs, monkeypatch):
    case = local_inputs
    original = command.prepare_local_inputs
    admitted = case.args.input
    replacement = admitted.with_name('rebound-alias.jsonl')
    replacement.write_bytes(b'42\n')

    def admit(args, data):
        result = original(args, data)
        args.input = replacement
        return result

    monkeypatch.setattr(command, 'prepare_local_inputs', admit)
    assert command.run(case.args)['complete'] is True
    header = json.loads((case.args.output / 'header.json').read_text())
    assert str(admitted) in header['input_sources'] and str(replacement) not in header['input_sources']
    assert ('split', 'dolly', list(range(128))) in case.calls


@pytest.mark.parametrize('fault', ['existing_output', 'inside_model', 'source_checkout', 'model_hash',
    'plan_hash', 'missing_plan', 'manifest_inside_model', 'cpu', 'missing_cuda', 'no_device_index', 'dirty_source',
    'short_slice', 'non_object_rows'])
def test_fair_entry_rejects_unbound_inputs_before_model_or_output(local_inputs, monkeypatch, fault):
    case, args = local_inputs, local_inputs.args
    if fault == 'existing_output':
        args.output.mkdir()
        (args.output / 'keep.txt').write_bytes(b'old Run must stay unchanged')
    elif fault == 'inside_model':
        args.output = args.model_root / 'output'
    elif fault == 'source_checkout':
        checkout = args.output.parent / 'other-checkout'
        (checkout / '.git').mkdir(parents=True)
        args.output = checkout / 'output'
    elif fault == 'model_hash':
        (args.model_root / 'config.json').write_bytes(b'changed')
    elif fault == 'plan_hash':
        args.plan_sha256 = '0' * 64
    elif fault == 'missing_plan':
        args.plan.unlink()
    elif fault == 'manifest_inside_model':
        path = args.model_root / 'sources.json'
        path.write_bytes(args.model_sources.read_bytes())
        args.model_sources = path
    elif fault == 'missing_cuda':
        monkeypatch.setattr(case.torch.cuda, 'is_available', lambda: False)
    elif fault == 'dirty_source':
        monkeypatch.setattr(case.protocol, 'identity', lambda: {'git_head': 'fixture', 'dirty': True})
    elif fault == 'short_slice':
        args.input.write_text(''.join(json.dumps(row) + '\n' for row in _rows('dolly')[:-1]), encoding='utf-8')
    elif fault == 'non_object_rows':
        args.input.write_bytes(b'42\n')
    else:
        args.device = 'cpu' if fault == 'cpu' else 'cuda'
    with pytest.raises((ValueError, RuntimeError, OSError)):
        command.run(args)
    assert not case.calls
    if fault == 'existing_output':
        assert (args.output / 'keep.txt').read_bytes() == b'old Run must stay unchanged'
    else:
        assert not args.output.exists()


@pytest.mark.parametrize('changed', ['input', 'plan', 'model', 'source'])
def test_fair_entry_drift_preserves_partial_run_without_completion(local_inputs, monkeypatch, changed):
    case, args = local_inputs, local_inputs.args

    def drift(*values):
        summary = case.split(*values)
        if changed == 'model':
            (args.model_root / 'config.json').write_bytes(b'changed model')
        elif changed == 'source':
            monkeypatch.setattr(case.protocol, 'identity', lambda: {'git_head': 'other', 'dirty': False})
        else:
            getattr(args, changed).write_bytes(b'changed input')
        return summary

    monkeypatch.setattr(case.protocol, 'run_split', drift)
    with pytest.raises(ValueError, match='changed'):
        command.run(args)
    assert (args.output / 'header.json').is_file()
    assert not (args.output / 'protocol_complete.json').exists()
    assert case.calls[-1] == ('closed',)


@pytest.mark.parametrize('stage', ['fit', 'tokenizer', 'close'])
def test_fair_entry_closes_the_owned_model_after_failure(local_inputs, monkeypatch, stage):
    from transformers import AutoTokenizer

    def fail(*args, **kwargs):
        raise RuntimeError('injected construction or fitting failure')

    if stage == 'fit':
        monkeypatch.setattr(local_inputs.protocol, 'calibrate', fail)
    elif stage == 'tokenizer':
        monkeypatch.setattr(AutoTokenizer, 'from_pretrained', fail)
    else:
        from spec_mtp.structure.model import hf_causal

        def close(model):
            local_inputs.calls.append(('closed',))
            fail()

        monkeypatch.setattr(hf_causal.HFCausalLM, 'close', close)
    with pytest.raises(RuntimeError, match='injected'):
        command.run(local_inputs.args)
    assert local_inputs.calls[-1] == ('closed',)
    assert not (local_inputs.args.output / 'protocol_complete.json').exists()


@pytest.mark.parametrize('missing', ['--model', '--split', '--model-root', '--model-sources', '--input',
                                    '--plan', '--plan-sha256', '--device', '--output'])
def test_fair_entry_requires_explicit_protocol_inputs_before_dispatch(tmp_path, monkeypatch, missing):
    values = {'--model': '4b', '--split': 'dolly', '--model-root': str(tmp_path),
        '--model-sources': str(tmp_path / 'sources.json'), '--input': str(tmp_path / 'rows.jsonl'),
        '--plan': str(tmp_path / 'plan.md'), '--plan-sha256': '0' * 64,
        '--device': 'cuda:0', '--output': str(tmp_path / 'output')}
    monkeypatch.setattr(command, 'run', lambda args: pytest.fail('invalid arguments must not dispatch'))
    with pytest.raises(SystemExit) as stopped:
        command.main([item for key, value in values.items() if key != missing for item in (key, value)])
    assert stopped.value.code == 2 and not list(tmp_path.iterdir())
