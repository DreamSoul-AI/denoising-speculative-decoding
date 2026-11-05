"""Explicit scientific CLI and independent read-only commands."""

from tests.conftest import cli_model_args as model_args


import json

import pytest

from spec_mtp.flow.cli import main

from tests.conftest import decode_request as request

def test_cli_audit_consumes_existing_rows_only(tmp_path, capsys, cuda_decode):
    run = cuda_decode.run
    req = request()
    _, _, sink = run(req)
    raw, sources = tmp_path / '观测.jsonl', tmp_path / '来源.json'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in sink.rows), encoding='utf-8')
    sources.write_text(json.dumps(dict(req.sources)), encoding='utf-8')
    assert main(['audit', str(raw), '--sources', str(sources)]) == 0
    assert json.loads(capsys.readouterr().out)['argmax_coverage'] is True

def test_cli_audit_rejects_corrupt_trace_without_model_calls(tmp_path, capsys, cuda_decode):
    run = cuda_decode.run
    req = request()
    _, model, sink = run(req)
    calls = model.calls
    sink.rows[-1]['stats']['trace'][0]['committed_tokens'][0] = 999
    raw, sources = tmp_path / 'bad.jsonl', tmp_path / 'sources.json'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in sink.rows), encoding='utf-8')
    sources.write_text(json.dumps(dict(req.sources)), encoding='utf-8')
    assert main(['audit', str(raw), '--sources', str(sources)]) == 1
    assert json.loads(capsys.readouterr().out)['argmax_coverage'] is False
    assert model.calls == calls

def test_cli_success_closes_its_model_and_keeps_auditable_output(tmp_path, monkeypatch, cuda_decode):
    from spec_mtp.flow.cli import _decode
    from spec_mtp.structure.artifact.observation import audit_run
    FakeSession = cuda_decode.FakeSession
    import spec_mtp.structure.model.hf_causal

    class Session(FakeSession):
        def describe(self):
            return {'device': str(self.device), 'dtype': 'torch.float32', 'model_class': 'FakeSession'}

        def close(self):
            self.closed = True

    session = Session()
    args = model_args(tmp_path)
    output = args.output
    monkeypatch.setattr(spec_mtp.structure.model.hf_causal, 'HFCausalLM', lambda *args, **kwargs: session)
    result = _decode(args)
    assert result['execution'] == 'completed' and result['evidence_complete']
    assert session.closed and session.calls > 0
    manifest = json.loads(args.model_sources.read_text())
    assert not audit_run(output, request().sources).source_integrity
    audit = audit_run(output, request().sources, expected_model_sources=manifest)
    assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness

@pytest.mark.parametrize('damage', ['modify', 'add', 'remove'])
def test_cli_model_drift_cannot_write_completed_end(tmp_path, monkeypatch, damage, cuda_decode):
    from spec_mtp.flow.cli import _decode
    from spec_mtp.structure.algorithm.decode.loop import DecodeFailure
    from spec_mtp.structure.artifact.observation import audit_run
    FakeSession = cuda_decode.FakeSession
    import spec_mtp.structure.model.hf_causal

    args = model_args(tmp_path)

    class Session(FakeSession):
        def describe(self):
            return {'device': str(self.device), 'dtype': 'torch.float32', 'model_class': 'FakeSession'}

        def __call__(self, *a, **kwargs):
            result = super().__call__(*a, **kwargs)
            if damage == 'modify':
                (args.model / 'config.json').write_text('{"changed":true}', encoding='utf-8')
            elif damage == 'add':
                (args.model / 'added.safetensors').write_bytes(b'added weights')
            else:
                (args.model / 'config.json').unlink(missing_ok=True)
            return result

        def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr(spec_mtp.structure.model.hf_causal, 'HFCausalLM', lambda *a, **k: session)
    with pytest.raises(DecodeFailure) as raised:
        _decode(args)
    assert isinstance(raised.value.__cause__, ValueError)
    assert not raised.value.result.evidence_complete
    assert session.calls > 0 and session.closed
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert rows[0]['model']['device'] == str(cuda_decode.device)
    assert not any(row['event'] == 'end' for row in rows)
    assert not audit_run(args.output, request().sources,
                         expected_model_sources=json.loads(args.model_sources.read_text())).source_integrity

def test_cli_real_local_model_sources_and_read_only_audit(tmp_path, capsys, monkeypatch, cuda_decode):
    import hashlib
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    args = model_args(tmp_path)
    with torch.random.fork_rng(devices=[cuda_decode.device.index]), torch.device(cuda_decode.device):
        torch.manual_seed(123)
        native = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, tie_word_embeddings=True, attn_implementation='eager')).eval()
    assert sum(p.numel() for p in native.parameters()) == 728
    assert next(native.parameters()).device == cuda_decode.device
    import spec_mtp.structure.model.hf_causal as hf_causal

    original_loader = hf_causal.HFCausalLM
    forward_devices = []

    def cuda_loader(*args, **kwargs):
        session = original_loader(*args, **kwargs)
        session.hf.to(cuda_decode.device)
        assert session.embedding_table.device == cuda_decode.device
        assert session.describe()['device'] == str(cuda_decode.device)

        def check_forward(module, args, kwargs, output):
            assert kwargs['inputs_embeds'].device == output.logits.device == cuda_decode.device
            forward_devices.append(str(output.logits.device))
            cuda_decode.observed_devices.append(str(output.logits.device))

        session.hf.register_forward_hook(check_forward, with_kwargs=True)
        return session

    monkeypatch.setattr(hf_causal, 'HFCausalLM', cuda_loader)
    native.save_pretrained(args.model)
    declared = {p.relative_to(args.model).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.model.rglob('*') if p.is_file()}
    args.model_sources.write_text(json.dumps(declared), encoding='utf-8')
    body = json.loads(args.request.read_text(encoding='utf-8'))
    body['spec']['bonus_slot'] = {'mode': 'hard'}
    args.request.write_text(json.dumps(body), encoding='utf-8')
    assert main(['decode', '--request', str(args.request), '--model', str(args.model),
                 '--model-sources', str(args.model_sources), '--output', str(args.output)]) == 0
    assert json.loads(capsys.readouterr().out)['evidence_complete']
    assert forward_devices and set(forward_devices) == {str(cuda_decode.device)}
    header = json.loads(args.output.read_text().splitlines()[0])
    assert header['sources'] == request().sources and header['model_sources'] == declared
    assert header['model']['device'] == str(cuda_decode.device) and header['model']['dtype'] == 'torch.float32'
    assert header['model']['config']['hidden_size'] == 8 and header['model']['tied_embeddings']
    assert header['model']['attention_backend'] == 'eager'
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps(dict(request().sources)), encoding='utf-8')
    command = ['audit', str(args.output), '--sources', str(labels)]
    assert main(command) == 1
    assert not json.loads(capsys.readouterr().out)['source_integrity']
    assert main(command + ['--model-sources', str(args.model_sources)]) == 0
    assert json.loads(capsys.readouterr().out)['source_integrity']
    declared['model.safetensors'] = '0' * 64
    args.model_sources.write_text(json.dumps(declared), encoding='utf-8')
    assert main(command + ['--model-sources', str(args.model_sources)]) == 1
    assert not json.loads(capsys.readouterr().out)['source_integrity']
