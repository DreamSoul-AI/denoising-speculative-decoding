"""Explicit scientific CLI and independent read-only commands."""
from dataclasses import asdict
import json

import pytest

from spec_mtp.cli import main
from test_decode import request, run


def model_args(tmp_path):
    import hashlib
    from types import SimpleNamespace
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}', encoding='utf-8')
    manifest = tmp_path / 'model-sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256(b'{}').hexdigest()}), encoding='utf-8')
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(asdict(request())), encoding='utf-8')
    return SimpleNamespace(request=path, model=model, model_sources=manifest, output=tmp_path / 'raw.jsonl')


def test_cli_audit_consumes_existing_rows_only(tmp_path, capsys):
    req = request()
    _, _, sink = run(req)
    raw, sources = tmp_path / '观测.jsonl', tmp_path / '来源.json'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in sink.rows), encoding='utf-8')
    sources.write_text(json.dumps(dict(req.sources)), encoding='utf-8')
    assert main(['audit', str(raw), '--sources', str(sources)]) == 0
    assert json.loads(capsys.readouterr().out)['argmax_coverage'] is True


def test_legacy_study_commands_explain_the_moved_entry(capsys):
    with pytest.raises(SystemExit) as error:
        main(['make', 'study'])
    assert error.value.code == 2
    assert 'studies._shared.flow.cli' in capsys.readouterr().err


def test_cli_request_identity_is_validated_before_model_import(tmp_path):
    from spec_mtp.cli import _decode
    from types import SimpleNamespace
    body = asdict(request())
    body['prompt']['qid'] = True
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='qid'):
        _decode(SimpleNamespace(request=path, model='missing', output='unused'))


def test_cli_audit_rejects_corrupt_trace_without_model_calls(tmp_path, capsys):
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


@pytest.mark.parametrize('keys', [[], ['factor']])
@pytest.mark.parametrize('change', ['none', 'input_sha256', 'factor'])
def test_cli_compare_keeps_required_and_additional_pairing_checks(tmp_path, capsys, keys, change):
    from test_analysis import row
    left, right = row(factor='f'), row(factor='f')
    if change != 'none':
        right[change] = 'changed'
    paths = [tmp_path/name for name in ('left.json', 'right.json', 'pairing.json')]
    for path, body in zip(paths, ([left], [right], {'keys':keys})):
        path.write_text(json.dumps(body), encoding='utf-8')
    assert main(['compare', str(paths[0]), str(paths[1]), '--pairing', str(paths[2])]) == 0
    result = json.loads(capsys.readouterr().out)
    mismatch = change == 'input_sha256' or (change == 'factor' and bool(keys))
    assert result['incomparable'] == int(mismatch) and result['ties'] == int(not mismatch)


@pytest.mark.parametrize('output_state', ['existing', 'missing_parent'])
def test_cli_output_admission_precedes_model_construction(tmp_path, monkeypatch, output_state):
    from spec_mtp.cli import _decode
    import spec_mtp.model

    args = model_args(tmp_path)
    output = tmp_path / 'raw.jsonl' if output_state == 'existing' else tmp_path / 'missing' / 'raw.jsonl'
    if output_state == 'existing':
        output.write_bytes(b'original evidence\n')
    constructed = []
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', lambda *args, **kwargs: constructed.append(args))
    with pytest.raises(OSError):
        args.output = output
        _decode(args)
    assert constructed == []
    if output_state == 'existing':
        assert output.read_bytes() == b'original evidence\n'
    else:
        assert not output.exists()


@pytest.mark.parametrize('close_fails', [False, True])
def test_cli_failed_model_construction_closes_reserved_incomplete_output(tmp_path, monkeypatch, close_fails):
    from spec_mtp.cli import _decode
    from spec_mtp.audit import JsonlSink, audit_run
    import spec_mtp.audit
    import spec_mtp.model

    args = model_args(tmp_path)
    output = args.output
    sinks = []
    close_error = OSError('sink close failed')

    class ObservedSink(JsonlSink):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            sinks.append(self)

        def close(self):
            super().close()
            if close_fails:
                raise close_error

    failure = RuntimeError('model construction failed')

    def fail_model(*args, **kwargs):
        raise failure

    monkeypatch.setattr(spec_mtp.audit, 'JsonlSink', ObservedSink)
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', fail_model)
    with pytest.raises(RuntimeError) as raised:
        _decode(args)
    assert raised.value is failure
    assert len(sinks) == 1 and sinks[0].handle.closed
    assert output.read_bytes() == b''
    assert not audit_run(output, request().sources).argmax_coverage
    assert getattr(failure, 'cleanup_failures', []) == ([close_error] if close_fails else [])


def test_cli_success_closes_its_model_and_keeps_auditable_output(tmp_path, monkeypatch):
    from spec_mtp.cli import _decode
    from spec_mtp.audit import audit_run
    from test_decode import FakeSession
    import spec_mtp.model

    class Session(FakeSession):
        def describe(self):
            return {'device': 'cpu', 'dtype': 'torch.float32', 'model_class': 'FakeSession'}

        def close(self):
            self.closed = True

    session = Session()
    args = model_args(tmp_path)
    output = args.output
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', lambda *args, **kwargs: session)
    result = _decode(args)
    assert result['execution'] == 'completed' and result['evidence_complete']
    assert session.closed and session.calls > 0
    manifest = json.loads(args.model_sources.read_text())
    assert not audit_run(output, request().sources).source_integrity
    audit = audit_run(output, request().sources, expected_model_sources=manifest)
    assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness


@pytest.mark.parametrize('damage', ['empty', 'list', 'bad_sha', 'stale', 'missing_member', 'extra_member', 'escape', 'inside_root', 'output_inside_root'])
def test_cli_rejects_model_manifest_before_constructing_model(tmp_path, monkeypatch, damage):
    from spec_mtp.cli import _decode
    import spec_mtp.model

    args = model_args(tmp_path)
    declared = json.loads(args.model_sources.read_text())
    if damage == 'empty':
        declared = {}
    elif damage == 'list':
        declared = []
    elif damage == 'bad_sha':
        declared['config.json'] = 'invalid'
    elif damage == 'stale':
        declared['config.json'] = '0' * 64
    elif damage == 'missing_member':
        declared['missing.safetensors'] = '0' * 64
    elif damage == 'extra_member':
        (args.model / 'extra.safetensors').write_bytes(b'new weights')
    elif damage == 'escape':
        declared['../request.json'] = '0' * 64
    elif damage == 'inside_root':
        args.model_sources = args.model / 'sources.json'
    else:
        args.output = args.model / 'raw.jsonl'
    args.model_sources.write_text(json.dumps(declared), encoding='utf-8')
    calls = []
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', lambda *a, **k: calls.append(a))
    with pytest.raises((ValueError, OSError)):
        _decode(args)
    assert calls == [] and not args.output.exists()


@pytest.mark.parametrize('damage', ['modify', 'add', 'remove'])
def test_cli_model_drift_cannot_write_completed_end(tmp_path, monkeypatch, damage):
    from spec_mtp.cli import _decode
    from spec_mtp.decode import DecodeFailure
    from spec_mtp.audit import audit_run
    from test_decode import FakeSession
    import spec_mtp.model

    args = model_args(tmp_path)

    class Session(FakeSession):
        def describe(self):
            return {'device': 'cpu', 'dtype': 'torch.float32', 'model_class': 'FakeSession'}

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
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', lambda *a, **k: session)
    with pytest.raises(DecodeFailure) as raised:
        _decode(args)
    assert isinstance(raised.value.__cause__, ValueError)
    assert not raised.value.result.evidence_complete
    assert session.calls > 0 and session.closed
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert rows[0]['model']['device'] == 'cpu'
    assert not any(row['event'] == 'end' for row in rows)
    assert not audit_run(args.output, request().sources,
                         expected_model_sources=json.loads(args.model_sources.read_text())).source_integrity


def test_cli_real_local_model_sources_and_read_only_audit(tmp_path, capsys):
    import hashlib
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    args = model_args(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        native = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, tie_word_embeddings=True, attn_implementation='eager')).eval()
    assert sum(p.numel() for p in native.parameters()) == 728
    native.save_pretrained(args.model)
    declared = {p.relative_to(args.model).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.model.rglob('*') if p.is_file()}
    args.model_sources.write_text(json.dumps(declared), encoding='utf-8')
    assert main(['decode', '--request', str(args.request), '--model', str(args.model),
                 '--model-sources', str(args.model_sources), '--output', str(args.output)]) == 0
    assert json.loads(capsys.readouterr().out)['evidence_complete']
    header = json.loads(args.output.read_text().splitlines()[0])
    assert header['sources'] == request().sources and header['model_sources'] == declared
    assert header['model']['device'] == 'cpu' and header['model']['dtype'] == 'torch.float32'
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
