"""Explicit scientific CLI and independent read-only commands."""
from dataclasses import asdict
import json

import pytest

from spec_mtp.cli import main
from test_decode import request, run


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
    from types import SimpleNamespace
    from spec_mtp.cli import _decode
    import spec_mtp.model

    path = tmp_path / 'request.json'
    path.write_text(json.dumps(asdict(request())), encoding='utf-8')
    output = tmp_path / 'raw.jsonl' if output_state == 'existing' else tmp_path / 'missing' / 'raw.jsonl'
    if output_state == 'existing':
        output.write_bytes(b'original evidence\n')
    constructed = []
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', lambda *args, **kwargs: constructed.append(args))
    with pytest.raises(OSError):
        _decode(SimpleNamespace(request=path, model=tmp_path, output=output))
    assert constructed == []
    if output_state == 'existing':
        assert output.read_bytes() == b'original evidence\n'
    else:
        assert not output.exists()


@pytest.mark.parametrize('close_fails', [False, True])
def test_cli_failed_model_construction_closes_reserved_incomplete_output(tmp_path, monkeypatch, close_fails):
    from types import SimpleNamespace
    from spec_mtp.cli import _decode
    from spec_mtp.audit import JsonlSink, audit_run
    import spec_mtp.audit
    import spec_mtp.model

    path, output = tmp_path / 'request.json', tmp_path / 'raw.jsonl'
    path.write_text(json.dumps(asdict(request())), encoding='utf-8')
    sinks = []
    close_error = OSError('sink close failed')

    class ObservedSink(JsonlSink):
        def __init__(self, path):
            super().__init__(path)
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
        _decode(SimpleNamespace(request=path, model=tmp_path, output=output))
    assert raised.value is failure
    assert len(sinks) == 1 and sinks[0].handle.closed
    assert output.read_bytes() == b''
    assert not audit_run(output, request().sources).argmax_coverage
    assert getattr(failure, 'cleanup_failures', []) == ([close_error] if close_fails else [])


def test_cli_success_closes_its_model_and_keeps_auditable_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from spec_mtp.cli import _decode
    from spec_mtp.audit import audit_run
    from test_decode import FakeSession
    import spec_mtp.model

    class Session(FakeSession):
        def close(self):
            self.closed = True

    session = Session()
    path, output = tmp_path / 'request.json', tmp_path / 'raw.jsonl'
    path.write_text(json.dumps(asdict(request())), encoding='utf-8')
    monkeypatch.setattr(spec_mtp.model, 'HFCausalLM', lambda *args, **kwargs: session)
    result = _decode(SimpleNamespace(request=path, model=tmp_path, output=output))
    assert result['execution'] == 'completed' and result['evidence_complete']
    assert session.closed and session.calls > 0
    audit = audit_run(output, request().sources)
    assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness
