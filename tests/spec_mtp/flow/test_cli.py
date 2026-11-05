"""Explicit scientific CLI and independent read-only commands."""

from tests.conftest import cli_model_args as model_args

from dataclasses import asdict

import json

import pytest

from spec_mtp.flow.cli import main

from tests.conftest import decode_request as request

def test_legacy_study_commands_explain_the_moved_entry(capsys):
    with pytest.raises(SystemExit) as error:
        main(['make', 'study'])
    assert error.value.code == 2
    assert 'studies._shared.flow.cli' in capsys.readouterr().err

def test_cli_request_identity_is_validated_before_model_import(tmp_path):
    from spec_mtp.flow.cli import _decode
    from types import SimpleNamespace
    body = asdict(request())
    body['prompt']['qid'] = True
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='qid'):
        _decode(SimpleNamespace(request=path, model='missing', output='unused'))

@pytest.mark.parametrize('keys', [[], ['factor']])
@pytest.mark.parametrize('change', ['none', 'input_sha256', 'factor'])
def test_cli_compare_keeps_required_and_additional_pairing_checks(tmp_path, capsys, keys, change):
    left = dict(qid='q', accepted=2, committed=3, decode_calls=2, new_tokens=4,
                input_sha256='input', protocol='v1', evidence_complete=True, factor='f')
    right = dict(left)
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
    from spec_mtp.flow.cli import _decode
    import spec_mtp.structure.model.hf_causal

    args = model_args(tmp_path)
    output = tmp_path / 'raw.jsonl' if output_state == 'existing' else tmp_path / 'missing' / 'raw.jsonl'
    if output_state == 'existing':
        output.write_bytes(b'original evidence\n')
    constructed = []
    monkeypatch.setattr(spec_mtp.structure.model.hf_causal, 'HFCausalLM', lambda *args, **kwargs: constructed.append(args))
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
    from spec_mtp.flow.cli import _decode
    from spec_mtp.structure.artifact.observation import JsonlSink, audit_run
    import spec_mtp.structure.artifact.observation
    import spec_mtp.structure.model.hf_causal

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

    monkeypatch.setattr(spec_mtp.structure.artifact.observation, 'JsonlSink', ObservedSink)
    monkeypatch.setattr(spec_mtp.structure.model.hf_causal, 'HFCausalLM', fail_model)
    with pytest.raises(RuntimeError) as raised:
        _decode(args)
    assert raised.value is failure
    assert len(sinks) == 1 and sinks[0].handle.closed
    assert output.read_bytes() == b''
    assert not audit_run(output, request().sources).argmax_coverage
    assert getattr(failure, 'cleanup_failures', []) == ([close_error] if close_fails else [])

@pytest.mark.parametrize('damage', ['empty', 'list', 'bad_sha', 'stale', 'missing_member', 'extra_member', 'escape', 'inside_root', 'output_inside_root'])
def test_cli_rejects_model_manifest_before_constructing_model(tmp_path, monkeypatch, damage):
    from spec_mtp.flow.cli import _decode
    import spec_mtp.structure.model.hf_causal

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
    monkeypatch.setattr(spec_mtp.structure.model.hf_causal, 'HFCausalLM', lambda *a, **k: calls.append(a))
    with pytest.raises((ValueError, OSError)):
        _decode(args)
    assert calls == [] and not args.output.exists()
