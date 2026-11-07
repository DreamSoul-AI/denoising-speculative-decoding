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


@pytest.mark.parametrize('primary_kind,sink_kind,model_kind', [
    ('none', 'none', 'none'),
    ('none', 'KeyboardInterrupt', 'none'),
    ('none', 'SystemExit', 'OSError'),
    ('none', 'OSError', 'RuntimeError'),
    ('ValueError', 'KeyboardInterrupt', 'OSError'),
    ('ValueError', 'SystemExit', 'none'),
    ('KeyboardInterrupt', 'OSError', 'none'),
    ('SystemExit', 'KeyboardInterrupt', 'RuntimeError'),
    ('ValueError', 'none', 'OSError'),
    ('none', 'none', 'OSError'),
], ids=['success', 'sink_interrupt', 'sink_exit_then_model_error', 'two_close_errors',
        'primary_then_interrupt_and_error', 'primary_then_exit', 'primary_interrupt',
        'primary_exit_then_interrupt_and_error', 'primary_then_model_error', 'model_error'])
def test_cli_cleanup_finishes_every_resource_and_preserves_primary(tmp_path, monkeypatch, primary_kind, sink_kind, model_kind):
    from dataclasses import dataclass
    from spec_mtp.flow.cli import _decode
    from spec_mtp.structure.algorithm.decode import loop
    from spec_mtp.structure.algorithm import mask
    from spec_mtp.structure.artifact import observation
    from spec_mtp.structure.model import hf_causal

    kinds = {'none': None, 'ValueError': ValueError, 'KeyboardInterrupt': KeyboardInterrupt,
             'SystemExit': SystemExit, 'OSError': OSError, 'RuntimeError': RuntimeError}
    primary = kinds[primary_kind]('body failure') if primary_kind != 'none' else None
    sink_error = kinds[sink_kind]('sink close failure') if sink_kind != 'none' else None
    model_error = kinds[model_kind]('model close failure') if model_kind != 'none' else None
    earlier = OSError('already recorded inner cleanup failure')
    if primary is not None:
        primary.cleanup_failures = [earlier]
    args = model_args(tmp_path)  # Real, valid local JSON/manifest admission only.
    calls, sinks, models = [], [], []
    actual_sink = observation.JsonlSink

    class Sink(actual_sink):
        def __init__(self, *values, **options):
            super().__init__(*values, **options)
            sinks.append(self)
        def close(self):
            calls.append('sink')
            super().close()
            if sink_error is not None:
                raise sink_error

    class Model:
        embedding_table = object()
        def __init__(self, *values, **options):
            models.append(self)
        def describe(self):
            return {'fixture': 'pure control; no model forward'}
        def close(self):
            calls.append('model')
            if model_error is not None:
                raise model_error

    @dataclass
    class Completed:
        marker: str = 'pure-control'

    generated = []
    def generate(*values, **options):
        generated.append(True)
        if primary is not None:
            raise primary
        return Completed()

    monkeypatch.setattr(observation, 'JsonlSink', Sink)
    monkeypatch.setattr(hf_causal, 'HFCausalLM', Model)
    monkeypatch.setattr(loop, 'generate', generate)
    monkeypatch.setattr(mask, 'build_mask_provider', lambda *a, **k: object())
    failures = [error for error in (sink_error, model_error) if error is not None]
    expected = primary if primary is not None else failures[0] if failures else None
    if expected is None:
        assert _decode(args) == {'marker': 'pure-control'}
    else:
        with pytest.raises(BaseException) as raised:
            _decode(args)
        assert raised.value is expected
        expected_cleanup = [earlier, *failures] if primary is not None else failures[1:]
        actual_cleanup = getattr(raised.value, 'cleanup_failures', [])
        assert len(actual_cleanup) == len(expected_cleanup)
        assert all(actual is wanted for actual, wanted in zip(actual_cleanup, expected_cleanup))
    assert generated == [True]
    assert calls == ['sink', 'model']  # Cancellation cannot abort the remaining cleanup.
    assert len(sinks) == len(models) == 1 and sinks[0].handle.closed
    assert args.output.read_bytes() == b''
