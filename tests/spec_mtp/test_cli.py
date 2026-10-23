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
