"""Untrusted observation envelopes fail as scientific verdicts."""
from copy import deepcopy
import json

import pytest

from spec_mtp.audit import audit_run
from test_decode import request, run


@pytest.mark.parametrize('raw', [[[], {'event': 'end'}], [None], [True], {}, ''])
def test_non_object_or_missing_envelope_is_rejected(raw):
    assert audit_run(raw, {}).argmax_coverage is False


@pytest.mark.parametrize('bad', [True, float('nan'), float('inf'), -1, '0'])
def test_invalid_temperature_is_rejected(bad):
    req = request()
    _, _, sink = run(req)
    sink.rows[0]['temperature'] = bad
    result = audit_run(sink.rows, req.sources)
    assert not result.argmax_coverage and result.reasons


@pytest.mark.parametrize('field', ['token', 'index', 'forward', 'row', 'argmax'])
def test_boolean_never_impersonates_integer_audit_fields(field):
    req = request()
    _, _, sink = run(req)
    sink.rows[1][field] = True
    assert not audit_run(sink.rows, req.sources).argmax_coverage


def test_missing_or_mutated_committed_row_is_rejected():
    req = request()
    _, _, sink = run(req)
    del sink.rows[2]
    assert not audit_run(sink.rows, req.sources).argmax_coverage


def test_source_mismatch_is_independent_of_correct_argmax():
    _, _, sink = run(request())
    result = audit_run(sink.rows, {'science': 'changed'})
    assert not result.source_integrity and result.argmax_coverage and result.argmax_correctness


def test_truncated_jsonl_never_becomes_complete(tmp_path):
    req = request()
    _, _, sink = run(req)
    raw = tmp_path / '观测.jsonl'
    raw.write_text('\n'.join(json.dumps(row) for row in sink.rows), encoding='utf-8')
    assert not audit_run(raw, req.sources).argmax_coverage


def test_duplicate_envelope_is_rejected():
    _, _, sink = run(request())
    sink.rows.insert(1, deepcopy(sink.rows[0]))
    assert not audit_run(sink.rows, request().sources).argmax_coverage
