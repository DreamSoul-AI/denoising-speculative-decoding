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


def test_model_sources_reject_resolved_members_outside_root(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from spec_mtp.audit import verify_model_files

    root = tmp_path / 'model'
    root.mkdir()
    member = root / 'config.json'
    member.write_bytes(b'{}')
    outside = tmp_path / 'external-config.json'
    original = Path.resolve
    # Simulate an external symlink resolution without requiring OS symlink privileges.
    monkeypatch.setattr(Path, 'resolve', lambda self, *a, **k: outside if self == member else original(self, *a, **k))
    with pytest.raises(ValueError, match='escapes'):
        verify_model_files(root, {'config.json': hashlib.sha256(b'{}').hexdigest()})


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


@pytest.mark.parametrize('field', ['accepted', 'committed', 'prefill_calls', 'decode_calls', 'new_tokens'])
@pytest.mark.parametrize('bad', [-1, True, 1.5, float('nan'), float('inf')])
def test_invalid_decode_counts_are_rejected(field, bad):
    req = request()
    _, _, sink = run(req)
    sink.rows[-1]['stats'][field] = bad
    result = audit_run(sink.rows, req.sources)
    assert not result.argmax_coverage and result.reasons


@pytest.mark.parametrize('change', ['prefill', 'decode', 'accepted', 'committed', 'trace_token', 'trace_bool', 'time', 'missing_trace', 'bad_trace', 'empty_commit', 'bad_node'])
def test_decode_trace_and_counts_must_match_stream(change):
    req = request()
    _, _, sink = run(req)
    stats = sink.rows[-1]['stats']
    if change == 'trace_token':
        stats['trace'][0]['committed_tokens'][0] = 999
    elif change == 'trace_bool':
        stats['trace'][0]['committed_tokens'][0] = True
    elif change == 'time':
        stats['decode_time'] = float('inf')
    elif change == 'missing_trace':
        del stats['trace']
    elif change == 'bad_trace':
        stats['trace'][0] = []
    elif change == 'empty_commit':
        stats['trace'][0]['committed_tokens'] = []
    elif change == 'bad_node':
        stats['trace'][0]['accepted_nodes'] = [True]
    else:
        field = {'prefill': 'prefill_calls', 'decode': 'decode_calls'}.get(change, change)
        stats[field] += 1
    result = audit_run(sink.rows, req.sources)
    assert not result.argmax_coverage and result.reasons


def test_budget_trimmed_acceptance_may_exceed_committed_count():
    from dataclasses import replace
    from spec_mtp.decode import generate
    from spec_mtp.masks import ESPMaskProvider
    from test_decode import FakeSession, MemorySink

    class MatchingMasks(ESPMaskProvider):
        def masks(self):
            return self.embedding_table[4:6]

    req = replace(request(), max_new_tokens=2, spec={
        'num_masks': 2, 'impl': 'efficient', 'tree': {'mode': 'static', 'branches': [1, 1]}})
    model, sink = FakeSession(), MemorySink()
    policy = MatchingMasks(2, model.embedding_table)
    generate(req, model, policy, sink)
    stats = sink.rows[-1]['stats']
    assert stats['accepted'] > stats['committed']
    assert audit_run(sink.rows, req.sources).argmax_coverage


@pytest.mark.parametrize('field', ['accepted', 'committed', 'prefill_calls', 'decode_calls', 'new_tokens'])
def test_empty_decode_still_checks_all_counts(field):
    from dataclasses import replace
    req = replace(request(), max_new_tokens=0)
    _, _, sink = run(req)
    sink.rows[-1]['stats'][field] = True
    assert not audit_run(sink.rows, req.sources).argmax_coverage
