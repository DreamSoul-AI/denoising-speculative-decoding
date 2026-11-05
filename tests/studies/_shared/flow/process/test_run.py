"""Study aggregation validates every declared Run and saved evidence."""
import json
from types import SimpleNamespace

import pytest

from spec_mtp.structure.artifact.observation import SpecMtpGateError
from studies._shared.flow.process.run import study_gate
from studies._shared.structure.validation import STUDY_CHECK_NAME


def test_study_gate_complete_pass(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    assert study_gate(ctx)['passed']
    assert json.loads((ctx.study_dir / STUDY_CHECK_NAME).read_text())['complete']


def test_study_gate_subset_is_incomplete(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd')])
    with pytest.raises(SpecMtpGateError, match='incomplete'):
        study_gate(ctx)
    assert ctx.state['spec_mtp']['expected_factors_match'] is False


def test_study_gate_missing_result_is_incomplete(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', None, None, None)])
    with pytest.raises(SpecMtpGateError, match='incomplete'):
        study_gate(ctx)
    assert ctx.state['spec_mtp']['missing_runs'] == ['run_b']


def test_study_gate_failed_run_fails(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'failed', False, 'd')])
    with pytest.raises(SpecMtpGateError, match='failed'):
        study_gate(ctx)


def test_study_gate_mixed_source_digest_fails(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd1'), ('b', 'succeeded', True, 'd2')])
    with pytest.raises(SpecMtpGateError):
        study_gate(ctx)


def test_study_gate_seed_mismatch_is_incomplete(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')], seeds=(123, 124))
    with pytest.raises(SpecMtpGateError, match='incomplete'):
        study_gate(ctx)


def test_study_gate_skips_run_scope():
    assert study_gate(SimpleNamespace(scope='run')) == {}


def test_study_gate_empty_index_is_incomplete(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [])
    with pytest.raises(SpecMtpGateError, match='incomplete'):
        study_gate(ctx)
    assert ctx.state['spec_mtp']['complete'] is False


def test_study_gate_explicit_expected_factors(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    expected = [{'algorithm.config.variant': 'b'}, {'algorithm.config.variant': 'a'}]
    assert study_gate(ctx, expected_factors=expected)['passed']


def test_study_gate_rechecks_missing_evidence(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    (ctx.study_dir / 'runs/run_b/assets/decode_observations.json').unlink()
    with pytest.raises(SpecMtpGateError):
        study_gate(ctx)
    assert ctx.state['spec_mtp']['failed_runs'] == ['run_b']


def test_study_quality_failure_marks_generic_envelope_incomplete(tmp_path, saved_decode_study):
    ctx = saved_decode_study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    ctx.state['process'] = {'complete': True}
    (ctx.study_dir / 'runs/run_b/assets/decode_observations.json').unlink()
    with pytest.raises(SpecMtpGateError):
        study_gate(ctx)
    envelope = json.loads((ctx.study_dir / 'process.json').read_text())
    assert not envelope['complete'] and not envelope['spec_mtp']['passed']
