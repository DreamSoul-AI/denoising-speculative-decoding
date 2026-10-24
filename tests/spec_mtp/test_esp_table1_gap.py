"""Small arithmetic regressions for the CPU-only Table1 readout."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('esp_gap_readout', Path(__file__).resolve().parents[2] / 'tools/analyze_esp_table1_gap.py')
gap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gap)


def row(qid, committed, calls, accepted):
    return dict(question_id=qid, committed=committed, decode_calls=calls,
        accepted=accepted, new_tokens=committed + 1, model_calls=calls + 1,
        argmax_checked_tokens=committed + 1)


def test_pooled_prompt_and_clipping_have_distinct_denominators():
    result = gap.summarize([row(1, 3, 2, 2), row(2, 8, 4, 4)], target=2)
    assert result['pooled_tau'] == 11 / 6
    assert result['prompt_mean_tau'] == 1.75
    assert result['end_clipped_tokens'] == 1
    assert result['clipping_tau_delta'] == 1 / 6
    assert result['extra_committed_at_same_calls'] == 1
    assert result['including_prefill_tau'] == 13 / 8


def test_pairing_uses_exact_ratio_ties_and_rejects_identity_mismatch():
    first, second = [row(1, 3, 2, 1)], [row(1, 6, 4, 2)]
    result = gap.paired(first, second)
    assert result['tau'] == dict(win=0, tie=1, loss=0)
    assert result['accepted']['win'] == result['calls']['loss'] == 1
    second[0]['question_id'] = 2
    with pytest.raises(AssertionError):
        gap.paired(first, second)


def test_bad_coverage_or_duplicate_rows_cannot_be_summarized():
    r = row(1, 3, 2, 1)
    with pytest.raises(AssertionError):
        gap.summarize([r, r])
    r['argmax_checked_tokens'] = 3
    with pytest.raises(AssertionError):
        gap.summarize([r])
