"""Slice, arm spec and summary for the Qwen3-8B office comparison."""

from pathlib import Path

import pytest

from studies.trajectory_observation.structure.office_qwen8b import (
    EVAL_QIDS, EVAL_SHA256, MODELS, SKIPPED_QIDS, TAKE, esp_spec, existing_output_dir,
    kalman_spec, office_summary, prompts_after_used, replace_spec,
)


def _creative(text, context=''):
    return {'category': 'creative_writing', 'instruction': text, 'context': context}


def test_prompts_after_256_skip_texts_already_used():
    rows = [_creative(f'write {index}') for index in range(257)]
    rows.append(_creative('write 0'))
    rows.append(_creative('fresh', context='ctx'))
    chosen, skipped = prompts_after_used(rows, take=1)
    assert skipped == (257,)
    assert chosen == [{'qid': 258, 'category': 'creative_writing', 'text': 'ctx\n\nfresh'}]


def test_short_pool_after_256_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_after_used([_creative(f'write {index}') for index in range(300)], take=128)


def test_registered_qid_span_skips_the_three_duplicates():
    assert len(EVAL_QIDS) == TAKE
    assert EVAL_QIDS[0] == 257 and EVAL_QIDS[-1] == 387
    assert SKIPPED_QIDS == (289, 304, 365)
    assert set(SKIPPED_QIDS).isdisjoint(EVAL_QIDS)
    assert len(EVAL_SHA256) == 64


def test_8b_table_points_at_the_local_qwen3_directory():
    from studies.trajectory_observation.structure.office_qwen8b import _load_model
    assert MODELS['8b'] == 'Qwen3-8B'
    with pytest.raises(ValueError, match='explicit local model root'):
        _load_model(None)


def test_replace_keeps_the_esp_tree_and_kalman_is_frozen_last():
    prior = {'k': [0.0], 'p': [1.0], 'r': [1.0], 'q': [0.01]}
    replace = replace_spec(prior)
    kalman = kalman_spec(prior)
    assert replace['tree']['branches'] == [4]
    assert replace['hybrid_gain']['mode'] == 'replace'
    assert 'observe' not in replace['hybrid_gain']
    assert kalman['method'] == 'esp'
    assert kalman['hybrid_gain']['mode'] == 'frozen'
    assert kalman['hybrid_gain']['observe'] == 'last'
    assert esp_spec()['method'] == 'esp'
    assert 'ema' not in kalman


def _arm(qid, accepted, nodes=4):
    return {'qid': qid, 'accepted': accepted, 'committed': accepted + 1, 'decode_calls': 2,
            'tau': (accepted + 1) / 2, 'argmax_passed': True, 'ar_equal': True, 'max_nodes': nodes}


def test_summary_reports_each_arm_against_esp():
    esp = [_arm(qid, 1) for qid in range(TAKE)]
    replaced = [_arm(qid, 2) for qid in range(TAKE)]
    kalman = [_arm(qid, 1 if qid < 40 else 0) for qid in range(TAKE)]
    summary = office_summary(esp, replaced, kalman)
    assert summary['replace_vs_esp']['wins_losses_ties'] == [128, 0, 0]
    assert summary['replace_vs_esp']['positive_trend'] is True
    assert summary['replace_vs_esp']['accepted_difference']['ci95'][0] > 0
    assert summary['kalman_vs_esp']['wins_losses_ties'][1] == 88
    assert summary['kalman_vs_esp']['direction'] == 'negative'
    assert summary['replace_vs_esp']['max_nodes'] == 4


def test_output_dir_must_exist_off_the_repo(tmp_path):
    with pytest.raises(RuntimeError):
        existing_output_dir(tmp_path / 'missing')
    with pytest.raises(RuntimeError, match='outside the repo'):
        existing_output_dir(Path(__file__).resolve().parents[4])
