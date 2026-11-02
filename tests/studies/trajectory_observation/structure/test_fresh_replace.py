"""Prompt slice and effect summary for the fresh equal-node replication."""

import pytest

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, direction, fresh_prompts, outside_repo, paired_deltas,
)


def _rows(n, blank_at=None):
    rows = []
    for index in range(n):
        if index == blank_at:
            rows.append({'category': 'creative_writing', 'instruction': '  ', 'context': ''})
        rows.append({'category': 'creative_writing', 'instruction': f'write {index}', 'context': ''})
    rows.append({'category': 'open_qa', 'instruction': 'not creative', 'context': ''})
    return rows


def test_fresh_prompts_skip_the_used_creative_prefix():
    prompts = fresh_prompts(_rows(300, blank_at=10), skip=128, take=128)
    assert [prompt['qid'] for prompt in prompts] == list(range(128, 256))
    assert prompts[0]['text'] == 'write 128'
    assert prompts[-1]['text'] == 'write 255'
    assert all(prompt['category'] == 'creative_writing' for prompt in prompts)


def test_context_is_prefixed_the_same_way_as_the_original_loader():
    prompts = fresh_prompts(
        [{'category': 'creative_writing', 'instruction': f'write {index}', 'context': ''}
         for index in range(128)]
        + [{'category': 'creative_writing', 'instruction': 'go', 'context': 'ctx'}],
        skip=128, take=1)
    assert prompts == [{'qid': 128, 'category': 'creative_writing', 'text': 'ctx\n\ngo'}]


def test_repeated_text_is_skipped_and_short_pools_are_rejected():
    with pytest.raises(RuntimeError):
        fresh_prompts(_rows(20), skip=128, take=128)
    rows = [{'category': 'creative_writing', 'instruction': f'write {index}', 'context': ''}
            for index in range(4)]
    rows.append({'category': 'creative_writing', 'instruction': 'write 1', 'context': ''})
    rows.append({'category': 'creative_writing', 'instruction': 'fresh', 'context': ''})
    prompts = fresh_prompts(rows, skip=2, take=3)
    assert [prompt['qid'] for prompt in prompts] == [2, 3, 5]
    assert [prompt['text'] for prompt in prompts] == ['write 2', 'write 3', 'fresh']


def test_output_inside_the_repo_is_rejected():
    with pytest.raises(RuntimeError, match='outside the repo'):
        outside_repo(__file__)


def test_direction_follows_the_sign_and_the_mean():
    assert direction(17, 4, 0.2) == 'positive'
    assert direction(4, 17, -0.2) == 'negative'
    assert direction(10, 10, 0) == 'flat'
    assert direction(12, 4, -0.5) == 'mixed'


def test_bootstrap_ci_is_deterministic_and_paired_by_qid():
    first = bootstrap_mean_ci([1, -1, 2, 0], seed=0, resamples=200)
    second = bootstrap_mean_ci([1, -1, 2, 0], seed=0, resamples=200)
    assert first == second
    assert first['ci95'][0] <= first['mean'] <= first['ci95'][1]
    left = [{'qid': 2, 'accepted': 3}, {'qid': 1, 'accepted': 5}]
    right = [{'qid': 1, 'accepted': 4}, {'qid': 2, 'accepted': 3}]
    assert paired_deltas(left, right, 'accepted') == [0, 1]
