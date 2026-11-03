"""Slice and observation switch for the Qwen3-8B trajectory ablation."""

import pytest
import torch

from spec_mtp.masks import build_mask_provider
from studies.trajectory_observation.structure.office_qwen8b import ESP
from studies.trajectory_observation.structure.office_traj import (
    EVAL_QIDS, EVAL_SHA256, GAIN_FULL, GAIN_TRAJ, PROTOCOL, SKIPPED_QIDS, TAKE,
    ablation_summary, arm_spec, arm_specs, prompts_from_647,
)


def _creative(text, context=''):
    return {'category': 'creative_writing', 'instruction': text, 'context': context}


def test_prompts_from_647_skip_texts_already_seen():
    rows = [_creative(f'write {index}') for index in range(647)]
    rows.append(_creative('write 408'))
    rows.append(_creative('fresh'))
    rows.append(_creative('write 198'))
    rows.append(_creative('later'))
    chosen, skipped = prompts_from_647(rows, take=2)
    assert skipped == (647, 649)
    assert [item['qid'] for item in chosen] == [648, 650]


def test_short_pool_from_647_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_from_647([_creative(f'write {index}') for index in range(660)], take=TAKE)


def test_registered_tail_is_sixty_prompts_after_two_skips():
    assert PROTOCOL == 'office-8b-traj-ablate-v1'
    assert TAKE == 60
    assert SKIPPED_QIDS == (658, 678)
    assert EVAL_QIDS == tuple(qid for qid in range(647, 709) if qid not in SKIPPED_QIDS)
    assert len(EVAL_QIDS) == TAKE
    assert EVAL_QIDS[0] == 647 and EVAL_QIDS[-1] == 708
    assert EVAL_SHA256 == 'd23ad590c45307fa7c1981b272adaec243bfaca5a2e390eda394bf2fe42a4cca'


def test_arms_share_the_formal_tree_and_only_the_observation_gain_changes():
    specs = arm_specs()
    assert specs['full']['tree'] == specs['traj']['tree'] == ESP['tree']
    assert specs['full']['mask'] == specs['traj']['mask'] == ESP['mask']
    assert specs['full']['num_masks'] == specs['traj']['num_masks'] == 1
    assert specs['full']['observation']['gain'] == GAIN_FULL == 0.5
    assert specs['traj']['observation']['gain'] == GAIN_TRAJ == 0.0
    assert specs['full']['observation']['condition'] == specs['traj']['observation']['condition'] == 'none'
    assert specs['full']['observation']['form'] == specs['traj']['observation']['form'] == 'mean'
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    full = build_mask_provider(specs['full'], table)
    traj = build_mask_provider(specs['traj'], table)
    assert full.protocol == traj.protocol == 'trajectory-observation-blend-v1'
    assert full.gain == 0.5 and traj.gain == 0.0
    with pytest.raises(RuntimeError):
        arm_spec(1.0)


def _arm(qid, accepted):
    return {'qid': qid, 'accepted': accepted, 'committed': accepted + 8, 'decode_calls': 8,
            'tau': (accepted + 8) / 8, 'argmax_passed': True, 'ar_equal': True,
            'max_nodes': 4, 'block_complexity': 10}


def test_summary_defines_traj_minus_full_and_the_opposite_full_gain():
    full = [_arm(qid, 2) for qid in range(TAKE)]
    traj = [_arm(qid, 1 if qid < 40 else 2) for qid in range(TAKE)]
    summary = ablation_summary({'full': full, 'traj': traj})
    comparison = summary['comparison']
    assert summary['role'] == 'exploration' and summary['confirmatory'] is False
    assert summary['prompts'] == TAKE
    assert comparison['wins_losses_ties'] == [0, 40, 20]
    assert comparison['direction'] == 'negative'
    assert comparison['accepted_difference']['mean'] == -40 / TAKE
    assert comparison['full_gain_difference']['mean'] == 40 / TAKE
    assert comparison['accepted_difference']['ci95'][1] < 0
    assert comparison['full_gain_difference']['ci95'][0] > 0
    assert comparison['max_nodes'] == 4
    assert comparison['block_complexity'] == {'full': [10], 'traj': [10]}
