"""Slice, supported shapes and summary for the tree-shape exploration."""

from pathlib import Path

import pytest
import torch

from spec_mtp.tree import build_static_tree
from studies.trajectory_observation.structure.office_tree import (
    EVAL_QIDS, EVAL_QIDS_8B, EVAL_SHA256, EVAL_SHA256_8B, NODE_BUDGET, PROTOCOL, PROTOCOL_8B,
    QWEN3_4B, SKIPPED_QIDS, SKIPPED_QIDS_8B, TAKE, arm_spec, arm_specs, block_complexity,
    existing_output_dir, prompts_from_388, prompts_from_516, require_budget, tree_summary,
)
from studies.trajectory_observation.structure.fair_branch import ESP


def _creative(text, context=''):
    return {'category': 'creative_writing', 'instruction': text, 'context': context}


def test_prompts_from_388_skip_prefix_and_selected_duplicates():
    rows = [_creative(f'write {index}') for index in range(388)]
    rows.append(_creative('write 0'))
    rows.append(_creative('fresh', context='ctx'))
    rows.append(_creative('ctx\n\nfresh'))
    rows.append(_creative('next'))
    chosen, skipped = prompts_from_388(rows, take=2)
    assert skipped == (388, 390)
    assert [item['qid'] for item in chosen] == [389, 391]
    assert chosen[0]['text'] == 'ctx\n\nfresh'


def test_short_pool_from_388_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_from_388([_creative(f'write {index}') for index in range(400)], take=128)


def test_registered_slice_is_the_unskipped_block_after_387():
    assert EVAL_QIDS == tuple(range(388, 516))
    assert len(EVAL_QIDS) == TAKE
    assert SKIPPED_QIDS == ()
    assert EVAL_SHA256 == '123702e2294e9339665cf752ab286297b670cc06e6af632d9b6912399b6cc4e3'


def test_prompts_from_516_skip_prefix_and_selected_duplicates():
    rows = [_creative(f'write {index}') for index in range(516)]
    rows.append(_creative('write 0'))
    rows.append(_creative('fresh', context='ctx'))
    rows.append(_creative('ctx\n\nfresh'))
    rows.append(_creative('next'))
    chosen, skipped = prompts_from_516(rows, take=2)
    assert skipped == (516, 518)
    assert [item['qid'] for item in chosen] == [517, 519]
    assert chosen[0]['text'] == 'ctx\n\nfresh'


def test_registered_8b_slice_starts_after_515_and_records_three_skips():
    assert PROTOCOL_8B == 'office-8b-tree-shape-v1'
    assert SKIPPED_QIDS_8B == (597, 608, 626)
    assert EVAL_QIDS_8B == tuple(qid for qid in range(516, 647) if qid not in SKIPPED_QIDS_8B)
    assert len(EVAL_QIDS_8B) == TAKE
    assert 597 not in EVAL_QIDS_8B and 608 not in EVAL_QIDS_8B and 626 not in EVAL_QIDS_8B
    assert EVAL_SHA256_8B == 'd4596e4f7404f47ac53e327d711f70edb7d49298b7583155c8050bfe27bed4dd'
    assert EVAL_QIDS[0] == 388 and EVAL_QIDS_8B[0] == 516


def test_shapes_keep_four_draft_nodes_and_the_baseline_is_existing_esp():
    specs = arm_specs()
    assert specs['b4']['tree'] == ESP['tree']
    assert specs['b4']['num_masks'] == ESP['num_masks'] == 1
    assert specs['b4']['mask'] == ESP['mask']
    assert specs['b4']['method'] == 'esp' and 'hybrid_gain' not in specs['b4']
    assert specs['b22']['tree']['branches'] == [2, 2] and specs['b22']['num_masks'] == 2
    assert specs['b31']['tree']['branches'] == [3, 1] and specs['b31']['num_masks'] == 2
    assert block_complexity(1, [4]) == 10
    assert block_complexity(2, [2, 2]) == 15
    assert block_complexity(2, [3, 1]) == 15
    with pytest.raises(RuntimeError):
        arm_spec([5], 1)
    with pytest.raises(RuntimeError):
        arm_spec([2, 2], 1)


def test_decoder_entry_builds_four_nodes_for_each_registered_shape():
    probs = torch.softmax(torch.arange(64, dtype=torch.float32).reshape(2, 32), dim=-1)
    wide = build_static_tree(probs[:1], [4], root_token=3, pruning=True)
    balanced = build_static_tree(probs, [2, 2], root_token=3, pruning=True)
    deep = build_static_tree(probs, [3, 1], root_token=3, pruning=True)
    assert wide.num_nodes == balanced.num_nodes == deep.num_nodes == NODE_BUDGET
    assert wide.branch_widths() == [4]
    assert balanced.branch_widths() == [2, 2]
    assert deep.branch_widths() == [3, 1]
    with pytest.raises(ValueError):
        build_static_tree(probs[:1], [2, 2], root_token=3, pruning=True)


def test_a_step_off_the_budget_is_rejected():
    assert require_budget([4, 4, 4]) == 4
    with pytest.raises(RuntimeError):
        require_budget([4, 5])
    with pytest.raises(RuntimeError):
        require_budget([])


def _arm(qid, accepted, complexity):
    return {'qid': qid, 'accepted': accepted, 'committed': accepted + 8, 'decode_calls': 8,
            'tau': (accepted + 8) / 8, 'argmax_passed': True, 'ar_equal': True,
            'max_nodes': 4, 'block_complexity': complexity}


def test_summary_compares_each_shape_with_the_wide_baseline():
    baseline = [_arm(qid, 2, 10) for qid in range(TAKE)]
    balanced = [_arm(qid, 3, 15) for qid in range(TAKE)]
    skewed = [_arm(qid, 2 if qid < 40 else 1, 15) for qid in range(TAKE)]
    summary = tree_summary({'b4': baseline, 'b22': balanced, 'b31': skewed})
    assert summary['role'] == 'exploration' and summary['confirmatory'] is False
    assert summary['protocol'] == PROTOCOL
    eight = tree_summary(
        {'b4': baseline, 'b22': balanced, 'b31': skewed},
        protocol=PROTOCOL_8B, digest=EVAL_SHA256_8B, skipped=SKIPPED_QIDS_8B,
    )
    assert eight['protocol'] == PROTOCOL_8B
    assert eight['prompt_sha256'] == EVAL_SHA256_8B
    assert eight['skipped_qids'] == [597, 608, 626]
    assert eight['comparisons']['b22']['wins_losses_ties'] == [128, 0, 0]
    assert summary['comparisons']['b22']['wins_losses_ties'] == [128, 0, 0]
    assert summary['comparisons']['b22']['accepted_difference']['ci95'][0] > 0
    assert summary['comparisons']['b31']['wins_losses_ties'] == [0, 88, 40]
    assert summary['comparisons']['b31']['direction'] == 'negative'
    assert summary['comparisons']['b22']['max_nodes'] == 4
    assert summary['comparisons']['b22']['p'] is not None


def test_output_dir_must_already_exist_off_the_repo(tmp_path):
    with pytest.raises(RuntimeError):
        existing_output_dir(tmp_path / 'missing')
    assert QWEN3_4B == Path('D' + ':' + '\\') / 'models' / 'Qwen3-4B'
