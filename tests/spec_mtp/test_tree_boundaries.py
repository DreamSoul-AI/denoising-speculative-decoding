"""tree.py error paths and edge cases (no weights).

Covers every public failure mode and the empty / zero-width / tiny-vocab
branches. The ancestor-closure loop in ``build_dynamic_tree`` (lines that
drop orphans when depth-i Top-1 is missing) is defensive under Top-1
expansion: a depth-(i+1) candidate always has lower cumulative probability
than its depth-i Top-1 parent, so that parent is always selected first.
Left uncovered on purpose — changing it would need a synthetic selected_set,
not a real tree.
"""

from __future__ import annotations

import pytest
import torch

from spec_mtp.tree import (
    DraftTree, _topk_with_pruning, build_dynamic_tree, build_static_tree,
    build_tree, node_budget_for,
)
from studies._shared.structure.config import _convert


def _probs(k, v, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(k, v, generator=g), dim=-1)


def test_empty_tree_helpers():
    tree = DraftTree(root_token=7)
    assert tree.num_nodes == 0 and tree.branch_widths() == []
    assert tree.block_size(2) == 3 and tree.block_complexity(2) == 3
    assert tree.children_of(0) == []


def test_node_budget_rejects_too_small_block_complexity():
    with pytest.raises(ValueError, match='too small'):
        node_budget_for(2, 2)  # 2 // 3 - 1 == 0
    with pytest.raises(ValueError, match='too small'):
        node_budget_for(1, 1)
    assert node_budget_for(30, 2) == 9


def test_static_rejects_more_branches_than_masks():
    with pytest.raises(ValueError, match='more branch widths than mask tokens'):
        build_static_tree(_probs(2, 16), [3, 2, 1], root_token=0)


def test_static_zero_width_stops_deeper_levels():
    tree = build_static_tree(_probs(3, 20), [4, 0, 3], root_token=1, pruning=False)
    assert tree.branch_widths() == [4]  # depth 2 width 0 breaks; depth 3 never built
    assert tree.num_nodes == 4


def test_topk_skips_banned_and_taken_tokens():
    probs = torch.tensor([0.05, 0.40, 0.35, 0.15, 0.05])  # prefer 1, then 2, then 3
    tokens, token_probs = _topk_with_pruning(probs, width=2, banned_token=1, taken_tokens={2})
    assert tokens == [3, 0] or tokens[0] == 3  # 1 banned, 2 taken → 3 then next
    assert 1 not in tokens and 2 not in tokens
    assert len(tokens) == 2


def test_dynamic_stops_when_pruning_exhausts_tiny_vocab():
    """Only one token in the vocab and it is the parent → pruning yields no candidates."""
    probs = torch.ones(2, 1)  # vocab size 1
    tree = build_dynamic_tree(probs, node_budget=3, root_token=0, pruning=True)
    assert tree.num_nodes == 0 and tree.branch_widths() == []


def test_build_tree_dispatch_and_unknown_mode():
    probs = _probs(2, 32, seed=5)
    static = build_tree(_convert({
        'num_masks': 2, 'tree': {'mode': 'static', 'branches': [3, 2], 'pruning': False},
    }), probs, root_token=4)
    assert static.branch_widths() == [3, 2]
    dynamic = build_tree(_convert({
        'num_masks': 2, 'block_complexity': 30, 'tree': {'mode': 'dynamic', 'pruning': False},
    }), probs, root_token=4)
    assert dynamic.num_nodes == node_budget_for(30, 2)
    with pytest.raises(ValueError, match='Unsupported tree.mode'):
        build_tree(_convert({'num_masks': 1, 'tree': {'mode': 'forest'}}), probs[:1], 0)


def test_children_of_and_block_layout_indices():
    tree = build_static_tree(_probs(2, 24, seed=8), [3, 2], root_token=9, pruning=False)
    assert tree.children_of(0) == [1, 2, 3]  # three depth-1 nodes
    # Only the Top-1 depth-1 node grows children (ESP Top-1 expansion).
    top1 = max(range(3), key=lambda j: tree.probs[j])
    kids = tree.children_of(top1 + 1)
    assert len(kids) == 2 and all(tree.depths[i - 1] == 2 for i in kids)
    assert tree.children_of(kids[0]) == []
