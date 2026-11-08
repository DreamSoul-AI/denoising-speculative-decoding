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


from spec_mtp.structure.algorithm.decode.tree import (
    DraftTree, node_budget_for,
)


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
