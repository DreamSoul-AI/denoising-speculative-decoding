"""Draft tree construction, depth-1 edits, and block attention."""

from spec_mtp.structure.algorithm.decode.tree import (
    DraftTree,
    add_depth1_branch,
    build_dynamic_tree,
    build_static_tree,
    build_tree,
    node_budget_for,
    replace_weakest_depth1,
)
from spec_mtp.structure.algorithm.decode.tree_attention import (
    DynamicTreeAttentionCache,
    TreeAttentionCache,
    block_layout,
    build_block_attention,
    mask_indices_of,
)

__all__ = [
    'DraftTree',
    'DynamicTreeAttentionCache',
    'TreeAttentionCache',
    'add_depth1_branch',
    'block_layout',
    'build_block_attention',
    'build_dynamic_tree',
    'build_static_tree',
    'build_tree',
    'mask_indices_of',
    'node_budget_for',
    'replace_weakest_depth1',
]
