"""Compatibility shim. Definitions live in spec_mtp.structure.algorithm.decode."""

from spec_mtp.structure.algorithm.decode.tree import (
    DraftTree,
    _depth1_indices,
    _topk_with_pruning,
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
    _block_structure,
    block_layout,
    build_block_attention,
    mask_indices_of,
)

__all__ = [
    'DraftTree',
    'DynamicTreeAttentionCache',
    'TreeAttentionCache',
    '_block_structure',
    '_depth1_indices',
    '_topk_with_pruning',
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
