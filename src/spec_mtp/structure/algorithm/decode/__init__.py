"""Draft tree construction, depth-1 edits, block attention, and the decode loop."""

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
from spec_mtp.structure.algorithm.decode.loop import (
    DecodeFailure,
    SpecDecoder,
    ar_generate,
    generate,
    select_token,
    tree_probs,
)

__all__ = [
    'DecodeFailure',
    'DraftTree',
    'DynamicTreeAttentionCache',
    'SpecDecoder',
    'TreeAttentionCache',
    'add_depth1_branch',
    'ar_generate',
    'block_layout',
    'build_block_attention',
    'build_dynamic_tree',
    'build_static_tree',
    'build_tree',
    'generate',
    'mask_indices_of',
    'node_budget_for',
    'replace_weakest_depth1',
    'select_token',
    'tree_probs',
]
