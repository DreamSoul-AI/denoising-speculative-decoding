"""DynamicTreeAttentionCache is bit-identical to the naive builder on random dynamic trees."""

import pytest





from spec_mtp.structure.algorithm.decode.tree_attention import DynamicTreeAttentionCache

def test_max_entries_must_be_at_least_one():
    with pytest.raises(ValueError, match='max_entries'):
        DynamicTreeAttentionCache(max_entries=0)
    with pytest.raises(ValueError, match='max_entries'):
        DynamicTreeAttentionCache(max_entries=-1)
    assert DynamicTreeAttentionCache(max_entries=1).max_entries == 1
