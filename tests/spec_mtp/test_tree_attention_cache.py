"""DynamicTreeAttentionCache is bit-identical to the naive builder on random dynamic trees."""

from spec_mtp.masks import build_mask_provider

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spec_mtp.tree import build_dynamic_tree, node_budget_for
from spec_mtp.tree import DynamicTreeAttentionCache, build_block_attention


def _tree(seed, num_masks, bc, vocab, pruning):
    g = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(num_masks, vocab, generator=g) * 3, dim=-1)
    return build_dynamic_tree(probs, node_budget_for(bc, num_masks), root_token=int(seed % vocab), pruning=pruning)


@settings(max_examples=80, deadline=None)
@given(seeds=st.lists(st.integers(0, 10**6), min_size=1, max_size=6),
       num_masks=st.integers(1, 3), bc=st.sampled_from([10, 30, 60, 120]),
       cache_len=st.integers(0, 50), root=st.integers(0, 200), pruning=st.booleans(),
       dtype=st.sampled_from([torch.float32, torch.bfloat16]))
def test_matches_naive_builder(seeds, num_masks, bc, cache_len, root, pruning, dtype):
    cache = DynamicTreeAttentionCache(max_entries=3)
    min_value = torch.finfo(dtype).min
    for seed in seeds + seeds:  # second pass exercises hits and LRU eviction
        tree = _tree(seed, num_masks, bc, 64, pruning)
        want_mask, want_pid = build_block_attention(tree, num_masks, cache_len, root, dtype, 'cpu', min_value)
        got_mask, got_pid = cache.get(tree, num_masks, cache_len, root, dtype, 'cpu', min_value)
        assert got_mask.dtype == want_mask.dtype and torch.equal(got_mask, want_mask)
        assert got_pid.dtype == want_pid.dtype and torch.equal(got_pid, want_pid)
    assert len(cache._entries) <= 3 and cache.hits + cache.misses == 2 * len(seeds)


def test_returned_tensors_do_not_alias_the_cache():
    cache = DynamicTreeAttentionCache()
    tree = _tree(1, 2, 30, 64, True)
    mask, pid = cache.get(tree, 2, 4, 10, torch.float32, 'cpu', -1e9)
    mask.fill_(7.0)
    pid.fill_(7)
    again_mask, again_pid = cache.get(tree, 2, 4, 10, torch.float32, 'cpu', -1e9)
    want_mask, want_pid = build_block_attention(tree, 2, 4, 10, torch.float32, 'cpu', -1e9)
    assert torch.equal(again_mask, want_mask) and torch.equal(again_pid, want_pid)
    assert cache.hits == 1 and cache.misses == 1


def test_decoder_uses_it_only_for_dynamic_trees():
    from studies._shared.structure.config import apply_variant
    from spec_mtp.decode import SpecDecoder
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM

    model = HFCausalLM(tiny_config={'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
                                    'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1,
                                    'max_position_embeddings': 256})
    eval_cfg = _convert({'max_new_tokens': 4, 'temperature': 0.0, 'stop_on_eos': False})
    dynamic = SpecDecoder(model, apply_variant({}, 'esp_dynamic'), eval_cfg, mask_policy=build_mask_provider(apply_variant({}, 'esp_dynamic'), model.embedding_table))
    naive_static = SpecDecoder(model, apply_variant({}, 'llama3/impl_naive_bc30'), eval_cfg, mask_policy=build_mask_provider(apply_variant({}, 'llama3/impl_naive_bc30'), model.embedding_table))
    efficient = SpecDecoder(model, apply_variant({}, 'esp_static'), eval_cfg, mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), model.embedding_table))
    assert dynamic._dynamic_attn is not None
    assert naive_static._dynamic_attn is None and naive_static._attn_cache is None  # Table 4 naive path kept
    assert efficient._dynamic_attn is None and efficient._attn_cache is not None


def test_max_entries_must_be_at_least_one():
    with pytest.raises(ValueError, match='max_entries'):
        DynamicTreeAttentionCache(max_entries=0)
    with pytest.raises(ValueError, match='max_entries'):
        DynamicTreeAttentionCache(max_entries=-1)
    assert DynamicTreeAttentionCache(max_entries=1).max_entries == 1


def test_static_efficient_cache_matches_naive_builder():
    """TreeAttentionCache (Appendix E) stays bit-identical after sharing _block_structure."""
    from spec_mtp.tree import build_static_tree
    from spec_mtp.tree import TreeAttentionCache

    g = torch.Generator().manual_seed(2)
    probs = torch.softmax(torch.randn(2, 48, generator=g), dim=-1)
    tree = build_static_tree(probs, [7, 2], root_token=3, pruning=True)
    cache = TreeAttentionCache()
    for cache_len, root in ((0, 5), (12, 40), (3, 0)):
        want_m, want_p = build_block_attention(tree, 2, cache_len, root, torch.float32, 'cpu', -1e4)
        got_m, got_p = cache.get(tree, 2, cache_len, root, torch.float32, 'cpu', -1e4)
        assert torch.equal(got_m, want_m) and torch.equal(got_p, want_p)


def test_static_cache_rebuilds_when_dtype_or_min_value_changes():
    from spec_mtp.tree import build_static_tree
    from spec_mtp.tree import TreeAttentionCache

    g = torch.Generator().manual_seed(0)
    tree = build_static_tree(torch.softmax(torch.randn(2, 32, generator=g), dim=-1), [4], root_token=1, pruning=False)
    cache = TreeAttentionCache()
    cache.get(tree, 2, 3, 10, torch.float32, 'cpu', torch.finfo(torch.float32).min)
    for dtype in (torch.bfloat16, torch.float32):
        min_value = torch.finfo(dtype).min
        got, _ = cache.get(tree, 2, 3, 10, dtype, 'cpu', min_value)
        want, _ = build_block_attention(tree, 2, 3, 10, dtype, 'cpu', min_value)
        assert got.dtype == dtype and torch.equal(got, want)
