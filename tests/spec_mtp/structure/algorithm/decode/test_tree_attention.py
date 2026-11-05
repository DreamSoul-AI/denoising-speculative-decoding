"""DynamicTreeAttentionCache is bit-identical to the naive builder on random dynamic trees."""


import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from spec_mtp.structure.algorithm.decode.tree import build_dynamic_tree, node_budget_for
from spec_mtp.structure.algorithm.decode.tree_attention import DynamicTreeAttentionCache, build_block_attention


def _tree(seed, num_masks, bc, vocab, pruning, *, device):
    g = torch.Generator(device=device).manual_seed(seed)
    probs = torch.softmax(torch.randn(num_masks, vocab, generator=g, device=device) * 3, dim=-1)
    assert probs.device == device
    return build_dynamic_tree(probs, node_budget_for(bc, num_masks), root_token=int(seed % vocab), pruning=pruning)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seeds=st.lists(st.integers(0, 10**6), min_size=1, max_size=6),
       num_masks=st.integers(1, 3), bc=st.sampled_from([10, 30, 60, 120]),
       cache_len=st.integers(0, 50), root=st.integers(0, 200), pruning=st.booleans(),
       dtype=st.sampled_from([torch.float32, torch.bfloat16]))
def test_matches_naive_builder(seeds, num_masks, bc, cache_len, root, pruning, dtype, cuda_decode, record_property):
    # The device is fixed; each example creates its own scientific cache.
    if not hasattr(cuda_decode, 'attention_tensor_devices'):
        cuda_decode.attention_tensor_devices = []
        record_property('attention_tensor_devices', cuda_decode.attention_tensor_devices)
    cache = DynamicTreeAttentionCache(max_entries=3)
    min_value = torch.finfo(dtype).min
    for seed in seeds + seeds:  # second pass exercises hits and LRU eviction
        tree = _tree(seed, num_masks, bc, 64, pruning, device=cuda_decode.device)
        want_mask, want_pid = build_block_attention(tree, num_masks, cache_len, root, dtype, cuda_decode.device, min_value)
        got_mask, got_pid = cache.get(tree, num_masks, cache_len, root, dtype, cuda_decode.device, min_value)
        assert got_mask.device == got_pid.device == want_mask.device == want_pid.device == cuda_decode.device
        if str(got_mask.device) not in cuda_decode.attention_tensor_devices:
            cuda_decode.attention_tensor_devices.append(str(got_mask.device))
        assert got_mask.dtype == want_mask.dtype and torch.equal(got_mask, want_mask)
        assert got_pid.dtype == want_pid.dtype and torch.equal(got_pid, want_pid)
    assert len(cache._entries) <= 3 and cache.hits + cache.misses == 2 * len(seeds)


def test_returned_tensors_do_not_alias_the_cache(cuda_decode, record_property):
    cache = DynamicTreeAttentionCache()
    tree = _tree(1, 2, 30, 64, True, device=cuda_decode.device)
    mask, pid = cache.get(tree, 2, 4, 10, torch.float32, cuda_decode.device, -1e9)
    assert mask.device == cuda_decode.device
    record_property('attention_tensor_device', str(mask.device))
    mask.fill_(7.0)
    pid.fill_(7)
    again_mask, again_pid = cache.get(tree, 2, 4, 10, torch.float32, cuda_decode.device, -1e9)
    want_mask, want_pid = build_block_attention(tree, 2, 4, 10, torch.float32, cuda_decode.device, -1e9)
    assert torch.equal(again_mask, want_mask) and torch.equal(again_pid, want_pid)
    assert cache.hits == 1 and cache.misses == 1


def test_max_entries_must_be_at_least_one():
    with pytest.raises(ValueError, match='max_entries'):
        DynamicTreeAttentionCache(max_entries=0)
    with pytest.raises(ValueError, match='max_entries'):
        DynamicTreeAttentionCache(max_entries=-1)
    assert DynamicTreeAttentionCache(max_entries=1).max_entries == 1


def test_static_efficient_cache_matches_naive_builder(cuda_decode, record_property):
    """TreeAttentionCache (Appendix E) stays bit-identical after sharing _block_structure."""
    from spec_mtp.structure.algorithm.decode.tree import build_static_tree
    from spec_mtp.structure.algorithm.decode.tree_attention import TreeAttentionCache

    g = torch.Generator(device=cuda_decode.device).manual_seed(2)
    probs = torch.softmax(torch.randn(2, 48, generator=g, device=cuda_decode.device), dim=-1)
    tree = build_static_tree(probs, [7, 2], root_token=3, pruning=True)
    cache = TreeAttentionCache()
    for cache_len, root in ((0, 5), (12, 40), (3, 0)):
        want_m, want_p = build_block_attention(tree, 2, cache_len, root, torch.float32, cuda_decode.device, -1e4)
        got_m, got_p = cache.get(tree, 2, cache_len, root, torch.float32, cuda_decode.device, -1e4)
        assert torch.equal(got_m, want_m) and torch.equal(got_p, want_p)
    assert got_m.device == got_p.device == cuda_decode.device
    record_property('attention_tensor_device', str(got_m.device))


def test_static_cache_rebuilds_when_dtype_or_min_value_changes(cuda_decode, record_property):
    from spec_mtp.structure.algorithm.decode.tree import build_static_tree
    from spec_mtp.structure.algorithm.decode.tree_attention import TreeAttentionCache

    g = torch.Generator(device=cuda_decode.device).manual_seed(0)
    tree = build_static_tree(torch.softmax(torch.randn(2, 32, generator=g, device=cuda_decode.device), dim=-1), [4], root_token=1, pruning=False)
    cache = TreeAttentionCache()
    initial, _ = cache.get(tree, 2, 3, 10, torch.float32, cuda_decode.device, torch.finfo(torch.float32).min)
    assert initial.device == cuda_decode.device
    record_property('attention_tensor_device', str(initial.device))
    for dtype in (torch.bfloat16, torch.float32):
        min_value = torch.finfo(dtype).min
        got, _ = cache.get(tree, 2, 3, 10, dtype, cuda_decode.device, min_value)
        want, _ = build_block_attention(tree, 2, 3, 10, dtype, cuda_decode.device, min_value)
        assert got.dtype == dtype and torch.equal(got, want)
