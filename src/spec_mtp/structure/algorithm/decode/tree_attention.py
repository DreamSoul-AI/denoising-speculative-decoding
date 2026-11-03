"""Block attention and position ids; no model state."""

from typing import Any

import torch


def block_layout(tree: Any, num_masks: int) -> tuple[list[set[int]], list[int], list[int], int]:
    """Returns (attend_sets, rel_positions, owner_of_mask, block_size).

    attend_sets[i] is the set of *block-local* indices row i attends to.
    rel_positions[i] is the position offset from the root position p0."""
    num_nodes = tree.num_nodes
    block = 1 + num_nodes
    # Attend sets and relative positions for root + nodes.
    attend: list[set[int]] = [set() for _ in range(block)]
    rel = [0] * block
    attend[0] = {0}
    for j in range(num_nodes):
        idx = 1 + j
        parent = tree.parents[j]
        attend[idx] = set(attend[parent]) | {idx}
        rel[idx] = tree.depths[j]

    # Masks, grouped by owner (root first, then nodes in block order).
    owner_of_mask: list[int] = []
    for owner in range(block):
        base = block + len(owner_of_mask)
        for j in range(num_masks):
            mask_idx = base + j
            attend.append(set(attend[owner]) | set(range(base, mask_idx + 1)))
            rel.append(rel[owner] + j + 1)
            owner_of_mask.append(owner)
    return attend, rel, owner_of_mask, len(attend)


def mask_indices_of(tree: Any, num_masks: int, owner_block_idx: int) -> list[int]:
    """Block indices of the k masks owned by `owner_block_idx`."""
    block = 1 + tree.num_nodes
    base = block + owner_block_idx * num_masks
    return list(range(base, base + num_masks))


def _block_structure(tree: Any, num_masks: int, dtype: Any, device: Any, min_value: float) -> tuple[Any, Any]:
    """Block-local additive mask [1,1,B,B] and relative PIDs [1,B].

    Same values as filling with a Python loop over ``block_layout`` attend sets;
    one indexed write. Used by both the static efficient cache and the dynamic
    memoised cache so the two stay bit-identical to each other and to the naive
    builder.
    """
    attend, rel, _, size = block_layout(tree, num_masks)
    rows = [i for i, cols in enumerate(attend) for _ in cols]
    cols = [j for cols_i in attend for j in cols_i]
    structure = torch.full((1, 1, size, size), min_value, dtype=dtype, device=device)
    if rows:
        structure[0, 0, torch.tensor(rows, dtype=torch.long, device=device),
                  torch.tensor(cols, dtype=torch.long, device=device)] = 0.0
    return structure, torch.tensor([rel], dtype=torch.long, device=device)



def build_block_attention(tree: Any, num_masks: int, cache_len: int, root_position: int, dtype: Any, device: Any,
                          min_value: float) -> tuple[Any, Any]:
    """Naive path: full 4D additive mask + PIDs from the tree structure."""
    attend, rel, _, size = block_layout(tree, num_masks)
    mask = torch.full((1, 1, size, cache_len + size), min_value, dtype=dtype, device=device)
    mask[..., :cache_len] = 0.0
    for i, cols in enumerate(attend):
        for j in cols:
            mask[0, 0, i, cache_len + j] = 0.0
    position_ids = torch.tensor(
        [[root_position + r for r in rel]], dtype=torch.long, device=device)
    return mask, position_ids


class TreeAttentionCache:
    """Efficient path (Appendix E): cache the block-local structure once;
    per step prepend attend-columns for the new cache length and shift PIDs.

    Valid only while the tree structure is unchanged, i.e. static branch
    configurations. Callers must re-`prepare` if the structure changes
    (e.g. the truncated final step)."""

    def __init__(self) -> None:
        self._structure: Any = None   # [1, 1, B, B] additive block-local mask
        self._rel: Any = None         # [1, B] relative PIDs
        self._signature: Any = None

    def _signature_key(self, tree: Any, num_masks: int, dtype: Any, device: Any, min_value: float) -> tuple:
        # dtype / device / min_value must be part of the key: a float32 structure
        # reused for a bf16 step would keep the wrong fill value (caught 2026-10-12).
        return (tuple(tree.depths), tuple(tree.parents), num_masks, dtype, str(device), float(min_value))

    def prepare(self, tree: Any, num_masks: int, dtype: Any, device: Any, min_value: float) -> None:
        self._structure, self._rel = _block_structure(tree, num_masks, dtype, device, min_value)
        self._signature = self._signature_key(tree, num_masks, dtype, device, min_value)

    def get(self, tree: Any, num_masks: int, cache_len: int, root_position: int, dtype: Any, device: Any, min_value: float) -> tuple[Any, Any]:
        if self._signature != self._signature_key(tree, num_masks, dtype, device, min_value):
            # First step, or tree / dtype / device / min_value changed: (re)build once.
            self.prepare(tree, num_masks, dtype, device, min_value)
        size = self._structure.size(-1)
        mask = torch.empty((1, 1, size, cache_len + size), dtype=dtype, device=device)
        mask[..., :cache_len] = 0.0                      # prepended attend columns
        mask[..., cache_len:] = self._structure          # unchanged block structure
        position_ids = self._rel + root_position         # uniform PID shift
        return mask, position_ids


class DynamicTreeAttentionCache:
    """Dynamic trees (``impl: naive``, ``tree.mode: dynamic``): same mask and PIDs as
    `build_block_attention`, built faster.

    Dynamic trees change shape between steps but repeat a small set of shapes,
    so the block-local structure is memoised per tree signature (LRU) and
    filled with one indexed write instead of one Python write per entry. Per
    step only the attend columns for the current cache are prepended and the
    PIDs shifted, as in `TreeAttentionCache`. The result is bit-identical to
    the naive builder (same dtype, same 0 / ``min_value`` entries, same PIDs);
    ``tests/.../test_tree_attention_cache.py`` checks it on random trees.

    Static trees with ``impl: naive`` keep `build_block_attention`, so the
    paper's naive vs efficient ablation (Table 4) still times the naive path.
    """

    def __init__(self, max_entries: int = 256) -> None:
        max_entries = int(max_entries)
        if max_entries < 1:
            raise ValueError(f'DynamicTreeAttentionCache.max_entries must be >= 1, got {max_entries}')
        self.max_entries = max_entries
        self._entries: dict[Any, tuple[Any, Any]] = {}  # signature -> (structure, rel); dict order = LRU
        self.hits = 0
        self.misses = 0

    def _structure(self, tree: Any, num_masks: int, dtype: Any, device: Any, min_value: float) -> tuple[Any, Any]:
        key = (tuple(tree.depths), tuple(tree.parents), num_masks, dtype, str(device), float(min_value))
        entry = self._entries.pop(key, None)
        if entry is None:
            self.misses += 1
            entry = _block_structure(tree, num_masks, dtype, device, min_value)
            if len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))
        else:
            self.hits += 1
        self._entries[key] = entry
        return entry

    def get(self, tree: Any, num_masks: int, cache_len: int, root_position: int, dtype: Any, device: Any, min_value: float) -> tuple[Any, Any]:
        structure, rel = self._structure(tree, num_masks, dtype, device, min_value)
        size = structure.size(-1)
        mask = torch.empty((1, 1, size, cache_len + size), dtype=dtype, device=device)
        mask[..., :cache_len] = 0.0
        mask[..., cache_len:] = structure
        return mask, rel + root_position
