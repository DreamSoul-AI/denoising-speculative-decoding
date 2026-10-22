"""Draft topology, owner mappings and attention; no model state."""


from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DraftTree:
    """Flattened draft tree. Block index 0 is the root (last generated token);
    node j (1-indexed by position in `tokens`) sits at block index j."""

    root_token: int
    tokens: list[int] = field(default_factory=list)       # node tokens
    depths: list[int] = field(default_factory=list)       # node depths, >= 1
    parents: list[int] = field(default_factory=list)      # parent block index
    probs: list[float] = field(default_factory=list)      # cumulative path prob

    @property
    def num_nodes(self):
        return len(self.tokens)

    def block_size(self, num_masks):
        return (1 + self.num_nodes) * (1 + num_masks)

    def block_complexity(self, num_masks):
        """Eq (7): (k+1)(1 + sum_i K_i)."""
        return self.block_size(num_masks)

    def children_of(self, block_index):
        return [j + 1 for j in range(self.num_nodes) if self.parents[j] == block_index]

    def branch_widths(self):
        if not self.depths:
            return []
        widths = [0] * max(self.depths)
        for d in self.depths:
            widths[d - 1] += 1
        return widths


def node_budget_for(block_complexity, num_masks):
    """Invert Eq (7): total draft nodes sum K_i for a given BC and k masks.

    The paper's grids keep BC divisible by (k+1); when it is not (e.g. the
    Table 3 three-mask row), we floor and the caller logs the actual BC."""
    budget = block_complexity // (num_masks + 1) - 1
    if budget < 1:
        raise ValueError(
            f'block_complexity={block_complexity} too small for {num_masks} masks')
    return budget


def _topk_with_pruning(probs, width, banned_token, taken_tokens):
    """Top-`width` tokens of one mask distribution, replacing the banned
    (parent-repeat) token and any already-taken token with the next best.

    Candidates from a single distribution are distinct by construction, so
    pulling width + |extras| top tokens always suffices."""
    extra = 2 + len(taken_tokens)
    k = min(width + extra, probs.size(-1))
    top = torch.topk(probs, k=k)
    picked_tokens, picked_probs = [], []
    for token, p in zip(top.indices.tolist(), top.values.tolist()):
        if token == banned_token or token in taken_tokens:
            continue
        picked_tokens.append(token)
        picked_probs.append(p)
        if len(picked_tokens) >= width:
            break
    return picked_tokens, picked_probs


def build_static_tree(mask_probs, branches, root_token, pruning=True):
    """Fixed [K_1..K_k] tree (Table 2 static configs).

    mask_probs: [k, V] — softmax of the k mask-token logits from the previous
    forward pass. branches: widths per depth; len(branches) <= k."""
    if len(branches) > mask_probs.size(0):
        raise ValueError('more branch widths than mask tokens')
    tree = DraftTree(root_token=int(root_token))
    parent_block_idx = 0            # root
    parent_token = int(root_token)
    parent_prob = 1.0
    for depth, width in enumerate(branches, start=1):
        if width <= 0:
            break
        probs = mask_probs[depth - 1]
        banned = parent_token if pruning else -1
        tokens, token_probs = _topk_with_pruning(probs, width, banned, set())
        top1_block_idx: Any
        top1_token: Any
        top1_block_idx, top1_prob, top1_token = None, -1.0, None
        for token, p in zip(tokens, token_probs):
            tree.tokens.append(token)
            tree.depths.append(depth)
            tree.parents.append(parent_block_idx)
            tree.probs.append(parent_prob * p)
            block_idx = tree.num_nodes  # 1-indexed block position of this node
            if p > top1_prob:
                top1_block_idx, top1_prob, top1_token = block_idx, p, token
        # Top-1 expansion: only the best node at this depth grows children.
        parent_block_idx = top1_block_idx
        parent_token = top1_token
        parent_prob = parent_prob * top1_prob
    return tree


def build_dynamic_tree(mask_probs, node_budget, root_token, pruning=True):
    """Algorithm 1 under Top-1 expansion.

    Per depth i the candidate pool is the Top-(budget) tokens of mask i's
    distribution chained below the Top-1 node of depth i-1 (multiplicative
    cumulative probability). The final tree keeps the `node_budget` highest
    cumulative-probability candidates; ancestor closure is automatic because
    each expanded parent is the Top-1 (hence highest-scoring) token of its
    depth. Widths [K_1..K_k] therefore adapt to model confidence."""
    num_masks = mask_probs.size(0)
    candidates = []   # (cum_prob, depth, token, order_within_depth)
    parent_token = int(root_token)
    parent_prob = 1.0
    per_depth_tokens = {}
    for depth in range(1, num_masks + 1):
        probs = mask_probs[depth - 1]
        banned = parent_token if pruning else -1
        tokens, token_probs = _topk_with_pruning(probs, node_budget, banned, set())
        if not tokens:
            break
        per_depth_tokens[depth] = list(zip(tokens, token_probs))
        for rank, (token, p) in enumerate(per_depth_tokens[depth]):
            candidates.append((parent_prob * p, depth, token, rank))
        # Chain below this depth's Top-1 (rank 0) token.
        parent_token = tokens[0]
        parent_prob = parent_prob * token_probs[0]

    # Rank all trajectories by cumulative probability; keep the budget best.
    # Tie-break by (depth, rank) so selection is deterministic.
    candidates.sort(key=lambda c: (-c[0], c[1], c[3]))
    selected = candidates[:node_budget]

    # A depth-(i+1) candidate requires its parent (depth-i Top-1). Enforce
    # closure explicitly in case of pathological ties.
    selected_set = {(depth, rank) for _, depth, _, rank in selected}
    changed = True
    while changed:
        changed = False
        for cand in list(selected_set):
            depth, _ = cand
            if depth > 1 and (depth - 1, 0) not in selected_set:
                selected_set.discard(cand)
                changed = True
    selected = [c for c in selected if (c[1], c[3]) in selected_set]

    # Emit depth by depth so the flattened block layout is BFS-ordered.
    tree = DraftTree(root_token=int(root_token))
    expanded_block_idx = {0: 0}  # depth -> block index of that depth's Top-1
    cum_prob_of_expanded = {0: 1.0}
    for depth in sorted({c[1] for c in selected}):
        depth_cands = sorted([c for c in selected if c[1] == depth], key=lambda c: c[3])
        parent_block_idx = expanded_block_idx[depth - 1]
        for cum_prob, _, token, rank in depth_cands:
            tree.tokens.append(token)
            tree.depths.append(depth)
            tree.parents.append(parent_block_idx)
            tree.probs.append(cum_prob)
            if rank == 0:
                expanded_block_idx[depth] = tree.num_nodes
                cum_prob_of_expanded[depth] = cum_prob
    return tree


def build_tree(spec_cfg, mask_probs, root_token):
    """Dispatch on spec.tree config; returns a DraftTree."""
    tree_cfg = spec_cfg.get('tree', {})
    mode = str(tree_cfg.get('mode', 'dynamic')).lower()
    pruning = bool(tree_cfg.get('pruning', True))
    num_masks = int(spec_cfg.get('num_masks', 1))
    if mode == 'static':
        branches = [int(b) for b in tree_cfg['branches']]
        return build_static_tree(mask_probs, branches, root_token, pruning=pruning)
    if mode == 'dynamic':
        budget = node_budget_for(int(spec_cfg['block_complexity']), num_masks)
        return build_dynamic_tree(mask_probs, budget, root_token, pruning=pruning)
    raise ValueError(f'Unsupported tree.mode: {mode}')




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
