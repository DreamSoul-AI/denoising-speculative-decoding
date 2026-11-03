"""Draft topology and tree construction; no model state."""

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

