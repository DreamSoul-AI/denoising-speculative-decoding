"""Budget-preserving depth-1 node edits; no model state."""

def _depth1_indices(tree):
    return [index for index, parent in enumerate(tree.parents)
            if parent == 0 and tree.depths[index] == 1]


def add_depth1_branch(tree, token, pruning=True):
    """Append one depth-1 sibling. Existing children stay.

    Returns False when pruning rejects the parent token or the token is already
    a depth-1 child. The added node's probability is unused by verification.
    """
    token = int(token)
    if pruning and token == int(tree.root_token):
        return False
    if any(tree.tokens[index] == token for index in _depth1_indices(tree)):
        return False
    tree.tokens.append(token)
    tree.depths.append(1)
    tree.parents.append(0)
    tree.probs.append(0.0)
    return True


def replace_weakest_depth1(tree, token, pruning=True):
    """Replace the weakest depth-1 leaf with token. The node count stays put.

    Weakest means the smallest cumulative probability, then the later sibling.
    A depth-1 node that already has children is left in place. Returns False
    when pruning rejects the root token, the token is already a depth-1 child,
    or no childless depth-1 node can be swapped.
    """
    token = int(token)
    if pruning and token == int(tree.root_token):
        return False
    depth1 = _depth1_indices(tree)
    if any(tree.tokens[index] == token for index in depth1):
        return False
    parents = set(tree.parents)
    leaves = [index for index in depth1 if (index + 1) not in parents]
    if not leaves:
        return False
    weakest = min(leaves, key=lambda index: (tree.probs[index], -index))
    tree.tokens[weakest] = token
    return True

