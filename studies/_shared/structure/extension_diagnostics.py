"""Library helpers folded from one-shot structure scripts."""
def choose_window(zero, one):
    a, b = ({t['root_position']: t for t in trace} for trace in (zero, one))
    common = sorted(set(a) & set(b))
    root, reason = (None, 'no continuing common root')
    for i, hit in enumerate(common):
        if i and a[hit]['accepted_nodes'] and b[hit]['accepted_nodes']:
            previous = common[i - 1]
            if not a[previous]['accepted_nodes'] and (not b[previous]['accepted_nodes']):
                root, reason = (previous, 'before earliest eligible shared hit')
                break
    if root is None:
        for position in common:
            if not a[position]['stopped'] and (not b[position]['stopped']) and (not a[position]['accepted_nodes']) and (not b[position]['accepted_nodes']):
                root, reason = (position, 'earliest continuing common rejection')
                break
    if root is None:
        for position in common:
            if not a[position]['stopped'] and (not b[position]['stopped']):
                root, reason = (position, 'unused-owner control')
                break
    return {'intervention_root': root, 'selection_reason': reason, 'targets': [r for r in sorted(set(a) | set(b)) if root is not None and r >= root]}

def frozen_targets(selection):
    entry = selection['intervention_root']
    later = [p for p in selection['targets'] if p > entry]
    return sorted(set([entry, *later[:3], *later[-2:]]))
