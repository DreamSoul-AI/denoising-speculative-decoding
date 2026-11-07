"""Pure fixed four-cell count metrics and verifier-argmax quality."""
from typing import Any

CELLS = ((10, 1, 'primary', None), (30, 1, 'primary', 1.63), (60, 1, 'primary', 1.74), (60, 2, 'control', 1.74))


def count_summary(rows):
    assert rows
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        assert row['decode_calls'] >= 0 and row['committed'] >= 0
        groups.setdefault(row['category'], []).append(row)
    def ratio(group):
        calls = sum(r['decode_calls'] for r in group)
        return sum(r['committed'] for r in group)/calls if calls else 0.
    return {'n':len(rows),'exact_count':sum(bool(r['exact']) for r in rows),
            'accepted':sum(r['accepted'] for r in rows),
            'main_calls':sum(r['decode_calls']+1 for r in rows),
            'pooled_tau':ratio(rows),'prompt_mean_tau':sum(r['committed']/max(r['decode_calls'],1) for r in rows)/len(rows),
            'category_mean_pooled_tau':sum(ratio(g) for g in groups.values())/len(groups)}


def summarize(rows):
    summary = count_summary(rows)
    return {**summary, 'AR_exact_rate': summary['exact_count'] / summary['n'],
            'argmax_checked_tokens': sum(row.get('argmax_checked_tokens', 0) for row in rows),
            'argmax_all_passed': all(row.get('argmax_passed') is True for row in rows)}


def ordered_cells(observed, *, declarations=CELLS):
    result=[]
    for bc,k,role,target in declarations:
        cell=dict(observed.get((bc,k),{}),BC=bc,num_masks=k,role=role,target=target)
        cell['pending']=(bc,k) not in observed
        for metric in ('pooled_tau','prompt_mean_tau','category_mean_pooled_tau'):
            value=cell.get(metric)
            cell[metric+'_target_delta']=value-target if value is not None and target is not None else None
        result.append(cell)
    return result
