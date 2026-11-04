"""Library helpers folded from one-shot structure scripts."""

def _load_theme(name):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        return importlib.import_module("studies._shared.structure." + name)
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

def count_repeats(observations):
    calls = accepted = repeats = nonterminal = prompts = 0
    for observation in observations:
        found = False
        for step in observation['stats']['trace']:
            calls += 1
            accepted += len(step['accepted_nodes'])
            if step['committed_tokens'][0] == observation['output_ids'][step['root_position']]:
                assert not step['accepted_nodes'], 'pruned single-mask tree cannot accept a parent repeat'
                repeats += 1
                found = True
                nonterminal += not step['stopped']
        prompts += found
    assert calls > 0
    return dict(calls=calls, accepted=accepted, repeat_root_fallbacks=repeats, nonterminal_repeat_root_fallbacks=nonterminal, prompts_with_repeat_root_fallback=prompts, repeat_fraction=repeats / calls, optimistic_extra_one_same_calls=nonterminal / calls)
MT_CATEGORIES = {'coding', 'extraction', 'humanities', 'math', 'reasoning', 'roleplay', 'stem', 'writing'}
FILES = {'eight_b': 'studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json', 'five_methods': 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json', 'directions': 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json'}

def summarize(rows, target=None):
    assert rows and len({r['question_id'] for r in rows}) == len(rows)
    totals = {k: sum((r[k] for r in rows)) for k in ('committed', 'decode_calls', 'accepted', 'new_tokens')}
    committed, calls, accepted, new = (totals[k] for k in ('committed', 'decode_calls', 'accepted', 'new_tokens'))
    assert calls > 0 and new == committed + len(rows)
    assert all((r['decode_calls'] > 0 and r['accepted'] + r['decode_calls'] >= r['committed'] for r in rows))
    clipped = accepted + calls - committed
    result = dict(n=len(rows), **totals, pooled_tau=committed / calls, prompt_mean_tau=sum((r['committed'] / r['decode_calls'] for r in rows)) / len(rows), accepted_per_decode_call=accepted / calls, end_clipped_tokens=clipped, unclipped_accounting_tau=(accepted + calls) / calls, clipping_tau_delta=clipped / calls, including_prefill_tau=new / (calls + len(rows)), output_cap_256_count=sum((r['new_tokens'] == 256 for r in rows)), shorter_than_256_count=sum((r['new_tokens'] < 256 for r in rows)), output_tokens_mean=new / len(rows), argmax_checked_tokens=sum((r['argmax_checked_tokens'] for r in rows)))
    assert result['argmax_checked_tokens'] == new
    if target is not None:
        result.update(target=target, pooled_delta_to_target=committed / calls - target, relative_shortfall=(target - committed / calls) / target, extra_committed_at_same_calls=target * calls - committed)
    return result

def paired(first, second):
    assert [r['question_id'] for r in first] == [r['question_id'] for r in second]
    result = {key: dict(win=0, tie=0, loss=0) for key in ('accepted', 'calls', 'tau')}
    for a, b in zip(first, second):
        changes = {'accepted': b['accepted'] - a['accepted'], 'calls': a['model_calls'] - b['model_calls'], 'tau': b['committed'] * a['decode_calls'] - a['committed'] * b['decode_calls']}
        for key, delta in changes.items():
            result[key]['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    result['delta_second_minus_first'] = {k: sum((r[k] for r in second)) - sum((r[k] for r in first)) for k in ('accepted', 'model_calls', 'new_tokens')}
    return result

def group_summary(rows, key, target=None):
    groups = {value: [r for r in rows if key(r) == value] for value in sorted({key(r) for r in rows})}
    result = {value: summarize(rs, target) for value, rs in groups.items()}
    if target is not None:
        calls = sum((r['decode_calls'] for r in rows))
        for entry in result.values():
            entry['decode_call_weight'] = entry['decode_calls'] / calls
            entry['uniform_target_gap_contribution'] = entry['extra_committed_at_same_calls'] / calls
    return result
import importlib.util
import json
VENDOR = ROOT / 'src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts'

def load_original():
    _folded_geometry_4b = _load_theme('geometry_4b')
    sha = _folded_geometry_4b.sha
    origin = json.loads((VENDOR / 'ORIGIN.json').read_text(encoding='utf-8'))
    assert origin['commit'] == 'ad76f8da29f3bfc7faa20488193c6aa83a9f708c' and (not origin['author_official'])
    for entry in origin['files']:
        assert sha(VENDOR / 'upstream' / entry['path']) == entry['sha256']
    spec = importlib.util.spec_from_file_location('esp_original_v6', VENDOR / 'upstream/code/esp_v6.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module, origin)

def audit_step(logits, prompt_length, draft_ids, result, remaining):
    """Check the positions preceding the emitted prefix, rather than draft-self/mask rows."""
    n_acc, acc_idx, bonus, _ = result
    assert prompt_length > 0 and logits.ndim == 2 and (remaining > 0)
    assert logits.shape[0] == prompt_length + len(draft_ids) + 1
    assert n_acc >= 0 and (n_acc == 0 or 0 <= acc_idx < len(draft_ids))
    proposed = (draft_ids[:acc_idx + 1] if n_acc else []) + [bonus]
    records = []
    for offset, token in enumerate(proposed[:remaining]):
        row = prompt_length - 1 + offset
        expected = int(logits[row].argmax())
        records.append({'offset': offset, 'verifier_row': row, 'chosen': int(token), 'argmax': expected, 'passed': int(token) == expected, 'argmax_minus_chosen_logit': float(logits[row, expected] - logits[row, int(token)])})
    return records

class ArgmaxRejected(Exception):
    pass
import math
STUDY = ROOT / 'studies/esp_table1_bf16_full_argmax'

def accounting(rows):
    """Exact weighting identities; contributions are accounting, not causes."""
    n = len(rows)
    calls = sum((r['decode_calls'] for r in rows))
    assert n and calls > 0 and all((r['decode_calls'] > 0 for r in rows))
    ratios = [r['committed'] / r['decode_calls'] for r in rows]
    pooled = sum((r['committed'] for r in rows)) / calls
    mean = sum(ratios) / n
    covariance_term = sum(((r['decode_calls'] - calls / n) * (t - mean) for r, t in zip(rows, ratios))) / calls
    within = between = category_mean = 0.0
    categories = sorted({r['category'] for r in rows})
    for category in categories:
        group = [r for r in rows if r['category'] == category]
        count = sum((r['decode_calls'] for r in group))
        tau = sum((r['committed'] for r in group)) / count
        prompt_mean = sum((r['committed'] / r['decode_calls'] for r in group)) / len(group)
        within += len(group) / n * (tau - prompt_mean)
        between += (count / calls - len(group) / n) * tau
        category_mean += tau / len(categories)
    assert math.isclose(pooled - mean, covariance_term, abs_tol=1e-12)
    assert math.isclose(pooled - mean, within + between, abs_tol=1e-12)
    return dict(pooled_tau=pooled, prompt_mean_tau=mean, category_mean_tau=category_mean, pooled_minus_prompt_mean=pooled - mean, covariance_over_mean_calls=covariance_term, within_category_call_weight_term=within, between_category_call_weight_term=between)

def prefix_accounting(block_sizes, limit=100):
    """Clip the observed path after its prefill token; not a new generation."""
    assert limit >= 1 and all((size > 0 for size in block_sizes))
    remaining = limit - 1
    committed = calls = partial = 0
    for size in block_sizes:
        if not remaining:
            break
        take = min(size, remaining)
        committed += take
        remaining -= take
        calls += 1
        partial += take < size
    return dict(committed=committed, decode_calls=calls, new_tokens=committed + 1, boundary_partial_blocks=partial)
