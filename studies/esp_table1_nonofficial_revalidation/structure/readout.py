"""Pure saved-repeat argmax reconciliation and paired count readout."""
from typing import Any
from studies.esp_table1_bf16_full_argmax.structure.readout import summarize as summarize
from studies._shared.structure.esp_controls import token_digest

TARGETS = {10: None, 30: 1.63, 60: 1.74}
METRICS = ('pooled_tau', 'prompt_mean_tau', 'category_mean_pooled_tau')


def first_difference(a, b):
    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def audited_rows(raw, audits, prompts, budget):
    assert len(raw['rows']) == len(raw['evidence']) == len(audits) == len(prompts)
    rows: list[dict[str, Any]] = []
    tokens = []
    for prompt, row, item, audit in zip(prompts, raw['rows'], raw['evidence'], audits):
        ids = prompt['input_ids']
        assert row['prompt'] == len(rows)
        assert item['input_ids'] == ids and item['output_ids'][:len(ids)] == item['ar_ids'][:len(ids)] == ids
        stats = item['stats']
        assert row['category'] == prompt['category'] and row['spec_time'] == row['ar_time'] == 0
        assert stats['prefill_calls'] == 1 and stats['block_complexity'] == budget
        for name in ('accepted', 'committed', 'decode_calls', 'new_tokens'):
            assert row[name] == stats[name]
        output, ar = item['output_ids'][len(ids):], item['ar_ids'][len(ids):]
        assert stats['new_tokens'] == len(output) and 0 < len(output) <= 256
        assert stats['committed'] == len(output) - 1
        assert audit == stats['verification_argmax_audit'] and audit['passed'] is True
        assert audit['checked_committed_tokens'] == len(audit['committed']) == len(output)
        assert audit['input_sha256'] == token_digest(ids) and audit['output_sha256'] == token_digest(item['output_ids'])
        assert all(a['chosen'] == a['argmax'] == token for a, token in zip(audit['committed'], output))
        exact = item['output_ids'] == item['ar_ids']
        assert bool(row['exact_match']) == exact
        first = first_difference(output, ar)
        rows.append(dict(row, question_id=prompt['question_id'], exact=exact, AR_new_tokens=len(ar),
            first_new_difference_index=first, argmax_passed=True, argmax_checked_tokens=len(output)))
        tokens.append({'question_id': prompt['question_id'], 'output_sha256': token_digest(item['output_ids']),
            'AR_sha256': token_digest(item['ar_ids']), 'first_new_difference_index': first})
    return rows, tokens


def compare(old, new):
    assert len(old['rows']) == len(new['rows']) == len(old['token_audits']) == len(new['token_audits'])
    rows = []
    for a, b, at, bt in zip(old['rows'], new['rows'], old['token_audits'], new['token_audits']):
        assert a['question_id'] == b['question_id'] == at['question_id'] == bt['question_id']
        assert a['category'] == b['category']
        change = {k: b[k] - a[k] for k in ('accepted', 'decode_calls', 'committed', 'new_tokens')}
        tau_cross = b['committed'] * max(a['decode_calls'], 1) - a['committed'] * max(b['decode_calls'], 1)
        label = lambda x: 'win' if x > 0 else 'loss' if x < 0 else 'tie'
        rows.append({'question_id': a['question_id'], 'category': a['category'],
            'delta_new_minus_old': change, 'accepted': label(change['accepted']),
            'calls': label(-change['decode_calls']), 'tau': label(tau_cross),
            'output_hash_equal': at['output_sha256'] == bt['output_sha256'],
            'AR_hash_equal': at['AR_sha256'] == bt['AR_sha256']})
    return {'n': len(rows), **{metric: {s: sum(r[metric] == s for r in rows) for s in ('win', 'tie', 'loss')}
            for metric in ('accepted', 'calls', 'tau')},
            'output_hash_equal_count': sum(r['output_hash_equal'] for r in rows),
            'AR_hash_equal_count': sum(r['AR_hash_equal'] for r in rows), 'rows': rows}


def verify_contract(inputs, previous, cfg):
    expected = dict(cfg['algorithm']['config']['eval'], device=cfg['system']['config']['device'])
    assert inputs['settings']['eval'] == previous['settings']['eval'] == expected
    assert inputs['settings']['spec'] == previous['settings']['spec'] == cfg['algorithm']['config']['spec']
    assert inputs['eos_ids'] == previous['eos_ids'] and inputs['eos_ids']
    assert inputs['model_digest'] == previous['model_digest']
