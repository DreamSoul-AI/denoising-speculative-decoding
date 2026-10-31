"""Count protocols and paired comparisons. Timing ratios stay in separate fields.

No execution or file IO. ``tau`` is committed tokens over decode calls.
Zero decode calls have no ratio. Wall-clock ratios are observations, not a
claim that the run is faster.
"""

def prompt_tau(committed, decode_calls):
    """Single tau formula. Zero decode calls are missing, not zero."""
    if type(committed) is not int or committed < 0 or type(decode_calls) is not int or decode_calls < 0:
        raise ValueError('tau counts must be nonnegative integers')
    if decode_calls == 0:
        return None
    return committed / decode_calls


def round_split(trace):
    """Decode rounds grouped by whether the previous round accepted a draft.

    The first decode round follows prefill and is its own group. Each group
    keeps rounds, accepted drafts and committed tokens; tau is per group.
    """
    groups = {name: {'rounds': 0, 'accepted': 0, 'committed': 0}
              for name in ('after_prefill', 'after_accept', 'after_reject')}
    previous = None
    for row in trace:
        accepted, committed = len(row['accepted_nodes']), len(row['committed_tokens'])
        if committed < 1 or committed > accepted + 1:
            raise ValueError('round must commit accepted drafts plus at most one bonus')
        name = 'after_prefill' if previous is None else 'after_accept' if previous else 'after_reject'
        group = groups[name]
        group['rounds'] += 1
        group['accepted'] += accepted
        group['committed'] += committed
        previous = accepted > 0
    for group in groups.values():
        group['tau'] = prompt_tau(group['committed'], group['rounds'])
    return groups


def _counts(row):
    for key in ('accepted', 'committed', 'decode_calls', 'new_tokens'):
        value = row[key]
        if type(value) is not int or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if row['committed'] > row['new_tokens'] or row['accepted'] < 0:
        raise ValueError('inconsistent committed/generated counts')


def summarize(records, protocol):
    """Pooled tau excludes prefill, exactly as declared by the protocol."""
    if protocol.get('tau') != 'decode_committed/decode_calls':
        raise ValueError('unknown tau protocol')
    rows = list(records)
    qids = [row['qid'] for row in rows]
    if len(set(qids)) != len(qids):
        raise ValueError('duplicate prompt identity')
    for row in rows:
        _counts(row)
    expected = tuple(protocol.get('qids', qids))
    missing = [qid for qid in expected if qid not in qids]
    unexpected = [qid for qid in qids if qid not in expected]
    counts = totals(rows)
    tau = prompt_tau(counts['committed'], counts['decode_calls'])
    return {'protocol': dict(protocol), 'prompts': len(rows), 'counts': counts, 'tau': tau,
            'missing': missing, 'unexpected': unexpected,
            'complete': not missing and not unexpected and all(row.get('evidence_complete') is True for row in rows)}


def compare(left, right, pairing):
    """Keep mismatched/missing prompts visible; never select only wins."""
    def indexed(records):
        out = {}
        for row in records:
            _counts(row)
            if row['qid'] in out:
                raise ValueError('duplicate prompt identity')
            out[row['qid']] = row
        return out
    lhs, rhs = indexed(left), indexed(right)
    pairs = []
    for qid in sorted(lhs.keys() | rhs.keys()):
        a, b = lhs.get(qid), rhs.get(qid)
        reasons = []
        if a is None or b is None:
            reasons.append('missing prompt')
        else:
            if a.get('evidence_complete') is not True or b.get('evidence_complete') is not True:
                reasons.append('incomplete evidence')
            for key in dict.fromkeys(('input_sha256', 'protocol', 'new_tokens', *pairing.get('keys', ()))):
                if key not in a or key not in b or a[key] != b[key]:
                    reasons.append(f'{key} mismatch')
        delta = None if reasons else a['accepted'] - b['accepted']
        pairs.append({'qid': qid, 'comparable': not reasons, 'reasons': reasons, 'accepted_delta': delta})
    return {'pairs': pairs, 'wins': sum(p['accepted_delta'] > 0 for p in pairs if p['comparable']),
            'ties': sum(p['accepted_delta'] == 0 for p in pairs if p['comparable']),
            'losses': sum(p['accepted_delta'] < 0 for p in pairs if p['comparable']),
            'incomparable': sum(not p['comparable'] for p in pairs)}

def _count_summary(group):
    """Acceptance accounting only. Depth rates describe attempts, not speed."""
    committed = sum(r['committed'] for r in group)
    decode_calls = sum(r['decode_calls'] for r in group)
    new_tokens = sum(r['new_tokens'] for r in group)
    tau = prompt_tau(committed, decode_calls)
    summary = {
        'prompts': len(group),
        'tau': tau,
        'call_reduction': 1.0 - 1.0 / tau if tau is not None and tau > 0 else None,
        'new_tokens_per_prompt': new_tokens / len(group),
    }
    for depth in (1, 2, 3):
        observed = all(f'attempt_d{depth}' in r and f'accept_d{depth}' in r for r in group)
        attempts = sum(int(r[f'attempt_d{depth}']) for r in group) if observed else None
        accepts = sum(int(r[f'accept_d{depth}']) for r in group) if observed else None
        summary[f'attempt_d{depth}'] = attempts
        summary[f'accept_d{depth}'] = accepts
        summary[f'accept_rate_d{depth}'] = accepts / attempts if attempts and accepts is not None else None
    return summary


def _real_number(value):
    return type(value) in (int, float) and value == value and value not in (float('inf'), float('-inf'))


def _timing_summary(group):
    """Wall-clock observations. Missing timing is not a zero-duration run."""
    times = [r.get('spec_time') for r in group]
    complete_spec = all(_real_number(value) and value >= 0 for value in times)
    spec_time = sum(times) if complete_spec else None
    new_tokens = sum(r['new_tokens'] for r in group)
    if not complete_spec:
        summary = {'tokens_per_s': None}
    else:
        summary = {'tokens_per_s': new_tokens / spec_time if spec_time > 0 else 0.0}
    if any('ar_time' in r for r in group):
        complete_ar = all(
            _real_number(r.get('ar_time')) and r.get('ar_time') >= 0 and _real_number(r.get('exact_match'))
            for r in group)
        summary['AR_observed_prompts'] = sum(
            _real_number(r.get('ar_time')) and _real_number(r.get('exact_match')) for r in group)
        summary['AR_complete'] = complete_ar
        ar_time = sum(r['ar_time'] for r in group) if complete_ar else None
        summary['speedup'] = ar_time / spec_time if complete_ar and spec_time else None
        summary['exact_match_rate'] = sum(r['exact_match'] for r in group) / len(group) if complete_ar else None
    return summary


def _aggregate(rows):
    """Per-category and overall view. Counts and timing stay in separate builders."""
    groups: dict = {}
    for row in rows:
        for key in dict.fromkeys((row['category'], 'overall')):
            groups.setdefault(key, []).append(row)
    out = {}
    for key, group in sorted(groups.items()):
        out[key] = {**_count_summary(group), **_timing_summary(group)}
    return out

def totals(rows):
    """Pooled counts over all prompts; `accepted` is what the reports compare."""
    keys = ('accepted', 'committed', 'decode_calls', 'new_tokens')
    return {key: int(sum(int(r[key]) for r in rows)) for key in keys}
