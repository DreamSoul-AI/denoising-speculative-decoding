"""Fixed hybrid-gain-bonus-aligned-v1 protocol, calibration and summary statistics."""

from __future__ import annotations


import torch

from spec_mtp.structure.algorithm.mask import calibrate_diagonal_gain


from studies.trajectory_observation.structure.output_hidden_kalman import (
    HEADS as HEADS, TAILS as TAILS, templates, sign_test, pooled_tau,
)


ARMS = ('esp', 'frozen_last', 'opened_last', 'opened_first')


def prompts_for(name, rows=None):
    if name == 'calibration':
        return [{'qid': f'template-{index}', 'text': text} for index, text in enumerate(templates()[:128])]
    if name != 'specbench':
        raise ValueError(name)
    if rows is None:
        raise ValueError('SpecBench rows must come from an explicitly bound local input')
    rows = [row for row in rows
            if isinstance(row['question_id'], int) and 465 <= row['question_id'] <= 560]
    rows.sort(key=lambda row: row['question_id'])
    if [row['question_id'] for row in rows] != list(range(465, 561)):
        raise RuntimeError('SpecBench 465-560 is not a contiguous local slice')
    return [{'qid': row['question_id'], 'text': row['turns'][0], 'category': row['category']} for row in rows]


def tree_spec(hybrid=None):
    spec = {'method': 'esp', 'num_masks': 1, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [4], 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True}}
    if hybrid is not None:
        spec['hybrid_gain'] = hybrid
    return spec


def arm_spec(arm, priors):
    if arm == 'esp':
        return tree_spec()
    if arm == 'frozen_last':
        return tree_spec({'mode': 'frozen', 'observe': 'last', **priors['last']})
    if arm == 'opened_last':
        return tree_spec({'mode': 'opened', 'observe': 'last', 'boost': 64, **priors['last']})
    if arm == 'opened_first':
        return tree_spec({'mode': 'opened', 'observe': 'first', 'boost': 64, **priors['first']})
    raise ValueError(arm)


def _cosine(left, right):
    num = (left * right).sum(dim=-1)
    den = left.norm(dim=-1) * right.norm(dim=-1)
    ok = den > 1e-12
    if not bool(ok.any()):
        return None
    return float((num[ok] / den[ok]).mean())


def describe_pairs(stored, device):
    m = torch.cat(stored['m']).to(device).float()
    first = torch.cat(stored['first']).to(device).float()
    last = torch.cat(stored['last']).to(device).float()
    differ = (first - last).abs().sum(dim=-1) > 0
    report = {'rows': int(m.shape[0]), 'rounds_first_ne_last': int(differ.sum()),
              'cosine_all': _cosine(first - m, last - m)}
    report['cosine_when_different'] = None if not bool(differ.any()) else _cosine(first[differ] - m[differ], last[differ] - m[differ])
    return report


def calibrate(stored, device):
    if stored['qid'] != [item['qid'] for item in prompts_for('calibration')]:
        raise RuntimeError('calibration pairs do not cover 128 prompts')
    m = torch.cat(stored['m']).to(device)
    zed = torch.cat(stored['z']).to(device)
    priors = {}
    for name, target in (('first', torch.cat(stored['first']).to(device)),
                         ('last', torch.cat(stored['last']).to(device))):
        prior = calibrate_diagonal_gain(m, zed, target)
        prior['k_mean'] = sum(prior['k']) / len(prior['k'])
        priors[name] = prior
    payload = {'first': priors['first'], 'last': priors['last'], 'pairs': describe_pairs(stored, device)}
    return payload


def mean_diag(rows, key):
    values = [row['diagnostics'][key] for row in rows if row.get('diagnostics') and row['diagnostics'].get(key) is not None]
    return None if not values else sum(values) / len(values)


def summarize(arms):
    expected = 96
    incomplete = None
    for arm in ARMS:
        rows = arms.get(arm, [])
        if len(rows) != expected:
            incomplete = {'complete': False, 'arm': arm, 'rows': len(rows), 'expected': expected}
            break
    if incomplete:
        summary = incomplete
    else:
        summary = {'complete': True, 'tau': {arm: pooled_tau(rows) for arm, rows in arms.items()},
                   'opened_last_versus_esp': sign_test(arms['opened_last'], arms['esp']),
                   'frozen_last_versus_esp': sign_test(arms['frozen_last'], arms['esp']),
                   'opened_first_versus_esp': sign_test(arms['opened_first'], arms['esp']),
                   'opened_last_versus_opened_first': sign_test(arms['opened_last'], arms['opened_first']),
                   'weight_mean': {arm: mean_diag(rows, 'weight_mean') for arm, rows in arms.items()},
                   'k_abs_delta': {arm: mean_diag(rows, 'k_abs_delta') for arm, rows in arms.items()},
                   'k_mean': {arm: mean_diag(rows, 'k_mean') for arm, rows in arms.items()}}
    return summary
