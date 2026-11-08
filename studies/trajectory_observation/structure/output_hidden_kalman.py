"""Fixed output-hidden-hybrid-kalman-v1 protocol, calibration and summary statistics."""

from __future__ import annotations


import torch

from spec_mtp.structure.algorithm.mask import calibrate_diagonal_gain


HEADS = (
    'Name the color of', 'A compiler turns', 'The Pacific Ocean is', 'After the rain',
    'Please sort these', 'An electron has', 'The word for cat in', 'When the loop ends',
    'A rectangle with sides', 'The train leaves at', 'Binary 1010 equals', 'Salt dissolves in',
    'The author of Hamlet is', 'Click the button to', 'Gravity pulls objects', 'A dictionary maps',
)
TAILS = (
    'a common example', 'the next word', 'two equal parts', 'the empty page',
    'one short line', 'the left side', 'three small items', 'the final step',
    'a closed door', 'the first row', 'four quiet notes', 'the open window',
    'a single page', 'the right margin', 'two fresh copies', 'the last stop',
    'one plain box', 'the middle line',
)
ARMS = ('esp', 'output_hidden', 'trajectory', 'frozen', 'scratch', 'hybrid')


def templates():
    texts = [f'{head} {tail}' for head in HEADS for tail in TAILS]
    if len(texts) != 288 or TAILS[15] != 'the last stop':
        raise RuntimeError('template grid changed')
    return texts


def prompts_for(name, rows=None):
    if name == 'calibration':
        return [{'qid': f'template-{index}', 'text': text} for index, text in enumerate(templates()[:128])]
    if name == 'holdout':
        return [{'qid': f'template-{index}', 'text': text}
                for index, text in enumerate(templates()[128:160], start=128)]
    if name != 'specbench':
        raise ValueError(name)
    if rows is None:
        raise ValueError('SpecBench rows must come from an explicitly bound local input')
    rows = [row for row in rows
            if isinstance(row['question_id'], int) and 209 <= row['question_id'] <= 336]
    rows.sort(key=lambda row: row['question_id'])
    if [row['question_id'] for row in rows] != list(range(209, 337)):
        raise RuntimeError('SpecBench 209-336 is not a contiguous local slice')
    return [{'qid': row['question_id'], 'text': row['turns'][0], 'category': row['category']} for row in rows]


def tree_spec(method, output_kalman=None):
    spec = {'method': method, 'num_masks': 1, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [4], 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
            'output': {'order': 1, 'beta': 0.9, 'beta_prime': 0.99}}
    if output_kalman is not None:
        spec['method'] = 'output_hidden'
        spec['output_kalman'] = output_kalman
    return spec


def arm_spec(arm, prior):
    if arm == 'esp':
        return tree_spec('esp')
    if arm == 'output_hidden':
        return tree_spec('output_hidden')
    if arm == 'trajectory':
        return tree_spec('output_hidden', {'mode': 'trajectory'})
    if arm == 'scratch':
        return tree_spec('output_hidden', {'mode': 'scratch'})
    if arm == 'collect':
        return tree_spec('output_hidden', {'mode': 'collect'})
    payload = {'k': prior['k'], 'p': prior['p'], 'r': prior['r'], 'q': prior['q']}
    if arm == 'frozen':
        return tree_spec('output_hidden', {'mode': 'frozen', **payload})
    if arm == 'hybrid':
        return tree_spec('output_hidden', {'mode': 'hybrid', 'boost': 64, **payload})
    raise ValueError(arm)


def calibrate(stored, device):
    if stored['qid'] != [item['qid'] for item in prompts_for('calibration')]:
        raise RuntimeError('calibration pairs do not cover 128 prompts')
    prior = calibrate_diagonal_gain(*(torch.cat(stored[key]).to(device) for key in ('h', 'z', 'y')))
    prior['k_mean'] = sum(prior['k']) / len(prior['k'])
    return prior


def sign_test(left, right):
    from scipy.stats import binomtest
    paired = {row['qid']: row for row in left}
    wins = losses = ties = 0
    for row in right:
        delta = paired[row['qid']]['accepted'] - row['accepted']
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    tested = wins + losses
    pvalue = None if tested == 0 else float(binomtest(wins, tested, 0.5, alternative='two-sided').pvalue)
    return {'wins': wins, 'losses': losses, 'ties': ties, 'p': pvalue}


def pooled_tau(rows):
    calls = sum(row['decode_calls'] for row in rows)
    return None if calls == 0 else sum(row['committed'] for row in rows) / calls


def summarize(split_rows, model_name):
    summary = {'model': model_name}
    for split in ('specbench', 'holdout'):
        expected = 128 if split == 'specbench' else 32
        arms = {}
        incomplete = None
        for arm in ARMS:
            rows = split_rows.get(split, {}).get(arm, [])
            if len(rows) != expected:
                incomplete = {'complete': False, 'arm': arm, 'rows': len(rows), 'expected': expected}
                break
            arms[arm] = rows
        if incomplete:
            summary[split] = incomplete
            continue
        summary[split] = {'complete': True, 'tau': {arm: pooled_tau(rows) for arm, rows in arms.items()},
                          'versus_esp': sign_test(arms['hybrid'], arms['esp']),
                          'versus_output_hidden': sign_test(arms['hybrid'], arms['output_hidden']),
                          'trajectory_versus_output_hidden': sign_test(arms['trajectory'], arms['output_hidden']),
                          'frozen_versus_esp': sign_test(arms['frozen'], arms['esp']),
                          'scratch_versus_esp': sign_test(arms['scratch'], arms['esp'])}
    return summary
