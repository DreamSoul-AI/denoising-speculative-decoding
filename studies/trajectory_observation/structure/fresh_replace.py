"""Fresh-prompt replication of the equal-node extra-branch replacement.

The comparison itself is `fair_branch.run_split`: frozen diagonal K, slot 0
left as the ESP mask, corrected top-1 swapped into the weakest childless
depth-1 leaf. This module selects the unused Dolly slice, summarizes the
paired effect. Command orchestration lives in flow/execute/fresh_replace.py.
Import does not mutate the process environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'extra-branch-replace-fresh-v1'
SKIP = 128
TAKE = 128
BOOT_SEED = 0
BOOT_RESAMPLES = 10000
FRESH_SHA256 = 'e37e3be985b5356acedeffc22a83b00a0484b1983d44ef6e67eebfe0c3f36a20'
FRESH_QIDS = tuple(index for index in range(128, 257) if index != 144)
SPLIT = 'dolly-creative-128-256-except-144'


def creative_texts(rows):
    texts = []
    for item in rows:
        if item.get('category') != 'creative_writing':
            continue
        text = item.get('instruction')
        context = item.get('context') or ''
        if context:
            text = f"{context}\n\n{text}"
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def fresh_prompts(rows, skip=SKIP, take=TAKE):
    """Next `take` creative texts after the used prefix, skipping repeated strings.

    `qid` is the index in the nonempty creative_writing stream. The first
    `skip` rows are the sample already measured by `extra-branch-replace-v1`.
    A later row whose text already occurred is not a new prompt.
    """
    texts = creative_texts(rows)
    if len(texts) < skip or len(set(texts[:skip])) != skip:
        raise RuntimeError('used Dolly prefix is missing or not unique')
    used = set(texts[:skip])
    chosen = []
    for index, text in enumerate(texts[skip:], start=skip):
        if text in used:
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': 'creative_writing', 'text': text})
        if len(chosen) == take:
            return chosen
    raise RuntimeError(f'creative_writing has only {len(chosen)} new prompts after the used prefix')


def prompt_sha256(prompts):
    body = '\n'.join(prompt['text'] for prompt in prompts)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def assert_known_slice(prompts):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if digest != FRESH_SHA256 or qids != FRESH_QIDS:
        raise RuntimeError(f'fresh prompt hash {digest} does not match the pre-registered slice')
    return digest


def outside_repo(path):
    output = Path(path).resolve()
    root = ROOT.resolve()
    if output == root or root in output.parents:
        raise RuntimeError('write the measurement outside the repo')
    return output


def bootstrap_mean_ci(deltas, seed=BOOT_SEED, resamples=BOOT_RESAMPLES):
    import numpy as np

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError('bootstrap needs a nonempty 1d sample')
    means = values[np.random.default_rng(seed).integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {'mean': float(values.mean()), 'ci95': [float(low), float(high)],
            'seed': int(seed), 'resamples': int(resamples),
            'excludes_zero': bool(low > 0.0 or high < 0.0)}


def direction(wins, losses, mean):
    if wins > losses and mean > 0:
        return 'positive'
    if wins < losses and mean < 0:
        return 'negative'
    if wins == losses and mean == 0:
        return 'flat'
    return 'mixed'


def paired_deltas(left, right, key):
    paired = {row['qid']: row for row in right}
    if set(paired) != {row['qid'] for row in left}:
        raise RuntimeError('paired rows do not share qids')
    return [row[key] - paired[row['qid']][key] for row in left]


def replication_summary(esp, replaced):
    from studies.trajectory_observation.structure.fair_branch import mean_tau, pooled_tau, sign_test

    compared = sign_test(replaced, esp)
    accepted = bootstrap_mean_ci(paired_deltas(replaced, esp, 'accepted'))
    tau = bootstrap_mean_ci(paired_deltas(replaced, esp, 'tau'))
    per_round = bootstrap_mean_ci(paired_deltas(
        [{'qid': row['qid'], 'per_round': row['accepted'] / row['decode_calls']} for row in replaced],
        [{'qid': row['qid'], 'per_round': row['accepted'] / row['decode_calls']} for row in esp],
        'per_round'))
    trend = direction(compared['wins'], compared['losses'], accepted['mean'])
    return {
        'protocol': PROTOCOL,
        'split': SPLIT,
        'prompts': len(esp),
        'tau_mean': {'esp': mean_tau(esp), 'replace': mean_tau(replaced)},
        'tau_pooled': {'esp': pooled_tau(esp), 'replace': pooled_tau(replaced)},
        'replace_vs_esp': compared,
        'accepted_difference': accepted,
        'tau_difference': tau,
        'accepted_per_round_difference': per_round,
        'direction': trend,
        'positive_trend': trend == 'positive',
        'argmax_all_passed': all(row['argmax_passed'] for row in esp + replaced),
        'ar_mismatches': {'esp': sum(not row['ar_equal'] for row in esp),
                          'replace': sum(not row['ar_equal'] for row in replaced)},
        'extra_added': sum(row['extra_added'] for row in replaced),
        'extra_present': sum(row['extra_present'] for row in replaced),
        'max_nodes': max(row['max_nodes'] for row in replaced),
    }
