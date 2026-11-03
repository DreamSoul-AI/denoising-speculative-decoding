"""Qwen3-8B trajectory-only versus joint observation ablation.

Protocols ``office-8b-traj-ablate-v1``, ``office-8b-traj-xcat-v1``,
``office-8b-traj-xcat2-v1``, ``office-8b-traj-xcat3-v1`` and ``office-8b-traj-xcat4-v1``. Prompt
selection, the two arm specs and the paired summary live here. The command that
loads weights is ``studies.trajectory_observation.flow.execute.office_traj``.
Both arms wrap the
current ESP baseline with the existing ``spec.observation`` switch. This module
does not fit K and does not mutate the process environment on import.
"""

from __future__ import annotations

import json
from pathlib import Path

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, direction, paired_deltas, prompt_sha256,
)
from studies.trajectory_observation.structure.office_tree import (
    NODE_BUDGET, EVAL as TREE_EVAL, SMOKE as TREE_SMOKE, block_complexity, prompts_after,
)

PROTOCOL = 'office-8b-traj-ablate-v1'
ROLE = 'exploration'
START = 647
END = 708
TAKE = 60
BOOT_SEED = 0
BOOT_RESAMPLES = 10000
SKIPPED_QIDS = (658, 678)
EVAL_SHA256 = 'd23ad590c45307fa7c1981b272adaec243bfaca5a2e390eda394bf2fe42a4cca'
EVAL_QIDS = tuple(qid for qid in range(START, END + 1) if qid not in SKIPPED_QIDS)
PROTOCOL_XCAT = 'office-8b-traj-xcat-v1'
CATEGORY_XCAT = 'open_qa'
TAKE_XCAT = 128
SKIPPED_QIDS_XCAT = ()
EVAL_SHA256_XCAT = 'f6e6b1c71a20f1170692e22a2c2bd79fb31eaeb810b861553f7c76a41df25231'
EVAL_QIDS_XCAT = tuple(range(TAKE_XCAT))
PROTOCOL_XCAT2 = 'office-8b-traj-xcat2-v1'
CATEGORY_XCAT2 = 'general_qa'
TAKE_XCAT2 = 128
SKIPPED_QIDS_XCAT2 = (91,)
EVAL_SHA256_XCAT2 = 'a6a306ea868767da6cf8b6006bb4d91c5360387ae8b16ba1a20142cdd35ec677'
EVAL_QIDS_XCAT2 = tuple(range(91)) + tuple(range(92, 129))
PROTOCOL_XCAT3 = 'office-8b-traj-xcat3-v1'
CATEGORY_XCAT3 = 'classification'
TAKE_XCAT3 = 128
SKIPPED_QIDS_XCAT3 = ()
EVAL_SHA256_XCAT3 = '01c21dbf4a774b2395bed3a8606d3fe20d92a0cb7d2562ff8e8d965b9f2bd35e'
EVAL_QIDS_XCAT3 = tuple(range(TAKE_XCAT3))
PROTOCOL_XCAT4 = 'office-8b-traj-xcat4-v1'
CATEGORY_XCAT4 = 'closed_qa'
TAKE_XCAT4 = 128
SKIPPED_QIDS_XCAT4 = ()
EVAL_SHA256_XCAT4 = '55ad2a82a6853587c7f28a641ba5acbdcc66d374c1b2cc3924c1563ce3fafdf9'
EVAL_QIDS_XCAT4 = tuple(range(TAKE_XCAT4))
GAIN_FULL = 0.5
GAIN_TRAJ = 0.0
MASK = {'init': 'mean_prompt', 'lam': 0.1, 'update': True}
TREE = {'mode': 'static', 'branches': [4], 'pruning': True}
EVAL = dict(TREE_EVAL)
SMOKE = dict(TREE_SMOKE)
ARMS = (
    ('full', GAIN_FULL),
    ('traj', GAIN_TRAJ),
)
ROOT = Path(__file__).resolve().parents[3]


def arm_spec(gain):
    """ESP baseline plus the existing observation blend switch."""
    value = float(gain)
    if value not in (GAIN_FULL, GAIN_TRAJ):
        raise RuntimeError('this ablation only uses the pre-registered gains')
    return {
        'method': 'esp',
        'num_masks': 1,
        'impl': 'efficient',
        'tree': dict(TREE),
        'mask': dict(MASK),
        'observation': {'gain': value, 'condition': 'none', 'form': 'mean'},
    }


def arm_specs():
    return {name: arm_spec(gain) for name, gain in ARMS}


def prompts_from_647(rows, take=TAKE):
    """Unused creative texts from qid 647 through the end of the stream."""
    return prompts_after(rows, START, take)


def dolly_text(item):
    text = item.get('instruction')
    context = item.get('context') or ''
    if context:
        text = f"{context}\n\n{text}"
    if isinstance(text, str) and text.strip():
        return text
    return None


def category_texts(rows, category):
    texts = []
    for item in rows:
        if item.get('category') != category:
            continue
        text = dolly_text(item)
        if text is not None:
            texts.append(text)
    return texts


def previously_used_texts(rows):
    """Creative stream, calibration templates, and local SpecBench turns."""
    from studies.trajectory_observation.structure.office_qwen8b import HEADS, TAILS

    used = set(category_texts(rows, 'creative_writing'))
    used.update(f'{head} {tail}' for head in HEADS for tail in TAILS)
    path = ROOT / 'datasets' / 'spec_bench' / 'question.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for turn in item.get('turns') or []:
            if isinstance(turn, str) and turn.strip():
                used.add(turn)
    return used


def prompts_open_qa(rows, take=TAKE_XCAT):
    """First unused nonempty open_qa texts. qid is the index in that stream."""
    texts = category_texts(rows, CATEGORY_XCAT)
    if len(texts) < take:
        raise RuntimeError(f'open_qa has only {len(texts)} nonempty prompts')
    used = previously_used_texts(rows)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'open_qa has only {len(chosen)} unused prompts')


def assert_registered_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if len(EVAL_QIDS) != TAKE or EVAL_QIDS[0] != START or EVAL_QIDS[-1] != END:
        raise RuntimeError('registered qids do not cover the unused tail')
    if digest != EVAL_SHA256 or qids != EVAL_QIDS or tuple(skipped) != SKIPPED_QIDS:
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered slice')
    return digest


def prompts_general_qa(rows, take=TAKE_XCAT2):
    """First unused nonempty general_qa texts. qid is the index in that stream."""
    prior, prior_skipped = prompts_open_qa(rows, take=TAKE_XCAT)
    if tuple(prior_skipped) != SKIPPED_QIDS_XCAT or len(prior) != TAKE_XCAT:
        raise RuntimeError('open_qa slice drifted before the general_qa draw')
    texts = category_texts(rows, CATEGORY_XCAT2)
    used = previously_used_texts(rows)
    used.update(item['text'] for item in prior)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT2, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'general_qa has only {len(chosen)} unused prompts')


def assert_registered_xcat_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT or digest != EVAL_SHA256_XCAT or qids != EVAL_QIDS_XCAT
            or tuple(skipped) != SKIPPED_QIDS_XCAT):
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered open_qa slice')
    if any(prompt['category'] != CATEGORY_XCAT for prompt in prompts):
        raise RuntimeError('cross-category slice left open_qa')
    return digest


def assert_registered_xcat2_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT2 or digest != EVAL_SHA256_XCAT2 or qids != EVAL_QIDS_XCAT2
            or tuple(skipped) != SKIPPED_QIDS_XCAT2):
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered general_qa slice')
    if any(prompt['category'] != CATEGORY_XCAT2 for prompt in prompts):
        raise RuntimeError('second cross-category slice left general_qa')
    return digest


def prompts_classification(rows, take=TAKE_XCAT3):
    """First unused nonempty classification texts. qid is the index in that stream."""
    prior, _prior_skipped = prompts_general_qa(rows, take=TAKE_XCAT2)
    if len(prior) != TAKE_XCAT2 or any(item['category'] != CATEGORY_XCAT2 for item in prior):
        raise RuntimeError('general_qa slice drifted before the classification draw')
    open_prior, open_skipped = prompts_open_qa(rows, take=TAKE_XCAT)
    if tuple(open_skipped) != SKIPPED_QIDS_XCAT or len(open_prior) != TAKE_XCAT:
        raise RuntimeError('open_qa slice drifted before the classification draw')
    texts = category_texts(rows, CATEGORY_XCAT3)
    used = previously_used_texts(rows)
    used.update(item['text'] for item in open_prior)
    used.update(item['text'] for item in prior)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT3, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'classification has only {len(chosen)} unused prompts')


def assert_registered_xcat3_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT3 or digest != EVAL_SHA256_XCAT3 or qids != EVAL_QIDS_XCAT3
            or tuple(skipped) != SKIPPED_QIDS_XCAT3):
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered classification slice')
    if any(prompt['category'] != CATEGORY_XCAT3 for prompt in prompts):
        raise RuntimeError('third cross-category slice left classification')
    return digest


def prompts_closed_qa(rows, take=TAKE_XCAT4):
    """First unused nonempty closed_qa texts. qid is the index in that stream."""
    prior, _prior_skipped = prompts_classification(rows, take=TAKE_XCAT3)
    if len(prior) != TAKE_XCAT3 or any(item['category'] != CATEGORY_XCAT3 for item in prior):
        raise RuntimeError('classification slice drifted before the closed_qa draw')
    general_prior, _general_skipped = prompts_general_qa(rows, take=TAKE_XCAT2)
    if len(general_prior) != TAKE_XCAT2 or any(item['category'] != CATEGORY_XCAT2 for item in general_prior):
        raise RuntimeError('general_qa slice drifted before the closed_qa draw')
    open_prior, open_skipped = prompts_open_qa(rows, take=TAKE_XCAT)
    if tuple(open_skipped) != SKIPPED_QIDS_XCAT or len(open_prior) != TAKE_XCAT:
        raise RuntimeError('open_qa slice drifted before the closed_qa draw')
    texts = category_texts(rows, CATEGORY_XCAT4)
    used = previously_used_texts(rows)
    used.update(item['text'] for item in open_prior)
    used.update(item['text'] for item in general_prior)
    used.update(item['text'] for item in prior)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT4, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'closed_qa has only {len(chosen)} unused prompts')


def assert_registered_xcat4_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT4 or digest != EVAL_SHA256_XCAT4 or qids != EVAL_QIDS_XCAT4
            or tuple(skipped) != SKIPPED_QIDS_XCAT4):
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered closed_qa slice')
    if any(prompt['category'] != CATEGORY_XCAT4 for prompt in prompts):
        raise RuntimeError('fourth cross-category slice left closed_qa')
    return digest


def _compare(treated_name, treated, baseline_name, baseline):
    from studies.trajectory_observation.structure.fair_branch import mean_tau, pooled_tau, sign_test

    compared = sign_test(treated, baseline)
    accepted = bootstrap_mean_ci(paired_deltas(treated, baseline, 'accepted'))
    full_gain = bootstrap_mean_ci(paired_deltas(baseline, treated, 'accepted'))
    tau = bootstrap_mean_ci(paired_deltas(treated, baseline, 'tau'))
    trend = direction(compared['wins'], compared['losses'], accepted['mean'])
    return {
        'baseline': baseline_name,
        'arm': treated_name,
        'delta': 'accepted_traj - accepted_full',
        'full_gain': 'accepted_full - accepted_traj',
        'wins': compared['wins'],
        'losses': compared['losses'],
        'ties': compared['ties'],
        'wins_losses_ties': [compared['wins'], compared['losses'], compared['ties']],
        'p': compared['p'],
        'tau_mean': {baseline_name: mean_tau(baseline), treated_name: mean_tau(treated)},
        'tau_pooled': {baseline_name: pooled_tau(baseline), treated_name: pooled_tau(treated)},
        'accepted_difference': accepted,
        'full_gain_difference': full_gain,
        'tau_difference': tau,
        'direction': trend,
        'confirmatory': False,
        'argmax_all_passed': all(row['argmax_passed'] for row in baseline + treated),
        'ar_mismatches': {
            baseline_name: sum(not row['ar_equal'] for row in baseline),
            treated_name: sum(not row['ar_equal'] for row in treated),
        },
        'max_nodes': max(row['max_nodes'] for row in baseline + treated),
        'block_complexity': {
            baseline_name: sorted({row['block_complexity'] for row in baseline}),
            treated_name: sorted({row['block_complexity'] for row in treated}),
        },
    }


def ablation_summary(rows_by_name, protocol=PROTOCOL, digest=EVAL_SHA256, skipped=SKIPPED_QIDS, take=TAKE):
    if set(rows_by_name) != {name for name, _gain in ARMS}:
        raise RuntimeError('summary is missing a registered arm')
    if any(len(rows) != take for rows in rows_by_name.values()):
        raise RuntimeError(f'summary needs {take} rows in every arm')
    expected = block_complexity(1, TREE['branches'])
    comparison = _compare('traj', rows_by_name['traj'], 'full', rows_by_name['full'])
    complexities = comparison['block_complexity']['full'] + comparison['block_complexity']['traj']
    passed = (
        comparison['argmax_all_passed']
        and comparison['max_nodes'] == NODE_BUDGET
        and complexities == [expected, expected]
    )
    return {
        'protocol': protocol,
        'role': ROLE,
        'confirmatory': False,
        'prompts': take,
        'prompt_sha256': digest,
        'skipped_qids': list(skipped),
        'baseline': 'full',
        'comparison': comparison,
        'argmax_all_passed': passed,
    }
