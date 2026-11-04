"""Qwen3-8B trajectory-only versus joint observation ablation.

Protocols ``office-8b-traj-ablate-v1``, ``office-8b-traj-xcat-v1``,
``office-8b-traj-xcat2-v1``, ``office-8b-traj-xcat3-v1``, ``office-8b-traj-xcat4-v1``, ``office-8b-traj-xcat5-v1``,
``office-8b-traj-xcat6-v1`` and ``office-8b-traj-xcat7-v1``. Prompt
selection, the two arm specs, the paired summary, and the command that loads
weights live here. Both arms wrap the current ESP baseline with the existing
``spec.observation`` switch. ``--protocol xcat7`` keeps the frozen summarization
slice and records per-arm ``wall_seconds`` for ``office-8b-traj-wall-v1``.
This module does not fit K. Import does not mutate the process environment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, direction, paired_deltas, prompt_sha256,
)
from studies.trajectory_observation.structure.office_tree import (
    NODE_BUDGET, EVAL as TREE_EVAL, SMOKE as TREE_SMOKE, _append, _identity, _load_model,
    _offline_caches, block_complexity, existing_output_dir, prompts_after, read_rows,
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
PROTOCOL_XCAT5 = 'office-8b-traj-xcat5-v1'
CATEGORY_XCAT5 = 'brainstorming'
TAKE_XCAT5 = 128
SKIPPED_QIDS_XCAT5 = ()
EVAL_SHA256_XCAT5 = '3dcf4c093d586cd329f7393213009962308365098a9790b009cafc32c7f37869'
EVAL_QIDS_XCAT5 = tuple(range(TAKE_XCAT5))
PROTOCOL_XCAT6 = 'office-8b-traj-xcat6-v1'
CATEGORY_XCAT6 = 'information_extraction'
TAKE_XCAT6 = 128
SKIPPED_QIDS_XCAT6 = ()
EVAL_SHA256_XCAT6 = 'b7094d9ecf62a73e299d742016a3fdfcf4608ef5a7a32f7adc7703b86706d7c3'
EVAL_QIDS_XCAT6 = tuple(range(TAKE_XCAT6))
PROTOCOL_XCAT7 = 'office-8b-traj-xcat7-v1'
CATEGORY_XCAT7 = 'summarization'
TAKE_XCAT7 = 128
SKIPPED_QIDS_XCAT7 = ()
EVAL_SHA256_XCAT7 = 'f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a'
EVAL_QIDS_XCAT7 = tuple(range(TAKE_XCAT7))
PROTOCOL_WALL = 'office-8b-traj-wall-v1'
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


def prompts_brainstorming(rows, take=TAKE_XCAT5):
    """First unused nonempty brainstorming texts. qid is the index in that stream."""
    prior, prior_skipped = prompts_closed_qa(rows, take=TAKE_XCAT4)
    if (len(prior) != TAKE_XCAT4 or tuple(prior_skipped) != SKIPPED_QIDS_XCAT4
            or any(item['category'] != CATEGORY_XCAT4 for item in prior)):
        raise RuntimeError('closed_qa slice drifted before the brainstorming draw')
    class_prior, _class_skipped = prompts_classification(rows, take=TAKE_XCAT3)
    if len(class_prior) != TAKE_XCAT3 or any(item['category'] != CATEGORY_XCAT3 for item in class_prior):
        raise RuntimeError('classification slice drifted before the brainstorming draw')
    general_prior, _general_skipped = prompts_general_qa(rows, take=TAKE_XCAT2)
    if len(general_prior) != TAKE_XCAT2 or any(item['category'] != CATEGORY_XCAT2 for item in general_prior):
        raise RuntimeError('general_qa slice drifted before the brainstorming draw')
    open_prior, open_skipped = prompts_open_qa(rows, take=TAKE_XCAT)
    if tuple(open_skipped) != SKIPPED_QIDS_XCAT or len(open_prior) != TAKE_XCAT:
        raise RuntimeError('open_qa slice drifted before the brainstorming draw')
    texts = category_texts(rows, CATEGORY_XCAT5)
    used = previously_used_texts(rows)
    used.update(item['text'] for item in open_prior)
    used.update(item['text'] for item in general_prior)
    used.update(item['text'] for item in class_prior)
    used.update(item['text'] for item in prior)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT5, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'brainstorming has only {len(chosen)} unused prompts')


def assert_registered_xcat5_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT5 or digest != EVAL_SHA256_XCAT5 or qids != EVAL_QIDS_XCAT5
            or tuple(skipped) != SKIPPED_QIDS_XCAT5):
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered brainstorming slice')
    if any(prompt['category'] != CATEGORY_XCAT5 for prompt in prompts):
        raise RuntimeError('fifth cross-category slice left brainstorming')
    return digest


def prompts_information_extraction(rows, take=TAKE_XCAT6):
    """First unused nonempty information_extraction texts. qid is the index in that stream."""
    prior, prior_skipped = prompts_brainstorming(rows, take=TAKE_XCAT5)
    if (len(prior) != TAKE_XCAT5 or tuple(prior_skipped) != SKIPPED_QIDS_XCAT5
            or any(item['category'] != CATEGORY_XCAT5 for item in prior)):
        raise RuntimeError('brainstorming slice drifted before the information_extraction draw')
    closed_prior, closed_skipped = prompts_closed_qa(rows, take=TAKE_XCAT4)
    if (len(closed_prior) != TAKE_XCAT4 or tuple(closed_skipped) != SKIPPED_QIDS_XCAT4
            or any(item['category'] != CATEGORY_XCAT4 for item in closed_prior)):
        raise RuntimeError('closed_qa slice drifted before the information_extraction draw')
    class_prior, _class_skipped = prompts_classification(rows, take=TAKE_XCAT3)
    if len(class_prior) != TAKE_XCAT3 or any(item['category'] != CATEGORY_XCAT3 for item in class_prior):
        raise RuntimeError('classification slice drifted before the information_extraction draw')
    general_prior, _general_skipped = prompts_general_qa(rows, take=TAKE_XCAT2)
    if len(general_prior) != TAKE_XCAT2 or any(item['category'] != CATEGORY_XCAT2 for item in general_prior):
        raise RuntimeError('general_qa slice drifted before the information_extraction draw')
    open_prior, open_skipped = prompts_open_qa(rows, take=TAKE_XCAT)
    if tuple(open_skipped) != SKIPPED_QIDS_XCAT or len(open_prior) != TAKE_XCAT:
        raise RuntimeError('open_qa slice drifted before the information_extraction draw')
    texts = category_texts(rows, CATEGORY_XCAT6)
    used = previously_used_texts(rows)
    used.update(item['text'] for item in open_prior)
    used.update(item['text'] for item in general_prior)
    used.update(item['text'] for item in class_prior)
    used.update(item['text'] for item in closed_prior)
    used.update(item['text'] for item in prior)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT6, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'information_extraction has only {len(chosen)} unused prompts')


def assert_registered_xcat6_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT6 or digest != EVAL_SHA256_XCAT6 or qids != EVAL_QIDS_XCAT6
            or tuple(skipped) != SKIPPED_QIDS_XCAT6):
        raise RuntimeError(
            f'prompt hash {digest} does not match the pre-registered information_extraction slice')
    if any(prompt['category'] != CATEGORY_XCAT6 for prompt in prompts):
        raise RuntimeError('sixth cross-category slice left information_extraction')
    return digest


def prompts_summarization(rows, take=TAKE_XCAT7):
    """First unused nonempty summarization texts. qid is the index in that stream."""
    prior, prior_skipped = prompts_information_extraction(rows, take=TAKE_XCAT6)
    if (len(prior) != TAKE_XCAT6 or tuple(prior_skipped) != SKIPPED_QIDS_XCAT6
            or any(item['category'] != CATEGORY_XCAT6 for item in prior)):
        raise RuntimeError('information_extraction slice drifted before the summarization draw')
    brain_prior, brain_skipped = prompts_brainstorming(rows, take=TAKE_XCAT5)
    if (len(brain_prior) != TAKE_XCAT5 or tuple(brain_skipped) != SKIPPED_QIDS_XCAT5
            or any(item['category'] != CATEGORY_XCAT5 for item in brain_prior)):
        raise RuntimeError('brainstorming slice drifted before the summarization draw')
    closed_prior, closed_skipped = prompts_closed_qa(rows, take=TAKE_XCAT4)
    if (len(closed_prior) != TAKE_XCAT4 or tuple(closed_skipped) != SKIPPED_QIDS_XCAT4
            or any(item['category'] != CATEGORY_XCAT4 for item in closed_prior)):
        raise RuntimeError('closed_qa slice drifted before the summarization draw')
    class_prior, _class_skipped = prompts_classification(rows, take=TAKE_XCAT3)
    if len(class_prior) != TAKE_XCAT3 or any(item['category'] != CATEGORY_XCAT3 for item in class_prior):
        raise RuntimeError('classification slice drifted before the summarization draw')
    general_prior, _general_skipped = prompts_general_qa(rows, take=TAKE_XCAT2)
    if len(general_prior) != TAKE_XCAT2 or any(item['category'] != CATEGORY_XCAT2 for item in general_prior):
        raise RuntimeError('general_qa slice drifted before the summarization draw')
    open_prior, open_skipped = prompts_open_qa(rows, take=TAKE_XCAT)
    if tuple(open_skipped) != SKIPPED_QIDS_XCAT or len(open_prior) != TAKE_XCAT:
        raise RuntimeError('open_qa slice drifted before the summarization draw')
    texts = category_texts(rows, CATEGORY_XCAT7)
    used = previously_used_texts(rows)
    used.update(item['text'] for item in open_prior)
    used.update(item['text'] for item in general_prior)
    used.update(item['text'] for item in class_prior)
    used.update(item['text'] for item in closed_prior)
    used.update(item['text'] for item in brain_prior)
    used.update(item['text'] for item in prior)
    skipped = []
    chosen = []
    for index, text in enumerate(texts):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': CATEGORY_XCAT7, 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'summarization has only {len(chosen)} unused prompts')


def assert_registered_xcat7_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if (len(prompts) != TAKE_XCAT7 or digest != EVAL_SHA256_XCAT7 or qids != EVAL_QIDS_XCAT7
            or tuple(skipped) != SKIPPED_QIDS_XCAT7):
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered summarization slice')
    if any(prompt['category'] != CATEGORY_XCAT7 for prompt in prompts):
        raise RuntimeError('seventh cross-category slice left summarization')
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


def finite_wall(value):
    """Seconds kept on a row. Missing, non-finite, and negative stay missing."""
    if type(value) not in (int, float):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def measure_generate(generate, prompt_ids, device):
    """Wall clock around one ``SpecDecoder.generate``.

    ``sync_device`` runs before the start read and again before the end read.
    A failed sync or clock leaves the wall missing and still returns the decode.
    """
    from spec_mtp.model import sync_device

    started = None
    try:
        sync_device(device)
        started = time.perf_counter()
    except Exception:
        started = None
    output, stats = generate(prompt_ids)
    if started is None:
        return output, stats, None
    try:
        sync_device(device)
        elapsed = time.perf_counter() - started
    except Exception:
        return output, stats, None
    return output, stats, finite_wall(elapsed)


def wall_direction(shorter, longer, mean):
    """Direction of ``wall_seconds_traj - wall_seconds_full``.

    ``shorter`` means traj used less wall time. This is not the acceptance
    ``direction``, which treats a larger count as positive.
    """
    if shorter > longer and mean < 0:
        return 'shorter'
    if longer > shorter and mean > 0:
        return 'longer'
    if shorter == longer and mean == 0:
        return 'flat'
    return 'mixed'


def wall_report(full_rows, traj_rows):
    """Paired wall summary. An incomplete clock is not a direction."""
    if any(finite_wall(row.get('wall_seconds')) is None for row in full_rows + traj_rows):
        return {'complete': False}
    deltas = paired_deltas(traj_rows, full_rows, 'wall_seconds')
    shorter = sum(delta < 0 for delta in deltas)
    longer = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    difference = bootstrap_mean_ci(deltas)
    tested = shorter + longer
    from scipy.stats import binomtest

    pvalue = None if tested == 0 else float(
        binomtest(shorter, tested, 0.5, alternative='two-sided').pvalue)
    return {
        'complete': True,
        'delta': 'wall_seconds_traj - wall_seconds_full',
        'shorter': shorter,
        'longer': longer,
        'ties': ties,
        'wins_losses_ties': [shorter, longer, ties],
        'p': pvalue,
        'wall_difference': difference,
        'wall_direction': wall_direction(shorter, longer, difference['mean']),
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
    summary = {
        'protocol': protocol,
        'role': ROLE,
        'confirmatory': False,
        'prompts': take,
        'prompt_sha256': digest,
        'skipped_qids': list(skipped),
        'baseline': 'full',
        'comparison': comparison,
        'argmax_all_passed': passed,
        'wall': wall_report(rows_by_name['full'], rows_by_name['traj']),
    }
    if protocol == PROTOCOL_XCAT7:
        summary['wall_protocol'] = PROTOCOL_WALL
    return summary


def round_config(protocol):
    if protocol == 'ablate':
        return {
            'protocol': PROTOCOL,
            'prefix': 'sd-office-8b-traj-ablate',
            'stamp': '20261103-0905',
            'select': prompts_from_647,
            'check': assert_registered_slice,
            'take': TAKE,
        }
    if protocol == 'xcat':
        return {
            'protocol': PROTOCOL_XCAT,
            'prefix': 'sd-office-8b-traj-xcat',
            'stamp': '20261103-0929',
            'select': prompts_open_qa,
            'check': assert_registered_xcat_slice,
            'take': TAKE_XCAT,
        }
    if protocol == 'xcat2':
        return {
            'protocol': PROTOCOL_XCAT2,
            'prefix': 'sd-office-8b-traj-xcat2',
            'stamp': '20261103-0958',
            'select': prompts_general_qa,
            'check': assert_registered_xcat2_slice,
            'take': TAKE_XCAT2,
        }
    if protocol == 'xcat3':
        return {
            'protocol': PROTOCOL_XCAT3,
            'prefix': 'sd-office-8b-traj-xcat3',
            'stamp': '20261103-1024',
            'select': prompts_classification,
            'check': assert_registered_xcat3_slice,
            'take': TAKE_XCAT3,
        }
    if protocol == 'xcat4':
        return {
            'protocol': PROTOCOL_XCAT4,
            'prefix': 'sd-office-8b-traj-xcat4',
            'stamp': '20261104-032457',
            'select': prompts_closed_qa,
            'check': assert_registered_xcat4_slice,
            'take': TAKE_XCAT4,
        }
    if protocol == 'xcat5':
        return {
            'protocol': PROTOCOL_XCAT5,
            'prefix': 'sd-office-8b-traj-xcat5',
            'stamp': '20261104-035230',
            'select': prompts_brainstorming,
            'check': assert_registered_xcat5_slice,
            'take': TAKE_XCAT5,
        }
    if protocol == 'xcat6':
        return {
            'protocol': PROTOCOL_XCAT6,
            'prefix': 'sd-office-8b-traj-xcat6',
            'stamp': '20261104-042546',
            'select': prompts_information_extraction,
            'check': assert_registered_xcat6_slice,
            'take': TAKE_XCAT6,
        }
    if protocol == 'xcat7':
        return {
            'protocol': PROTOCOL_XCAT7,
            'prefix': 'sd-office-8b-traj-wall-v1',
            'stamp': '20261104-0758',
            'select': prompts_summarization,
            'check': assert_registered_xcat7_slice,
            'take': TAKE_XCAT7,
        }
    raise RuntimeError(f'unknown trajectory protocol {protocol}')


def _file(dest, name, suffix, prefix, stamp):
    return dest / f'{prefix}-{name}-{stamp}.{suffix}'


def _specs():
    specs = arm_specs()
    return tuple((name, specs[name]) for name, _gain in ARMS)


def _model_device(model):
    return getattr(model, 'device', 'cpu')


def _store_clock(row, stats, wall):
    kept = finite_wall(wall)
    if kept is not None:
        row['wall_seconds'] = kept
    if isinstance(stats, dict):
        for key in ('prefill_time', 'decode_time'):
            if key not in stats:
                continue
            kept = finite_wall(stats[key])
            if kept is not None:
                row[key] = kept
    return row


def _measured_arm(model, spec, ids, limit):
    """One new decoder. The wall covers only this arm's generate."""
    import spec_mtp.decode as decode_mod
    from studies.trajectory_observation.structure.office_tree import arm_row

    captured = {}
    original = decode_mod.SpecDecoder.generate

    def generate(self, prompt_ids):
        output, stats, wall = measure_generate(
            lambda tokens: original(self, tokens), prompt_ids, _model_device(model))
        captured['stats'] = stats
        captured['wall'] = wall
        return output, stats

    decode_mod.SpecDecoder.generate = generate
    try:
        row = arm_row(model, spec, ids, limit)
    finally:
        decode_mod.SpecDecoder.generate = original
    return _store_clock(row, captured.get('stats', {}), captured.get('wall'))


def decode_measured_prompt(model, tokenizer, prompt, specs, limit):
    """Independent AR, then full, then traj. Only the two arms get a wall clock."""
    from spec_mtp.decode import ar_generate
    from studies.trajectory_observation.structure.office_tree import token_hash

    ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
    reference, _stats = ar_generate(model, ids, limit)
    reference_hash = token_hash(reference)
    rows = {}
    for name, spec in specs:
        row = _measured_arm(model, spec, ids, limit)
        row['qid'] = prompt['qid']
        row['category'] = prompt['category']
        row['ar_equal'] = row['output_sha256'] == reference_hash
        rows[name] = row
    return rows


def _warmup_arms(model, tokenizer, prompt):
    """Untimed 8-token pass. Each arm builds a decoder and drops it, then sync."""
    from spec_mtp.model import sync_device
    from studies.trajectory_observation.structure.office_tree import arm_row

    ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
    for _name, spec in _specs():
        arm_row(model, spec, ids, SMOKE)
    sync_device(_model_device(model))


def _smoke_view(row):
    recorded = {
        'accepted': row['accepted'],
        'argmax_passed': row['argmax_passed'],
        'max_nodes': row['max_nodes'],
        'block_complexity': row['block_complexity'],
    }
    wall = finite_wall(row.get('wall_seconds'))
    if wall is not None:
        recorded['wall_seconds'] = wall
    for key in ('prefill_time', 'decode_time'):
        if key in row and finite_wall(row[key]) is not None:
            recorded[key] = row[key]
    return recorded


def _smoke(model, tokenizer, prompts, dest, cfg):
    rows = []
    for prompt in prompts[:2]:
        started = time.perf_counter()
        decoded = decode_measured_prompt(model, tokenizer, prompt, _specs(), SMOKE)
        rows.append({name: _smoke_view(row) for name, row in decoded.items()})
        print(f"smoke {prompt['qid']} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {'protocol': cfg['protocol'], 'role': 'exploration', 'prompts': 2,
               'qids': [prompt['qid'] for prompt in prompts[:2]],
               'tokens': SMOKE['max_new_tokens'], 'arms': rows,
               'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values()),
               'wall_all_recorded': all('wall_seconds' in arm for item in rows for arm in item.values())}
    if cfg['protocol'] == PROTOCOL_XCAT7:
        payload['wall_protocol'] = PROTOCOL_WALL
    _file(dest, 'smoke', 'json', cfg['prefix'], cfg['stamp']).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    if not payload['wall_all_recorded']:
        raise RuntimeError('smoke wall clock missing')
    print('smoke passed', flush=True)
    return payload


def _shown_wall(row):
    wall = finite_wall(row.get('wall_seconds'))
    if wall is None:
        return 'missing'
    return f'{wall:.3f}'


def _formal(model, tokenizer, prompts, skipped, digest, dest, cfg):
    paths = {name: _file(dest, name, 'jsonl', cfg['prefix'], cfg['stamp']) for name, _gain in ARMS}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('arms resumed at different prompts')
    header = _file(dest, 'header', 'json', cfg['prefix'], cfg['stamp'])
    if not header.exists():
        body = {
            'protocol': cfg['protocol'], 'role': 'exploration', 'source': _identity(),
            'eval': EVAL, 'arms': {name: spec for name, spec in _specs()},
            'prompt_sha256': digest, 'qids': [prompt['qid'] for prompt in prompts],
            'skipped_qids': list(skipped),
            'delta': 'accepted_traj - accepted_full',
            'wall_seconds': 'sync_device and perf_counter around this arm SpecDecoder.generate',
            'wall_delta': 'wall_seconds_traj - wall_seconds_full',
            'bootstrap': {'seed': 0, 'resamples': 10000},
        }
        if cfg['protocol'] == PROTOCOL_XCAT7:
            body['wall_protocol'] = PROTOCOL_WALL
        header.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
    _warmup_arms(model, tokenizer, prompts[0])
    finished = done['full']
    for prompt in prompts:
        if prompt['qid'] in finished:
            continue
        started = time.perf_counter()
        decoded = decode_measured_prompt(model, tokenizer, prompt, _specs(), EVAL)
        for name, row in decoded.items():
            _append(paths[name], row)
        print(f"eval {prompt['qid']} full={decoded['full']['accepted']} "
              f"traj={decoded['traj']['accepted']} nodes={decoded['full']['max_nodes']} "
              f"wall={_shown_wall(decoded['full'])}/{_shown_wall(decoded['traj'])} "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    summary = ablation_summary(
        {name: read_rows(path) for name, path in paths.items()},
        protocol=cfg['protocol'], digest=digest, skipped=skipped, take=cfg['take'],
    )
    _file(dest, 'summary', 'json', cfg['prefix'], cfg['stamp']).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--only', choices=('all', 'smoke', 'formal'), default='all')
    parser.add_argument(
        '--protocol', choices=('ablate', 'xcat', 'xcat2', 'xcat3', 'xcat4', 'xcat5', 'xcat6', 'xcat7'),
        default='ablate')
    args = parser.parse_args(argv)
    dest = existing_output_dir(args.dest)
    cfg = round_config(args.protocol)
    _offline_caches(dest)
    from studies.trajectory_observation.structure.office_qwen8b import load_dolly_rows

    prompts, skipped = cfg['select'](load_dolly_rows())
    digest = cfg['check'](prompts, skipped)
    print('load 8b', flush=True)
    model, tokenizer, torch_version = _load_model('8b')
    if args.only in ('all', 'smoke'):
        _smoke(model, tokenizer, prompts, dest, cfg)
    if args.only in ('all', 'formal'):
        if _identity()['dirty']:
            raise RuntimeError('formal run requires a clean worktree')
        _formal(model, tokenizer, prompts, skipped, digest, dest, cfg)
    print('torch', torch_version, flush=True)


if __name__ == '__main__':
    main()
