"""Qwen3-8B trajectory-only versus joint observation ablation.

Protocols ``office-8b-traj-ablate-v1``, ``office-8b-traj-xcat-v1``,
``office-8b-traj-xcat2-v1``, ``office-8b-traj-xcat3-v1``, ``office-8b-traj-xcat4-v1``, ``office-8b-traj-xcat5-v1``,
``office-8b-traj-xcat6-v1`` and ``office-8b-traj-xcat7-v1``. Prompt
selection, the two arm specs, the paired summary, and the command that loads
weights live here. Both arms wrap the current ESP baseline with the existing
``spec.observation`` switch. ``--protocol xcat7`` keeps the frozen summarization
slice and records per-arm ``wall_seconds`` for ``office-8b-traj-wall-v1``.
This module does not fit K. Import does not mutate the process environment.
Offline helpers for ``office-8b-traj-failset-v1`` and ``office-ar-inconsistency-v1``
live here. They do not load a generator or start a decode, and they have no command entry.
``office-ar-first-fork-v1`` reuses this command. It records the first fork against the
BF16 AR reference. A float32 AR load runs only when the resident copy fits in free GPU memory.
``office-ar-bf16-determinism-v1`` reuses this command on the archived mismatch prompts.
It compares two BF16 ``ar_generate`` runs, the whole-prompt path with prompt-tokenwise
decoding, and the top1-top2 gap at each archived fork. It does not load float32 weights.
``office-8b-traj-wall-by-length-v1`` reuses this command on the summarization slice.
It records ``wall_seconds`` at 64, 256 and 512 tokens. ``overhead_seconds`` is
``prefill_time``. ``draft_seconds`` and ``verify_seconds`` come from the decode loop.
``residual_seconds`` is wall minus those three buckets and may be negative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import time

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, direction, paired_deltas, prompt_sha256,
)
from studies.trajectory_observation.structure.office_qwen8b import QWEN3_8B
from studies.trajectory_observation.structure.office_tree import (
    NODE_BUDGET, EVAL as TREE_EVAL, SMOKE as TREE_SMOKE, _append, _identity as _run_identity,
    _load_model, _offline_caches, block_complexity, existing_output_dir, prompts_after, read_rows,
    token_hash,
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
PROTOCOL_FORK = 'office-ar-first-fork-v1'
FORK_PREFIX = 'sd-office-ar-first-fork-v1'
FORK_STAMP = '20261104-1244'
FORK_RESERVE_MIB = 4096
PROTOCOL_DET = 'office-ar-bf16-determinism-v1'
DET_PREFIX = 'sd-office-ar-bf16-determinism-v1'
PROTOCOL_LENGTH = 'office-8b-traj-wall-by-length-v1'
LENGTH_PREFIX = 'sd-office-8b-traj-wall-by-length-v1'
LENGTH_STAMP = '20261104-1420'
LENGTHS = (64, 256, 512)
LENGTH_FLAT_RATIO = 1.5
LENGTH_BUCKET_SHARE = 0.5
LENGTH_BUCKET_ORDER = (
    'draft_seconds', 'verify_seconds', 'overhead_seconds', 'residual_seconds',
)
DET_NEAR_GAP = 0.5
DET_LARGE_GAP = 2.0
DET_NEAR_SHARE_MAX = 0.20
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


def row_buckets(row):
    """Split one arm into the pre-registered buckets.

    ``overhead_seconds`` is the existing unsynced ``prefill_time``.
    ``residual_seconds`` is wall minus overhead, draft and verify. It may be
    negative and is not rewritten to zero.
    """
    wall = finite_wall(row.get('wall_seconds'))
    overhead = finite_wall(row.get('prefill_time'))
    draft = finite_wall(row.get('draft_seconds'))
    verify = finite_wall(row.get('verify_seconds'))
    if wall is None or overhead is None or draft is None or verify is None:
        return None
    return {
        'wall_seconds': wall,
        'overhead_seconds': overhead,
        'draft_seconds': draft,
        'verify_seconds': verify,
        'residual_seconds': wall - overhead - draft - verify,
    }


def bucket_report(full_rows, traj_rows):
    """Paired bucket deltas. An incomplete bucket is not a share."""
    full_buckets = [row_buckets(row) for row in full_rows]
    traj_buckets = [row_buckets(row) for row in traj_rows]
    if any(item is None for item in full_buckets + traj_buckets):
        return {'complete': False}
    report = {'complete': True}
    for key in LENGTH_BUCKET_ORDER:
        deltas = [traj[key] - full[key] for traj, full in zip(traj_buckets, full_buckets)]
        report[key] = {
            'delta': f'{key}_traj - {key}_full',
            'difference': bootstrap_mean_ci(deltas),
        }
    return report


def _decomp_label(piece, wall_mean):
    if wall_mean == 0:
        return 'decomp-split'
    means = {key: piece['buckets'][key]['difference']['mean'] for key in LENGTH_BUCKET_ORDER}
    same = {
        key: value for key, value in means.items()
        if value != 0 and (value < 0) == (wall_mean < 0)
    }
    if not same:
        return 'decomp-split'
    best = max(LENGTH_BUCKET_ORDER, key=lambda key: abs(same[key]) if key in same else -1.0)
    if abs(same[best]) < LENGTH_BUCKET_SHARE * abs(wall_mean):
        return 'decomp-split'
    return {
        'draft_seconds': 'decomp-draft',
        'verify_seconds': 'decomp-verify',
        'overhead_seconds': 'decomp-overhead',
        'residual_seconds': 'decomp-residual',
    }[best]


def length_decision(pieces):
    """Length label, then the bucket label on the largest absolute wall gap.

    A partial or failed length stays ``incomplete`` and does not pick a bucket.
    """
    if [piece.get('length') for piece in pieces] != list(LENGTHS):
        return {'length_label': 'incomplete', 'decomp_label': None, 'decomp_length': None}
    if any(not piece.get('complete') for piece in pieces):
        return {'length_label': 'incomplete', 'decomp_label': None, 'decomp_length': None}
    directions = [piece['wall']['wall_direction'] for piece in pieces]
    means = [piece['wall']['wall_difference']['mean'] for piece in pieces]
    absolutes = [abs(mean) for mean in means]
    ratio = None if min(absolutes) == 0 else max(absolutes) / min(absolutes)
    growing = absolutes[0] < absolutes[1] < absolutes[2]
    if (
        all(direction == 'shorter' for direction in directions)
        and growing
        and ratio is not None
        and ratio >= LENGTH_FLAT_RATIO
    ):
        label = 'length-dependent'
    elif len(set(directions)) == 1 and (
        directions[0] == 'flat' or (ratio is not None and ratio < LENGTH_FLAT_RATIO)
    ):
        label = 'length-flat'
    elif directions[1] != 'shorter' and directions[2] != 'shorter':
        label = 'dead-metric'
    else:
        label = 'mixed'
    host = max(range(len(pieces)), key=lambda index: absolutes[index])
    return {
        'length_label': label,
        'decomp_label': _decomp_label(pieces[host], means[host]),
        'decomp_length': pieces[host]['length'],
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
    if protocol == 'first_fork':
        return {
            'protocol': PROTOCOL_FORK,
            'prefix': FORK_PREFIX,
            'stamp': FORK_STAMP,
            'select': prompts_summarization,
            'check': assert_registered_xcat7_slice,
            'take': TAKE_XCAT7,
        }
    if protocol == 'wall_by_length':
        return {
            'protocol': PROTOCOL_LENGTH,
            'prefix': LENGTH_PREFIX,
            'stamp': LENGTH_STAMP,
            'select': prompts_summarization,
            'check': assert_registered_xcat7_slice,
            'take': TAKE_XCAT7,
            'lengths': LENGTHS,
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
        for key in ('prefill_time', 'decode_time', 'draft_seconds', 'verify_seconds'):
            if key not in stats:
                continue
            kept = finite_wall(stats[key])
            if kept is not None:
                row[key] = kept
    return row


def _id_list(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    import torch
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in value]


def hash_token_ids(ids):
    return hashlib.sha256(json.dumps(_id_list(ids)).encode()).hexdigest()


def first_fork(reference, candidate):
    """First differing index in the hashed token sequences. Equal sequences leave the index empty."""
    left = _id_list(reference)
    right = _id_list(candidate)
    shared = 0
    limit = min(len(left), len(right))
    while shared < limit and left[shared] == right[shared]:
        shared += 1
    if shared == len(left) == len(right):
        return {
            'first_divergence_index': None,
            'common_prefix_length': shared,
            'side_tokens': None,
        }
    sides = {}
    if shared < len(left):
        sides['reference'] = int(left[shared])
    if shared < len(right):
        sides['candidate'] = int(right[shared])
    return {
        'first_divergence_index': shared,
        'common_prefix_length': shared,
        'side_tokens': sides,
    }


def attach_fork(row, reference, candidate, prefix=''):
    fork = first_fork(reference, candidate)
    row[f'{prefix}first_divergence_index'] = fork['first_divergence_index']
    row[f'{prefix}common_prefix_length'] = fork['common_prefix_length']
    key = f'{prefix}side_tokens'
    if fork['side_tokens'] is None:
        row.pop(key, None)
    else:
        row[key] = fork['side_tokens']
    return row


def fp32_weight_mib(parameter_count):
    return math.ceil(int(parameter_count) * 4 / (1024 * 1024))


def fp32_allocation_fits(parameter_count, free_mib, reserve_mib=FORK_RESERVE_MIB):
    need = fp32_weight_mib(parameter_count)
    return {
        'fp32_weight_mib': need,
        'free_mib': int(free_mib),
        'reserve_mib': int(reserve_mib),
        'fits': int(free_mib) >= need + int(reserve_mib),
    }


def resident_parameter_count(model):
    return sum(int(parameter.numel()) for parameter in model.hf.parameters())


def fp32_fields(reference_ids, output_ids, fp32_ids):
    fields = {
        'fp32_ar_equal': hash_token_ids(fp32_ids) == hash_token_ids(output_ids),
        'ar_fp32_equal': hash_token_ids(fp32_ids) == hash_token_ids(reference_ids),
    }
    attach_fork(fields, fp32_ids, output_ids, prefix='fp32_')
    attach_fork(fields, reference_ids, fp32_ids, prefix='ar_fp32_')
    return fields


def _measured_arm(model, spec, ids, limit):
    """One new decoder. The wall covers only this arm's generate."""
    import spec_mtp.structure.algorithm.decode.loop as decode_mod
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
        row = arm_row(model, spec, ids, limit, keep_output=True)
    finally:
        decode_mod.SpecDecoder.generate = original
    output_ids = row.pop('output_ids')
    return _store_clock(row, captured.get('stats', {}), captured.get('wall')), output_ids


def decode_measured_prompt(model, tokenizer, prompt, specs, limit, keep_sequences=False):
    """Independent AR, then full, then traj. Only the two arms get a wall clock.

    The fork fields compare each arm with that AR sequence. Argmax stays on the arm row.
    """
    from spec_mtp.structure.algorithm.decode.loop import ar_generate

    ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
    reference, _stats = ar_generate(model, ids, limit)
    reference_ids = _id_list(reference)
    reference_hash = token_hash(reference)
    rows = {}
    for name, spec in specs:
        row, output_ids = _measured_arm(model, spec, ids, limit)
        row['qid'] = prompt['qid']
        row['category'] = prompt['category']
        row['prompt_length'] = int(ids.shape[1])
        row['ar_equal'] = row['output_sha256'] == reference_hash
        attach_fork(row, reference_ids, output_ids)
        if keep_sequences:
            row['output_ids'] = output_ids
            row['reference_ids'] = reference_ids
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
    for key in ('prefill_time', 'decode_time', 'draft_seconds', 'verify_seconds'):
        if key in row and finite_wall(row[key]) is not None:
            recorded[key] = row[key]
    for key in ('prompt_length', 'first_divergence_index', 'common_prefix_length'):
        if key in row:
            recorded[key] = row[key]
    if 'side_tokens' in row:
        recorded['side_tokens'] = row['side_tokens']
    return recorded


def _smoke(model, tokenizer, prompts, dest, cfg):
    rows = []
    for prompt in prompts[:2]:
        started = time.perf_counter()
        decoded = decode_measured_prompt(model, tokenizer, prompt, _specs(), SMOKE)
        if cfg['protocol'] == PROTOCOL_FORK:
            for row in decoded.values():
                if 'first_divergence_index' not in row or 'common_prefix_length' not in row:
                    raise RuntimeError('smoke first fork missing')
        rows.append({name: _smoke_view(row) for name, row in decoded.items()})
        print(f"smoke {prompt['qid']} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {'protocol': cfg['protocol'], 'role': 'exploration', 'prompts': 2,
               'qids': [prompt['qid'] for prompt in prompts[:2]],
               'tokens': SMOKE['max_new_tokens'], 'arms': rows,
               'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values()),
               'wall_all_recorded': all('wall_seconds' in arm for item in rows for arm in item.values())}
    if cfg['protocol'] == PROTOCOL_XCAT7:
        payload['wall_protocol'] = PROTOCOL_WALL
    if cfg['protocol'] == PROTOCOL_LENGTH:
        payload['wall_protocol'] = PROTOCOL_LENGTH
        payload['lengths'] = list(LENGTHS)
    _file(dest, 'smoke', 'json', cfg['prefix'], cfg['stamp']).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    if not payload['wall_all_recorded']:
        raise RuntimeError('smoke wall clock missing')
    if cfg['protocol'] == PROTOCOL_LENGTH:
        needed = ('wall_seconds', 'prefill_time', 'draft_seconds', 'verify_seconds')
        recorded = all(
            finite_wall(arm.get(key)) is not None
            for item in rows for arm in item.values() for key in needed
        )
        if not recorded:
            raise RuntimeError('smoke bucket missing')
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
            'protocol': cfg['protocol'], 'role': 'exploration', 'source': _run_identity(),
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
        decoded = decode_measured_prompt(
            model, tokenizer, prompt, _specs(), EVAL,
            keep_sequences=cfg['protocol'] == PROTOCOL_FORK,
        )
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


def _length_path(dest, name, length, suffix):
    return dest / f'{LENGTH_PREFIX}-{name}-{length}-{LENGTH_STAMP}.{suffix}'


def _length_piece(full_rows, traj_rows, length, digest, skipped):
    if len(full_rows) != TAKE_XCAT7 or len(traj_rows) != TAKE_XCAT7:
        return {'length': length, 'complete': False, 'prompts': len(full_rows)}
    inner = ablation_summary(
        {'full': full_rows, 'traj': traj_rows},
        protocol=PROTOCOL_LENGTH, digest=digest, skipped=skipped, take=TAKE_XCAT7,
    )
    buckets = bucket_report(full_rows, traj_rows)
    walls = [finite_wall(row.get('wall_seconds')) for row in full_rows]
    full_mean = None if any(value is None for value in walls) else statistics.mean(walls)
    wall_mean = inner['wall']['wall_difference']['mean'] if inner['wall'].get('complete') else None
    relative = None
    if full_mean not in (None, 0) and wall_mean is not None:
        relative = abs(wall_mean) / full_mean
    return {
        'length': length,
        'complete': bool(inner['argmax_all_passed'] and inner['wall'].get('complete') and buckets.get('complete')),
        'prompts': TAKE_XCAT7,
        'argmax_all_passed': inner['argmax_all_passed'],
        'comparison': inner['comparison'],
        'wall': inner['wall'],
        'buckets': buckets,
        'relative_abs_delta': relative,
    }


def _write_length_summary(dest, pieces, digest, skipped):
    payload = {
        'protocol': PROTOCOL_LENGTH,
        'role': ROLE,
        'confirmatory': False,
        'prompts': TAKE_XCAT7,
        'prompt_sha256': digest,
        'skipped_qids': list(skipped),
        'lengths': list(LENGTHS),
        'by_length': {str(piece['length']): piece for piece in pieces},
    }
    payload.update(length_decision(pieces))
    _file(dest, 'summary', 'json', LENGTH_PREFIX, LENGTH_STAMP).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'length_label': payload['length_label'],
        'decomp_label': payload['decomp_label'],
        'decomp_length': payload['decomp_length'],
    }, ensure_ascii=False), flush=True)
    return payload


def _formal_by_length(model, tokenizer, prompts, skipped, digest, dest, cfg):
    """One warmup, then lengths 64, 256, 512. An incomplete length stops the rest."""
    if cfg['protocol'] != PROTOCOL_LENGTH:
        raise RuntimeError('length formal is only for the length protocol')
    pieces = []
    _warmup_arms(model, tokenizer, prompts[0])
    for length in LENGTHS:
        paths = {name: _length_path(dest, name, length, 'jsonl') for name, _gain in ARMS}
        done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
        if len(set(map(frozenset, done.values()))) != 1:
            raise RuntimeError('arms resumed at different prompts')
        header = _length_path(dest, 'header', length, 'json')
        if not header.exists():
            header.write_text(json.dumps({
                'protocol': PROTOCOL_LENGTH,
                'role': 'exploration',
                'source': _run_identity(),
                'length': length,
                'lengths': list(LENGTHS),
                'eval': {**EVAL, 'max_new_tokens': length},
                'arms': {name: spec for name, spec in _specs()},
                'prompt_sha256': digest,
                'qids': [prompt['qid'] for prompt in prompts],
                'skipped_qids': list(skipped),
                'delta': 'accepted_traj - accepted_full',
                'wall_seconds': 'sync_device and perf_counter around this arm SpecDecoder.generate',
                'wall_delta': 'wall_seconds_traj - wall_seconds_full',
                'overhead_seconds': 'prefill_time',
                'draft_seconds': 'unsynced tree build and block assembly before the decode forward',
                'verify_seconds': 'unsynced decode forward plus the sample-matching walk',
                'residual_seconds': 'wall_seconds - overhead_seconds - draft_seconds - verify_seconds',
                'bootstrap': {'seed': 0, 'resamples': 10000},
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        limit = dict(EVAL)
        limit['max_new_tokens'] = length
        finished = done['full']
        for prompt in prompts:
            if prompt['qid'] in finished:
                continue
            started = time.perf_counter()
            decoded = decode_measured_prompt(model, tokenizer, prompt, _specs(), limit)
            for name, row in decoded.items():
                row['max_new_tokens'] = length
                _append(paths[name], row)
            print(
                f"eval {length} {prompt['qid']} full={decoded['full']['accepted']} "
                f"traj={decoded['traj']['accepted']} nodes={decoded['full']['max_nodes']} "
                f"wall={_shown_wall(decoded['full'])}/{_shown_wall(decoded['traj'])} "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        piece = _length_piece(read_rows(paths['full']), read_rows(paths['traj']), length, digest, skipped)
        pieces.append(piece)
        _write_length_summary(dest, pieces, digest, skipped)
        if not piece['complete']:
            print(f'length {length} incomplete', flush=True)
            return pieces
    return pieces


# Offline analysis of archived jsonl. No command entry.
PROTOCOL_FAILSET = 'office-8b-traj-failset-v1'
PROTOCOL_AR = 'office-ar-inconsistency-v1'
WALL_PREFIX = 'sd-office-8b-traj-wall-v1'
WALL_STAMP = '20261104-0758'
MISSING_FIELDS = (
    'first_divergence_index',
    'common_prefix_length',
    'side_tokens',
)
# Chapter 9 B table. xcat7 keeps the archived acceptance files. The wall-clock
# run writes a different prefix and stamp, and those files are not this subset.
GROUPS = (
    {
        'key': 'ablate', 'protocol': PROTOCOL, 'category': 'creative_writing', 'take': TAKE,
        'sha256': EVAL_SHA256, 'archived_losses': 5,
        'prefix': 'sd-office-8b-traj-ablate', 'stamp': '20261103-0905',
        'select': prompts_from_647, 'check': assert_registered_slice,
    },
    {
        'key': 'xcat', 'protocol': PROTOCOL_XCAT, 'category': 'open_qa', 'take': TAKE_XCAT,
        'sha256': EVAL_SHA256_XCAT, 'archived_losses': 12,
        'prefix': 'sd-office-8b-traj-xcat', 'stamp': '20261103-0929',
        'select': prompts_open_qa, 'check': assert_registered_xcat_slice,
    },
    {
        'key': 'xcat2', 'protocol': PROTOCOL_XCAT2, 'category': 'general_qa', 'take': TAKE_XCAT2,
        'sha256': EVAL_SHA256_XCAT2, 'archived_losses': 17,
        'prefix': 'sd-office-8b-traj-xcat2', 'stamp': '20261103-0958',
        'select': prompts_general_qa, 'check': assert_registered_xcat2_slice,
    },
    {
        'key': 'xcat3', 'protocol': PROTOCOL_XCAT3, 'category': 'classification', 'take': TAKE_XCAT3,
        'sha256': EVAL_SHA256_XCAT3, 'archived_losses': 23,
        'prefix': 'sd-office-8b-traj-xcat3', 'stamp': '20261103-1024',
        'select': prompts_classification, 'check': assert_registered_xcat3_slice,
    },
    {
        'key': 'xcat4', 'protocol': PROTOCOL_XCAT4, 'category': 'closed_qa', 'take': TAKE_XCAT4,
        'sha256': EVAL_SHA256_XCAT4, 'archived_losses': 22,
        'prefix': 'sd-office-8b-traj-xcat4', 'stamp': '20261104-032457',
        'select': prompts_closed_qa, 'check': assert_registered_xcat4_slice,
    },
    {
        'key': 'xcat5', 'protocol': PROTOCOL_XCAT5, 'category': 'brainstorming', 'take': TAKE_XCAT5,
        'sha256': EVAL_SHA256_XCAT5, 'archived_losses': 17,
        'prefix': 'sd-office-8b-traj-xcat5', 'stamp': '20261104-035230',
        'select': prompts_brainstorming, 'check': assert_registered_xcat5_slice,
    },
    {
        'key': 'xcat6', 'protocol': PROTOCOL_XCAT6, 'category': 'information_extraction',
        'take': TAKE_XCAT6, 'sha256': EVAL_SHA256_XCAT6, 'archived_losses': 10,
        'prefix': 'sd-office-8b-traj-xcat6', 'stamp': '20261104-042546',
        'select': prompts_information_extraction, 'check': assert_registered_xcat6_slice,
    },
    {
        'key': 'xcat7', 'protocol': PROTOCOL_XCAT7, 'category': 'summarization', 'take': TAKE_XCAT7,
        'sha256': EVAL_SHA256_XCAT7, 'archived_losses': 17,
        'prefix': 'sd-office-8b-traj-xcat7', 'stamp': '20261104-045236',
        'select': prompts_summarization, 'check': assert_registered_xcat7_slice,
    },
)


def archived_paths(dest, group):
    root = Path(dest)
    prefix = group['prefix']
    stamp = group['stamp']
    return (
        root / f'{prefix}-full-{stamp}.jsonl',
        root / f'{prefix}-traj-{stamp}.jsonl',
    )


def _identity(group, status, reason):
    return {
        'protocol': group['protocol'],
        'category': group['category'],
        'status': status,
        'reason': reason,
        'prefix': group['prefix'],
        'stamp': group['stamp'],
        'archived_losses': group['archived_losses'],
        'expected_sha256': group['sha256'],
    }


def _center(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None}
    return {
        'count': len(values),
        'mean': sum(values) / len(values),
        'median': statistics.median(values),
    }


def _delta_summary(deltas):
    return {
        'count': len(deltas),
        'min': min(deltas),
        'median': statistics.median(deltas),
        'max': max(deltas),
        'mean': sum(deltas) / len(deltas),
        'sum': sum(deltas),
    }


def _by_qid(rows):
    found = {}
    for row in rows:
        if 'qid' not in row or 'accepted' not in row or 'ar_equal' not in row:
            return None
        qid = row['qid']
        if qid in found or type(row['ar_equal']) is not bool:
            return None
        found[qid] = row
    return found


def _false_counts(items):
    return {
        'full': sum(not item['ar_equal_full'] for item in items),
        'traj': sum(not item['ar_equal_traj'] for item in items),
        'either': sum((not item['ar_equal_full']) or (not item['ar_equal_traj']) for item in items),
    }


def _bucket_lengths(items, tokenizer):
    texts = [item['text'] for item in items]
    payload = {'chars': _center([len(text) for text in texts])}
    if tokenizer is None:
        payload['tokens'] = None
        return payload
    try:
        payload['tokens'] = _center([len(tokenizer(text)['input_ids']) for text in texts])
    except Exception:
        payload['tokens'] = None
    return payload


def describe_paired(group, prompts, full_rows, traj_rows, digest, tokenizer=None):
    """Describe one archived pair. A count or qid mismatch stops that group."""
    base = _identity(group, 'stopped', '')
    base['prompt_sha256'] = digest
    if digest != group['sha256']:
        base['reason'] = 'prompt sha256 does not match the frozen slice'
        return base
    full = _by_qid(full_rows)
    traj = _by_qid(traj_rows)
    expected = tuple(sorted(prompt['qid'] for prompt in prompts))
    if (full is None or traj is None or len(prompts) != group['take']
            or tuple(sorted(full)) != expected or tuple(sorted(traj)) != expected):
        base['reason'] = 'archived qids do not match the frozen slice'
        return base
    texts = {prompt['qid']: prompt['text'] for prompt in prompts}
    paired = []
    for qid in tuple(prompt['qid'] for prompt in prompts):
        accepted_full = full[qid]['accepted']
        accepted_traj = traj[qid]['accepted']
        paired.append({
            'qid': qid,
            'text': texts[qid],
            'accepted_full': accepted_full,
            'accepted_traj': accepted_traj,
            'delta': accepted_full - accepted_traj,
            'ar_equal_full': full[qid]['ar_equal'],
            'ar_equal_traj': traj[qid]['ar_equal'],
        })
    subset = [item for item in paired if item['delta'] > 0]
    higher = [item for item in paired if item['delta'] < 0]
    tied = [item for item in paired if item['delta'] == 0]
    if len(subset) != group['archived_losses']:
        base['reason'] = 'subset count does not match archived losses'
        base['observed_subset_count'] = len(subset)
        return base
    rest = higher + tied
    return {
        'protocol': group['protocol'],
        'category': group['category'],
        'status': 'described',
        'prefix': group['prefix'],
        'stamp': group['stamp'],
        'prompt_sha256': digest,
        'prompts': len(paired),
        'archived_losses': group['archived_losses'],
        'subset': [{'qid': item['qid'], 'delta': item['delta']} for item in subset],
        'lengths': {
            'subset': _bucket_lengths(subset, tokenizer),
            'traj_higher': _bucket_lengths(higher, tokenizer),
            'tie': _bucket_lengths(tied, tokenizer),
        },
        'ar_equal_false': {
            'subset': _false_counts(subset),
            'rest': _false_counts(rest),
        },
        'delta': _delta_summary([item['delta'] for item in subset]),
        'counts': {'subset': len(subset), 'traj_higher': len(higher), 'tie': len(tied)},
    }


def report_group(group, dest, prompts, digest, tokenizer=None):
    full_path, traj_path = archived_paths(dest, group)
    if not full_path.is_file() or not traj_path.is_file():
        missing = _identity(group, 'unknown', 'archived jsonl missing')
        missing['prompt_sha256'] = digest
        return missing
    return describe_paired(
        group, prompts, read_rows(full_path), read_rows(traj_path), digest, tokenizer)


def analyze_archive(dest, rows, tokenizer=None):
    """One report per frozen group. A bad group does not rewrite another group."""
    reports = []
    for group in GROUPS:
        prompts, skipped = group['select'](rows)
        digest = group['check'](prompts, skipped)
        reports.append(report_group(group, dest, prompts, digest, tokenizer))
    return {'protocol': PROTOCOL_FAILSET, 'groups': reports}


def load_local_tokenizer(path=None, loader=None):
    """Qwen3-8B tokenizer only. ``local_files_only`` refuses a download."""
    root = QWEN3_8B if path is None else Path(path)
    if not root.is_dir():
        return None
    if loader is None:
        from transformers import AutoTokenizer

        loader = AutoTokenizer.from_pretrained
    try:
        return loader(str(root), local_files_only=True)
    except Exception:
        return None


def _eighth(n):
    return (n + 7) // 8


def _sixteenth(n):
    return (n + 15) // 16


def pair_relation(n, false_full, false_traj, only_one, both_false):
    """Separation, overlap, or all-equal. Anything else stays unlabeled."""
    if false_full == 0 and false_traj == 0:
        return '全一致'
    gap = abs(false_full - false_traj)
    if gap >= _eighth(n) and only_one >= _eighth(n):
        return '分离'
    if both_false >= only_one and gap < _sixteenth(n):
        return '重叠'
    return None


def jaccard(left, right):
    """Intersection over union. Two empty mismatch sets are 1."""
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_band(value):
    if value >= 0.80:
        return '稳定'
    if value < 0.50:
        return '不稳定'
    return None


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def acceptance_label(mean_inconsistent, mean_consistent, n_inconsistent, n_consistent):
    """Acceptance-gap movement. Tau is not an input. Short groups stay unlabeled."""
    if n_inconsistent < 8 or n_consistent < 8:
        return None
    if mean_inconsistent is None or mean_consistent is None:
        return None
    diff = abs(mean_inconsistent - mean_consistent)
    same = ((mean_inconsistent > 0 and mean_consistent > 0)
            or (mean_inconsistent < 0 and mean_consistent < 0))
    opposite = ((mean_inconsistent > 0 and mean_consistent < 0)
                or (mean_inconsistent < 0 and mean_consistent > 0))
    if same and diff < 0.5:
        return '不随分组变动'
    if opposite or diff >= 1:
        return '随分组变动'
    return None


def choose_conclusion(groups, stability):
    """First matching chapter-10 label. Unknown and invalid come before the three."""
    unknown = sum(item['status'] == 'unknown' for item in groups)
    if stability.get('status') == 'unknown' or unknown > 2:
        return '未知'
    if any(item['status'] == 'invalid' for item in groups) or stability.get('status') == 'invalid':
        return '无效'
    described = [item for item in groups if item['status'] == 'described']
    separated = sum(item.get('relation') == '分离' for item in described)
    shifting = sum(item.get('acceptance_label') == '随分组变动' for item in described)
    bands = (stability['full']['band'], stability['traj']['band'])
    if separated == 0 and shifting == 0 and bands == ('不稳定', '不稳定'):
        return '保持诊断'
    if '稳定' in bands and (separated >= 4 or shifting >= 4):
        return '正确性风险需处理'
    return '可隔离的口径问题'


def wall_group():
    """Summarization sentences stay on the frozen xcat7 slice. Files are the wall pair."""
    source = next(group for group in GROUPS if group['key'] == 'xcat7')
    return {
        'key': 'wall',
        'protocol': PROTOCOL_WALL,
        'category': source['category'],
        'take': source['take'],
        'sha256': source['sha256'],
        'prefix': WALL_PREFIX,
        'stamp': WALL_STAMP,
        'select': source['select'],
        'check': source['check'],
    }


def _number(value):
    return type(value) in (int, float)


def _arm_index(rows):
    """Map qid to the fields this protocol reads. A bad row is not an index."""
    found = {}
    argmax_failed = False
    for row in rows:
        if 'qid' not in row or 'accepted' not in row or 'ar_equal' not in row:
            return None, False
        if type(row['ar_equal']) is not bool or not _number(row['accepted']):
            return None, False
        qid = row['qid']
        if qid in found:
            return None, False
        if row.get('argmax_passed') is not True:
            argmax_failed = True
        tau = row.get('tau')
        if row.get('decode_calls') == 0 or not _number(tau):
            tau = None
        found[qid] = {
            'accepted': row['accepted'],
            'ar_equal': row['ar_equal'],
            'tau': tau,
        }
    return found, argmax_failed


def _tau_mean(items, key):
    return _mean([item[key] for item in items if item[key] is not None])


def classify_pair(group, prompts, full_rows, traj_rows, digest):
    """One full/traj pair. Missing files are handled by the caller."""
    base = {
        'key': group['key'],
        'protocol': group['protocol'],
        'category': group['category'],
        'prefix': group['prefix'],
        'stamp': group['stamp'],
        'missing_fields': list(MISSING_FIELDS),
    }
    if digest != group['sha256']:
        base.update(status='unknown', reason='prompt sha256 does not match the frozen slice')
        return base
    full, full_argmax = _arm_index(full_rows)
    traj, traj_argmax = _arm_index(traj_rows)
    expected = tuple(sorted(prompt['qid'] for prompt in prompts))
    if (full is None or traj is None or len(prompts) != group['take']
            or tuple(sorted(full)) != expected or tuple(sorted(traj)) != expected):
        base.update(status='unknown', reason='archived qids do not match the frozen slice')
        return base
    n = len(expected)
    if full_argmax or traj_argmax:
        base.update(status='invalid', reason='argmax failed', n=n, argmax_all_passed=False)
        return base
    false_qids = {'full': [], 'traj': []}
    only_one = 0
    both_false = 0
    inconsistent = []
    consistent = []
    for qid in expected:
        left, right = full[qid], traj[qid]
        if not left['ar_equal']:
            false_qids['full'].append(qid)
        if not right['ar_equal']:
            false_qids['traj'].append(qid)
        left_false = not left['ar_equal']
        right_false = not right['ar_equal']
        if left_false and right_false:
            both_false += 1
        elif left_false or right_false:
            only_one += 1
        bucket = inconsistent if left_false or right_false else consistent
        bucket.append({
            'delta': right['accepted'] - left['accepted'],
            'tau_full': left['tau'],
            'tau_traj': right['tau'],
        })
    false_full = len(false_qids['full'])
    false_traj = len(false_qids['traj'])
    mean_in = _mean([item['delta'] for item in inconsistent])
    mean_ok = _mean([item['delta'] for item in consistent])
    label = acceptance_label(mean_in, mean_ok, len(inconsistent), len(consistent))
    base.update(
        status='described',
        n=n,
        argmax_all_passed=True,
        inconsistent={
            'full': {'count': false_full, 'rate': false_full / n},
            'traj': {'count': false_traj, 'rate': false_traj / n},
        },
        only_one=only_one,
        both_false=both_false,
        relation=pair_relation(n, false_full, false_traj, only_one, both_false),
        acceptance_label=label,
        acceptance={
            'inconsistent': {
                'count': len(inconsistent),
                'mean_delta': mean_in,
                'tau_full': _tau_mean(inconsistent, 'tau_full'),
                'tau_traj': _tau_mean(inconsistent, 'tau_traj'),
            },
            'consistent': {
                'count': len(consistent),
                'mean_delta': mean_ok,
                'tau_full': _tau_mean(consistent, 'tau_full'),
                'tau_traj': _tau_mean(consistent, 'tau_traj'),
            },
        },
        false_qids=false_qids,
        qids=list(expected),
    )
    return base


def _stability(xcat7, wall):
    if xcat7.get('status') == 'invalid' or wall.get('status') == 'invalid':
        return {'status': 'invalid', 'reason': 'summarization argmax failed'}
    if (xcat7.get('status') != 'described' or wall.get('status') != 'described'
            or xcat7.get('qids') != wall.get('qids')):
        return {'status': 'unknown', 'reason': 'summarization pair is not aligned'}
    report = {'status': 'compared'}
    for arm in ('full', 'traj'):
        value = jaccard(xcat7['false_qids'][arm], wall['false_qids'][arm])
        report[arm] = {
            'jaccard': value,
            'band': jaccard_band(value),
            'xcat7': len(xcat7['false_qids'][arm]),
            'wall': len(wall['false_qids'][arm]),
        }
    return report


def _classify_files(group, dest, prompts, digest):
    full_path, traj_path = archived_paths(dest, group)
    if not full_path.is_file() or not traj_path.is_file():
        return {
            'key': group['key'],
            'protocol': group['protocol'],
            'category': group['category'],
            'status': 'unknown',
            'reason': 'archived jsonl missing',
            'prefix': group['prefix'],
            'stamp': group['stamp'],
            'missing_fields': list(MISSING_FIELDS),
        }
    return classify_pair(group, prompts, read_rows(full_path), read_rows(traj_path), digest)


def classify_archive(dest, rows):
    """Chapter-10 report. One bad group does not rewrite the others."""
    reports = []
    for group in GROUPS:
        prompts, skipped = group['select'](rows)
        digest = group['check'](prompts, skipped)
        reports.append(_classify_files(group, dest, prompts, digest))
    wall = wall_group()
    prompts, skipped = wall['select'](rows)
    digest = wall['check'](prompts, skipped)
    wall_report = _classify_files(wall, dest, prompts, digest)
    xcat7 = next(item for item in reports if item['key'] == 'xcat7')
    stability = _stability(xcat7, wall_report)
    return {
        'protocol': PROTOCOL_AR,
        'conclusion': choose_conclusion(reports, stability),
        'missing_fields': list(MISSING_FIELDS),
        'groups': reports,
        'wall': wall_report,
        'stability': stability,
    }


def _same_fork(left, right):
    if left.get('first_divergence_index') != right.get('first_divergence_index'):
        return False
    if left.get('first_divergence_index') is None:
        return False
    left_sides = left.get('side_tokens') or {}
    right_sides = right.get('side_tokens') or {}
    return (
        'candidate' in left_sides and 'candidate' in right_sides
        and left_sides['candidate'] == right_sides['candidate']
        and left_sides.get('reference') == right_sides.get('reference')
    )


def _median(values):
    if not values:
        return None
    return statistics.median(values)


def _flag_rate(flags):
    if not flags:
        return None
    return sum(bool(flag) for flag in flags) / len(flags)


def _fp32_summary(full, traj):
    rows = list(full.values()) + list(traj.values())
    if any('fp32_ar_equal' not in row or 'ar_fp32_equal' not in row for row in rows):
        return None
    for qid in full:
        if full[qid]['ar_fp32_equal'] != traj[qid]['ar_fp32_equal']:
            return {'status': 'invalid', 'reason': 'bf16 and fp32 comparison differs across arms'}
    full_bf16 = _flag_rate([row['ar_equal'] for row in full.values()])
    traj_bf16 = _flag_rate([row['ar_equal'] for row in traj.values()])
    full_fp32 = _flag_rate([row['fp32_ar_equal'] for row in full.values()])
    traj_fp32 = _flag_rate([row['fp32_ar_equal'] for row in traj.values()])
    bf16_fp32 = _flag_rate([row['ar_fp32_equal'] for row in full.values()])
    full_gap = full_fp32 - full_bf16
    traj_gap = traj_fp32 - traj_bf16
    if full_gap >= 0.10 and traj_gap >= 0.10:
        label = 'bf16参照离开fp32'
        direction = 'positive'
    elif bf16_fp32 >= 0.80 and abs(full_gap) < 0.10 and abs(traj_gap) < 0.10:
        label = 'fp32解释不了这条不一致'
        direction = None
    else:
        label = None
        direction = None
    return {
        'status': 'described',
        'n': len(full),
        'bf16_fp32_equal_rate': bf16_fp32,
        'full_bf16_equal_rate': full_bf16,
        'traj_bf16_equal_rate': traj_bf16,
        'full_fp32_equal_rate': full_fp32,
        'traj_fp32_equal_rate': traj_fp32,
        'full_gap': full_gap,
        'traj_gap': traj_gap,
        'label': label,
        'h3_direction': direction,
    }


def describe_first_fork(full_rows, traj_rows):
    """Paired first-fork summary. Argmax failure stays on its own gate."""
    full = {}
    traj = {}
    for source, rows in ((full, full_rows), (traj, traj_rows)):
        for row in rows:
            qid = row.get('qid')
            if qid in source or qid is None:
                return {'status': 'unknown', 'reason': 'qids do not match'}
            source[qid] = row
    if set(full) != set(traj):
        return {'status': 'unknown', 'reason': 'qids do not match'}
    required = (
        'ar_equal', 'argmax_passed', 'first_divergence_index',
        'common_prefix_length', 'prompt_length', 'output_sha256',
    )
    combined = list(full.values()) + list(traj.values())
    for row in combined:
        if any(key not in row for key in required):
            return {'status': 'unknown', 'reason': 'first fork fields missing'}
        if type(row['ar_equal']) is not bool or type(row['argmax_passed']) is not bool:
            return {'status': 'unknown', 'reason': 'flags are not bool'}
    n = len(full)
    if any(not row['argmax_passed'] for row in combined):
        return {
            'status': 'invalid', 'reason': 'argmax failed', 'n': n, 'argmax_all_passed': False,
        }
    for row in combined:
        diverged = row['first_divergence_index'] is not None
        if row['ar_equal'] == diverged:
            return {'status': 'invalid', 'reason': 'fork disagrees with ar_equal', 'n': n}
    mismatched = []
    inside = 0
    offsets = {'full': [], 'traj': []}
    for qid in sorted(full):
        left, right = full[qid], traj[qid]
        for name, row in (('full', left), ('traj', right)):
            index = row['first_divergence_index']
            if index is not None and index < row['prompt_length']:
                inside += 1
            if not row['ar_equal']:
                offsets[name].append(index - row['prompt_length'])
        if (not left['ar_equal']) or (not right['ar_equal']):
            mismatched.append((left, right))
    mismatch_count = len(mismatched)
    shared = sum(1 for left, right in mismatched if _same_fork(left, right))
    hash_equal = sum(1 for left, right in mismatched if left['output_sha256'] == right['output_sha256'])
    rate = None if mismatch_count == 0 else shared / mismatch_count
    hash_rate = None if mismatch_count == 0 else hash_equal / mismatch_count
    if mismatch_count == 0:
        label = None
        direction = None
        relation = '全一致'
    elif rate >= 0.80 and inside == 0:
        label = '共享参照'
        direction = 'positive'
        relation = label
    elif rate < 0.50:
        label = '分叉分开'
        direction = 'negative'
        relation = label
    else:
        label = None
        direction = None
        relation = None
    report = {
        'status': 'described',
        'n': n,
        'argmax_all_passed': True,
        'mismatch_count': mismatch_count,
        'shared_fork_count': shared,
        'shared_fork_rate': rate,
        'arms_hash_equal_rate': hash_rate,
        'fork_inside_prompt': inside,
        'continuation_offset_median': {
            'full': _median(offsets['full']),
            'traj': _median(offsets['traj']),
        },
        'label': label,
        'h1_direction': direction,
        'relation': relation,
    }
    fp32 = _fp32_summary(full, traj)
    if fp32 is not None:
        report['fp32'] = fp32
    return report


def fill_fp32_rows(model, tokenizer, prompts, rows_by_name, limit, generate):
    import torch
    if model.dtype != torch.float32:
        raise RuntimeError(f'fp32 control expected float32, got {model.dtype}')
    by_prompt = {prompt['qid']: prompt for prompt in prompts}
    produced = {}
    updated = {}
    for name, rows in rows_by_name.items():
        arm_rows = []
        for row in rows:
            qid = row['qid']
            if 'output_ids' not in row or 'reference_ids' not in row:
                raise RuntimeError('fp32 control needs the recorded token sequences')
            if qid not in produced:
                if qid not in by_prompt:
                    raise RuntimeError('fp32 control prompt is missing')
                ids = tokenizer(by_prompt[qid]['text'], return_tensors='pt').input_ids
                reference, _stats = generate(model, ids, limit)
                produced[qid] = _id_list(reference)
            merged = dict(row)
            merged.update(fp32_fields(row['reference_ids'], row['output_ids'], produced[qid]))
            arm_rows.append(merged)
        updated[name] = arm_rows
    return updated


def _load_fp32_model():
    import torch
    from spec_mtp.model import HFCausalLM

    path = QWEN3_8B
    if not path.is_dir():
        raise RuntimeError('Qwen3-8B weights are not at the local 8B directory')
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='float32', attn_implementation='eager')
    model.hf.to('cuda')
    if model.dtype != torch.float32 or not str(model.device).startswith('cuda'):
        raise RuntimeError(f'fp32 AR is on {model.device} with dtype {model.dtype}')
    return model


def gpu_free_mib():
    import subprocess
    free = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if free.returncode != 0 or not free.stdout.strip():
        raise RuntimeError('nvidia-smi could not report free GPU memory')
    return int(free.stdout.strip().splitlines()[0])


def release_session(model):
    import gc
    close = getattr(model, 'close', None)
    if close is not None:
        close()
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish_first_fork(dest, parameter_count, free_mib, loader=None, generate=None,
                      tokenizer=None, prompts=None, limit=None):
    """Record float32 AR only when its resident weights fit. Otherwise leave the gate result."""
    gate = fp32_allocation_fits(parameter_count, free_mib)
    payload = {'protocol': PROTOCOL_FORK, 'parameter_count': int(parameter_count), **gate}
    full_path = _file(dest, 'full', 'jsonl', FORK_PREFIX, FORK_STAMP)
    traj_path = _file(dest, 'traj', 'jsonl', FORK_PREFIX, FORK_STAMP)
    if not gate['fits']:
        payload['status'] = 'not_loaded'
        payload['reason'] = 'fp32 resident weights exceed free GPU memory'
    elif not full_path.exists() or not traj_path.exists():
        payload['status'] = 'not_loaded'
        payload['reason'] = 'no arm sequences to compare'
    else:
        if loader is None or generate is None:
            raise RuntimeError('fp32 control needs a loader')
        model = loader()
        try:
            rows = fill_fp32_rows(
                model, tokenizer, prompts,
                {'full': read_rows(full_path), 'traj': read_rows(traj_path)},
                limit, generate,
            )
        finally:
            release_session(model)
        for name, path in (('full', full_path), ('traj', traj_path)):
            path.write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[name]),
                encoding='utf-8')
        payload['status'] = 'recorded'
    if full_path.exists() and traj_path.exists():
        payload['fork'] = describe_first_fork(read_rows(full_path), read_rows(traj_path))
    _file(dest, 'fork', 'json', FORK_PREFIX, FORK_STAMP).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def _prompt_length(ids):
    import torch
    if torch.is_tensor(ids):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise RuntimeError('prompt ids must be [1, length]')
        return int(ids.shape[1])
    values = _id_list(ids)
    if not values:
        raise RuntimeError('prompt ids must be [1, length]')
    return len(values)


def top2_gap(logits_row):
    """top1 minus top2 on one logits row. The subtraction is float32."""
    import torch
    row = logits_row.detach().float().reshape(-1)
    if int(row.numel()) < 2:
        raise RuntimeError('logit row needs two candidates')
    values, indices = torch.topk(row, k=2)
    gap = float(values[0] - values[1])
    if not math.isfinite(gap):
        raise RuntimeError('logit gap is not finite')
    return {'gap': gap, 'top1': int(indices[0]), 'top2': int(indices[1])}


def determinism_record(qid, prompt_length, whole_a, whole_b, tokenwise, gaps, new_tokens):
    """Identity of two whole-prompt runs against the tokenwise prompt path."""
    expected = int(prompt_length) + int(new_tokens)
    sequences = (whole_a, whole_b, tokenwise)
    record = {
        'qid': qid,
        'prompt_length': int(prompt_length),
        'double_run_equal': hash_token_ids(whole_a) == hash_token_ids(whole_b),
        'path_equal': hash_token_ids(whole_a) == hash_token_ids(tokenwise),
        'length_ok': all(len(_id_list(seq)) == expected for seq in sequences),
        'whole_sha256': hash_token_ids(whole_a),
        'repeat_sha256': hash_token_ids(whole_b),
        'tokenwise_sha256': hash_token_ids(tokenwise),
        'gaps': gaps,
    }
    if not record['double_run_equal']:
        record['run_fork'] = first_fork(whole_a, whole_b)
    if not record['path_equal']:
        record['path_fork'] = first_fork(whole_a, tokenwise)
    return record


def measure_determinism_prompt(model, ids, targets, limit, generate, logits_at):
    """Two whole-prompt AR runs, one tokenwise run, and archived-fork logit gaps."""
    prompt_length = _prompt_length(ids)
    if not targets or any(int(item['prompt_length']) != prompt_length for item in targets):
        raise RuntimeError('tokenizer length does not match the archived prompt')
    qid = targets[0]['qid']
    if any(item['qid'] != qid for item in targets):
        raise RuntimeError('gap targets mix prompts')
    reference = targets[0]['reference_ids']
    if any(item['reference_ids'] != reference for item in targets):
        raise RuntimeError('archived AR reference differs inside one prompt')
    whole_a, _stats = generate(model, ids, limit)
    whole_b, _stats = generate(model, ids, limit)
    tokenwise, _stats = generate(model, ids, limit, prompt_tokenwise=True)
    seen = {}
    gaps = []
    for item in targets:
        index = int(item['first_divergence_index'])
        if index not in seen:
            seen[index] = top2_gap(logits_at(model, reference, prompt_length, index))
        gap = dict(seen[index])
        gap['arm'] = item['arm']
        gap['first_divergence_index'] = index
        gap['reference_token'] = int(item['reference_token'])
        gap['top1_matches'] = gap['top1'] == gap['reference_token']
        gaps.append(gap)
    return determinism_record(qid, prompt_length, whole_a, whole_b, tokenwise, gaps, limit['max_new_tokens'])


def _finite_gap(value):
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value)


def describe_ar_determinism(rows):
    """Identity rates and the logit-gap band. Argmax, acceptance, and tau stay out."""
    if not rows:
        return {'status': 'unknown', 'reason': 'no prompts'}
    seen = set()
    parsed = []
    for row in rows:
        qid = row.get('qid')
        if qid in seen or qid is None:
            return {'status': 'unknown', 'reason': 'qids do not match'}
        seen.add(qid)
        if type(row.get('double_run_equal')) is not bool or type(row.get('path_equal')) is not bool:
            return {'status': 'unknown', 'reason': 'flags are not bool'}
        if type(row.get('length_ok')) is not bool:
            return {'status': 'unknown', 'reason': 'length flag missing'}
        gaps = row.get('gaps')
        if not isinstance(gaps, list) or not gaps:
            return {'status': 'unknown', 'reason': 'divergence gap missing'}
        parsed.append(row)
    if any(not row['length_ok'] for row in parsed):
        return {
            'status': 'invalid', 'reason': 'sequence length is not the budget', 'n': len(parsed),
        }
    flat = []
    collapsed = {}
    for row in parsed:
        for gap in row['gaps']:
            if gap.get('arm') not in ('full', 'traj'):
                return {'status': 'unknown', 'reason': 'gap arm missing'}
            if type(gap.get('top1_matches')) is not bool:
                return {'status': 'unknown', 'reason': 'replay flag is not bool'}
            index = gap.get('first_divergence_index')
            top1 = gap.get('top1')
            reference_token = gap.get('reference_token')
            if type(index) is not int or type(top1) is not int or type(reference_token) is not int:
                return {'status': 'unknown', 'reason': 'divergence index missing'}
            value = gap.get('gap')
            if not _finite_gap(value):
                return {'status': 'unknown', 'reason': 'logit gap is not finite'}
            current = (float(value), gap['top1_matches'], top1, reference_token)
            key = (row['qid'], index)
            if key in collapsed and collapsed[key] != current:
                return {'status': 'unknown', 'reason': 'repeated fork disagrees'}
            collapsed[key] = current
            flat.append(gap)
    values = [item[0] for item in collapsed.values()]
    matches = [item[1] for item in collapsed.values()]
    n = len(parsed)
    double_count = sum(row['double_run_equal'] for row in parsed)
    path_count = sum(row['path_equal'] for row in parsed)
    replay_count = sum(matches)
    double_rate = double_count / n
    path_rate = path_count / n
    replay_rate = replay_count / len(matches)
    near_rate = sum(value <= DET_NEAR_GAP for value in values) / len(values)
    large_rate = sum(value > DET_LARGE_GAP for value in values) / len(values)
    median = float(statistics.median(values))
    unstable = double_rate < 1 or path_rate < 1 or replay_rate < 1
    if unstable:
        label = 'AR路径不稳定'
        direction = 'positive'
    elif median <= DET_NEAR_GAP:
        label = '分叉贴近并列'
        direction = 'positive'
    elif median > DET_LARGE_GAP and near_rate < DET_NEAR_SHARE_MAX:
        label = '差距大且路径稳定'
        direction = 'negative'
    else:
        label = None
        direction = None
    return {
        'status': 'described',
        'n': n,
        'double_run_identity_count': double_count,
        'double_run_identity_rate': double_rate,
        'path_identity_count': path_count,
        'path_identity_rate': path_rate,
        'arm_gap_count': len(flat),
        'gap_count': len(collapsed),
        'replay_top1_match_count': replay_count,
        'replay_top1_match_rate': replay_rate,
        'gap_min': float(min(values)),
        'gap_median': median,
        'gap_max': float(max(values)),
        'near_gap_rate': near_rate,
        'large_gap_rate': large_rate,
        'near_gap': DET_NEAR_GAP,
        'large_gap': DET_LARGE_GAP,
        'near_share_max': DET_NEAR_SHARE_MAX,
        'label': label,
        'noise_direction': direction,
    }


def archived_mismatch_qids(full_rows, traj_rows):
    """Union of prompts where either arm left the archived BF16 AR."""
    full = {}
    traj = {}
    for source, rows in ((full, full_rows), (traj, traj_rows)):
        for row in rows:
            qid = row.get('qid')
            if type(qid) is not int or qid in source:
                raise RuntimeError('fork qids do not match')
            source[qid] = row
    if set(full) != set(traj):
        raise RuntimeError('fork qids do not match')
    found = []
    for qid in sorted(full):
        if type(full[qid].get('ar_equal')) is not bool or type(traj[qid].get('ar_equal')) is not bool:
            raise RuntimeError('ar_equal is not bool')
        if (not full[qid]['ar_equal']) or (not traj[qid]['ar_equal']):
            found.append(qid)
    return tuple(found), full, traj


def gap_targets_for_qids(full, traj, qids):
    """Archived continuation forks. The qid list must be that union, sorted."""
    if not isinstance(qids, list) or tuple(qids) != tuple(sorted(set(qids))):
        raise RuntimeError('mismatch qid list is not sorted and unique')
    found = tuple(
        qid for qid in sorted(full)
        if (not full[qid]['ar_equal']) or (not traj[qid]['ar_equal'])
    )
    if tuple(qids) != found:
        raise RuntimeError('mismatch qid list is not the archived union')
    targets = []
    for qid in qids:
        left, right = full[qid], traj[qid]
        if left.get('reference_ids') != right.get('reference_ids'):
            raise RuntimeError('archived AR reference differs across arms')
        reference = left.get('reference_ids')
        if not isinstance(reference, list) or not reference:
            raise RuntimeError('archived AR reference is missing')
        if type(left.get('prompt_length')) is not int or left['prompt_length'] != right.get('prompt_length'):
            raise RuntimeError('prompt length differs across arms')
        prompt_length = left['prompt_length']
        for arm, row in (('full', left), ('traj', right)):
            if row['ar_equal']:
                continue
            index = row.get('first_divergence_index')
            token = (row.get('side_tokens') or {}).get('reference')
            if type(index) is not int or type(token) is not int:
                raise RuntimeError('archived fork has no reference token')
            if index < prompt_length or index >= len(reference) or reference[index] != token:
                raise RuntimeError('archived fork index is outside the continuation')
            targets.append({
                'qid': qid,
                'arm': arm,
                'prompt_length': prompt_length,
                'first_divergence_index': index,
                'reference_token': token,
                'reference_ids': reference,
            })
    if not targets:
        raise RuntimeError('archived fork has no reference token')
    return targets


def load_mismatch_targets(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    full_rows = read_rows(Path(payload['source_full']))
    traj_rows = read_rows(Path(payload['source_traj']))
    _found, full, traj = archived_mismatch_qids(full_rows, traj_rows)
    return gap_targets_for_qids(full, traj, payload.get('mismatch_qids'))


def run_ar_determinism_main(dest, mismatch):
    """BF16-only AR determinism on the archived mismatch prompts. No fp32 load."""
    if mismatch is None or not Path(mismatch).is_file():
        raise RuntimeError('ar determinism needs the archived mismatch json')
    identity = _run_identity()
    if identity['dirty']:
        raise RuntimeError('formal run requires a clean worktree')
    from spec_mtp.structure.algorithm.decode.loop import ar_context_logits, ar_generate
    from studies.trajectory_observation.structure.office_qwen8b import load_dolly_rows

    prompts, skipped = prompts_summarization(load_dolly_rows())
    digest = assert_registered_xcat7_slice(prompts, skipped)
    targets = load_mismatch_targets(mismatch)
    by_qid = {}
    for item in targets:
        by_qid.setdefault(item['qid'], []).append(item)
    by_prompt = {prompt['qid']: prompt for prompt in prompts}
    chosen = []
    for qid in by_qid:
        if qid not in by_prompt:
            raise RuntimeError('mismatch qid is outside the summarization slice')
        chosen.append(by_prompt[qid])
    stamp = time.strftime('%Y%m%d-%H%M%S')
    header = {
        'protocol': PROTOCOL_DET,
        'role': 'exploration',
        'source': identity,
        'prompt_sha256': digest,
        'parent_prompts': len(prompts),
        'skipped_qids': list(skipped),
        'qids': [prompt['qid'] for prompt in chosen],
        'limit': dict(EVAL),
        'dtype': 'bfloat16',
        'fp32': 'not_requested',
        'near_gap': DET_NEAR_GAP,
        'large_gap': DET_LARGE_GAP,
        'near_share_max': DET_NEAR_SHARE_MAX,
    }
    _file(dest, 'header', 'json', DET_PREFIX, stamp).write_text(
        json.dumps(header, ensure_ascii=False, indent=2), encoding='utf-8')
    print('load 8b', flush=True)
    model, tokenizer, torch_version = _load_model('8b')
    started = time.perf_counter()
    try:
        path = _file(dest, 'rows', 'jsonl', DET_PREFIX, stamp)
        for prompt in chosen:
            prompt_started = time.perf_counter()
            ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
            row = measure_determinism_prompt(
                model, ids, by_qid[prompt['qid']], EVAL, ar_generate, ar_context_logits)
            _append(path, row)
            print(
                f"det {prompt['qid']} double={row['double_run_equal']} path={row['path_equal']} "
                f"gaps={len(row['gaps'])} {time.perf_counter() - prompt_started:.1f}s",
                flush=True,
            )
        rows = read_rows(path)
    finally:
        release_session(model)
    report = describe_ar_determinism(rows)
    payload = {
        'protocol': PROTOCOL_DET,
        'stamp': stamp,
        'source': identity,
        'torch': torch_version,
        'elapsed_seconds': time.perf_counter() - started,
        'report': report,
    }
    _file(dest, 'summary', 'json', DET_PREFIX, stamp).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('torch', torch_version, report.get('label'), flush=True)
    if report['status'] != 'described':
        raise RuntimeError(report.get('reason', report['status']))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--only', choices=('all', 'smoke', 'formal'), default='all')
    parser.add_argument('--mismatch', type=Path)
    parser.add_argument(
        '--protocol',
        choices=(
            'ablate', 'xcat', 'xcat2', 'xcat3', 'xcat4', 'xcat5', 'xcat6', 'xcat7',
            'first_fork', 'ar_determinism', 'wall_by_length',
        ),
        default='ablate')
    args = parser.parse_args(argv)
    dest = existing_output_dir(args.dest)
    if args.protocol == 'ar_determinism':
        run_ar_determinism_main(dest, args.mismatch)
        return
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
        if _run_identity()['dirty']:
            raise RuntimeError('formal run requires a clean worktree')
        if args.protocol == 'wall_by_length':
            _formal_by_length(model, tokenizer, prompts, skipped, digest, dest, cfg)
        else:
            _formal(model, tokenizer, prompts, skipped, digest, dest, cfg)
    print('torch', torch_version, flush=True)
    if args.protocol == 'first_fork':
        from spec_mtp.structure.algorithm.decode.loop import ar_generate
        count = resident_parameter_count(model)
        release_session(model)
        finish_first_fork(
            dest, count, gpu_free_mib(), loader=_load_fp32_model, generate=ar_generate,
            tokenizer=tokenizer, prompts=prompts, limit=EVAL,
        )


if __name__ == '__main__':
    main()
