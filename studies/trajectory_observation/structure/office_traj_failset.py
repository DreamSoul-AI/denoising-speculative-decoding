"""Offline description of full-win prompts for ``office-8b-traj-failset-v1``.

Prompt text and SHA256 stay in ``office_traj``. This module only pairs the
archived jsonl. It does not load a generator, fit K, or start a decode.
Import does not mutate the process environment. There is no command entry.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from studies.trajectory_observation.structure.office_qwen8b import QWEN3_8B
from studies.trajectory_observation.structure.office_traj import (
    EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2, EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4,
    EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7, PROTOCOL, PROTOCOL_XCAT,
    PROTOCOL_XCAT2, PROTOCOL_XCAT3, PROTOCOL_XCAT4, PROTOCOL_XCAT5, PROTOCOL_XCAT6, PROTOCOL_XCAT7,
    TAKE, TAKE_XCAT, TAKE_XCAT2, TAKE_XCAT3, TAKE_XCAT4, TAKE_XCAT5, TAKE_XCAT6, TAKE_XCAT7,
    assert_registered_slice, assert_registered_xcat2_slice, assert_registered_xcat3_slice,
    assert_registered_xcat4_slice, assert_registered_xcat5_slice, assert_registered_xcat6_slice,
    assert_registered_xcat7_slice, assert_registered_xcat_slice, prompts_brainstorming,
    prompts_classification, prompts_closed_qa, prompts_from_647, prompts_general_qa,
    prompts_information_extraction, prompts_open_qa, prompts_summarization,
)
from studies.trajectory_observation.structure.office_tree import read_rows

PROTOCOL_FAILSET = 'office-8b-traj-failset-v1'
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
