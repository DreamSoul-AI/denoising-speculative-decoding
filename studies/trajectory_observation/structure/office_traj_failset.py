"""Offline reads of archived office jsonl.

``office-8b-traj-failset-v1`` describes full-win prompts.
``office-ar-inconsistency-v1`` classifies ``ar_equal`` on those eight pairs,
and compares the summarization arm against the wall-clock pair. Prompt text
and SHA256 stay in ``office_traj``. This module does not load a generator,
fit K, or start a decode. Import does not mutate the process environment.
There is no command entry.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from studies.trajectory_observation.structure.office_qwen8b import QWEN3_8B
from studies.trajectory_observation.structure.office_traj import (
    EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2, EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4,
    EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7, PROTOCOL, PROTOCOL_WALL, PROTOCOL_XCAT,
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
