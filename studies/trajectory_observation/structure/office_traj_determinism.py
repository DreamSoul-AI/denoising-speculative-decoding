"""BF16 AR determinism helpers. The command stays on office_traj."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from studies.trajectory_observation.structure.office_traj import (
    EVAL, _file, assert_registered_xcat7_slice, prompts_summarization,
)
from studies.trajectory_observation.structure.office_traj_fork import (
    _id_list, first_fork, hash_token_ids, release_session,
)
from studies.trajectory_observation.structure.office_tree import (
    _append, _identity as _run_identity, _load_model, read_rows,
)

PROTOCOL_DET = 'office-ar-bf16-determinism-v1'
DET_PREFIX = 'sd-office-ar-bf16-determinism-v1'
DET_NEAR_GAP = 0.5
DET_LARGE_GAP = 2.0
DET_NEAR_SHARE_MAX = 0.20

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
