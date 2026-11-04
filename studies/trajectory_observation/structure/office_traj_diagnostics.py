"""Post-hoc office_traj diagnostic helpers.

First-fork, BF16 AR determinism, and wall-by-length. The command stays on office_traj.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from studies.trajectory_observation.structure.fresh_replace import bootstrap_mean_ci
from studies.trajectory_observation.structure.office_qwen8b import QWEN3_8B
from studies.trajectory_observation.structure.office_traj import (
    ARMS, EVAL, ROLE, TAKE_XCAT7, _file, _shown_wall, _specs, _warmup_arms,
    ablation_summary, assert_registered_xcat7_slice, decode_measured_prompt, finite_wall,
    prompts_summarization,
)
from studies.trajectory_observation.structure.office_tree import (
    _append, _identity as _run_identity, _load_model, read_rows,
)

PROTOCOL_FORK = 'office-ar-first-fork-v1'
FORK_PREFIX = 'sd-office-ar-first-fork-v1'
FORK_STAMP = '20261104-1244'
FORK_RESERVE_MIB = 4096

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

PROTOCOL_LENGTH = 'office-8b-traj-wall-by-length-v1'
LENGTH_PREFIX = 'sd-office-8b-traj-wall-by-length-v1'
LENGTH_STAMP = '20261104-1420'
LENGTHS = (64, 256, 512)
LENGTH_FLAT_RATIO = 1.5
LENGTH_BUCKET_SHARE = 0.5
LENGTH_BUCKET_ORDER = (
    'draft_seconds', 'verify_seconds', 'overhead_seconds', 'residual_seconds',
)

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
