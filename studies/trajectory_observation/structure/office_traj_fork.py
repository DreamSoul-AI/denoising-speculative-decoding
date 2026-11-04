"""First-fork and float32-control helpers. The command stays on office_traj."""
from __future__ import annotations

from pathlib import Path

import hashlib
import json
import math
import statistics

from studies.trajectory_observation.structure.office_traj import _file
from studies.trajectory_observation.structure.office_tree import read_rows

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

def _load_fp32_model(model_root=None):
    if model_root is None:
        raise ValueError('an explicit local model root is required for fp32 replay')
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM

    path = Path(model_root).resolve()
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
