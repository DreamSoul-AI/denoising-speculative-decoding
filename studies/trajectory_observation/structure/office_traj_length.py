"""Wall-by-length helpers. The command stays on office_traj."""
from __future__ import annotations

import json
import statistics
import time

from studies.trajectory_observation.structure.fresh_replace import bootstrap_mean_ci
from studies.trajectory_observation.structure.office_traj import (
    ARMS, EVAL, ROLE, TAKE_XCAT7, _file, _shown_wall, _specs, _warmup_arms,
    ablation_summary, decode_measured_prompt, finite_wall,
)
from studies.trajectory_observation.structure.office_tree import (
    _append, _identity as _run_identity, read_rows,
)

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
