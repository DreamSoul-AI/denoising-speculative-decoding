"""Observation-gain scan helpers. The command stays on office_traj."""
from __future__ import annotations

import json
import statistics
import time

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, direction, paired_deltas,
)
from studies.trajectory_observation.structure.office_traj import (
    EVAL, MASK, ROLE, SMOKE, TAKE_XCAT7, TREE, _file, _model_device, _shown_wall,
    decode_measured_prompt, finite_wall,
)
from studies.trajectory_observation.structure.office_tree import (
    NODE_BUDGET, _append, _identity as _run_identity, block_complexity, read_rows,
)

PROTOCOL_GSCAN = 'office-8b-traj-g-scan-v1'
GSCAN_PREFIX = 'sd-office-8b-traj-g-scan-v1'
# Frozen so a resumed formal run appends the same jsonl. The CLI writes this
# stamp into artifact names outside the source tree.
GSCAN_STAMP = '20261104-1813'
GSCAN_TOKENS = 256
GSCAN_BLOCK = 10
BLEND = 'trajectory-observation-blend-v1'
GSCAN_ARMS = (
    ('g0', 0.0),
    ('g025', 0.25),
    ('g05', 0.5),
    ('g075', 0.75),
)
GSCAN_BASELINE = 'g0'
_COMPARED = ('g025', 'g05', 'g075')


def gscan_arm_spec(name):
    """Same ESP draft mask as the xcat7 ablation. Only the registered gain changes."""
    gains = {arm: gain for arm, gain in GSCAN_ARMS}
    if name not in gains:
        raise RuntimeError('this scan only uses the pre-registered arms')
    if block_complexity(1, TREE['branches']) != GSCAN_BLOCK or sum(TREE['branches']) != NODE_BUDGET:
        raise RuntimeError('g-scan keeps the registered ESP mask')
    return {
        'method': 'esp',
        'num_masks': 1,
        'impl': 'efficient',
        'tree': {'mode': TREE['mode'], 'branches': list(TREE['branches']), 'pruning': TREE['pruning']},
        'mask': dict(MASK),
        'observation': {'gain': gains[name], 'condition': 'none', 'form': 'mean'},
    }


def gscan_specs():
    return tuple((name, gscan_arm_spec(name)) for name, _gain in GSCAN_ARMS)


def gscan_limit():
    limit = dict(EVAL)
    limit['max_new_tokens'] = GSCAN_TOKENS
    if limit['temperature'] != 0.0 or limit['stop_on_eos'] is not False:
        raise RuntimeError('g-scan keeps temperature 0 and stop_on_eos false')
    return limit


def _indexed(rows):
    found = {}
    for row in rows:
        qid = row.get('qid')
        if qid in found or qid is None:
            return None
        found[qid] = row
    return found


def _tau_mean(rows):
    values = [row.get('tau') for row in rows]
    if any(type(value) not in (int, float) for value in values):
        return None
    return float(sum(values) / len(values))


def _mismatches(rows):
    if any(type(row.get('ar_equal')) is not bool for row in rows):
        return None
    return sum(not row['ar_equal'] for row in rows)


def _descriptive_wall(name, treated, baseline):
    """Wall is descriptive. A missing, non-finite, or negative clock stays missing."""
    if any(finite_wall(row.get('wall_seconds')) is None for row in treated + baseline):
        return {'present': False}
    deltas = paired_deltas(treated, baseline, 'wall_seconds')
    return {
        'present': True,
        'delta': f'wall_seconds_{name} - wall_seconds_{GSCAN_BASELINE}',
        'mean': float(statistics.mean(deltas)),
    }


def _pair(name, treated, baseline):
    from studies.trajectory_observation.structure.fair_branch import sign_test

    compared = sign_test(treated, baseline)
    accepted = bootstrap_mean_ci(paired_deltas(treated, baseline, 'accepted'))
    pair = {
        'baseline': GSCAN_BASELINE,
        'arm': name,
        'delta': f'accepted_{name} - accepted_{GSCAN_BASELINE}',
        'wins': compared['wins'],
        'losses': compared['losses'],
        'ties': compared['ties'],
        'wins_losses_ties': [compared['wins'], compared['losses'], compared['ties']],
        'p': compared['p'],
        'exploratory': True,
        'confirmatory': False,
        'tau_mean': {GSCAN_BASELINE: _tau_mean(baseline), name: _tau_mean(treated)},
        'accepted_difference': accepted,
        'direction': direction(compared['wins'], compared['losses'], accepted['mean']),
        'wall': _descriptive_wall(name, treated, baseline),
        'ar_mismatches': {
            GSCAN_BASELINE: _mismatches(baseline),
            name: _mismatches(treated),
        },
    }
    if pair['tau_mean'][GSCAN_BASELINE] is not None and pair['tau_mean'][name] is not None:
        pair['tau_difference'] = bootstrap_mean_ci(paired_deltas(treated, baseline, 'tau'))
    return pair


def _gate(rows_by_name):
    if set(rows_by_name) != {name for name, _gain in GSCAN_ARMS}:
        return {'valid': False, 'reason': 'arms', 'argmax_all_passed': False, 'nodes_all_four': False}
    indexed = {name: _indexed(rows) for name, rows in rows_by_name.items()}
    expected = set(range(TAKE_XCAT7))
    if any(item is None or set(item) != expected for item in indexed.values()):
        return {'valid': False, 'reason': 'row count', 'argmax_all_passed': False, 'nodes_all_four': False}
    rows = [row for arm in indexed.values() for row in arm.values()]
    argmax = all(row.get('argmax_passed') is True for row in rows)
    nodes = all(row.get('max_nodes') == NODE_BUDGET and row.get('min_nodes') == NODE_BUDGET for row in rows)
    complexity = all(row.get('block_complexity') == GSCAN_BLOCK for row in rows)
    length = all(row.get('max_new_tokens') == GSCAN_TOKENS for row in rows)
    if not argmax:
        reason = 'argmax'
    elif not nodes:
        reason = 'draft nodes'
    elif not complexity:
        reason = 'block complexity'
    elif not length:
        reason = 'length'
    else:
        reason = None
    return {
        'valid': reason is None,
        'reason': reason,
        'argmax_all_passed': argmax,
        'nodes_all_four': nodes,
        'indexed': indexed,
    }


def gscan_summary(rows_by_name, digest, skipped):
    """Paired acceptance versus g=0. An invalid group does not enter the sign test."""
    gate = _gate(rows_by_name)
    summary = {
        'protocol': PROTOCOL_GSCAN,
        'role': ROLE,
        'confirmatory': False,
        'prompts': TAKE_XCAT7,
        'prompt_sha256': digest,
        'skipped_qids': list(skipped),
        'baseline': GSCAN_BASELINE,
        'gains': {name: gain for name, gain in GSCAN_ARMS},
        'blend': BLEND,
        'max_new_tokens': GSCAN_TOKENS,
        'valid': gate['valid'],
        'reason': gate['reason'],
        'argmax_all_passed': gate['argmax_all_passed'],
        'nodes_all_four': gate['nodes_all_four'],
    }
    indexed = gate.get('indexed')
    if indexed is not None:
        summary['tau_mean'] = {
            name: _tau_mean(list(indexed[name].values())) for name, _gain in GSCAN_ARMS
        }
        summary['ar_mismatches'] = {
            name: _mismatches(list(indexed[name].values())) for name, _gain in GSCAN_ARMS
        }
    if not gate['valid']:
        return summary
    summary['comparisons'] = {
        name: _pair(name, list(indexed[name].values()), list(indexed[GSCAN_BASELINE].values()))
        for name in _COMPARED
    }
    return summary


def _warmup_gscan(model, tokenizer, prompt):
    """Untimed 8-token pass on all four gains, then one sync."""
    from spec_mtp.model import sync_device
    from studies.trajectory_observation.structure.office_tree import arm_row

    ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
    for _name, spec in gscan_specs():
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
    return recorded


def gscan_smoke(model, tokenizer, prompts, dest, cfg):
    """First two prompts, 8 tokens, four gains. Argmax failure blocks the formal run."""
    if cfg['protocol'] != PROTOCOL_GSCAN:
        raise RuntimeError('g-scan smoke is only for the g-scan protocol')
    specs = gscan_specs()
    rows = []
    for prompt in prompts[:2]:
        started = time.perf_counter()
        decoded = decode_measured_prompt(model, tokenizer, prompt, specs, SMOKE)
        if [name for name, _spec in specs] != list(decoded):
            raise RuntimeError('smoke arms left the frozen order')
        rows.append({name: _smoke_view(row) for name, row in decoded.items()})
        print(f"smoke {prompt['qid']} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {
        'protocol': PROTOCOL_GSCAN,
        'role': 'exploration',
        'prompts': 2,
        'qids': [prompt['qid'] for prompt in prompts[:2]],
        'tokens': SMOKE['max_new_tokens'],
        'arms': rows,
        'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values()),
        'nodes_all_four': all(arm['max_nodes'] == NODE_BUDGET for item in rows for arm in item.values()),
    }
    _file(dest, 'smoke', 'json', cfg['prefix'], cfg['stamp']).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    if not payload['nodes_all_four']:
        raise RuntimeError('smoke draft nodes')
    print('smoke passed', flush=True)
    return payload


def gscan_formal(model, tokenizer, prompts, skipped, digest, dest, cfg):
    """256 tokens on the frozen xcat7 slice. One failed arm leaves the group invalid."""
    if cfg['protocol'] != PROTOCOL_GSCAN:
        raise RuntimeError('g-scan formal is only for the g-scan protocol')
    specs = gscan_specs()
    paths = {name: _file(dest, name, 'jsonl', cfg['prefix'], cfg['stamp']) for name, _gain in GSCAN_ARMS}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('arms resumed at different prompts')
    header = _file(dest, 'header', 'json', cfg['prefix'], cfg['stamp'])
    limit = gscan_limit()
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': PROTOCOL_GSCAN,
            'role': 'exploration',
            'source': _run_identity(),
            'eval': limit,
            'blend': BLEND,
            'arms': {name: spec for name, spec in specs},
            'prompt_sha256': digest,
            'qids': [prompt['qid'] for prompt in prompts],
            'skipped_qids': list(skipped),
            'delta': 'accepted_<arm> - accepted_g0',
            'wall_seconds': 'descriptive only; missing, non-finite, and negative stay missing',
            'sign_test': 'exploratory two-sided sign test; no p<0.05 gate',
            'bootstrap': {'seed': 0, 'resamples': 10000},
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    _warmup_gscan(model, tokenizer, prompts[0])
    finished = done[GSCAN_BASELINE]
    for prompt in prompts:
        if prompt['qid'] in finished:
            continue
        started = time.perf_counter()
        decoded = decode_measured_prompt(model, tokenizer, prompt, specs, limit)
        for name, row in decoded.items():
            row['max_new_tokens'] = GSCAN_TOKENS
            _append(paths[name], row)
        walls = '/'.join(_shown_wall(decoded[name]) for name, _gain in GSCAN_ARMS)
        accepted = ' '.join(f"{name}={decoded[name]['accepted']}" for name, _gain in GSCAN_ARMS)
        print(
            f"eval {prompt['qid']} {accepted} nodes={decoded[GSCAN_BASELINE]['max_nodes']} "
            f"wall={walls} {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    summary = gscan_summary(
        {name: read_rows(path) for name, path in paths.items()}, digest, skipped,
    )
    _file(dest, 'summary', 'json', cfg['prefix'], cfg['stamp']).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'valid': summary['valid'],
        'reason': summary['reason'],
        'argmax_all_passed': summary['argmax_all_passed'],
        'nodes_all_four': summary['nodes_all_four'],
    }, ensure_ascii=False), flush=True)
    return summary
