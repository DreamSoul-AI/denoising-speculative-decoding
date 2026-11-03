"""Run a pre-registered Qwen3-8B trajectory-only versus full-blend comparison.

``--protocol ablate`` is the finished creative_writing tail. ``--protocol xcat``
is the finished open_qa cross-category exploration. ``--protocol xcat2`` is the finished general_qa cross-category exploration.
``--protocol xcat3`` is the classification cross-category exploration. Writes
flat files into an existing directory. It does not create directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from studies.trajectory_observation.flow.execute.office_tree import (
    _append, _identity, _load_model, _offline_caches,
)
from studies.trajectory_observation.structure.office_traj import (
    ARMS, EVAL, PROTOCOL, PROTOCOL_XCAT, PROTOCOL_XCAT2, PROTOCOL_XCAT3, SMOKE, TAKE,
    TAKE_XCAT, TAKE_XCAT2, TAKE_XCAT3, ablation_summary, arm_specs, assert_registered_slice,
    assert_registered_xcat_slice, assert_registered_xcat2_slice, assert_registered_xcat3_slice,
    prompts_classification, prompts_from_647, prompts_general_qa, prompts_open_qa,
)
from studies.trajectory_observation.structure.office_tree import decode_prompt, existing_output_dir, read_rows


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
    raise RuntimeError(f'unknown trajectory protocol {protocol}')


def _file(dest, name, suffix, prefix, stamp):
    return dest / f'{prefix}-{name}-{stamp}.{suffix}'


def _specs():
    specs = arm_specs()
    return tuple((name, specs[name]) for name, _gain in ARMS)


def _smoke(model, tokenizer, prompts, dest, cfg):
    rows = []
    for prompt in prompts[:2]:
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, _specs(), SMOKE)
        rows.append({name: {'accepted': row['accepted'], 'argmax_passed': row['argmax_passed'],
                            'max_nodes': row['max_nodes'], 'block_complexity': row['block_complexity']}
                     for name, row in decoded.items()})
        print(f"smoke {prompt['qid']} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {'protocol': cfg['protocol'], 'role': 'exploration', 'prompts': 2,
               'qids': [prompt['qid'] for prompt in prompts[:2]],
               'tokens': SMOKE['max_new_tokens'], 'arms': rows,
               'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values())}
    _file(dest, 'smoke', 'json', cfg['prefix'], cfg['stamp']).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    print('smoke passed', flush=True)
    return payload


def _formal(model, tokenizer, prompts, skipped, digest, dest, cfg):
    paths = {name: _file(dest, name, 'jsonl', cfg['prefix'], cfg['stamp']) for name, _gain in ARMS}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('arms resumed at different prompts')
    header = _file(dest, 'header', 'json', cfg['prefix'], cfg['stamp'])
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': cfg['protocol'], 'role': 'exploration', 'source': _identity(),
            'eval': EVAL, 'arms': {name: spec for name, spec in _specs()},
            'prompt_sha256': digest, 'qids': [prompt['qid'] for prompt in prompts],
            'skipped_qids': list(skipped),
            'delta': 'accepted_traj - accepted_full',
            'bootstrap': {'seed': 0, 'resamples': 10000},
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    finished = done['full']
    for prompt in prompts:
        if prompt['qid'] in finished:
            continue
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, _specs(), EVAL)
        for name, row in decoded.items():
            _append(paths[name], row)
        print(f"eval {prompt['qid']} full={decoded['full']['accepted']} "
              f"traj={decoded['traj']['accepted']} nodes={decoded['full']['max_nodes']} "
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
    parser.add_argument('--protocol', choices=('ablate', 'xcat', 'xcat2', 'xcat3'), default='ablate')
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
