"""Run the pre-registered Qwen3-8B trajectory-only versus full-blend ablation.

Writes flat files into an existing directory. It does not create directories.
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
    ARMS, EVAL, PROTOCOL, SMOKE, ablation_summary, arm_specs, assert_registered_slice,
    prompts_from_647,
)
from studies.trajectory_observation.structure.office_tree import decode_prompt, existing_output_dir, read_rows

STAMP = '20261103-0905'
PREFIX = 'sd-office-8b-traj-ablate'


def _file(dest, name, suffix):
    return dest / f'{PREFIX}-{name}-{STAMP}.{suffix}'


def _specs():
    specs = arm_specs()
    return tuple((name, specs[name]) for name, _gain in ARMS)


def _smoke(model, tokenizer, prompts, dest):
    rows = []
    for prompt in prompts[:2]:
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, _specs(), SMOKE)
        rows.append({name: {'accepted': row['accepted'], 'argmax_passed': row['argmax_passed'],
                            'max_nodes': row['max_nodes'], 'block_complexity': row['block_complexity']}
                     for name, row in decoded.items()})
        print(f"smoke {prompt['qid']} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {'protocol': PROTOCOL, 'role': 'exploration', 'prompts': 2,
               'qids': [prompt['qid'] for prompt in prompts[:2]],
               'tokens': SMOKE['max_new_tokens'], 'arms': rows,
               'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values())}
    _file(dest, 'smoke', 'json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    print('smoke passed', flush=True)
    return payload


def _formal(model, tokenizer, prompts, skipped, digest, dest):
    paths = {name: _file(dest, name, 'jsonl') for name, _gain in ARMS}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('arms resumed at different prompts')
    header = _file(dest, 'header', 'json')
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': PROTOCOL, 'role': 'exploration', 'source': _identity(),
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
        protocol=PROTOCOL, digest=digest, skipped=skipped,
    )
    _file(dest, 'summary', 'json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--only', choices=('all', 'smoke', 'formal'), default='all')
    args = parser.parse_args(argv)
    dest = existing_output_dir(args.dest)
    _offline_caches(dest)
    from studies.trajectory_observation.structure.office_qwen8b import load_dolly_rows

    prompts, skipped = prompts_from_647(load_dolly_rows())
    digest = assert_registered_slice(prompts, skipped)
    print('load 8b', flush=True)
    model, tokenizer, torch_version = _load_model('8b')
    if args.only in ('all', 'smoke'):
        _smoke(model, tokenizer, prompts, dest)
    if args.only in ('all', 'formal'):
        if _identity()['dirty']:
            raise RuntimeError('formal run requires a clean worktree')
        _formal(model, tokenizer, prompts, skipped, digest, dest)
    print('torch', torch_version, flush=True)


if __name__ == '__main__':
    main()
