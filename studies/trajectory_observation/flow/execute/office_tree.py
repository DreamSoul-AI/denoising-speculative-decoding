"""Run the pre-registered Qwen3-4B tree-shape exploration.

Writes flat files into an existing directory. It does not create directories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from studies.trajectory_observation.structure.office_tree import (
    ARMS, EVAL, PROTOCOL, QWEN3_4B, SMOKE, arm_specs, assert_registered_slice,
    decode_prompt, existing_output_dir, prompts_from_388, read_rows, tree_summary,
)

PREFIX = 'sd-office-tree'
STAMP = '20261103'


def _file(dest, name, suffix):
    return dest / f'{PREFIX}-{name}-{STAMP}.{suffix}'


def _offline_caches(dest):
    for key in ('HF_HOME', 'HF_DATASETS_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TORCH_HOME', 'TEMP', 'TMP'):
        value = os.environ.get(key)
        if not value:
            raise RuntimeError(f'{key} must already point at an existing directory off C:')
        path = Path(value)
        if not path.is_dir() or path.drive.upper() == 'C:':
            raise RuntimeError(f'{key} must already point at an existing directory off C:')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    if dest.drive.upper() == 'C:':
        raise RuntimeError('measurement output must stay off C:')


def _identity():
    from studies.trajectory_observation.structure.office_tree import ROOT

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=20)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    if head.returncode != 0:
        raise RuntimeError('git head is unavailable')
    return {'git_head': head.stdout.strip(), 'dirty': bool(dirty.stdout.strip())}


def _load_model():
    import torch
    from transformers import AutoTokenizer

    from spec_mtp.model import HFCausalLM

    path = QWEN3_4B
    if not path.is_dir():
        raise RuntimeError('Qwen3-4B weights are not at the local 4B directory')
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    if not str(model.device).startswith('cuda') or model.dtype != torch.bfloat16:
        raise RuntimeError(f'4B is on {model.device} with dtype {model.dtype}')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer, torch.__version__


def _append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()


def _specs():
    specs = arm_specs()
    return tuple((name, specs[name]) for name, _branches, _masks in ARMS)


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
    paths = {name: _file(dest, name, 'jsonl') for name, _branches, _masks in ARMS}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('shapes resumed at different prompts')
    header = _file(dest, 'header', 'json')
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': PROTOCOL, 'role': 'exploration', 'source': _identity(),
            'eval': EVAL, 'arms': {name: spec for name, spec in _specs()},
            'prompt_sha256': digest, 'qids': [prompt['qid'] for prompt in prompts],
            'skipped_qids': list(skipped),
            'bootstrap': {'seed': 0, 'resamples': 10000},
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    finished = done['b4']
    for prompt in prompts:
        if prompt['qid'] in finished:
            continue
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, _specs(), EVAL)
        for name, row in decoded.items():
            _append(paths[name], row)
        print(f"eval {prompt['qid']} b4={decoded['b4']['accepted']} "
              f"b22={decoded['b22']['accepted']} b31={decoded['b31']['accepted']} "
              f"nodes={decoded['b4']['max_nodes']} {time.perf_counter() - started:.1f}s", flush=True)
    summary = tree_summary({name: read_rows(path) for name, path in paths.items()})
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

    prompts, skipped = prompts_from_388(load_dolly_rows())
    digest = assert_registered_slice(prompts, skipped)
    print('load 4b', flush=True)
    model, tokenizer, torch_version = _load_model()
    if args.only in ('all', 'smoke'):
        _smoke(model, tokenizer, prompts, dest)
    if args.only in ('all', 'formal'):
        if _identity()['dirty']:
            raise RuntimeError('formal run requires a clean worktree')
        _formal(model, tokenizer, prompts, skipped, digest, dest)
    print('torch', torch_version, flush=True)


if __name__ == '__main__':
    main()
