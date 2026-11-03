"""Run a pre-registered equal-node tree-shape exploration.

``--model 4b`` is the finished Qwen3-4B slice. ``--model 8b`` is the Qwen3-8B
exploration. Writes flat files into an existing directory. It does not create
directories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from studies.trajectory_observation.structure.office_tree import (
    ARMS, EVAL, PROTOCOL, PROTOCOL_8B, QWEN3_4B, SMOKE, arm_specs,
    assert_registered_8b_slice, assert_registered_slice, decode_prompt,
    existing_output_dir, prompts_from_388, prompts_from_516, read_rows, tree_summary,
)

STAMP = '20261103'
PREFIXES = {'4b': 'sd-office-tree', '8b': 'sd-office-8b-tree'}


def _file(dest, name, suffix, prefix):
    return dest / f'{prefix}-{name}-{STAMP}.{suffix}'


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


def _require_idle_gpu(minimum_free_mib=18000):
    free = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=20, check=False,
    )
    apps = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if free.returncode != 0 or apps.returncode != 0:
        raise RuntimeError('nvidia-smi could not report GPU occupancy')
    free_mib = int(free.stdout.strip().splitlines()[0])
    busy = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2:
            continue
        memory = parts[1].replace('[', '').replace(']', '')
        if memory.isdigit():
            busy.append(line.strip())
    if busy or free_mib < minimum_free_mib:
        raise RuntimeError(f'GPU is not free for 8B: free_mib={free_mib} busy={busy}')
    return free_mib


def _weights(kind):
    if kind == '4b':
        return QWEN3_4B, '4B'
    from studies.trajectory_observation.structure.office_qwen8b import QWEN3_8B
    return QWEN3_8B, '8B'


def _load_model(kind):
    import torch
    from transformers import AutoTokenizer

    from spec_mtp.model import HFCausalLM

    path, label = _weights(kind)
    if kind == '8b':
        _require_idle_gpu()
    if not path.is_dir():
        raise RuntimeError(f'Qwen3-{label} weights are not at the local {label} directory')
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    if not str(model.device).startswith('cuda') or model.dtype != torch.bfloat16:
        raise RuntimeError(f'{label} is on {model.device} with dtype {model.dtype}')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer, torch.__version__


def _append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()


def _specs():
    specs = arm_specs()
    return tuple((name, specs[name]) for name, _branches, _masks in ARMS)


def _registered(kind, rows):
    if kind == '4b':
        prompts, skipped = prompts_from_388(rows)
        return prompts, skipped, assert_registered_slice(prompts, skipped), PROTOCOL
    prompts, skipped = prompts_from_516(rows)
    return prompts, skipped, assert_registered_8b_slice(prompts, skipped), PROTOCOL_8B


def _smoke(model, tokenizer, prompts, dest, protocol, prefix):
    rows = []
    for prompt in prompts[:2]:
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, _specs(), SMOKE)
        rows.append({name: {'accepted': row['accepted'], 'argmax_passed': row['argmax_passed'],
                            'max_nodes': row['max_nodes'], 'block_complexity': row['block_complexity']}
                     for name, row in decoded.items()})
        print(f"smoke {prompt['qid']} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {'protocol': protocol, 'role': 'exploration', 'prompts': 2,
               'qids': [prompt['qid'] for prompt in prompts[:2]],
               'tokens': SMOKE['max_new_tokens'], 'arms': rows,
               'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values())}
    _file(dest, 'smoke', 'json', prefix).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    print('smoke passed', flush=True)
    return payload


def _formal(model, tokenizer, prompts, skipped, digest, dest, protocol, prefix):
    paths = {name: _file(dest, name, 'jsonl', prefix) for name, _branches, _masks in ARMS}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('shapes resumed at different prompts')
    header = _file(dest, 'header', 'json', prefix)
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': protocol, 'role': 'exploration', 'source': _identity(),
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
    summary = tree_summary(
        {name: read_rows(path) for name, path in paths.items()},
        protocol=protocol, digest=digest, skipped=skipped,
    )
    _file(dest, 'summary', 'json', prefix).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--model', choices=tuple(PREFIXES), default='4b')
    parser.add_argument('--only', choices=('all', 'smoke', 'formal'), default='all')
    args = parser.parse_args(argv)
    dest = existing_output_dir(args.dest)
    _offline_caches(dest)
    from studies.trajectory_observation.structure.office_qwen8b import load_dolly_rows

    prompts, skipped, digest, protocol = _registered(args.model, load_dolly_rows())
    prefix = PREFIXES[args.model]
    print(f'load {args.model}', flush=True)
    model, tokenizer, torch_version = _load_model(args.model)
    if args.only in ('all', 'smoke'):
        _smoke(model, tokenizer, prompts, dest, protocol, prefix)
    if args.only in ('all', 'formal'):
        if _identity()['dirty']:
            raise RuntimeError('formal run requires a clean worktree')
        _formal(model, tokenizer, prompts, skipped, digest, dest, protocol, prefix)
    print('torch', torch_version, flush=True)


if __name__ == '__main__':
    main()
