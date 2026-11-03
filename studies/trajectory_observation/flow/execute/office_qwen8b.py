"""Run the pre-registered Qwen3-8B comparison. Writes flat files, no new directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from studies.trajectory_observation.structure.office_qwen8b import (
    ESP, EVAL, FIT, MODELS, PROTOCOL, SMOKE, TREE, collect_last_pairs,
    decode_prompt, esp_spec, existing_output_dir, kalman_spec, load_dolly_rows,
    office_summary, prior_from_triples, prompts_after_used, read_rows, replace_spec,
    replay_block, templates, assert_registered_slice,
)

PREFIX = 'sd-office-8b'
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
    from studies.trajectory_observation.structure.office_qwen8b import ROOT

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=20)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    if head.returncode != 0:
        raise RuntimeError('git head is unavailable')
    return {'git_head': head.stdout.strip(), 'dirty': bool(dirty.stdout.strip())}


def _load_model():
    import torch
    from transformers import AutoTokenizer

    from spec_mtp.model import HFCausalLM

    path = MODELS['8b']
    if not path.is_dir():
        raise RuntimeError('Qwen3-8B weights are not at the 8b table path')
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    if not str(model.device).startswith('cuda'):
        raise RuntimeError(f'8b is on {model.device}')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer, torch.__version__


def _write_prior(path, prior):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    path.write_text(json.dumps(prior), encoding='utf-8')
    return prior


def _fit(model, tokenizer, dest):
    from spec_mtp.decode import ar_generate

    replace_path = _file(dest, 'fit-replace', 'json')
    kalman_path = _file(dest, 'fit-kalman', 'json')
    if replace_path.exists() and kalman_path.exists():
        return (json.loads(replace_path.read_text(encoding='utf-8')),
                json.loads(kalman_path.read_text(encoding='utf-8')))
    texts = templates()[:128]
    blocks = []
    last_m, last_z, last_y = [], [], []
    for index, text in enumerate(texts):
        ids = tokenizer(text, return_tensors='pt').input_ids
        full, _stats = ar_generate(model, ids, FIT)
        blocks.append(replay_block(model, ids, full))
        harvested = collect_last_pairs(model, tokenizer, text, FIT['max_new_tokens'])
        last_m.append(harvested[0])
        last_z.append(harvested[1])
        last_y.append(harvested[2])
        if index % 16 == 0:
            print(f'fit {index}', flush=True)
    import torch

    stacked = tuple(torch.cat([block[axis] for block in blocks], dim=0) for axis in range(3))
    replace_prior = prior_from_triples(*stacked, prompts=128)
    kalman_prior = prior_from_triples(torch.cat(last_m), torch.cat(last_z), torch.cat(last_y), prompts=128)
    print(f"fitted replace_k={replace_prior['k_mean']:.6f} kalman_k={kalman_prior['k_mean']:.6f} "
          f"rows={replace_prior['rows']}/{kalman_prior['rows']}", flush=True)
    return _write_prior(replace_path, replace_prior), _write_prior(kalman_path, kalman_prior)


def _append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()


def _smoke(model, tokenizer, replace_prior, kalman_prior, dest):
    specs = (
        ('esp', esp_spec()),
        ('replace', replace_spec(replace_prior)),
        ('kalman', kalman_spec(kalman_prior)),
    )
    rows = []
    for index, text in enumerate(templates()[:2]):
        prompt = {'qid': index, 'category': 'calibration', 'text': text}
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, specs, SMOKE)
        rows.append({name: {'accepted': row['accepted'], 'argmax_passed': row['argmax_passed'],
                            'max_nodes': row.get('max_nodes')} for name, row in decoded.items()})
        print(f"smoke {index} {time.perf_counter() - started:.1f}s", flush=True)
    payload = {'protocol': PROTOCOL, 'prompts': 2, 'tokens': SMOKE['max_new_tokens'], 'arms': rows,
               'argmax_all_passed': all(arm['argmax_passed'] for item in rows for arm in item.values())}
    _file(dest, 'smoke', 'json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not payload['argmax_all_passed']:
        raise RuntimeError('smoke argmax failed')
    print('smoke passed', flush=True)
    return payload


def _formal(model, tokenizer, replace_prior, kalman_prior, dest):
    prompts, skipped = prompts_after_used(load_dolly_rows())
    digest = assert_registered_slice(prompts, skipped)
    paths = {name: _file(dest, name, 'jsonl') for name in ('esp', 'replace', 'kalman')}
    done = {name: {row['qid'] for row in read_rows(path)} for name, path in paths.items()}
    if len(set(map(frozenset, done.values()))) != 1:
        raise RuntimeError('arms resumed at different prompts')
    specs = (
        ('esp', esp_spec()),
        ('replace', replace_spec(replace_prior)),
        ('kalman', kalman_spec(kalman_prior)),
    )
    header = _file(dest, 'header', 'json')
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': PROTOCOL, 'source': _identity(), 'eval': EVAL, 'tree': TREE,
            'esp': {'method': ESP['method'], 'num_masks': ESP['num_masks']},
            'prompt_sha256': digest, 'skipped_qids': list(skipped),
            'bootstrap': {'seed': 0, 'resamples': 10000},
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    for prompt in prompts:
        if prompt['qid'] in done['esp']:
            continue
        started = time.perf_counter()
        decoded = decode_prompt(model, tokenizer, prompt, specs, EVAL)
        for name, row in decoded.items():
            _append(paths[name], row)
        print(f"eval {prompt['qid']} esp={decoded['esp']['accepted']} "
              f"replace={decoded['replace']['accepted']} kalman={decoded['kalman']['accepted']} "
              f"nodes={decoded['replace']['max_nodes']} {time.perf_counter() - started:.1f}s", flush=True)
    esp = read_rows(paths['esp'])
    replaced = read_rows(paths['replace'])
    kalman = read_rows(paths['kalman'])
    summary = office_summary(esp, replaced, kalman)
    summary['k_mean'] = {'replace': replace_prior['k_mean'], 'kalman': kalman_prior['k_mean']}
    summary['fit_rows'] = {'replace': replace_prior['rows'], 'kalman': kalman_prior['rows']}
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
    print('load 8b', flush=True)
    model, tokenizer, torch_version = _load_model()
    replace_prior, kalman_prior = _fit(model, tokenizer, dest)
    if args.only in ('all', 'smoke'):
        _smoke(model, tokenizer, replace_prior, kalman_prior, dest)
    if args.only in ('all', 'formal'):
        if _identity()['dirty']:
            raise RuntimeError('formal run requires a clean worktree')
        _formal(model, tokenizer, replace_prior, kalman_prior, dest)
    print('torch', torch_version, flush=True)


if __name__ == '__main__':
    main()
