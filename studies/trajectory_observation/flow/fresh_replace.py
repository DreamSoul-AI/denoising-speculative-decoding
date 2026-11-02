"""Command for the fresh-prompt equal-node replacement replication.

Selection, hashing, and the effect summary stay in
``studies.trajectory_observation.structure.fresh_replace``. This module parses
arguments and runs that measurement. The comparison math is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from studies.trajectory_observation.structure.fresh_replace import (
    BOOT_RESAMPLES, BOOT_SEED, PROTOCOL, SKIP, SPLIT, TAKE, assert_known_slice,
    fresh_prompts, model_path, outside_repo, replication_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('4b', '0.6b'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer

    from studies.trajectory_observation.structure.fair_branch import (
        ESP, EVAL, TREE, calibrate, identity, load_dolly, read_rows, run_split, templates,
    )

    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    prompts = fresh_prompts(load_dolly())
    digest = assert_known_slice(prompts)
    if set(item['text'] for item in prompts) & set(templates()[:128]):
        raise RuntimeError('fresh prompts overlap calibration text')
    weights = model_path(args.model)
    header = output / 'header.json'
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': PROTOCOL, 'source': identity(), 'torch': torch.__version__,
            'eval': EVAL, 'tree': TREE, 'esp': {'method': ESP['method'], 'num_masks': ESP['num_masks']},
            'split': SPLIT, 'skip': SKIP, 'take': TAKE, 'prompt_sha256': digest,
            'bootstrap': {'seed': BOOT_SEED, 'resamples': BOOT_RESAMPLES},
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    model_dir = output / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'prompt-index.json').write_text(json.dumps(
        [{'qid': prompt['qid'], 'sha256': hashlib.sha256(prompt['text'].encode('utf-8')).hexdigest()}
         for prompt in prompts], ensure_ascii=False), encoding='utf-8')
    print('load', args.model, flush=True)
    from spec_mtp.model import HFCausalLM

    model = HFCausalLM(str(weights), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    if not str(model.device).startswith('cuda'):
        raise RuntimeError(f'{args.model} is on {model.device}')
    tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
    prior = calibrate(model, tokenizer, model_dir)
    run_split(model, tokenizer, prompts, prior, model_dir, SPLIT)
    esp = read_rows(model_dir / f'{SPLIT}-esp.jsonl')
    replaced = read_rows(model_dir / f'{SPLIT}-replace.jsonl')
    if len(esp) != TAKE or len(replaced) != TAKE:
        raise RuntimeError('replication finished without both arms complete')
    summary = replication_summary(esp, replaced)
    summary['model'] = args.model
    summary['weights'] = str(weights)
    summary['prompt_sha256'] = digest
    summary['k_mean'] = prior['k_mean']
    (model_dir / 'replication-summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
