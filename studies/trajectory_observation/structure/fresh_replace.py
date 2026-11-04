"""Fresh-prompt replication of the equal-node extra-branch replacement.

The comparison itself is `fair_branch.run_split`: frozen diagonal K, slot 0
left as the ESP mask, corrected top-1 swapped into the weakest childless
depth-1 leaf. This module selects the unused Dolly slice, summarizes the
paired effect, and runs that comparison from ``main``. Import does not mutate
the process environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'extra-branch-replace-fresh-v1'
SKIP = 128
TAKE = 128
BOOT_SEED = 0
BOOT_RESAMPLES = 10000
FRESH_SHA256 = 'e37e3be985b5356acedeffc22a83b00a0484b1983d44ef6e67eebfe0c3f36a20'
FRESH_QIDS = tuple(index for index in range(128, 257) if index != 144)
SPLIT = 'dolly-creative-128-256-except-144'


def creative_texts(rows):
    texts = []
    for item in rows:
        if item.get('category') != 'creative_writing':
            continue
        text = item.get('instruction')
        context = item.get('context') or ''
        if context:
            text = f"{context}\n\n{text}"
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def fresh_prompts(rows, skip=SKIP, take=TAKE):
    """Next `take` creative texts after the used prefix, skipping repeated strings.

    `qid` is the index in the nonempty creative_writing stream. The first
    `skip` rows are the sample already measured by `extra-branch-replace-v1`.
    A later row whose text already occurred is not a new prompt.
    """
    texts = creative_texts(rows)
    if len(texts) < skip or len(set(texts[:skip])) != skip:
        raise RuntimeError('used Dolly prefix is missing or not unique')
    used = set(texts[:skip])
    chosen = []
    for index, text in enumerate(texts[skip:], start=skip):
        if text in used:
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': 'creative_writing', 'text': text})
        if len(chosen) == take:
            return chosen
    raise RuntimeError(f'creative_writing has only {len(chosen)} new prompts after the used prefix')


def prompt_sha256(prompts):
    body = '\n'.join(prompt['text'] for prompt in prompts)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def assert_known_slice(prompts):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if digest != FRESH_SHA256 or qids != FRESH_QIDS:
        raise RuntimeError(f'fresh prompt hash {digest} does not match the pre-registered slice')
    return digest


def outside_repo(path):
    output = Path(path).resolve()
    root = ROOT.resolve()
    if output == root or root in output.parents:
        raise RuntimeError('write the measurement outside the repo')
    return output


def bootstrap_mean_ci(deltas, seed=BOOT_SEED, resamples=BOOT_RESAMPLES):
    import numpy as np

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError('bootstrap needs a nonempty 1d sample')
    means = values[np.random.default_rng(seed).integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {'mean': float(values.mean()), 'ci95': [float(low), float(high)],
            'seed': int(seed), 'resamples': int(resamples),
            'excludes_zero': bool(low > 0.0 or high < 0.0)}


def direction(wins, losses, mean):
    if wins > losses and mean > 0:
        return 'positive'
    if wins < losses and mean < 0:
        return 'negative'
    if wins == losses and mean == 0:
        return 'flat'
    return 'mixed'


def paired_deltas(left, right, key):
    paired = {row['qid']: row for row in right}
    if set(paired) != {row['qid'] for row in left}:
        raise RuntimeError('paired rows do not share qids')
    return [row[key] - paired[row['qid']][key] for row in left]


def replication_summary(esp, replaced):
    from studies.trajectory_observation.structure.fair_branch import mean_tau, pooled_tau, sign_test

    compared = sign_test(replaced, esp)
    accepted = bootstrap_mean_ci(paired_deltas(replaced, esp, 'accepted'))
    tau = bootstrap_mean_ci(paired_deltas(replaced, esp, 'tau'))
    per_round = bootstrap_mean_ci(paired_deltas(
        [{'qid': row['qid'], 'per_round': row['accepted'] / row['decode_calls']} for row in replaced],
        [{'qid': row['qid'], 'per_round': row['accepted'] / row['decode_calls']} for row in esp],
        'per_round'))
    trend = direction(compared['wins'], compared['losses'], accepted['mean'])
    return {
        'protocol': PROTOCOL,
        'split': SPLIT,
        'prompts': len(esp),
        'tau_mean': {'esp': mean_tau(esp), 'replace': mean_tau(replaced)},
        'tau_pooled': {'esp': pooled_tau(esp), 'replace': pooled_tau(replaced)},
        'replace_vs_esp': compared,
        'accepted_difference': accepted,
        'tau_difference': tau,
        'accepted_per_round_difference': per_round,
        'direction': trend,
        'positive_trend': trend == 'positive',
        'argmax_all_passed': all(row['argmax_passed'] for row in esp + replaced),
        'ar_mismatches': {'esp': sum(not row['ar_equal'] for row in esp),
                          'replace': sum(not row['ar_equal'] for row in replaced)},
        'extra_added': sum(row['extra_added'] for row in replaced),
        'extra_present': sum(row['extra_present'] for row in replaced),
        'max_nodes': max(row['max_nodes'] for row in replaced),
    }


def model_path(name):
    from studies.trajectory_observation.structure.fair_branch import MODELS

    if name == '4b':
        path = MODELS['4b']
    elif name == '0.6b':
        office = MODELS['0.6b']
        path = office if office.exists() else (
            ROOT / 'studies/input_mask_baseline/shared/model/modelscope/Qwen/Qwen3-0.6B')
    else:
        raise ValueError(name)
    if not path.exists():
        raise RuntimeError(f'{name} weights are not on this machine')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('4b', '0.6b'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from studies.trajectory_observation.structure.fair_branch import (
        ESP, EVAL, TREE, calibrate, identity, load_dolly, read_rows, run_split, templates,
    )

    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=True)
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
