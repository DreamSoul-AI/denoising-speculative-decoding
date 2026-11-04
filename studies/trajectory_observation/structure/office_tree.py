"""Equal-node static tree shapes for the Qwen3 explorations.

``office-tree-shape-v1`` is the finished Qwen3-4B slice. ``office-8b-tree-shape-v1``
reuses the same three shapes on a later Dolly slice. Prompt selection, the
paired summary, and the command that loads weights live here. This module does
not fit K. Import does not mutate the process environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, creative_texts, direction, paired_deltas, prompt_sha256,
)

PROTOCOL = 'office-tree-shape-v1'
ROLE = 'exploration'
START = 388
TAKE = 128
NODE_BUDGET = 4
BOOT_SEED = 0
BOOT_RESAMPLES = 10000
SKIPPED_QIDS = ()
EVAL_SHA256 = '123702e2294e9339665cf752ab286297b670cc06e6af632d9b6912399b6cc4e3'
EVAL_QIDS = tuple(range(START, START + TAKE))
PROTOCOL_8B = 'office-8b-tree-shape-v1'
START_8B = 516
END_8B = 646
SKIPPED_QIDS_8B = (597, 608, 626)
EVAL_SHA256_8B = 'd4596e4f7404f47ac53e327d711f70edb7d49298b7583155c8050bfe27bed4dd'
EVAL_QIDS_8B = tuple(qid for qid in range(START_8B, END_8B + 1) if qid not in SKIPPED_QIDS_8B)
MASK = {'init': 'mean_prompt', 'lam': 0.1, 'update': True}
EVAL = {'max_new_tokens': 64, 'temperature': 0.0, 'stop_on_eos': False}
SMOKE = {'max_new_tokens': 8, 'temperature': 0.0, 'stop_on_eos': False}
QWEN3_4B = Path('D' + ':' + '\\') / 'models' / 'Qwen3-4B'
ARMS = (
    ('b4', [4], 1),
    ('b22', [2, 2], 2),
    ('b31', [3, 1], 2),
)
ROOT = Path(__file__).resolve().parents[3]


def block_complexity(num_masks, branches):
    return (int(num_masks) + 1) * (1 + sum(int(width) for width in branches))


def arm_spec(branches, num_masks):
    """One static ESP shape whose draft-node count is exactly the budget."""
    widths = [int(width) for width in branches]
    masks = int(num_masks)
    if not widths or any(width <= 0 for width in widths) or sum(widths) != NODE_BUDGET:
        raise RuntimeError(f'draft node budget must stay {NODE_BUDGET}')
    if len(widths) > masks:
        raise RuntimeError('branch depths exceed the mask slots the decoder can feed')
    return {
        'method': 'esp',
        'num_masks': masks,
        'impl': 'efficient',
        'tree': {'mode': 'static', 'branches': widths, 'pruning': True},
        'mask': dict(MASK),
    }


def arm_specs():
    return {name: arm_spec(branches, masks) for name, branches, masks in ARMS}


def prompts_after(rows, start, take=TAKE):
    """Take unused creative texts from ``start``, skipping strings already seen."""
    texts = creative_texts(rows)
    if len(texts) <= start - 1:
        raise RuntimeError(f'creative stream does not reach qid {start}')
    used = set(texts[:start])
    skipped = []
    chosen = []
    for index, text in enumerate(texts[start:], start=start):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': 'creative_writing', 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'creative_writing has only {len(chosen)} new prompts from qid {start}')


def prompts_from_388(rows, take=TAKE):
    """Take unused creative texts from qid 388, skipping strings already seen."""
    return prompts_after(rows, START, take)


def prompts_from_516(rows, take=TAKE):
    """Take unused creative texts from qid 516, skipping strings already seen."""
    return prompts_after(rows, START_8B, take)


def _assert_slice(prompts, skipped, digest_expected, qids_expected, skipped_expected, label):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if digest != digest_expected or qids != qids_expected or tuple(skipped) != skipped_expected:
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered {label} slice')
    return digest


def assert_registered_slice(prompts, skipped):
    return _assert_slice(prompts, skipped, EVAL_SHA256, EVAL_QIDS, SKIPPED_QIDS, '4b')


def assert_registered_8b_slice(prompts, skipped):
    return _assert_slice(prompts, skipped, EVAL_SHA256_8B, EVAL_QIDS_8B, SKIPPED_QIDS_8B, '8b')


def existing_output_dir(path):
    output = Path(path).resolve()
    if not output.is_dir():
        raise RuntimeError('output directory must already exist')
    if output.drive.upper() == 'C:':
        raise RuntimeError('measurement output must stay off C:')
    root = ROOT.resolve()
    if output == root or root in output.parents:
        raise RuntimeError('write the measurement outside the repo')
    return output


def require_budget(counts, budget=NODE_BUDGET):
    if not counts or any(int(count) != budget for count in counts):
        raise RuntimeError(f'draft nodes {list(counts)} are not all {budget}')
    return int(budget)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids.detach().cpu().reshape(-1).tolist()).encode()).hexdigest()


def arm_row(model, spec, ids, limit, *, keep_output=False):
    import spec_mtp.structure.algorithm.decode.loop as decode_mod
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    from spec_mtp.structure.algorithm.mask import build_mask_provider

    seen = []
    original = decode_mod.build_tree

    def wrapped(spec_cfg, mask_probs, root_token):
        tree = original(spec_cfg, mask_probs, root_token)
        count = int(tree.num_nodes)
        seen.append(count)
        if count != NODE_BUDGET:
            raise RuntimeError(f'draft nodes {count} != {NODE_BUDGET}')
        return tree

    decode_mod.build_tree = wrapped
    try:
        policy = build_mask_provider(spec, model.embedding_table)
        output, stats = SpecDecoder(model, spec, limit, mask_policy=policy).generate(ids)
    finally:
        decode_mod.build_tree = original
    require_budget(seen)
    if not stats['verification_argmax_audit']['passed']:
        raise RuntimeError('argmax audit failed')
    expected = block_complexity(spec['num_masks'], spec['tree']['branches'])
    if int(stats['block_complexity']) != expected:
        raise RuntimeError(f"block complexity {stats['block_complexity']} != {expected}")
    calls = int(stats['decode_calls'])
    row = {
        'accepted': int(stats['accepted']),
        'committed': int(stats['committed']),
        'decode_calls': calls,
        'tau': None if calls == 0 else stats['committed'] / calls,
        'argmax_passed': True,
        'output_sha256': token_hash(output),
        'max_nodes': NODE_BUDGET,
        'min_nodes': min(seen),
        'tree_steps': len(seen),
        'block_complexity': int(stats['block_complexity']),
    }
    if keep_output:
        row['output_ids'] = output.detach().cpu().reshape(-1).tolist()
    return row


def decode_prompt(model, tokenizer, prompt, specs, limit):
    from spec_mtp.structure.algorithm.decode.loop import ar_generate

    ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
    reference, _stats = ar_generate(model, ids, limit)
    reference_hash = token_hash(reference)
    rows = {}
    for name, spec in specs:
        row = arm_row(model, spec, ids, limit)
        row['qid'] = prompt['qid']
        row['category'] = prompt['category']
        row['ar_equal'] = row['output_sha256'] == reference_hash
        rows[name] = row
    return rows


def _compare(name, treated, baseline):
    from studies.trajectory_observation.structure.fair_branch import mean_tau, pooled_tau, sign_test

    compared = sign_test(treated, baseline)
    accepted = bootstrap_mean_ci(paired_deltas(treated, baseline, 'accepted'))
    tau = bootstrap_mean_ci(paired_deltas(treated, baseline, 'tau'))
    trend = direction(compared['wins'], compared['losses'], accepted['mean'])
    return {
        'baseline': 'b4',
        'arm': name,
        'wins': compared['wins'],
        'losses': compared['losses'],
        'ties': compared['ties'],
        'wins_losses_ties': [compared['wins'], compared['losses'], compared['ties']],
        'p': compared['p'],
        'tau_mean': {'b4': mean_tau(baseline), name: mean_tau(treated)},
        'tau_pooled': {'b4': pooled_tau(baseline), name: pooled_tau(treated)},
        'accepted_difference': accepted,
        'tau_difference': tau,
        'direction': trend,
        'positive_direction': trend == 'positive',
        'confirmatory': False,
        'argmax_all_passed': all(row['argmax_passed'] for row in baseline + treated),
        'ar_mismatches': {'b4': sum(not row['ar_equal'] for row in baseline),
                          name: sum(not row['ar_equal'] for row in treated)},
        'max_nodes': max(row['max_nodes'] for row in baseline + treated),
        'block_complexity': {
            'b4': sorted({row['block_complexity'] for row in baseline}),
            name: sorted({row['block_complexity'] for row in treated}),
        },
    }


def tree_summary(rows_by_name, protocol=PROTOCOL, digest=EVAL_SHA256, skipped=SKIPPED_QIDS):
    if set(rows_by_name) != {name for name, _branches, _masks in ARMS}:
        raise RuntimeError('summary is missing a registered shape')
    if any(len(rows) != TAKE for rows in rows_by_name.values()):
        raise RuntimeError('summary needs 128 rows in every shape')
    baseline = rows_by_name['b4']
    comparisons = {name: _compare(name, rows_by_name[name], baseline) for name in ('b22', 'b31')}
    passed = all(item['argmax_all_passed'] and item['max_nodes'] == NODE_BUDGET for item in comparisons.values())
    return {
        'protocol': protocol,
        'role': ROLE,
        'confirmatory': False,
        'prompts': TAKE,
        'prompt_sha256': digest,
        'skipped_qids': list(skipped),
        'baseline': 'b4',
        'comparisons': comparisons,
        'argmax_all_passed': passed,
    }


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
