"""Equal-node replacement of the embedding extra branch.

Protocol `extra-branch-replace-v1`. Calibration matches the 2026-11-02 extra
branch run. The corrected top-1 replaces the weakest depth-1 leaf; it does not
add a node. Writes only under an explicit directory outside the repo.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

TEMP = Path(r'D:\360安全浏览器下载\Agent\DreamSoul\temp')
for _key, _sub in (
    ('TEMP', 'tmp'), ('TMP', 'tmp'), ('HF_HOME', 'hf'),
    ('HF_DATASETS_CACHE', 'hf/datasets'), ('HUGGINGFACE_HUB_CACHE', 'hf/hub'),
    ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'cache'),
):
    _path = TEMP / _sub
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_key] = str(_path)

import torch
from transformers import AutoTokenizer

from spec_mtp.decode import SpecDecoder, ar_generate
from spec_mtp.masks import build_mask_provider, calibrate_diagonal_gain
from spec_mtp.model import HFCausalLM


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'extra-branch-replace-v1'
HEADS = (
    'Name the color of', 'A compiler turns', 'The Pacific Ocean is', 'After the rain',
    'Please sort these', 'An electron has', 'The word for cat in', 'When the loop ends',
    'A rectangle with sides', 'The train leaves at', 'Binary 1010 equals', 'Salt dissolves in',
    'The author of Hamlet is', 'Click the button to', 'Gravity pulls objects', 'A dictionary maps',
)
TAILS = (
    'a common example', 'the next word', 'two equal parts', 'the empty page',
    'one short line', 'the left side', 'three small items', 'the final step',
    'a closed door', 'the first row', 'four quiet notes', 'the open window',
    'a single page', 'the right margin', 'two fresh copies', 'the last stop',
    'one plain box', 'the middle line',
)
MODELS = {
    '4b': Path(r'D:\models\Qwen3-4B'),
    '0.6b': Path(r'D:\models\Qwen3-0.6B'),
}
EVAL = {'max_new_tokens': 64, 'temperature': 0.0, 'stop_on_eos': False}
FIT = {'max_new_tokens': 24, 'temperature': 0.0, 'stop_on_eos': False}
TREE = {'mode': 'static', 'branches': [4], 'pruning': True}
ESP = {
    'method': 'esp', 'num_masks': 1, 'impl': 'efficient', 'tree': dict(TREE),
    'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
}


def templates():
    texts = [f'{head} {tail}' for head in HEADS for tail in TAILS]
    if len(texts) != 288 or len(set(texts)) != 288 or TAILS[15] != 'the last stop':
        raise RuntimeError('template grid changed')
    return texts


def specbench_prompts():
    path = ROOT / 'datasets/spec_bench/question.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    chosen = [row for row in rows if isinstance(row.get('question_id'), int) and 337 <= row['question_id'] <= 464]
    chosen.sort(key=lambda row: row['question_id'])
    if [row['question_id'] for row in chosen] != list(range(337, 465)):
        raise RuntimeError('SpecBench 337-464 is not the local contiguous slice')
    if any(209 <= row['question_id'] <= 336 for row in chosen):
        raise RuntimeError('SpecBench slice overlaps the extra-branch run')
    prompts = [{'qid': row['question_id'], 'category': row['category'], 'text': row['turns'][0]} for row in chosen]
    if set(item['text'] for item in prompts) & set(templates()[:128]):
        raise RuntimeError('SpecBench prompts overlap calibration text')
    return prompts


def dolly_prompts():
    from datasets import load_dataset

    data = load_dataset('databricks/databricks-dolly-15k', split='train')
    prompts = []
    for item in data:
        if item['category'] != 'creative_writing':
            continue
        text = item['instruction']
        if item.get('context'):
            text = f"{item['context']}\n\n{text}"
        if not isinstance(text, str) or not text.strip():
            continue
        prompts.append({'qid': len(prompts), 'category': 'creative_writing', 'text': text})
        if len(prompts) == 128:
            break
    if len(prompts) != 128:
        raise RuntimeError(f'dolly creative_writing yielded {len(prompts)} prompts, need 128')
    if set(item['text'] for item in prompts) & set(templates()[:128]):
        raise RuntimeError('dolly prompts overlap calibration text')
    return prompts


def prompts_for(name):
    if name == 'specbench':
        return specbench_prompts()
    if name == 'dolly':
        return dolly_prompts()
    raise ValueError(name)


def identity():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=20)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    if head.returncode != 0:
        raise RuntimeError('git head is unavailable')
    return {'git_head': head.stdout.strip(), 'dirty': bool(dirty.stdout.strip())}


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids.detach().cpu().reshape(-1).tolist()).encode()).hexdigest()


def replay_block(model, prompt_ids, full_ids):
    prompt_len = int(prompt_ids.shape[1])
    embeds = model.embed(full_ids)[0]
    spec = dict(ESP)
    spec['tree'] = {'mode': 'static', 'branches': [1], 'pruning': True}
    policy = build_mask_provider(spec, model.embedding_table)
    policy.init_from_prompt(embeds[:prompt_len])
    policy.on_token_committed(embeds[prompt_len], embeds[prompt_len - 1])
    predicted, observed, target = [], [], []
    for index in range(prompt_len + 1, embeds.size(0)):
        predicted.append(policy.masks()[0].detach().float().cpu())
        observed.append(embeds[index - 1].detach().float().cpu())
        target.append(embeds[index].detach().float().cpu())
        policy.on_token_committed(embeds[index], embeds[index - 1])
    if not predicted:
        raise RuntimeError('calibration replay produced no rows')
    return torch.stack(predicted), torch.stack(observed), torch.stack(target)


def calibrate(model, tokenizer, output):
    prior_path = output / 'prior.json'
    if prior_path.exists():
        return json.loads(prior_path.read_text(encoding='utf-8'))
    blocks = []
    texts = templates()[:128]
    for index, text in enumerate(texts):
        ids = tokenizer(text, return_tensors='pt').input_ids
        full, _stats = ar_generate(model, ids, FIT)
        blocks.append(replay_block(model, ids, full))
        if index % 32 == 0:
            print(f'fit {index}', flush=True)
    stacked = tuple(torch.cat([block[axis] for block in blocks], dim=0) for axis in range(3))
    prior = calibrate_diagonal_gain(*stacked)
    prior['prompts'] = 128
    prior['k_mean'] = sum(prior['k']) / len(prior['k'])
    prior['p_mean'] = sum(prior['p']) / len(prior['p'])
    prior['r_mean'] = sum(prior['r']) / len(prior['r'])
    prior['q_mean'] = sum(prior['q']) / len(prior['q'])
    prior_path.write_text(json.dumps(prior), encoding='utf-8')
    print(f"calibrated rows={prior['rows']} k_mean={prior['k_mean']:.6f}", flush=True)
    return prior


def sign_test(left, right):
    from scipy.stats import binomtest

    paired = {row['qid']: row for row in right}
    wins = losses = ties = 0
    for row in left:
        delta = row['accepted'] - paired[row['qid']]['accepted']
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    tested = wins + losses
    pvalue = None if tested == 0 else float(binomtest(wins, tested, 0.5, alternative='two-sided').pvalue)
    return {'wins': wins, 'losses': losses, 'ties': ties, 'p': pvalue,
            'wins_exceed_losses': wins > losses,
            'p_below_0.05': pvalue is not None and pvalue < 0.05}


def mean_tau(rows):
    return sum(row['tau'] for row in rows) / len(rows)


def pooled_tau(rows):
    calls = sum(row['decode_calls'] for row in rows)
    return None if calls == 0 else sum(row['committed'] for row in rows) / calls


def run_arm(model, spec, ids):
    policy = build_mask_provider(spec, model.embedding_table)
    output, stats = SpecDecoder(model, spec, EVAL, mask_policy=policy).generate(ids)
    if not stats['verification_argmax_audit']['passed']:
        raise RuntimeError('argmax audit failed')
    calls = int(stats['decode_calls'])
    row = {
        'accepted': int(stats['accepted']),
        'committed': int(stats['committed']),
        'decode_calls': calls,
        'tau': None if calls == 0 else stats['committed'] / calls,
        'argmax_passed': True,
        'output_sha256': token_hash(output),
    }
    if hasattr(policy, 'diagnostics') and policy.diagnostics().get('mode'):
        diag = policy.diagnostics()
        row['extra_added'] = diag['extra_added']
        row['extra_present'] = diag['extra_present']
        row['max_nodes'] = diag['max_nodes']
        if row['max_nodes'] > 4:
            raise RuntimeError(f"replace arm exceeded 4 draft nodes ({row['max_nodes']})")
    return row


def run_split(model, tokenizer, prompts, prior, output, split):
    esp_path = output / f'{split}-esp.jsonl'
    replace_path = output / f'{split}-replace.jsonl'
    esp_done = {row['qid'] for row in read_rows(esp_path)}
    replace_done = {row['qid'] for row in read_rows(replace_path)}
    if esp_done != replace_done:
        raise RuntimeError(f'{split} arms resumed at different prompts')
    payload = {key: prior[key] for key in ('k', 'p', 'r', 'q')}
    replace_spec = {**ESP, 'hybrid_gain': {'mode': 'replace', **payload}}
    output.mkdir(parents=True, exist_ok=True)
    with esp_path.open('a', encoding='utf-8') as esp_stream, replace_path.open('a', encoding='utf-8') as replace_stream:
        for prompt in prompts:
            if prompt['qid'] in esp_done:
                continue
            ids = tokenizer(prompt['text'], return_tensors='pt').input_ids
            started = time.perf_counter()
            reference, _stats = ar_generate(model, ids, EVAL)
            reference_hash = token_hash(reference)
            esp = run_arm(model, dict(ESP), ids)
            replaced = run_arm(model, replace_spec, ids)
            esp['qid'] = replaced['qid'] = prompt['qid']
            esp['category'] = replaced['category'] = prompt['category']
            esp['ar_equal'] = esp['output_sha256'] == reference_hash
            replaced['ar_equal'] = replaced['output_sha256'] == reference_hash
            esp_stream.write(json.dumps(esp) + '\n')
            replace_stream.write(json.dumps(replaced) + '\n')
            esp_stream.flush()
            replace_stream.flush()
            print(f"{split} {prompt['qid']} esp={esp['accepted']} replace={replaced['accepted']} "
                  f"nodes={replaced['max_nodes']} {time.perf_counter() - started:.1f}s", flush=True)
    summarize(output, split)


def summarize(output, split):
    expected = len(prompts_for(split))
    esp = read_rows(output / f'{split}-esp.jsonl')
    replaced = read_rows(output / f'{split}-replace.jsonl')
    if len(esp) != expected or len(replaced) != expected:
        summary = {'complete': False, 'esp': len(esp), 'replace': len(replaced), 'expected': expected}
    else:
        compared = sign_test(replaced, esp)
        summary = {
            'complete': True, 'protocol': PROTOCOL, 'prompts': expected,
            'tau_mean': {'esp': mean_tau(esp), 'replace': mean_tau(replaced)},
            'tau_pooled': {'esp': pooled_tau(esp), 'replace': pooled_tau(replaced)},
            'replace_vs_esp': compared,
            'argmax_all_passed': all(row['argmax_passed'] for row in esp + replaced),
            'ar_mismatches': {'esp': sum(not row['ar_equal'] for row in esp),
                              'replace': sum(not row['ar_equal'] for row in replaced)},
            'extra_added': sum(row['extra_added'] for row in replaced),
            'extra_present': sum(row['extra_present'] for row in replaced),
            'max_nodes': max(row['max_nodes'] for row in replaced),
            'passed': compared['wins_exceed_losses'] and compared['p_below_0.05'],
        }
    (output / f'{split}-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--split', choices=('specbench', 'dolly'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    output = args.output.resolve()
    if 'DreamSoul' not in str(output) or output.is_relative_to(ROOT):
        raise RuntimeError('write the measurement outside the repo, on D:')
    output.mkdir(parents=True, exist_ok=True)
    header = output / 'header.json'
    if not header.exists():
        header.write_text(json.dumps({
            'protocol': PROTOCOL, 'source': identity(), 'torch': torch.__version__,
            'python': platform.python_version(), 'eval': EVAL, 'tree': TREE,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    model_dir = output / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    print('load', args.model, flush=True)
    model = HFCausalLM(str(MODELS[args.model]), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    if not str(model.device).startswith('cuda'):
        raise RuntimeError(f'{args.model} is on {model.device}')
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS[args.model]), local_files_only=True)
    prior = calibrate(model, tokenizer, model_dir)
    run_split(model, tokenizer, prompts_for(args.split), prior, model_dir, args.split)


if __name__ == '__main__':
    main()
