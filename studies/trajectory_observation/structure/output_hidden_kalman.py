"""Qwen measurement for output-hidden-hybrid-kalman-v1.

Reads only local weights. Writes under an explicit directory outside the repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from spec_mtp.structure.data.prompts import load_questions, render
from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
from spec_mtp.structure.algorithm.mask import build_mask_provider, calibrate_diagonal_gain
from spec_mtp.structure.model.hf_causal import HFCausalLM


ROOT = Path(__file__).resolve().parents[3]
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
    '0.6b': ROOT / 'studies/input_mask_baseline/shared/model/modelscope/Qwen/Qwen3-0.6B',
}
ARMS = ('esp', 'output_hidden', 'trajectory', 'frozen', 'scratch', 'hybrid')


def templates():
    texts = [f'{head} {tail}' for head in HEADS for tail in TAILS]
    if len(texts) != 288 or TAILS[15] != 'the last stop':
        raise RuntimeError('template grid changed')
    return texts


def prompts_for(name):
    if name == 'calibration':
        return [{'qid': f'template-{index}', 'text': text} for index, text in enumerate(templates()[:128])]
    if name == 'holdout':
        return [{'qid': f'template-{index}', 'text': text}
                for index, text in enumerate(templates()[128:160], start=128)]
    if name != 'specbench':
        raise ValueError(name)
    rows = [row for row in load_questions(ROOT / 'datasets/spec_bench/question.jsonl')
            if isinstance(row['question_id'], int) and 209 <= row['question_id'] <= 336]
    rows.sort(key=lambda row: row['question_id'])
    if [row['question_id'] for row in rows] != list(range(209, 337)):
        raise RuntimeError('SpecBench 209-336 is not a contiguous local slice')
    return [{'qid': row['question_id'], 'text': row['turns'][0], 'category': row['category']} for row in rows]


def tree_spec(method, output_kalman=None):
    spec = {'method': method, 'num_masks': 1, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [4], 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
            'output': {'order': 1, 'beta': 0.9, 'beta_prime': 0.99}}
    if output_kalman is not None:
        spec['method'] = 'output_hidden'
        spec['output_kalman'] = output_kalman
    return spec


def load_model(path):
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    aliased = False
    if not model.tied_embeddings:
        left = model.embedding_table.detach()
        right = model.hf.get_output_embeddings().weight.detach()
        if left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError('output and input embeddings differ; not tying them')
        model.hf.get_output_embeddings().weight = model.embedding_table
        aliased = True
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer, {'config_tie': bool(model.config.tie_word_embeddings),
                              'storage_tie': model.tied_embeddings, 'aliased_equal_weights': aliased}


def arm_spec(arm, prior):
    if arm == 'esp':
        return tree_spec('esp')
    if arm == 'output_hidden':
        return tree_spec('output_hidden')
    if arm == 'trajectory':
        return tree_spec('output_hidden', {'mode': 'trajectory'})
    if arm == 'scratch':
        return tree_spec('output_hidden', {'mode': 'scratch'})
    if arm == 'collect':
        return tree_spec('output_hidden', {'mode': 'collect'})
    payload = {'k': prior['k'], 'p': prior['p'], 'r': prior['r'], 'q': prior['q']}
    if arm == 'frozen':
        return tree_spec('output_hidden', {'mode': 'frozen', **payload})
    if arm == 'hybrid':
        return tree_spec('output_hidden', {'mode': 'hybrid', 'boost': 64, **payload})
    raise ValueError(arm)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def run_prompts(model, tokenizer, prompts, spec, tokens, path, keep_pairs=False):
    done = {row['qid'] for row in read_rows(path)}
    provider = build_mask_provider(spec, model.embedding_table)
    eval_cfg = {'max_new_tokens': tokens, 'temperature': 0.0, 'stop_on_eos': False}
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = path.with_suffix('.pt')
    stored = torch.load(pairs, map_location='cpu') if keep_pairs and pairs.exists() else {'qid': [], 'h': [], 'z': [], 'y': []}
    known = set(stored['qid'])
    with path.open('a', encoding='utf-8') as stream:
        for prompt in prompts:
            if prompt['qid'] in done:
                continue
            ids = render(tokenizer, prompt['text']).to('cuda')
            started = time.perf_counter()
            _out, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=provider).generate(ids)
            if not stats['verification_argmax_audit']['passed']:
                raise RuntimeError(f"argmax audit failed for {prompt['qid']}")
            row = {'qid': prompt['qid'], 'accepted': int(stats['accepted']), 'committed': int(stats['committed']),
                   'decode_calls': int(stats['decode_calls']), 'new_tokens': int(stats['new_tokens']),
                   'seconds': time.perf_counter() - started}
            if hasattr(provider, 'diagnostics'):
                row['diagnostics'] = provider.diagnostics()
            if keep_pairs:
                if prompt['qid'] not in known:
                    harvested = provider.calibration_rows()
                    stored['qid'].append(prompt['qid'])
                    stored['h'].append(harvested['h'])
                    stored['z'].append(harvested['z'])
                    stored['y'].append(harvested['y'])
                    torch.save(stored, pairs)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"{path.name} {prompt['qid']} accepted={row['accepted']} calls={row['decode_calls']} "
                  f"{row['seconds']:.1f}s", flush=True)
    return stored


def calibrate(model, tokenizer, output):
    prior_path = output / 'prior.json'
    if prior_path.exists():
        return json.loads(prior_path.read_text(encoding='utf-8'))
    stored = run_prompts(model, tokenizer, prompts_for('calibration'), arm_spec('collect', None), 24,
                         output / 'calibration.jsonl', keep_pairs=True)
    if len(stored['qid']) != 128:
        raise RuntimeError('calibration pairs do not cover 128 prompts')
    prior = calibrate_diagonal_gain(torch.cat(stored['h']), torch.cat(stored['z']), torch.cat(stored['y']))
    prior['k_mean'] = sum(prior['k']) / len(prior['k'])
    prior_path.write_text(json.dumps(prior), encoding='utf-8')
    print(f"calibrated rows={prior['rows']} k_mean={prior['k_mean']:.6f}", flush=True)
    return prior


def sign_test(left, right):
    from scipy.stats import binomtest
    paired = {row['qid']: row for row in left}
    wins = losses = ties = 0
    for row in right:
        delta = paired[row['qid']]['accepted'] - row['accepted']
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    tested = wins + losses
    pvalue = None if tested == 0 else float(binomtest(wins, tested, 0.5, alternative='two-sided').pvalue)
    return {'wins': wins, 'losses': losses, 'ties': ties, 'p': pvalue}


def pooled_tau(rows):
    calls = sum(row['decode_calls'] for row in rows)
    return None if calls == 0 else sum(row['committed'] for row in rows) / calls


def summarize(output, model_name):
    summary = {'model': model_name}
    for split in ('specbench', 'holdout'):
        expected = len(prompts_for(split))
        arms = {}
        incomplete = None
        for arm in ARMS:
            rows = read_rows(output / f'{split}-{arm}.jsonl')
            if len(rows) != expected:
                incomplete = {'complete': False, 'arm': arm, 'rows': len(rows), 'expected': expected}
                break
            arms[arm] = rows
        if incomplete:
            summary[split] = incomplete
            continue
        summary[split] = {'complete': True, 'tau': {arm: pooled_tau(rows) for arm, rows in arms.items()},
                          'versus_esp': sign_test(arms['hybrid'], arms['esp']),
                          'versus_output_hidden': sign_test(arms['hybrid'], arms['output_hidden']),
                          'trajectory_versus_output_hidden': sign_test(arms['trajectory'], arms['output_hidden']),
                          'frozen_versus_esp': sign_test(arms['frozen'], arms['esp']),
                          'scratch_versus_esp': sign_test(arms['scratch'], arms['esp'])}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    if 'DreamSoul' not in str(args.output) or args.output.resolve().is_relative_to(ROOT):
        raise RuntimeError('write the measurement outside the repo, on D:')
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer, tie = load_model(MODELS[args.model])
    (args.output / 'tie.json').write_text(json.dumps(tie), encoding='utf-8')
    print(json.dumps(tie), flush=True)
    prior = calibrate(model, tokenizer, args.output)
    for split in ('specbench', 'holdout'):
        prompts = prompts_for(split)
        for arm in ARMS:
            run_prompts(model, tokenizer, prompts, arm_spec(arm, prior), 64, args.output / f'{split}-{arm}.jsonl')
    summarize(args.output, args.model)


if __name__ == '__main__':
    main()
