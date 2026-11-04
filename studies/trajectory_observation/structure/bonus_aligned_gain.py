"""Qwen measurement for hybrid-gain-bonus-aligned-v1.

Reads only local weights. Writes under an explicit directory outside the repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from spec_mtp.data import load_questions, render
from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
from spec_mtp.structure.algorithm.mask import build_mask_provider, calibrate_diagonal_gain
from spec_mtp.model import HFCausalLM


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
ARMS = ('esp', 'frozen_last', 'opened_last', 'opened_first')


def templates():
    texts = [f'{head} {tail}' for head in HEADS for tail in TAILS]
    if len(texts) != 288 or TAILS[15] != 'the last stop':
        raise RuntimeError('template grid changed')
    return texts


def prompts_for(name):
    if name == 'calibration':
        return [{'qid': f'template-{index}', 'text': text} for index, text in enumerate(templates()[:128])]
    if name != 'specbench':
        raise ValueError(name)
    rows = [row for row in load_questions(ROOT / 'datasets/spec_bench/question.jsonl')
            if isinstance(row['question_id'], int) and 465 <= row['question_id'] <= 560]
    rows.sort(key=lambda row: row['question_id'])
    if [row['question_id'] for row in rows] != list(range(465, 561)):
        raise RuntimeError('SpecBench 465-560 is not a contiguous local slice')
    return [{'qid': row['question_id'], 'text': row['turns'][0], 'category': row['category']} for row in rows]


def tree_spec(hybrid=None):
    spec = {'method': 'esp', 'num_masks': 1, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [4], 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True}}
    if hybrid is not None:
        spec['hybrid_gain'] = hybrid
    return spec


def load_model(path):
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def arm_spec(arm, priors):
    if arm == 'esp':
        return tree_spec()
    if arm == 'frozen_last':
        return tree_spec({'mode': 'frozen', 'observe': 'last', **priors['last']})
    if arm == 'opened_last':
        return tree_spec({'mode': 'opened', 'observe': 'last', 'boost': 64, **priors['last']})
    if arm == 'opened_first':
        return tree_spec({'mode': 'opened', 'observe': 'first', 'boost': 64, **priors['first']})
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
    stored = torch.load(pairs, map_location='cpu') if keep_pairs and pairs.exists() else {'qid': [], 'm': [], 'z': [], 'first': [], 'last': []}
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
            if keep_pairs and prompt['qid'] not in known:
                harvested = provider.calibration_rows()
                stored['qid'].append(prompt['qid'])
                for key in ('m', 'z', 'first', 'last'):
                    stored[key].append(harvested[key])
                torch.save(stored, pairs)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"{path.name} {prompt['qid']} accepted={row['accepted']} calls={row['decode_calls']} "
                  f"{row['seconds']:.1f}s", flush=True)
    return stored


def _cosine(left, right):
    num = (left * right).sum(dim=-1)
    den = left.norm(dim=-1) * right.norm(dim=-1)
    ok = den > 1e-12
    if not bool(ok.any()):
        return None
    return float((num[ok] / den[ok]).mean())


def describe_pairs(stored):
    m = torch.cat(stored['m']).float()
    first = torch.cat(stored['first']).float()
    last = torch.cat(stored['last']).float()
    differ = (first - last).abs().sum(dim=-1) > 0
    report = {'rows': int(m.shape[0]), 'rounds_first_ne_last': int(differ.sum()),
              'cosine_all': _cosine(first - m, last - m)}
    report['cosine_when_different'] = None if not bool(differ.any()) else _cosine(first[differ] - m[differ], last[differ] - m[differ])
    return report


def calibrate(model, tokenizer, output):
    prior_path = output / 'priors.json'
    if prior_path.exists():
        return json.loads(prior_path.read_text(encoding='utf-8'))
    stored = run_prompts(model, tokenizer, prompts_for('calibration'), tree_spec({'mode': 'collect'}), 24,
                         output / 'calibration.jsonl', keep_pairs=True)
    if len(stored['qid']) != 128:
        raise RuntimeError('calibration pairs do not cover 128 prompts')
    m = torch.cat(stored['m'])
    zed = torch.cat(stored['z'])
    priors = {}
    for name, target in (('first', torch.cat(stored['first'])), ('last', torch.cat(stored['last']))):
        prior = calibrate_diagonal_gain(m, zed, target)
        prior['k_mean'] = sum(prior['k']) / len(prior['k'])
        priors[name] = prior
    payload = {'first': priors['first'], 'last': priors['last'], 'pairs': describe_pairs(stored)}
    prior_path.write_text(json.dumps(payload), encoding='utf-8')
    print(f"calibrated rows={priors['last']['rows']} k_first={priors['first']['k_mean']:.6f} "
          f"k_last={priors['last']['k_mean']:.6f} differ={payload['pairs']['rounds_first_ne_last']}", flush=True)
    return payload


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


def mean_diag(rows, key):
    values = [row['diagnostics'][key] for row in rows if row.get('diagnostics') and row['diagnostics'].get(key) is not None]
    return None if not values else sum(values) / len(values)


def summarize(output):
    expected = len(prompts_for('specbench'))
    arms = {}
    incomplete = None
    for arm in ARMS:
        rows = read_rows(output / f'specbench-{arm}.jsonl')
        if len(rows) != expected:
            incomplete = {'complete': False, 'arm': arm, 'rows': len(rows), 'expected': expected}
            break
        arms[arm] = rows
    if incomplete:
        summary = incomplete
    else:
        summary = {'complete': True, 'tau': {arm: pooled_tau(rows) for arm, rows in arms.items()},
                   'opened_last_versus_esp': sign_test(arms['opened_last'], arms['esp']),
                   'frozen_last_versus_esp': sign_test(arms['frozen_last'], arms['esp']),
                   'opened_first_versus_esp': sign_test(arms['opened_first'], arms['esp']),
                   'opened_last_versus_opened_first': sign_test(arms['opened_last'], arms['opened_first']),
                   'weight_mean': {arm: mean_diag(rows, 'weight_mean') for arm, rows in arms.items()},
                   'k_abs_delta': {arm: mean_diag(rows, 'k_abs_delta') for arm, rows in arms.items()},
                   'k_mean': {arm: mean_diag(rows, 'k_mean') for arm, rows in arms.items()}}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() == 'C:' or ROOT == output or ROOT in output.parents:
        raise SystemExit('output must stay off C: and outside the repo')
    if not torch.cuda.is_available():
        raise RuntimeError('this measurement requires CUDA')
    if not MODELS[args.model].exists():
        raise FileNotFoundError(MODELS[args.model])
    model, tokenizer = load_model(MODELS[args.model])
    priors = calibrate(model, tokenizer, output)
    usable = {name: {key: priors[name][key] for key in ('k', 'p', 'r', 'q')} for name in ('first', 'last')}
    prompts = prompts_for('specbench')
    for arm in ARMS:
        run_prompts(model, tokenizer, prompts, arm_spec(arm, usable), 64, output / f'specbench-{arm}.jsonl')
    summarize(output)


if __name__ == '__main__':
    main()
