"""Offline R^2 of signals known before the forward against the bonus residual.

Reads only local weights. Writes under an explicit directory outside the repo.
The map is fit on the template prefix and scored on SpecBench 81-208.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from spec_mtp.data import load_questions, render
from spec_mtp.decode import SpecDecoder
from spec_mtp.masks import (
    ESPMaskProvider, PreforwardSignalCollector, RIDGE_GRID, apply_lowrank_delta,
    centered_r2, fit_lowrank_delta,
)
from spec_mtp.model import HFCausalLM


ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = Path(r'D:\360安全浏览器下载\Agent\DreamSoul\temp')
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
FEATURES = ('z', 'expected', 'hidden', 'topk')
PRIMARY = ('expected', 'hidden', 'topk')
RANKS = (4, 16)
GATE_R2 = 0.10
GATE_MARGIN = 0.05
FOLDS = 5
SEED = 0


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
            if isinstance(row['question_id'], int) and 81 <= row['question_id'] <= 208]
    rows.sort(key=lambda row: row['question_id'])
    if [row['question_id'] for row in rows] != list(range(81, 209)):
        raise RuntimeError('SpecBench 81-208 is not a contiguous local slice')
    return [{'qid': row['question_id'], 'text': row['turns'][0], 'category': row['category']} for row in rows]


def tree_spec():
    return {'method': 'esp', 'num_masks': 1, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [4], 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True}}


def load_model(path):
    model = HFCausalLM(str(path), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    model.hf.to('cuda')
    original = model.forward

    def forward(*args, **kwargs):
        kwargs['last_hidden_only'] = True
        return original(*args, **kwargs)

    model.forward = forward
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _empty_store():
    keys = ('m', 'z', 'expected', 'topk', 'hidden', 'first', 'last', 'entropy', 'top_mass', 'prev_accepted', 'accepted')
    return {'qid': [], **{key: [] for key in keys}}


def run_prompts(model, prompts, tokens, path):
    done = {row['qid'] for row in read_rows(path)}
    spec = tree_spec()
    eval_cfg = {'max_new_tokens': tokens, 'temperature': 0.0, 'stop_on_eos': False}
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = path.with_suffix('.pt')
    stored = torch.load(pairs, map_location='cpu') if pairs.exists() else _empty_store()
    known = set(stored['qid'])
    if known != done:
        raise RuntimeError(f'{path.name} and its pairs cover different prompts')
    with path.open('a', encoding='utf-8') as stream:
        for prompt in prompts:
            if prompt['qid'] in done:
                continue
            provider = PreforwardSignalCollector(ESPMaskProvider(1, model.embedding_table), model.embedding_table)
            ids = render(tokenizer_of(model), prompt['text']).to('cuda')
            started = time.perf_counter()
            _out, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=provider).generate(ids)
            if not stats['verification_argmax_audit']['passed']:
                raise RuntimeError(f"argmax audit failed for {prompt['qid']}")
            harvested = _stack_rows(provider.rows())
            if harvested is None:
                raise RuntimeError(f"no preforward rows for {prompt['qid']}")
            stored['qid'].append(prompt['qid'])
            for key, value in harvested.items():
                stored[key].append(value)
            torch.save(stored, pairs)
            row = {'qid': prompt['qid'], 'accepted': int(stats['accepted']), 'committed': int(stats['committed']),
                   'decode_calls': int(stats['decode_calls']), 'rounds': int(harvested['accepted'].shape[0]),
                   'seconds': time.perf_counter() - started}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"{path.name} {prompt['qid']} rounds={row['rounds']} accepted={row['accepted']} "
                  f"{row['seconds']:.1f}s", flush=True)
    return stored


def tokenizer_of(model):
    if not hasattr(model, '_preforward_tokenizer'):
        raise RuntimeError('tokenizer was not attached to the model')
    return model._preforward_tokenizer


def _stack_rows(rows):
    if not rows:
        return None
    stacked = {}
    for key in ('m', 'z', 'expected', 'topk', 'hidden', 'first', 'last'):
        stacked[key] = torch.stack([row[key] for row in rows])
    for key in ('entropy', 'top_mass', 'prev_accepted', 'accepted'):
        stacked[key] = torch.tensor([row[key] for row in rows], dtype=torch.float32)
    return stacked


def _pairs(stored, feature):
    blocks = []
    for index in range(len(stored['qid'])):
        prior = stored['m'][index].float()
        bonus = stored['last'][index].float() - prior
        if feature == 'z':
            delta = stored['z'][index].float() - prior
        elif feature == 'expected':
            delta = stored['expected'][index].float() - prior
        elif feature == 'topk':
            delta = stored['topk'][index].float() - prior
        elif feature == 'hidden':
            delta = stored['hidden'][index].float()
        else:
            raise ValueError(feature)
        blocks.append((delta, bonus))
    return blocks


def _cat(blocks):
    return torch.cat([block[0] for block in blocks]), torch.cat([block[1] for block in blocks])


def _folds(count):
    if count < FOLDS:
        raise ValueError('ridge selection needs at least one prompt per fold')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(SEED)).tolist()
    sizes = [count // FOLDS] * FOLDS
    for index in range(count % FOLDS):
        sizes[index] += 1
    chunks, cursor = [], 0
    for size in sizes:
        chunks.append(set(order[cursor:cursor + size]))
        cursor += size
    return chunks


def _cv_mse(blocks, rank, ridge, device):
    squared = rows = 0.0
    for held in _folds(len(blocks)):
        train = [block for index, block in enumerate(blocks) if index not in held]
        val = [block for index, block in enumerate(blocks) if index in held]
        train_delta, train_error = _cat(train)
        fit = fit_lowrank_delta(train_delta.to(device), train_error.to(device), rank, ridge)
        delta, error = _cat(val)
        delta, error = delta.to(device), error.to(device)
        squared += float((apply_lowrank_delta(delta, fit) - error).pow(2).sum())
        rows += int(error.numel())
    return squared / rows


def _select(blocks, device):
    chosen = None
    report = []
    for rank in RANKS:
        scores = []
        for ridge in RIDGE_GRID:
            mse = _cv_mse(blocks, rank, ridge, device)
            scores.append({'rank': rank, 'ridge': ridge, 'mse': mse})
            print(f"  cv rank={rank} ridge={ridge:.0e} mse={mse:.6e}", flush=True)
        best = min(scores, key=lambda item: (item['mse'], -item['ridge']))
        report.extend(scores)
        if chosen is None or (best['mse'], best['rank'], -best['ridge']) < (chosen['mse'], chosen['rank'], -chosen['ridge']):
            chosen = best
    return chosen, report


def _diagonal_predict(delta, gain):
    return delta * gain


def _fit_diagonal(delta, error):
    den = (delta * delta).sum(dim=0).clamp_min(1e-8)
    return (error * delta).sum(dim=0) / den


def _score(train_blocks, test_blocks, fit, device):
    train_delta, train_error = _cat(train_blocks)
    test_delta, test_error = _cat(test_blocks)
    center = train_error.mean(dim=0)
    train_pred = apply_lowrank_delta(train_delta.to(device), fit).cpu()
    test_pred = apply_lowrank_delta(test_delta.to(device), fit).cpu()
    return {'train_r2': centered_r2(train_error, train_pred, center),
            'holdout_r2': centered_r2(test_error, test_pred, center),
            'rows_train': int(train_error.shape[0]), 'rows_holdout': int(test_error.shape[0])}


def _score_diagonal(train_blocks, test_blocks):
    train_delta, train_error = _cat(train_blocks)
    test_delta, test_error = _cat(test_blocks)
    gain = _fit_diagonal(train_delta, train_error)
    center = train_error.mean(dim=0)
    return {'train_r2': centered_r2(train_error, _diagonal_predict(train_delta, gain), center),
            'holdout_r2': centered_r2(test_error, _diagonal_predict(test_delta, gain), center)}


def _pearson(left, right):
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    den = left.norm() * right.norm()
    if float(den) <= 1e-12:
        return None
    return float((left * right).sum() / den)


def _describe(stored):
    last = torch.cat(stored['last']).float()
    first = torch.cat(stored['first']).float()
    prior = torch.cat(stored['m']).float()
    accepted = torch.cat(stored['accepted'])
    entropy = torch.cat(stored['entropy'])
    top_mass = torch.cat(stored['top_mass'])
    prev = torch.cat(stored['prev_accepted'])
    seen = prev >= 0
    same = (last - first).abs().sum(dim=-1) == 0
    residual = (last - prior).norm(dim=-1)
    return {
        'rows': int(last.shape[0]),
        'prompts': len(stored['qid']),
        'rejection_rounds': float((accepted == 0).float().mean()),
        'bonus_equals_first': float(same.float().mean()),
        'pearson_entropy_residual_norm': _pearson(entropy, residual),
        'pearson_top_mass_residual_norm': _pearson(top_mass, residual),
        'pearson_prev_accepted_residual_norm': None if int(seen.sum()) < 2 else _pearson(prev[seen], residual[seen]),
        'pearson_entropy_accepted': _pearson(entropy, accepted),
        'pearson_top_mass_accepted': _pearson(top_mass, accepted),
    }


def _stratify(train_blocks, test_stored, feature, fit, device):
    _delta, train_error = _cat(train_blocks)
    center = train_error.mean(dim=0)
    blocks = _pairs(test_stored, feature)
    accepted = torch.cat(test_stored['accepted'])
    delta, error = _cat(blocks)
    pred = apply_lowrank_delta(delta.to(device), fit).cpu()
    out = {}
    for name, mask in (('accepted_0', accepted == 0), ('accepted_gt_0', accepted > 0)):
        if int(mask.sum()) < 2:
            out[name] = None
            continue
        try:
            value = centered_r2(error[mask], pred[mask], center)
        except ValueError:
            value = None
        out[name] = {'rows': int(mask.sum()), 'holdout_r2': value}
    return out


def fit_and_summarize(calibration, holdout, output):
    device = torch.device('cuda')
    summary = {'protocol': 'preforward-signal-r2-v1', 'features': {}, 'scalars': _describe(holdout)}
    for feature in FEATURES:
        print(f'fit {feature}', flush=True)
        train = _pairs(calibration, feature)
        test = _pairs(holdout, feature)
        chosen, cv = _select(train, device)
        train_delta, train_error = _cat(train)
        fit = fit_lowrank_delta(train_delta.to(device), train_error.to(device), chosen['rank'], chosen['ridge'])
        scored = _score(train, test, fit, device)
        entry = {'selected_rank': chosen['rank'], 'selected_ridge': chosen['ridge'], 'cv_mse': chosen['mse'],
                 'cv': cv, **scored, 'stratified_holdout': _stratify(train, holdout, feature, fit, device)}
        if feature in ('z', 'expected'):
            entry['diagonal'] = _score_diagonal(train, test)
        summary['features'][feature] = entry
        print(f"  selected rank={chosen['rank']} ridge={chosen['ridge']:.0e} "
              f"holdout_r2={scored['holdout_r2']:.4f} train_r2={scored['train_r2']:.4f}", flush=True)
        torch.save({'U': fit['U'].cpu(), 'V': fit['V'].cpu(), 'b': fit['b'].cpu(),
                    'rank': fit['rank'], 'ridge': fit['ridge']}, output / f'fit-{feature}.pt')
    control = summary['features']['z']['holdout_r2']
    chosen_name = None
    for name in PRIMARY:
        value = summary['features'][name]['holdout_r2']
        if value >= GATE_R2 and value >= control + GATE_MARGIN:
            chosen_name = name
            break
    summary['gate'] = {'r2': GATE_R2, 'margin': GATE_MARGIN, 'order': list(PRIMARY),
                       'control_holdout_r2': control, 'signal': chosen_name,
                       'open_kalman': chosen_name is not None}
    (output / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    print(json.dumps(summary['gate'], ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if TEMP_ROOT.resolve() not in output.parents and output != TEMP_ROOT.resolve():
        raise SystemExit(f'output must stay under {TEMP_ROOT}')
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(MODELS[args.model])
    model._preforward_tokenizer = tokenizer
    calibration = run_prompts(model, prompts_for('calibration'), 24, output / 'calibration.jsonl')
    holdout = run_prompts(model, prompts_for('specbench'), 64, output / 'specbench.jsonl')
    if len(calibration['qid']) != 128 or len(holdout['qid']) != 128:
        raise RuntimeError('collection did not cover 128 calibration and 128 SpecBench prompts')
    fit_and_summarize(calibration, holdout, output)


if __name__ == '__main__':
    main()
