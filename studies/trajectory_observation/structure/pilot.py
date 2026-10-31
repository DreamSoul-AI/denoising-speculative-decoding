"""CPU synthetic check of trajectory-observation-blend-v1.

A tiny tied LLaMA is trained on CPU on two synthetic order-2 Markov languages
that share one vocabulary. The joint gain is chosen on language-A fit prompts
only, frozen, and reported on held-out A prompts and on language-B prompts.
This is a synthetic engineering pilot, not a real-model or speed result.

Usage: python -m studies.trajectory_observation.structure.pilot --output <new dir outside the repo>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from spec_mtp.analysis import compare, prompt_tau, round_split
from spec_mtp.decode import SpecDecoder, ar_generate
from spec_mtp.masks import build_mask_provider
from spec_mtp.model import HFCausalLM

PROTOCOL = 'trajectory-observation-blend-v1'
JOINT_GAINS = (0.25, 0.5, 0.75)
BASES = ('esp', 'ema')

DEFAULT = {
    'content_vocab': 48, 'branching': 3, 'sharpness': 2.0,
    'model': {'hidden_size': 64, 'intermediate_size': 128, 'num_hidden_layers': 2,
              'num_attention_heads': 4, 'num_key_value_heads': 2, 'max_position_embeddings': 256},
    'train': {'sequences': 4096, 'length': 96, 'steps': 1500, 'batch': 32, 'lr': 3e-3},
    'prompts': {'fit': 24, 'holdout': 24, 'cross': 24, 'length': 24},
    'max_new_tokens': 48,
    'trees': [{'num_masks': 1, 'branches': [14]}, {'num_masks': 2, 'branches': [7, 2]},
              {'num_masks': 3, 'branches': [7, 5, 3]}],
    'seeds': [11, 12, 13],
}


def make_language(content_vocab, branching, sharpness, generator):
    """Order-2 transitions: each (a, b) keeps `branching` successors, softmax(sharpness * N(0, 1)) weights."""
    probs = torch.zeros(content_vocab, content_vocab, content_vocab)
    for a in range(content_vocab):
        for b in range(content_vocab):
            successors = torch.randperm(content_vocab, generator=generator)[:branching]
            weights = torch.softmax(sharpness * torch.randn(branching, generator=generator), dim=0)
            probs[a, b, successors] = weights
    return probs


def sample(language, marker, count, length, generator):
    content = language.shape[0]
    rows = torch.empty(count, length, dtype=torch.long)
    rows[:, 0] = marker
    rows[:, 1:3] = torch.randint(0, content, (count, 2), generator=generator)
    for position in range(3, length):
        rows[:, position] = torch.multinomial(language[rows[:, position - 2], rows[:, position - 1]], 1,
                                              generator=generator).squeeze(1)
    return rows


def train_model(config, data, seed):
    from transformers import LlamaConfig, LlamaForCausalLM

    train = config['train']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        native = LlamaForCausalLM(LlamaConfig(vocab_size=config['content_vocab'] + 2, tie_word_embeddings=True,
                                              attn_implementation='eager', **config['model']))
        optimizer = torch.optim.AdamW(native.parameters(), lr=train['lr'], weight_decay=0.0)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train['steps'])
        native.train()
        losses = []
        for step in range(train['steps']):
            batch = data[torch.randint(0, data.shape[0], (train['batch'],))]
            loss = native(input_ids=batch, labels=batch).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % 100 == 0 or step == train['steps'] - 1:
                losses.append([step, float(loss.detach())])
    native.eval()
    session = HFCausalLM(module=native)
    if not session.tied_embeddings:
        raise ValueError('pilot requires tied embeddings')
    return session, losses


def spec_for(base, tree, gain):
    spec = {'method': 'esp' if base == 'esp' else 'ema_velocity', 'num_masks': tree['num_masks'],
            'impl': 'efficient', 'tree': {'mode': 'static', 'branches': list(tree['branches']), 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
            'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False, 'update': True, 'level': 'last'}}
    if gain:
        spec['observation'] = {'gain': gain}
    return spec


def decode_rows(session, spec, prompts, max_new_tokens, references):
    rows = []
    eval_cfg = {'max_new_tokens': max_new_tokens, 'temperature': 0.0, 'stop_on_eos': False}
    for qid, prompt in prompts:
        decoder = SpecDecoder(session, spec, eval_cfg, generator=torch.Generator().manual_seed(0),
                              mask_policy=build_mask_provider(spec, session.embedding_table))
        output, stats = decoder.generate(prompt)
        rows.append({'qid': qid, 'input_sha256': stats['verification_argmax_audit']['input_sha256'],
                     'protocol': PROTOCOL, 'evidence_complete': True,
                     'accepted': stats['accepted'], 'committed': stats['committed'],
                     'decode_calls': stats['decode_calls'], 'new_tokens': stats['new_tokens'],
                     'argmax_passed': stats['verification_argmax_audit']['passed'],
                     'ar_equal': bool(torch.equal(output, references[qid])),
                     'rounds': round_split(stats['trace'])})
    return rows


def pooled(rows):
    counts = {key: sum(row[key] for row in rows) for key in ('accepted', 'committed', 'decode_calls', 'new_tokens')}
    groups = {}
    for name in ('after_prefill', 'after_accept', 'after_reject'):
        group = {key: sum(row['rounds'][name][key] for row in rows) for key in ('rounds', 'accepted', 'committed')}
        group['tau'] = prompt_tau(group['committed'], group['rounds'])
        groups[name] = group
    return {**counts, 'tau': prompt_tau(counts['committed'], counts['decode_calls']),
            'argmax_all_passed': all(row['argmax_passed'] for row in rows),
            'ar_equal_all': all(row['ar_equal'] for row in rows), 'rounds': groups}


def arm_rows(session, base, tree, gain, prompts, config, references):
    return decode_rows(session, spec_for(base, tree, gain), prompts, config['max_new_tokens'], references)


def run_seed(config, seed):
    generator = torch.Generator().manual_seed(seed)
    content = config['content_vocab']
    lang_a = make_language(content, config['branching'], config['sharpness'], generator)
    lang_b = make_language(content, config['branching'], config['sharpness'], generator)
    train = config['train']
    half = train['sequences'] // 2
    data = torch.cat([sample(lang_a, content, half, train['length'], generator),
                      sample(lang_b, content + 1, half, train['length'], generator)])
    started = time.perf_counter()
    session, losses = train_model(config, data, seed)
    train_seconds = time.perf_counter() - started
    plan = config['prompts']
    splits = {'fit': sample(lang_a, content, plan['fit'], plan['length'], generator),
              'holdout': sample(lang_a, content, plan['holdout'], plan['length'], generator),
              'cross': sample(lang_b, content + 1, plan['cross'], plan['length'], generator)}
    prompts = {name: [(f'{name}-{index}', rows[index:index + 1]) for index in range(rows.shape[0])]
               for name, rows in splits.items()}
    eval_cfg = {'max_new_tokens': config['max_new_tokens'], 'temperature': 0.0, 'stop_on_eos': False}
    references = {qid: ar_generate(session, prompt, eval_cfg)[0]
                  for split in prompts.values() for qid, prompt in split}
    result = {'seed': seed, 'train_losses': losses, 'train_seconds': train_seconds, 'trees': []}
    for tree in config['trees']:
        entry = {'tree': tree, 'fit_grid': {}, 'selected_gain': {}, 'splits': {}}
        for base in BASES:
            grid = {}
            for gain in (0.0, *JOINT_GAINS, 1.0):
                grid[str(gain)] = pooled(arm_rows(session, base, tree, gain, prompts['fit'], config, references))
            entry['fit_grid'][base] = grid
            best = max(JOINT_GAINS, key=lambda g: (grid[str(g)]['tau'], -g))
            entry['selected_gain'][base] = best
        for split in ('holdout', 'cross'):
            arms, raw = {}, {}
            for base in BASES:
                for name, gain in (('trajectory', 0.0), ('observation', 1.0),
                                   ('joint', entry['selected_gain'][base])):
                    rows = arm_rows(session, base, tree, gain, prompts[split], config, references)
                    raw[f'{base}_{name}'] = rows
                    arms[f'{base}_{name}'] = {**pooled(rows), 'gain': gain}
            pairs = {name: {key: value for key, value in compare(raw[name], raw['esp_trajectory'], {}).items()
                            if key != 'pairs'}
                     for name in raw if name != 'esp_trajectory'}
            entry['splits'][split] = {'arms': arms, 'paired_vs_esp_trajectory': pairs}
        result['trees'].append(entry)
    return result


def source_identity():
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=20).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True,
                                             timeout=20).strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {'git_head': head, 'dirty': dirty}


def run(config, output):
    output = Path(output)
    root = Path(__file__).resolve().parents[3]
    if output.resolve().is_relative_to(root):
        raise ValueError('pilot output must stay outside the source tree')
    output.mkdir(parents=True, exist_ok=False)
    header = {'protocol': PROTOCOL, 'scope': 'CPU synthetic pilot; not a real-model, speed or sampling result',
              'config': config, 'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              'source': source_identity(), 'torch': torch.__version__, 'python': platform.python_version(),
              'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32'}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, indent=2)
    results = []
    for seed in config['seeds']:
        result = run_seed(config, seed)
        with (output / f'seed-{seed}.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        results.append(result)
    return header, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='*')
    args = parser.parse_args(argv)
    config = json.loads(json.dumps(DEFAULT))
    if args.seeds:
        config['seeds'] = args.seeds
    torch.set_num_threads(max(1, torch.get_num_threads()))
    run(config, args.output)


if __name__ == '__main__':
    main()
