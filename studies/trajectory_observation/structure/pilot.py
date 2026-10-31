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

PROTOCOL = 'trajectory-observation-pilot-v2'
JOINT_GAINS = (0.25, 0.5, 0.75)
BASES = ('esp', 'ema')
CONDITIONS = ('none', 'rejection')
SUFFIX = {('mean', 'none'): '', ('mean', 'rejection'): '_rejection',
          ('top1', 'none'): '_top1', ('top1', 'rejection'): '_top1_rejection'}

DEFAULT = {
    'content_vocab': 48, 'branching': 3, 'sharpness': 2.0,
    'model': {'hidden_size': 64, 'intermediate_size': 128, 'num_hidden_layers': 2,
              'num_attention_heads': 4, 'num_key_value_heads': 2, 'max_position_embeddings': 256},
    'train': {'sequences': 4096, 'length': 96, 'steps': 1500, 'batch': 32, 'lr': 3e-3},
    'prompts': {'fit': 32, 'holdout': 32, 'cross': 32, 'length': 24},
    'max_new_tokens': 48,
    'trees': [{'num_masks': 1, 'branches': [14]}, {'num_masks': 2, 'branches': [7, 2]},
              {'num_masks': 3, 'branches': [7, 5, 3]}],
    'seeds': [11, 12, 13],
    'variants': [['mean', 'none'], ['mean', 'rejection']],
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


def spec_for(base, tree, gain, condition='none', form='mean'):
    spec = {'method': 'esp' if base == 'esp' else 'ema_velocity', 'num_masks': tree['num_masks'],
            'impl': 'efficient', 'tree': {'mode': 'static', 'branches': list(tree['branches']), 'pruning': True},
            'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
            'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False, 'update': True, 'level': 'last'}}
    if gain:
        spec['observation'] = {'gain': gain, 'condition': condition, 'form': form}
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


def arm_rows(session, base, tree, gain, prompts, config, references, condition='none', form='mean'):
    return decode_rows(session, spec_for(base, tree, gain, condition, form), prompts, config['max_new_tokens'],
                       references)


def paired(rows, reference):
    return {key: value for key, value in compare(rows, reference, {}).items() if key != 'pairs'}


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
        variants = [tuple(variant) for variant in config['variants']]
        for base in BASES:
            trajectory = pooled(arm_rows(session, base, tree, 0.0, prompts['fit'], config, references))
            for form, condition in variants:
                key = f'{base}{SUFFIX[form, condition] or "_none"}'
                grid = {'0.0': trajectory}
                for gain in (*JOINT_GAINS, 1.0):
                    grid[str(gain)] = pooled(arm_rows(session, base, tree, gain, prompts['fit'], config,
                                                      references, condition, form))
                entry['fit_grid'][key] = grid
                entry['selected_gain'][key] = max(JOINT_GAINS, key=lambda g: (grid[str(g)]['tau'], -g))
        for split in ('holdout', 'cross'):
            arms, raw = {}, {}
            for base in BASES:
                plan_arms = [('trajectory', 0.0, 'none', 'mean')]
                for form, condition in variants:
                    suffix = SUFFIX[form, condition]
                    gain = entry['selected_gain'][f'{base}{suffix or "_none"}']
                    plan_arms += [(f'observation{suffix}', 1.0, condition, form),
                                  (f'joint{suffix}', gain, condition, form)]
                for name, gain, condition, form in plan_arms:
                    rows = arm_rows(session, base, tree, gain, prompts[split], config, references, condition, form)
                    raw[f'{base}_{name}'] = rows
                    arms[f'{base}_{name}'] = {**pooled(rows), 'gain': gain, 'condition': condition, 'form': form}
            entry['splits'][split] = {
                'arms': arms, 'rows': raw,
                'paired_vs_esp_trajectory': {name: paired(raw[name], raw['esp_trajectory'])
                                             for name in raw if name != 'esp_trajectory'},
                'paired_vs_own_trajectory': {name: paired(raw[name], raw[name.split('_')[0] + '_trajectory'])
                                             for name in raw if not name.endswith('_trajectory')}}
        result['trees'].append(entry)
    return result


class OracleMaskProvider:
    """Upper bound only: slot j of a depth-d owner gets the true embedding of the token it stands for.

    Reads the AR reference continuation, i.e. future tokens. Positions past the
    reference fall back to the wrapped trajectory masks. Not a decoding method.
    """
    needs_hidden = False

    def __init__(self, base, embedding_table, continuation):
        self.base, self.num_masks = base, base.num_masks
        self.embedding_table = embedding_table
        self.continuation = list(continuation)
        self.committed = 0

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self.committed = 0

    def masks(self):
        return self.base.masks()

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)
        self.committed += 1

    def masks_for_tree(self, tree):
        trajectory = self.base.masks()
        owners = trajectory.unsqueeze(0).repeat(1 + tree.num_nodes, 1, 1)
        for owner, depth in enumerate([0] + list(tree.depths)):
            for slot in range(self.num_masks):
                index = self.committed - 1 + depth + slot + 1
                if index < len(self.continuation):
                    owners[owner, slot] = self.embedding_table[self.continuation[index]].to(trajectory.dtype)
        return owners


def run_oracle(config, seed):
    """Trajectory versus oracle masks on the same model, prompts and trees as run_seed."""
    generator = torch.Generator().manual_seed(seed)
    content = config['content_vocab']
    lang_a = make_language(content, config['branching'], config['sharpness'], generator)
    lang_b = make_language(content, config['branching'], config['sharpness'], generator)
    train = config['train']
    half = train['sequences'] // 2
    data = torch.cat([sample(lang_a, content, half, train['length'], generator),
                      sample(lang_b, content + 1, half, train['length'], generator)])
    session, losses = train_model(config, data, seed)
    plan = config['prompts']
    sample(lang_a, content, plan['fit'], plan['length'], generator)
    splits = {'holdout': sample(lang_a, content, plan['holdout'], plan['length'], generator),
              'cross': sample(lang_b, content + 1, plan['cross'], plan['length'], generator)}
    eval_cfg = {'max_new_tokens': config['max_new_tokens'], 'temperature': 0.0, 'stop_on_eos': False}
    result = {'seed': seed, 'train_losses': losses, 'trees': []}
    for tree in config['trees']:
        entry = {'tree': tree, 'splits': {}}
        for split, rows in splits.items():
            arms = {}
            for name in ('esp_trajectory', 'esp_oracle'):
                out = []
                for index in range(rows.shape[0]):
                    prompt = rows[index:index + 1]
                    reference = ar_generate(session, prompt, eval_cfg)[0]
                    spec = spec_for('esp', tree, 0.0)
                    provider = build_mask_provider(spec, session.embedding_table)
                    if name == 'esp_oracle':
                        provider = OracleMaskProvider(provider, session.embedding_table,
                                                      reference[0, prompt.shape[1]:].tolist())
                    decoder = SpecDecoder(session, spec, eval_cfg, generator=torch.Generator().manual_seed(0),
                                          mask_policy=provider)
                    output, stats = decoder.generate(prompt)
                    out.append({'qid': f'{split}-{index}', 'accepted': stats['accepted'],
                                'committed': stats['committed'], 'decode_calls': stats['decode_calls'],
                                'new_tokens': stats['new_tokens'], 'rounds': round_split(stats['trace']),
                                'argmax_passed': stats['verification_argmax_audit']['passed'],
                                'ar_equal': bool(torch.equal(output, reference))})
                arms[name] = {**pooled(out), 'rows': out}
            entry['splits'][split] = arms
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


def run(config, output, oracle=False):
    output = Path(output)
    root = Path(__file__).resolve().parents[3]
    if output.resolve().is_relative_to(root):
        raise ValueError('pilot output must stay outside the source tree')
    output.mkdir(parents=True, exist_ok=False)
    header = {'protocol': PROTOCOL + ('-oracle-bound' if oracle else ''),
              'scope': 'CPU synthetic pilot; not a real-model, speed or sampling result',
              'config': config, 'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              'source': source_identity(), 'torch': torch.__version__, 'python': platform.python_version(),
              'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32'}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, indent=2)
    results = []
    for seed in config['seeds']:
        result = (run_oracle if oracle else run_seed)(config, seed)
        with (output / f'seed-{seed}.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        results.append(result)
    return header, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='*')
    parser.add_argument('--oracle-bound', action='store_true',
                        help='trajectory versus future-token oracle masks; an upper bound, not a method')
    parser.add_argument('--variants', nargs='*', help='observation variants as form:condition, e.g. top1:rejection')
    args = parser.parse_args(argv)
    config = json.loads(json.dumps(DEFAULT))
    if args.seeds:
        config['seeds'] = args.seeds
    if args.variants:
        variants = [item.split(':') for item in args.variants]
        if any(tuple(variant) not in SUFFIX for variant in variants) or len(set(map(tuple, variants))) != len(variants):
            raise ValueError('variants must be unique form:condition pairs from ' + repr(sorted(SUFFIX)))
        config['variants'] = variants
    run(config, args.output, oracle=args.oracle_bound)


if __name__ == '__main__':
    main()
