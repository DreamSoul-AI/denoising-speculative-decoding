"""Qwen3-8B equal-node replacement and frozen last-token Kalman.

Protocol ``office-qwen3-8b-v1``. Prompt selection, arm specs, calibration and
the paired summary live here. The command that loads weights and writes files
is ``studies.trajectory_observation.flow.execute.office_qwen8b``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, creative_texts, direction, paired_deltas, prompt_sha256,
)

PROTOCOL = 'office-qwen3-8b-v1'
USED_THROUGH = 256
TAKE = 128
BOOT_SEED = 0
BOOT_RESAMPLES = 10000
SKIPPED_QIDS = (289, 304, 365)
EVAL_SHA256 = 'f4c5423ce250ae517abfe8ae04b3356f7bf725f66766d4f9ce323b30554766f4'
EVAL_QIDS = (
    tuple(range(257, 289)) + tuple(range(290, 304)) + tuple(range(305, 365)) + tuple(range(366, 388))
)
QWEN3_8B = Path('D' + ':' + '\\') / 'models' / 'Qwen3-8B'
MODELS = {'8b': QWEN3_8B}
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
EVAL = {'max_new_tokens': 64, 'temperature': 0.0, 'stop_on_eos': False}
FIT = {'max_new_tokens': 24, 'temperature': 0.0, 'stop_on_eos': False}
SMOKE = {'max_new_tokens': 8, 'temperature': 0.0, 'stop_on_eos': False}
TREE = {'mode': 'static', 'branches': [4], 'pruning': True}
ESP = {
    'method': 'esp', 'num_masks': 1, 'impl': 'efficient', 'tree': dict(TREE),
    'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
}
ROOT = Path(__file__).resolve().parents[3]


def templates():
    texts = [f'{head} {tail}' for head in HEADS for tail in TAILS]
    if len(texts) != 288 or len(set(texts)) != 288 or TAILS[15] != 'the last stop':
        raise RuntimeError('template grid changed')
    return texts


def prompts_after_used(rows, take=TAKE):
    """Take unused creative texts after qid 256, skipping strings already seen."""
    texts = creative_texts(rows)
    if len(texts) <= USED_THROUGH:
        raise RuntimeError('creative stream does not pass qid 256')
    used = set(texts[:USED_THROUGH + 1])
    skipped = []
    chosen = []
    for index, text in enumerate(texts[USED_THROUGH + 1:], start=USED_THROUGH + 1):
        if text in used:
            skipped.append(index)
            continue
        used.add(text)
        chosen.append({'qid': index, 'category': 'creative_writing', 'text': text})
        if len(chosen) == take:
            return chosen, tuple(skipped)
    raise RuntimeError(f'creative_writing has only {len(chosen)} new prompts after qid 256')


def assert_registered_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if digest != EVAL_SHA256 or qids != EVAL_QIDS or tuple(skipped) != SKIPPED_QIDS:
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered slice')
    if set(prompt['text'] for prompt in prompts) & set(templates()[:128]):
        raise RuntimeError('eval prompts overlap the calibration sentences')
    return digest


def esp_spec():
    return {'method': ESP['method'], 'num_masks': ESP['num_masks'], 'impl': ESP['impl'],
            'tree': dict(TREE), 'mask': dict(ESP['mask'])}


def _payload(prior):
    return {key: prior[key] for key in ('k', 'p', 'r', 'q')}


def replace_spec(prior):
    spec = esp_spec()
    spec['hybrid_gain'] = {'mode': 'replace', **_payload(prior)}
    return spec


def kalman_spec(prior):
    spec = esp_spec()
    spec['hybrid_gain'] = {'mode': 'frozen', 'observe': 'last', **_payload(prior)}
    return spec


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


def read_rows(path):
    file = Path(path)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]


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
    return {'wins': wins, 'losses': losses, 'ties': ties, 'p': pvalue}


def mean_tau(rows):
    return sum(row['tau'] for row in rows) / len(rows)


def pooled_tau(rows):
    calls = sum(row['decode_calls'] for row in rows)
    return None if calls == 0 else sum(row['committed'] for row in rows) / calls


def _arm_summary(name, treated, esp):
    compared = sign_test(treated, esp)
    accepted = bootstrap_mean_ci(paired_deltas(treated, esp, 'accepted'))
    trend = direction(compared['wins'], compared['losses'], accepted['mean'])
    return {
        'versus': 'esp',
        'wins_losses_ties': [compared['wins'], compared['losses'], compared['ties']],
        'p': compared['p'],
        'tau_mean': {'esp': mean_tau(esp), name: mean_tau(treated)},
        'tau_pooled': {'esp': pooled_tau(esp), name: pooled_tau(treated)},
        'accepted_difference': accepted,
        'direction': trend,
        'positive_trend': trend == 'positive',
        'argmax_all_passed': all(row['argmax_passed'] for row in esp + treated),
        'ar_mismatches': {'esp': sum(not row['ar_equal'] for row in esp),
                          name: sum(not row['ar_equal'] for row in treated)},
    }


def office_summary(esp, replaced, kalman):
    if not (len(esp) == len(replaced) == len(kalman) == TAKE):
        raise RuntimeError('summary needs all three arms complete')
    replace = _arm_summary('replace', replaced, esp)
    replace['max_nodes'] = max(row['max_nodes'] for row in replaced)
    kalman_summary = _arm_summary('kalman', kalman, esp)
    return {
        'protocol': PROTOCOL,
        'prompts': TAKE,
        'prompt_sha256': EVAL_SHA256,
        'replace_vs_esp': replace,
        'kalman_vs_esp': kalman_summary,
        'argmax_all_passed': replace['argmax_all_passed'] and kalman_summary['argmax_all_passed'],
    }


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids.detach().cpu().reshape(-1).tolist()).encode()).hexdigest()


def replay_block(model, prompt_ids, full_ids):
    """Diagonal pairs along an AR continuation. Same rows as fair_branch.calibrate."""
    import torch

    from spec_mtp.masks import build_mask_provider

    prompt_len = int(prompt_ids.shape[1])
    embeds = model.embed(full_ids)[0]
    spec = esp_spec()
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


def prior_from_triples(predicted, observed, target, prompts):
    from spec_mtp.masks import calibrate_diagonal_gain

    prior = calibrate_diagonal_gain(predicted, observed, target)
    prior['prompts'] = int(prompts)
    for name in ('k', 'p', 'r', 'q'):
        prior[f'{name}_mean'] = sum(prior[name]) / len(prior[name])
    return prior


def collect_last_pairs(model, tokenizer, text, tokens):
    """ESP rounds on one calibration sentence. y is the last committed token."""
    import torch

    from spec_mtp.decode import SpecDecoder
    from spec_mtp.masks import build_mask_provider

    spec = esp_spec()
    spec['hybrid_gain'] = {'mode': 'collect', 'observe': 'last'}
    ids = tokenizer(text, return_tensors='pt').input_ids
    policy = build_mask_provider(spec, model.embedding_table)
    limit = {'max_new_tokens': int(tokens), 'temperature': 0.0, 'stop_on_eos': False}
    _output, stats = SpecDecoder(model, spec, limit, mask_policy=policy).generate(ids)
    if not stats['verification_argmax_audit']['passed']:
        raise RuntimeError('argmax audit failed during Kalman calibration')
    harvested = policy.calibration_rows()
    return harvested['m'], harvested['z'], harvested['last']


def arm_row(model, spec, ids, limit):
    from spec_mtp.decode import SpecDecoder
    from spec_mtp.masks import build_mask_provider

    policy = build_mask_provider(spec, model.embedding_table)
    output, stats = SpecDecoder(model, spec, limit, mask_policy=policy).generate(ids)
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
    mode = spec.get('hybrid_gain', {}).get('mode')
    if mode == 'replace':
        diag = policy.diagnostics()
        row['extra_added'] = diag['extra_added']
        row['extra_present'] = diag['extra_present']
        row['max_nodes'] = diag['max_nodes']
        if row['max_nodes'] != 4:
            raise RuntimeError(f"replace arm draft nodes are {row['max_nodes']}, not 4")
    return row


def decode_prompt(model, tokenizer, prompt, specs, limit):
    from spec_mtp.decode import ar_generate

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


def load_dolly_rows():
    import sys
    from importlib import import_module

    sys.modules.pop('datasets', None)
    root = ROOT.resolve()
    kept = [entry for entry in sys.path if Path(entry or '.').resolve() != root]
    saved = sys.path[:]
    sys.path[:] = kept
    try:
        return import_module('datasets').load_dataset('databricks/databricks-dolly-15k', split='train')
    finally:
        sys.path[:] = saved
