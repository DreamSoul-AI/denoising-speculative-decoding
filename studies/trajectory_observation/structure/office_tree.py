"""Equal-node static tree shapes for the Qwen3 explorations.

``office-tree-shape-v1`` is the finished Qwen3-4B slice. ``office-8b-tree-shape-v1``
reuses the same three shapes on a later Dolly slice. Prompt selection and the
paired summary live here. The command that loads weights is
``studies.trajectory_observation.flow.execute.office_tree``. This module does
not fit K and does not mutate the process environment on import.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def arm_row(model, spec, ids, limit):
    import spec_mtp.decode as decode_mod
    from spec_mtp.decode import SpecDecoder
    from spec_mtp.masks import build_mask_provider

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
    return {
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
