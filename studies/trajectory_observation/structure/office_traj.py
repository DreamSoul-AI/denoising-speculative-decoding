"""Qwen3-8B trajectory-only versus joint observation ablation.

Protocol ``office-8b-traj-ablate-v1``. Prompt selection, the two arm specs and
the paired summary live here. The command that loads weights is
``studies.trajectory_observation.flow.execute.office_traj``. Both arms wrap the
current ESP baseline with the existing ``spec.observation`` switch. This module
does not fit K and does not mutate the process environment on import.
"""

from __future__ import annotations

from studies.trajectory_observation.structure.fresh_replace import (
    bootstrap_mean_ci, direction, paired_deltas, prompt_sha256,
)
from studies.trajectory_observation.structure.office_tree import (
    NODE_BUDGET, EVAL as TREE_EVAL, SMOKE as TREE_SMOKE, block_complexity, prompts_after,
)

PROTOCOL = 'office-8b-traj-ablate-v1'
ROLE = 'exploration'
START = 647
END = 708
TAKE = 60
BOOT_SEED = 0
BOOT_RESAMPLES = 10000
SKIPPED_QIDS = (658, 678)
EVAL_SHA256 = 'd23ad590c45307fa7c1981b272adaec243bfaca5a2e390eda394bf2fe42a4cca'
EVAL_QIDS = tuple(qid for qid in range(START, END + 1) if qid not in SKIPPED_QIDS)
GAIN_FULL = 0.5
GAIN_TRAJ = 0.0
MASK = {'init': 'mean_prompt', 'lam': 0.1, 'update': True}
TREE = {'mode': 'static', 'branches': [4], 'pruning': True}
EVAL = dict(TREE_EVAL)
SMOKE = dict(TREE_SMOKE)
ARMS = (
    ('full', GAIN_FULL),
    ('traj', GAIN_TRAJ),
)


def arm_spec(gain):
    """ESP baseline plus the existing observation blend switch."""
    value = float(gain)
    if value not in (GAIN_FULL, GAIN_TRAJ):
        raise RuntimeError('this ablation only uses the pre-registered gains')
    return {
        'method': 'esp',
        'num_masks': 1,
        'impl': 'efficient',
        'tree': dict(TREE),
        'mask': dict(MASK),
        'observation': {'gain': value, 'condition': 'none', 'form': 'mean'},
    }


def arm_specs():
    return {name: arm_spec(gain) for name, gain in ARMS}


def prompts_from_647(rows, take=TAKE):
    """Unused creative texts from qid 647 through the end of the stream."""
    return prompts_after(rows, START, take)


def assert_registered_slice(prompts, skipped):
    digest = prompt_sha256(prompts)
    qids = tuple(prompt['qid'] for prompt in prompts)
    if len(EVAL_QIDS) != TAKE or EVAL_QIDS[0] != START or EVAL_QIDS[-1] != END:
        raise RuntimeError('registered qids do not cover the unused tail')
    if digest != EVAL_SHA256 or qids != EVAL_QIDS or tuple(skipped) != SKIPPED_QIDS:
        raise RuntimeError(f'prompt hash {digest} does not match the pre-registered slice')
    return digest


def _compare(treated_name, treated, baseline_name, baseline):
    from studies.trajectory_observation.structure.fair_branch import mean_tau, pooled_tau, sign_test

    compared = sign_test(treated, baseline)
    accepted = bootstrap_mean_ci(paired_deltas(treated, baseline, 'accepted'))
    full_gain = bootstrap_mean_ci(paired_deltas(baseline, treated, 'accepted'))
    tau = bootstrap_mean_ci(paired_deltas(treated, baseline, 'tau'))
    trend = direction(compared['wins'], compared['losses'], accepted['mean'])
    return {
        'baseline': baseline_name,
        'arm': treated_name,
        'delta': 'accepted_traj - accepted_full',
        'full_gain': 'accepted_full - accepted_traj',
        'wins': compared['wins'],
        'losses': compared['losses'],
        'ties': compared['ties'],
        'wins_losses_ties': [compared['wins'], compared['losses'], compared['ties']],
        'p': compared['p'],
        'tau_mean': {baseline_name: mean_tau(baseline), treated_name: mean_tau(treated)},
        'tau_pooled': {baseline_name: pooled_tau(baseline), treated_name: pooled_tau(treated)},
        'accepted_difference': accepted,
        'full_gain_difference': full_gain,
        'tau_difference': tau,
        'direction': trend,
        'confirmatory': False,
        'argmax_all_passed': all(row['argmax_passed'] for row in baseline + treated),
        'ar_mismatches': {
            baseline_name: sum(not row['ar_equal'] for row in baseline),
            treated_name: sum(not row['ar_equal'] for row in treated),
        },
        'max_nodes': max(row['max_nodes'] for row in baseline + treated),
        'block_complexity': {
            baseline_name: sorted({row['block_complexity'] for row in baseline}),
            treated_name: sorted({row['block_complexity'] for row in treated}),
        },
    }


def ablation_summary(rows_by_name, protocol=PROTOCOL, digest=EVAL_SHA256, skipped=SKIPPED_QIDS):
    if set(rows_by_name) != {name for name, _gain in ARMS}:
        raise RuntimeError('summary is missing a registered arm')
    if any(len(rows) != TAKE for rows in rows_by_name.values()):
        raise RuntimeError(f'summary needs {TAKE} rows in every arm')
    expected = block_complexity(1, TREE['branches'])
    comparison = _compare('traj', rows_by_name['traj'], 'full', rows_by_name['full'])
    complexities = comparison['block_complexity']['full'] + comparison['block_complexity']['traj']
    passed = (
        comparison['argmax_all_passed']
        and comparison['max_nodes'] == NODE_BUDGET
        and complexities == [expected, expected]
    )
    return {
        'protocol': protocol,
        'role': ROLE,
        'confirmatory': False,
        'prompts': TAKE,
        'prompt_sha256': digest,
        'skipped_qids': list(skipped),
        'baseline': 'full',
        'comparison': comparison,
        'argmax_all_passed': passed,
    }
