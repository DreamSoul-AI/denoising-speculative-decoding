"""Library helpers folded from one-shot structure scripts."""

import torch  # before numpy on Windows
import hashlib
import json
from pathlib import Path
import numpy as np
POSITION_EDGES = (0, 32, 64, 128, 192, 256)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def distribution(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {'n': 0}
    assert np.isfinite(values).all()
    return dict(n=int(values.size), mean=float(values.mean()), **{name: float(np.quantile(values, q)) for name, q in [('min', 0), ('p05', 0.05), ('p25', 0.25), ('median', 0.5), ('p75', 0.75), ('p95', 0.95), ('max', 1)]})

def trace_metrics(evidence, prompt_length):
    stats = evidence['stats']
    traces = stats['trace']
    assert len(traces) == stats['decode_calls']
    assert sum((len(t['accepted_nodes']) for t in traces)) == stats['accepted']
    assert sum((len(t['committed_tokens']) for t in traces)) == stats['committed']
    assert sum(stats['accept_by_depth']) == stats['accepted']
    events, offset = ([], 0)
    for t in traces:
        root = t['root_position'] - prompt_length
        assert root == offset and t['cache_before'] == t['root_position']
        accepted = len(t['accepted_nodes'])
        assert accepted <= 2
        events.append((root, accepted, len(t['committed_tokens'])))
        offset += len(t['committed_tokens'])
    assert stats['accepted'] + stats['decode_calls'] >= stats['committed']
    return events

def summarize_geometry(capture):
    assert capture['quality_passed'] and capture['status'] == 'complete'
    methods = ('input_first', 'output_zero', 'output_first')
    metrics = ('mask_norm', 'raw_norm', 'target_norm', 'mask_target_cos', 'raw_target_cos', 'mask_target_norm_ratio', 'slot_pair_cos', 'velocity_anchor_ratio', 'real_hidden_norm', 'real_hidden_target_cos', 'predicted_hidden_real_cos', 'mask_scale_relative_error', 'reconstruction_direction_cos')
    result = {'groups': {}, 'slot_effects': {}, 'matched_shared_prefix': {}, 'capture_cases': len(capture['cases'])}
    for cohort in ('all', 'shared_prefix'):
        points = [p for p in capture['points'] if cohort == 'all' or p['shared_prefix']]
        for method in methods:
            indexed_slots = {(p['question_id'], p['root_index'], p['depth'], p['slot']): p for p in points if p['case'] == method}
            pairs = [(p, indexed_slots[key[:-1] + (1,)]) for key, p in indexed_slots.items() if key[-1] == 0 and key[:-1] + (1,) in indexed_slots]
            result['slot_effects'][f'{cohort}/{method}'] = {f'{field}_slot1_over_slot0': distribution([b[field] / a[field] for a, b in pairs]) for field in ('raw_norm', 'mask_norm')}
            for slot in (0, 1):
                for depth in (None, 0, 1, 2):
                    rows = [p for p in points if p['case'] == method and p['slot'] == slot and (depth is None or p['depth'] == depth)]
                    key = f'{cohort}/{method}/depth{depth}/slot{slot}'
                    result['groups'][key] = {m: distribution([p[m] for p in rows if m in p]) for m in metrics}
    indexed = {}
    for method in methods:
        rows = [p for p in capture['points'] if p['case'] == method and p['shared_prefix']]
        indexed[method] = {(p['question_id'], p['root_index'], p['depth'], p['slot']): p for p in rows}
        assert len(rows) == len(indexed[method]), 'Duplicate observed owner/slot'
    common = set.intersection(*(set(indexed[m]) for m in methods))
    for first, second in [('input_first', 'output_zero'), ('input_first', 'output_first'), ('output_zero', 'output_first')]:
        for slot in (0, 1):
            keys = sorted((k for k in common if k[-1] == slot))
            for key in keys:
                assert abs(indexed[first][key]['target_norm'] - indexed[second][key]['target_norm']) < 1e-06
            result['matched_shared_prefix'][f'{second}-{first}/slot{slot}'] = {metric: distribution([indexed[second][k][metric] - indexed[first][k][metric] for k in keys]) for metric in ('mask_target_cos', 'mask_target_norm_ratio', 'velocity_anchor_ratio')}
    return result

def summarize_cell(raw):
    rows, evidence = (raw['out']['rows'], raw['out']['evidence'])
    assert len(rows) == len(evidence) == 480
    bins = [{'root_index_range': [a, b], 'calls': 0, 'accept_ge1': 0, 'accept_ge2': 0, 'accepted': 0, 'committed': 0} for a, b in zip(POSITION_EDGES, POSITION_EDGES[1:])]
    attempts, accepts = ([0, 0, 0, 0], [0, 0, 0, 0])
    norm_min, norm_max, norm_errors, owner_diffs, scales = ([], [], [], [], [])
    by_category = {}
    for row, e in zip(rows, evidence):
        stats = e['stats']
        events = trace_metrics(e, len(e['input_ids']))
        assert len(e['output_ids']) - len(e['input_ids']) == row['new_tokens'] == stats['new_tokens']
        assert stats['verification_argmax_audit']['passed']
        for d in (1, 2, 3):
            attempts[d] += row[f'attempt_d{d}']
            accepts[d] += row[f'accept_d{d}']
            assert row[f'attempt_d{d}'] == stats['attempt_by_depth'][d]
            assert row[f'accept_d{d}'] == stats['accept_by_depth'][d]
        category = by_category.setdefault(row['category'], {'calls': 0, 'accepted': 0, 'committed': 0, 'prompts': 0})
        category['prompts'] += 1
        for root, accepted, committed in events:
            cell = next((b for b in bins if b['root_index_range'][0] <= root < b['root_index_range'][1]))
            for counter in (cell, category):
                counter['calls'] += 1
                counter['accepted'] += accepted
                counter['committed'] += committed
            cell['accept_ge1'] += accepted >= 1
            cell['accept_ge2'] += accepted >= 2
        if 'output_hidden_prefill' in stats:
            scale = stats['output_hidden_prefill']['scale']
            scales.append(scale)
            for t in stats['trace']:
                m = t['output_hidden']
                norm_min.append(m['mask_norm_min'])
                norm_max.append(m['mask_norm_max'])
                norm_errors.extend([abs(m['mask_norm_min'] / scale - 1), abs(m['mask_norm_max'] / scale - 1)])
                owner_diffs.append(m['last_owner_diff_from_root'])
    calls, accepted, committed = [sum((r[k] for r in rows)) for k in ('decode_calls', 'accepted', 'committed')]
    assert accepts[1] + accepts[2] == accepted and attempts[1] == calls
    trim = accepted + calls - committed
    tau = committed / calls
    assert abs(tau - (1 + accepts[1] / calls + accepts[2] / calls - trim / calls)) < 1e-12
    for b in bins:
        b.update(accept_ge1_per_call=b['accept_ge1'] / b['calls'], accept_ge2_per_call=b['accept_ge2'] / b['calls'], accepted_per_call=b['accepted'] / b['calls'])
    return {'prompts': len(rows), 'decode_calls': calls, 'accepted': accepted, 'committed': committed, 'tau': tau, 'trim_loss': trim, 'tau_components': {'d1': accepts[1] / calls, 'd2': accepts[2] / calls, 'negative_trim': -trim / calls}, 'depths': [{'depth': d, 'attempts': attempts[d], 'accepted': accepts[d], 'conditional_accept_rate': accepts[d] / attempts[d] if attempts[d] else None, 'attempts_per_decode_call': attempts[d] / calls} for d in (1, 2)], 'position_bins': bins, 'categories': by_category, 'norm_audit': {'scale': distribution(scales), 'block_min_norm': distribution(norm_min), 'block_max_norm': distribution(norm_max), 'relative_error_extrema': distribution(norm_errors), 'last_owner_diff_from_root': distribution(owner_diffs)}, 'hidden_vectors_saved': False}

def audit_saved(study, paired):
    """Recheck saved tokens and first blocks; do not re-sign native Runs."""
    import importlib.util
    import sys
    from studies._shared.structure import validation
    root = Path(__file__).resolve().parents[3]
    norm = root / 'studies/qwen3_4b_bf16_bias_norm_scan'
    sys.path.insert(0, str(norm))
    spec = importlib.util.spec_from_file_location('geometry_norm_helpers', norm / 'analyze_results.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    from norm_control import check_refs
    check_refs()
    assert sha(study / 'docs/paired_evidence_v1.json') == sha(root / 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json')
    prompts = helpers.read(root / 'studies/qwen3_4b_bf16_five_methods/docs/prompts_manifest.json')['rendered_prompts']
    assert len(prompts) == 480 and set(paired['cells']) == set(helpers._pair.CASES)
    blocks, cells, reference = ({}, {}, [])
    for name in helpers._pair.CASES:
        cell = paired['cells'][name]
        folder = study / 'runs' / cell['run_id']
        assert cell['native_status'] == 'succeeded'
        from spec_mtp.structure.artifact.observation import verify_files
        verify_files(folder, cell['files_sha256'])
        raw = helpers.read(folder / 'assets/decode_observations.json')['out']['evidence']
        rows = cell['rows']
        assert len(raw) == len(rows) == len(prompts) == 480
        assert helpers._pair.summarize(rows, name == 'AR') == cell['totals']
        checked = 0
        for item, prompt, row in zip(raw, prompts, rows):
            assert item['input_ids'] == prompt['input_ids'] and row['question_id'] == prompt['question_id']
            output = item['output_ids'][len(prompt['input_ids']):]
            assert hashlib.sha256(json.dumps(output).encode()).hexdigest() == row['output_sha256']
            assert len(output) == row['new_tokens']
            if name == 'AR':
                assert item['stats']['ar_calls'] == row['model_calls']
                reference.append((output, row['model_calls']))
            else:
                checked += validation._check_argmax(item)
                assert all((item['stats'][k] == row[k] for k in ('accepted', 'committed', 'decode_calls', 'new_tokens')))
                assert row['model_calls'] == item['stats']['decode_calls'] + item['stats']['prefill_calls']
        if name != 'AR':
            assert cell['contract']['dtype'] == 'torch.bfloat16' and cell['contract']['tied_storage']
            assert cell['model_digest'] == paired['cells']['AR']['model_digest']
            settings = cell['actual_settings']
            assert settings['spec']['num_masks'] == 2 and settings['spec']['block_complexity'] == int(name.split('/')[0])
            assert settings['eval']['temperature'] == 0 and settings['eval']['max_new_tokens'] == 256
            assert checked == cell['totals']['argmax_checked_tokens']
            for item, row, (ar, calls) in zip(raw, rows, reference):
                output = item['output_ids'][len(item['input_ids']):]
                assert row['AR_exact'] == (output == ar) and row['AR_new_tokens'] == len(ar)
                assert row['AR_model_calls'] == calls and row['calls_delta_vs_AR'] == row['model_calls'] - calls
                assert row['first_new_difference_index'] == helpers._pair.first_difference(output, ar)
            blocks[name] = helpers.depth_rows(raw, prompts)
        cells[name] = dict(run_id=cell['run_id'], files_sha256=cell['files_sha256'], raw_argmax_checked_tokens=checked)
    pairs = [dict(first=p['first'], second=p['second'], first_block=helpers.first_block_compare(blocks[p['first']], blocks[p['second']])) for p in paired['pairs']]
    assert len(pairs) == 20
    return dict(cells=cells, pairs=pairs, raw_files_checked=sum((len(c['files_sha256']) for c in cells.values())), raw_argmax_checked_tokens=sum((c['raw_argmax_checked_tokens'] for c in cells.values())), shared_AR_tokens=sum((len(ids) for ids, _ in reference)), helpers_sha256=sha(norm / 'analyze_results.py'), math_checker_sha256=sha(validation.__file__), limitations='Saved token/owner/hash evidence only; no logits recomputation, new forward or source re-signing. First-block ties do not imply zero velocity.')

def velocity_geometry(anchor, velocity):
    a, v = (anchor.float(), velocity.float())
    assert a.norm() > 0 and v.norm() > 0 and ((a + v).norm() > 0)
    radial = a * (v @ a) / (a @ a)
    tangent = v - radial
    normalized_move = (a + v) / (a + v).norm() - a / a.norm()
    return {'velocity_over_anchor': float(v.norm() / a.norm()), 'radial_signed_over_velocity': float(v @ a / (a.norm() * v.norm())), 'tangent_over_velocity': float(tangent.norm() / v.norm()), 'normalized_movement_retention': float(normalized_move.norm() / (v.norm() / a.norm())), 'raw_slot_norm_ratio': float((a + v).norm() / a.norm())}

def transition_counts(evidence):
    result = {str(d): {'transitions': 0, 'next_accept_d1': 0, 'next_accept_d2': 0} for d in (0, 1, 2)}
    for e in evidence:
        trace_metrics(e, len(e['input_ids']))
        for previous, current in zip(e['stats']['trace'], e['stats']['trace'][1:]):
            assert not previous['stopped']
            cell = result[str(len(previous['accepted_nodes']))]
            cell['transitions'] += 1
            cell['next_accept_d1'] += bool(current['accepted_nodes'])
            cell['next_accept_d2'] += len(current['accepted_nodes']) == 2
    for cell in result.values():
        n = cell['transitions']
        cell['next_accept_d1_rate'] = cell['next_accept_d1'] / n if n else None
        cell['next_accept_d2_rate'] = cell['next_accept_d2'] / n if n else None
    return result

def template_metrics(level, velocity, direction, masks):
    fixed = torch.stack([level + i * direction for i in (1, 2)])
    cosine = torch.nn.functional.cosine_similarity
    return {'velocity_initial_cos': float(cosine(velocity.float(), direction.float(), dim=0)), 'velocity_initial_norm_ratio': float(velocity.float().norm() / direction.float().norm()), 'velocity_initial_relative_error': float((velocity.float() - direction.float()).norm() / direction.float().norm()), 'constant_slot0_cos': float(cosine(fixed[0].float(), masks[0].float(), dim=0)), 'constant_slot1_cos': float(cosine(fixed[1].float(), masks[1].float(), dim=0)), 'constant_slot0_relative_error': float((fixed[0].float() - masks[0].float()).norm() / masks[0].float().norm()), 'constant_slot1_relative_error': float((fixed[1].float() - masks[1].float()).norm() / masks[1].float().norm())}

def rank_interval(logits, target, banned):
    if target == banned:
        return None
    greater = int((logits > logits[target]).sum()) - int(logits[banned] > logits[target])
    equal = int((logits == logits[target]).sum()) - int(logits[banned] == logits[target])
    return (greater + 1, greater + equal)

def coverage(logits, root, truth, budget=9, tree=None):
    from spec_mtp.structure.algorithm.decode.loop import tree_probs
    from spec_mtp.structure.algorithm.decode.tree import build_dynamic_tree
    if tree is None:
        tree = build_dynamic_tree(tree_probs(logits, 0.0), budget, root, pruning=True)
    first = tree.children_of(0)
    correct = next((c for c in first if tree.tokens[c - 1] == truth[0]), None)
    depth2_width = sum((d == 2 for d in tree.depths))
    d1_rank = rank_interval(logits[0], truth[0], root)
    d2_rank = rank_interval(logits[1], truth[1], truth[0])
    return (tree, {'actual_d1_width': len(first), 'actual_d2_width': depth2_width, 'reference_d1_is_expanded_top1': bool(first) and tree.tokens[first[0] - 1] == truth[0], 'reference_d1_in_tree': correct is not None, 'reference_d2_in_actual_tree': correct is not None and any((tree.tokens[c - 1] == truth[1] for c in tree.children_of(correct))), 'd1_pruned_rank_interval': d1_rank, 'd2_oracle_parent_pruned_rank_interval': d2_rank, 'all_budget_to_d1_definite_coverage_only': d1_rank is not None and d1_rank[1] <= budget, 'all_budget_to_d1_possible_coverage_only': d1_rank is not None and d1_rank[0] <= budget, 'oracle_parent_d2_definite_coverage_only': correct is not None and d2_rank is not None and (d2_rank[1] <= depth2_width), 'oracle_parent_d2_possible_coverage_only': correct is not None and d2_rank is not None and (d2_rank[0] <= depth2_width), 'repeat_pruning_bans_d1': truth[0] == root})

def plot_results(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    methods = ('zero', 'input_first', 'first', 'shifted', 'radial', 'both')
    qids = list(dict.fromkeys((r['question_id'] for r in report['rows'])))
    indexed = {(r['question_id'], r['variant']): r for r in report['rows']}
    assert len(indexed) == len(qids) * len(methods) == len(report['rows'])
    ranks = np.array([[indexed[q, m]['mask_logit_target_ranks'][0] for q in qids] for m in methods])
    accepted = np.array([[indexed[q, m]['next_accept_depth'] for q in qids] for m in methods])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, values, title, options in zip(axes, (ranks, accepted), ('Target rank in mask-1 logits', 'Next-block verified draft tokens'), ({'norm': LogNorm(vmin=1, vmax=max(1000, int(ranks.max()))), 'cmap': 'viridis_r'}, {'vmin': 0, 'vmax': 2, 'cmap': 'Blues'})):
        chart = ax.imshow(values, aspect='auto', **options)
        ax.set_title(title)
        ax.set_yticks(range(6), ['O0', 'I1 same history', 'O1 same history', 'O1 shifted', 'O1 radial', 'O1 both'])
        ax.set_xticks(range(len(qids)), [f"q{q}\nR{indexed[q, 'zero']['root_index']}" for q in qids])
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, str(values[i, j]), ha='center', va='center', color='red', fontsize=11)
        fig.colorbar(chart, ax=ax, shrink=0.8, **{'ticks': [0, 1, 2]} if values is accepted else {})
    fig.savefig(path, dpi=160)
    plt.close(fig)
import importlib.util
import ast
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
STUDY = ROOT / 'studies/qwen3_4b_bf16_directions'
RECOVERY = ROOT / 'studies/qwen3_4b_bf16_directions_recovery'
_spec = importlib.util.spec_from_file_location('original_five_method_readout', BASE / 'analyze_results.py')
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
load_cell, compare_rows = (_original.load_cell, _original.compare_rows)
NAMES = ('template', 'random0', 'random1', 'random2', 'zero', 'content')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha__analyze_4b_full_directions(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def category_mean(rows):
    categories = sorted({r['category'] for r in rows})
    return sum((sum((r['committed'] for r in rows if r['category'] == c)) / max(sum((r['decode_calls'] for r in rows if r['category'] == c)), 1) for c in categories)) / len(categories)

def compare(first, second):
    result = compare_rows(first, second)
    for a, b, r in zip(first, second, result['rows']):
        diff = b['committed'] * max(a['decode_calls'], 1) - a['committed'] * max(b['decode_calls'], 1)
        r['tau'] = 'win' if diff > 0 else 'loss' if diff < 0 else 'tie'
    result['tau'] = {s: sum((r['tau'] == s for r in result['rows'])) for s in ('win', 'tie', 'loss')}
    return result

def require_resource_failure(previous, error):
    assert previous['native_status'] == 'failed', 'Never replace a completed or running factor'
    assert any((text in error for text in ('Foreign GPU work appeared', 'Nonpreemptive GPU admission deferred'))), error

def control_source(path):
    source = path.read_text(encoding='utf-8')
    node = next((n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'control'))
    return ast.get_source_segment(source, node)

def plot(body, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert body['complete']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    try:
        values = [body['cells'][n]['totals']['pooled_tau'] for n in NAMES]
        axes[0].barh(NAMES, values)
        for i, value in enumerate(values):
            axes[0].text(value, i, f' {value:.6f}', va='center')
        axes[0].set_xlim(0, max(values) * 1.16)
        axes[0].set_xlabel('Pooled decode committed / decode calls')
        axes[0].set_title('BC30, 480 original prompts')
        names = [p['second'] for p in body['vs_template']]
        left = [0] * len(names)
        for key, color in [('win', '#3b8c6e'), ('tie', '#bbbbbb'), ('loss', '#c16b64')]:
            count = [p['accepted'][key] for p in body['vs_template']]
            axes[1].barh(names, count, left=left, label=key, color=color)
            left = [a + b for a, b in zip(left, count)]
        assert left == [480] * len(names)
        axes[1].set_xlabel('Prompts: accepted W/T/L versus fixed e2-e1')
        axes[1].legend()
        fig.supxlabel('Fixed data, BF16/T0. Different output paths/lengths; counts are not timing or significance.')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
from spec_mtp.structure.algorithm.mask import build_mask_provider
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

def admission(initial=True):
    """Observe available capacity; this is not an atomic cross-agent reservation."""
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert len(gpu) == 3, 'Only the declared single GPU is supported'
    command = "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True)
    processes = json.loads(raw) if raw.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    foreign = [p for p in processes if p['ProcessId'] != os.getpid()]
    cpu_ci = [p for p in foreign if any((name in (p['CommandLine'] or '') for name in ('watch-cleanup.py', 'watch-third.py', 'release-ci.py'))) or ('RPipe' in (p['CommandLine'] or '') and 'and not gpu and not slow' in (p['CommandLine'] or ''))]
    cpu_scripts = {'download_layers.py': '10689bac808d0e5fa1958d306508f459d3c4637dc92a51bf5b6c1bb915545026', 'build_official_container_v2.py': 'b22cbe9814bcfcf85e9e6caade2a60a2258d2bb83f5e1d856124753bf8976e44'}
    verified_cpu = []
    for p in foreign:
        command = p['CommandLine'] or ''
        for name, digest in cpu_scripts.items():
            script = Path('.tmp/paroquant-reference') / name
            if 'quantization\\.tmp\\paroquant-paper-env\\python.exe' in command and f'.tmp/paroquant-reference/{name}' in command and script.is_file() and (hashlib.sha256(script.read_bytes()).hexdigest() == digest):
                verified_cpu.append(p)
    for _ in foreign:
        known_ids = {p['ProcessId'] for p in cpu_ci + verified_cpu}
        children = [p for p in foreign if p not in cpu_ci and p not in verified_cpu and (p.get('ParentProcessId') in known_ids)]
        if not children:
            break
        cpu_ci.extend(children)
    unknown = [p for p in foreign if p not in cpu_ci and p not in verified_cpu]
    passed = not unknown and (not initial or int(gpu[1]) >= 18 * 1024)
    return {'passed': passed, 'gpu_uuid': gpu[0], 'free_mib': int(gpu[1]), 'used_mib': int(gpu[2]), 'foreign_python_count': len(foreign), 'recognized_cpu_ci_count': len(cpu_ci), 'verified_cpu_preparation_count': len(verified_cpu), 'unclassified_python_count': len(unknown), 'initial_capacity_check': initial}

def raw_output_slots(anchor, velocity, proxies, order, beta_prime):
    """Reconstruct unscaled predictions for observation, without changing the provider."""
    import torch
    a, v = (anchor.float().clone(), velocity.float().clone())
    result, ratios = ([], [])
    for depth in range(3):
        if depth < 2:
            y = proxies[depth].float()
            if order:
                v = beta_prime * v + (1 - beta_prime) * (y - a)
            a = y
        else:
            a = a + v
        result.append(torch.stack([a, a + v]))
        ratios.append(float(v.norm() / a.norm()))
    return (torch.stack(result), ratios)

def observe(model, prompt, spec, eval_cfg, eos, seed):
    import torch
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    from studies._shared.structure.config import _convert
    original_provider, original_tree = (build_mask_provider, decoding.build_tree)
    original_masks = OutputHiddenMaskProvider.masks_per_owner
    active, frames = ({}, [])
    calls = 0

    def provider_builder(*args, **kwargs):
        provider = original_provider(*args, **kwargs)
        active['provider'] = provider
        return provider

    def tree_builder(*args, **kwargs):
        tree = original_tree(*args, **kwargs)
        active['tree'] = tree
        return tree

    def masks_spy(provider, proxies):
        active['state'] = {k: getattr(provider, k).detach().cpu().clone() for k in ('anchor', 'velocity')}
        active['state']['proxies'] = proxies.detach().cpu().clone()
        return original_masks(provider, proxies)

    def before(_module, args, kwargs):
        nonlocal calls
        calls += 1
        active['prefill'] = kwargs.get('attention_mask_4d') is None
        if not active['prefill']:
            embeds, positions = args[:2]
            tree = active['tree']
            active['masks'] = embeds[0, 1 + tree.num_nodes:].reshape(1 + tree.num_nodes, 2, -1)
            active['root_index'] = int(positions[0, 0]) - prompt.shape[1]

    def norm_hook(_module, _args, output):
        active['hidden'] = output

    def after(_module, _args, _kwargs, output):
        if active['prefill']:
            return
        tree, owner, accepted = (active['tree'], 0, [])
        while True:
            target = int(output.logits[0, owner].argmax())
            match = next((child for child in tree.children_of(owner) if tree.tokens[child - 1] == target), None)
            if match is None:
                break
            accepted.append(match)
            owner = match
        owners = [0] + accepted
        depths = [0] + [tree.depths[owner - 1] for owner in accepted]
        provider = active['provider']
        frame = {'root_index': active['root_index'], 'accepted_nodes': accepted, 'depths': depths, 'masks': active['masks'][owners].detach().cpu().clone(), 'real_hidden': active['hidden'][0, owners].detach().cpu().clone()}
        if spec['method'] == 'output_hidden':
            state = active['state']
            raw, ratios = raw_output_slots(state['anchor'], state['velocity'], state['proxies'], spec['output']['order'], spec['output']['beta_prime'])
            frame.update(raw_predictions=raw[depths], velocity_ratios=[ratios[d] for d in depths], target_scale=float(provider.scale), provider_state=state)
        else:
            frame.update(raw_predictions=frame['masks'].float().clone(), velocity_ratios=[float(provider._velocity.float().norm() / provider._level.float().norm())] * len(owners), input_velocity=provider._velocity.detach().cpu().clone(), input_level=provider._level.detach().cpu().clone())
        frames.append(frame)
    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.hf.model.norm.register_forward_hook(norm_hook), model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch.object(decoding, 'build_tree', tree_builder), patch.object(OutputHiddenMaskProvider, 'masks_per_owner', masks_spy), patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.0)):
            output, stats = decoding.SpecDecoder(model, _convert(spec), _convert(eval_cfg), eos_ids=eos, generator=torch.Generator(device=model.device).manual_seed(seed), mask_policy=provider_builder(_convert(spec), model.embedding_table)).generate(prompt)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == stats['prefill_calls'] + stats['decode_calls'] == 1 + len(frames)
    assert stats['verification_argmax_audit']['passed']
    for frame, trace in zip(frames, stats['trace']):
        assert frame['root_index'] == trace['root_position'] - prompt.shape[1]
        assert frame['accepted_nodes'] == trace['accepted_nodes']
    return (output[0].detach().cpu(), stats, frames, calls)

def cosine(a, b):
    import torch
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1))

def point_rows(frames, generated, embedding_table, case, qid, common_prefix):
    import torch
    points, skipped = ([], 0)
    for frame in frames:
        for i, depth in enumerate(frame['depths']):
            owner_index = frame['root_index'] + depth
            for slot in (0, 1):
                target_index = owner_index + slot + 1
                if target_index >= len(generated):
                    skipped += 1
                    continue
                target = embedding_table[int(generated[target_index])].detach().cpu().float()
                mask, raw = (frame['masks'][i, slot].float(), frame['raw_predictions'][i, slot].float())
                h = frame['real_hidden'][i].float()
                row = {'case': case, 'question_id': qid, 'root_index': frame['root_index'], 'depth': depth, 'slot': slot, 'target_index': target_index, 'shared_prefix': target_index < common_prefix, 'mask_norm': float(mask.norm()), 'raw_norm': float(raw.norm()), 'target_norm': float(target.norm()), 'mask_target_cos': cosine(mask, target), 'raw_target_cos': cosine(raw, target), 'mask_target_norm_ratio': float(mask.norm() / target.norm()), 'slot_pair_cos': cosine(frame['masks'][i, 0], frame['masks'][i, 1]), 'velocity_anchor_ratio': frame['velocity_ratios'][i]}
                if slot == 0:
                    row.update(real_hidden_norm=float(h.norm()), real_hidden_target_cos=cosine(h, target), predicted_hidden_real_cos=cosine(raw, h))
                if 'target_scale' in frame:
                    row['mask_scale_relative_error'] = abs(float(mask.norm()) / frame['target_scale'] - 1)
                    row['reconstruction_direction_cos'] = cosine(mask, raw)
                    assert row['mask_scale_relative_error'] < 0.005 and row['reconstruction_direction_cos'] > 0.99999
                assert all((torch.isfinite(v).all() for v in (target, mask, raw, h)))
                points.append(row)
    return (points, skipped)
from copy import deepcopy
VARIANTS = ('zero', 'input_first', 'first', 'shifted', 'radial', 'both')
TEMPLATE_VARIANTS = ('zero', 'input_first', 'input_template', 'input_mean')
RECIPROCAL_VARIANTS = ('zero', 'input_first', 'first')
DIRECTION_VARIANTS = ('zero', 'input_first', 'input_template', 'input_mean', 'input_random_0', 'input_random_1', 'input_random_2', 'input_content')
EARLY_ROOTS = {121: 1, 401: 0, 241: 1, 161: 2}

def admission__diagnose_4b_mask_windows(initial=True):
    import csv
    gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    raw = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True)
    foreign = []
    for row in csv.reader(raw.splitlines()):
        if not row:
            continue
        pid, name = row
        if int(pid) != os.getpid() and Path(name.strip().replace('\\', '/')).name.lower().startswith('python'):
            foreign.append(int(pid))
    required = 14 * 1024
    return {'passed': not foreign and (not initial or int(gpu[0]) >= required), 'free_mib': int(gpu[0]), 'used_mib': int(gpu[1]), 'required_free_mib': required, 'foreign_gpu_python_pids': foreign, 'gpu_index': 0, 'initial': initial}

def intervention_masks(shadow, proxies, zero):
    """Native state simulation; only the slot offset and radial mapping vary."""
    a, v = (shadow.anchor.clone(), shadow.velocity.clone())
    anchors, velocities = ([], [])
    for depth in range(3):
        if depth < 2:
            y = proxies[depth].detach().float()
            v = shadow.beta_prime * v + (1 - shadow.beta_prime) * (y - a)
            a = y
        else:
            a = a + v
        anchors.append(a)
        velocities.append(v)
    a, v = (torch.stack(anchors)[:, None], torch.stack(velocities)[:, None])
    masks = {'zero': zero}
    for name in ('first', 'shifted', 'radial', 'both'):
        offset = int(name in ('shifted', 'both'))
        slots = a + (torch.arange(2, device=a.device) + offset)[None, :, None] * v
        norms = a.norm(dim=-1, keepdim=True) if name in ('radial', 'both') else slots.norm(dim=-1, keepdim=True)
        assert torch.isfinite(slots).all() and (norms > 0).all()
        masks[name] = (shadow.scale * slots / norms).to(zero.dtype)
    masks['first'] = shadow.masks_per_owner(proxies).to(zero.dtype)
    return masks

def walk(tree, logits):
    from spec_mtp.structure.algorithm.decode.loop import select_token
    owner, accepted, chosen = (0, [], [])
    while True:
        token = select_token(logits[owner], 0.0, None)
        assert token == int(logits[owner].argmax())
        chosen.append(token)
        child = next((c for c in tree.children_of(owner) if tree.tokens[c - 1] == token), None)
        if child is None:
            return (accepted, chosen)
        accepted.append(child)
        owner = child

def active_predecessor(trace, prompt_length):
    for previous, current in zip(trace, trace[1:]):
        if current['accepted_nodes'] == [1]:
            return previous['root_position'] - prompt_length
    raise ValueError('No predefined top1-only control opportunity; do not replace the prompt')

@torch.no_grad()
def input_control(model, prompt, generated_prefix, spec, velocity_mode='native', random_seed=None, content_start=None, state_report=None):
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    from studies._shared.structure.config import _convert
    provider = build_mask_provider(_convert(spec), model.embedding_table)
    embeds = model.embed(prompt)[0]
    provider.init_from_prompt(embeds)
    if velocity_mode == 'content':
        assert content_start is not None and 0 <= content_start < len(embeds) - 1
        level = provider._level
        provider.init_from_prompt(embeds[content_start:])
        provider._level = level
    previous = embeds[-1]
    for token in generated_prefix:
        current = model.embedding_table[token]
        provider.on_token_committed(current, previous)
        previous = current
    if velocity_mode == 'template':
        provider._velocity = (embeds[1] - embeds[0]).clone()
    elif velocity_mode == 'zero':
        provider._velocity = torch.zeros_like(provider._velocity)
    elif velocity_mode == 'random':
        assert random_seed is not None
        direction = torch.randn(embeds.shape[-1], generator=torch.Generator().manual_seed(random_seed), dtype=torch.float32)
        direction *= (embeds[1] - embeds[0]).float().norm().cpu() / direction.norm()
        provider._velocity = direction.to(device=embeds.device, dtype=embeds.dtype)
        assert abs(float(provider._velocity.float().norm() / (embeds[1] - embeds[0]).float().norm()) - 1) < 0.01
    else:
        assert velocity_mode in ('native', 'content')
    if state_report is not None:
        v, fixed = (provider._velocity.float(), (embeds[1] - embeds[0]).float())
        state_report.update(random_seed=random_seed, content_start=content_start, velocity_norm=float(v.norm()), template_norm=float(fixed.norm()), template_norm_ratio=float(v.norm() / fixed.norm()), level_norm=float(provider._level.float().norm()), template_cosine=float(torch.nn.functional.cosine_similarity(v, fixed, dim=0)))
    return provider.masks().to(model.dtype)[None].expand(3, -1, -1).clone()

def content_start_of(tokenizer, body, expected_ids):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == expected_ids and rendered.count(body) == 1
    start, end = (rendered.index(body), rendered.index(body) + len(body))
    indices = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if start <= a < b <= end]
    assert len(indices) >= 2 and indices[1] == indices[0] + 1
    return indices[0]

class WindowCaptured(Exception):
    pass

@torch.no_grad()
def capture_window(model, prompt, spec, eval_cfg, eos, seed, minimum=8):
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    from studies._shared.structure.config import _convert
    original_build, original_tree = (build_mask_provider, decoding.build_tree)
    original_observe, original_masks = (OutputHiddenMaskProvider.observe_committed, OutputHiddenMaskProvider.masks_per_owner)
    active, packet, generated = ({}, {}, [])

    def build(*args, **kwargs):
        provider = original_build(*args, **kwargs)
        active.update(provider=provider, shadow=OutputHiddenMaskProvider(2, model.embedding_table, order=1, beta=0.9, beta_prime=0.99))
        return provider

    def observe(provider, hidden):
        original_observe(provider, hidden)
        if provider is active['provider']:
            original_observe(active['shadow'], hidden)

    def masks(provider, proxies):
        zero = original_masks(provider, proxies)
        if provider is active['provider']:
            active.update(proxies=proxies, zero=zero)
        return zero

    def tree_builder(*args, **kwargs):
        tree = original_tree(*args, **kwargs)
        active['tree'] = tree
        return tree

    def before(_module, args, kwargs):
        active['prefill'] = kwargs.get('attention_mask_4d') is None
        if not active['prefill']:
            root = int(args[1][0, 0]) - prompt.shape[1]
            if root >= minimum and (not packet):
                packet.update(root_index=root, inputs=args[0].clone(), positions=args[1].clone(), kwargs={**kwargs, 'past_key_values': deepcopy(kwargs['past_key_values'])}, tree=deepcopy(active['tree']), masks=intervention_masks(active['shadow'], active['proxies'], active['zero'].to(model.dtype)))

    def after(_module, _args, _kwargs, output):
        if active['prefill']:
            generated.append(int(output.logits[0, prompt.shape[1] - 1].argmax()))
            return
        accepted, chosen = walk(active['tree'], output.logits[0])
        generated.extend(chosen)
        if packet:
            packet.update(baseline=output, prefix_generated=list(generated), accepted=accepted, chosen=chosen, zero_provider=active['provider'])
            raise WindowCaptured()
    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch.object(decoding, 'build_tree', tree_builder), patch.object(OutputHiddenMaskProvider, 'observe_committed', observe), patch.object(OutputHiddenMaskProvider, 'masks_per_owner', masks), patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.0)):
            try:
                decoding.SpecDecoder(model, _convert(spec), _convert(dict(eval_cfg, max_new_tokens=32)), eos, torch.Generator(device=model.device).manual_seed(seed), mask_policy=build(_convert(spec), model.embedding_table)).generate(prompt)
            except WindowCaptured:
                pass
    finally:
        for handle in handles:
            handle.remove()
    assert packet and packet['root_index'] < 32, 'No admissible fixed window; do not choose a result-dependent replacement'
    return packet

@torch.no_grad()
def capture_input_window(model, prompt, spec, eval_cfg, eos, seed, minimum):
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    from spec_mtp.structure.algorithm.decode.tree_attention import mask_indices_of
    from studies._shared.structure.config import _convert
    original_tree = decoding.build_tree
    shadows = [OutputHiddenMaskProvider(2, model.embedding_table, order=i, beta=0.9, beta_prime=0.99) for i in (0, 1)]
    active, packet, generated = ({}, {}, [])

    def tree_builder(*args, **kwargs):
        tree = original_tree(*args, **kwargs)
        active['tree'] = tree
        return tree

    def before(_module, args, kwargs):
        kwargs['last_hidden_only'] = True
        active['prefill'] = kwargs.get('attention_mask_4d') is None
        if not active['prefill'] and int(args[1][0, 0]) - prompt.shape[1] >= minimum:
            zero = shadows[0].masks_per_owner(active['proxies']).to(model.dtype)
            packet.update(root_index=int(args[1][0, 0]) - prompt.shape[1], inputs=args[0].clone(), positions=args[1].clone(), tree=deepcopy(active['tree']), zero_provider=shadows[0], kwargs={**kwargs, 'past_key_values': deepcopy(kwargs['past_key_values'])}, masks=intervention_masks(shadows[1], active['proxies'], zero))
        return (args, kwargs)

    def after(_module, _args, _kwargs, output):
        if active['prefill']:
            for shadow in shadows:
                shadow.init_hidden(output.last_hidden_state[0, :prompt.shape[1]])
            active['proxies'] = output.last_hidden_state[0, prompt.shape[1]:].clone()
            generated.append(int(output.logits[0, prompt.shape[1] - 1].argmax()))
            return
        accepted, chosen = walk(active['tree'], output.logits[0])
        generated.extend(chosen)
        if packet:
            packet.update(baseline=output, prefix_generated=list(generated), accepted=accepted, chosen=chosen)
            raise WindowCaptured()
        for row in [0] + accepted:
            for shadow in shadows:
                shadow.observe_committed(output.last_hidden_state[0, row])
        owner = accepted[-1] if accepted else 0
        active['proxies'] = output.last_hidden_state[0, mask_indices_of(active['tree'], 2, owner)].clone()
    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch.object(decoding, 'build_tree', tree_builder), patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.0)):
            try:
                decoding.SpecDecoder(model, _convert(spec), _convert(dict(eval_cfg, max_new_tokens=32)), eos, torch.Generator(device=model.device).manual_seed(seed), mask_policy=build_mask_provider(_convert(spec), model.embedding_table)).generate(prompt)
            except WindowCaptured:
                pass
    finally:
        for handle in handles:
            handle.remove()
    assert packet and packet['root_index'] < 32
    return packet

@torch.no_grad()
def probe_window(model, packet, spec, reference, variants=VARIANTS, control='zero'):
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from spec_mtp.structure.model import hf_causal as kv_cache
    from spec_mtp.structure.algorithm.decode.tree_attention import DynamicTreeAttentionCache, mask_indices_of
    from studies._shared.structure.config import _convert
    tree, baseline = (packet['tree'], packet['baseline'])
    accepted, chosen = (packet['accepted'], packet['chosen'])
    assert packet['prefix_generated'] == reference[:len(packet['prefix_generated'])]
    owner = accepted[-1] if accepted else 0
    seed_rows = mask_indices_of(tree, 2, owner)
    next_zero = packet.get('next_control_masks')
    if next_zero is None:
        next_zero = packet['zero_provider'].masks_per_owner(baseline.last_hidden_state[0, seed_rows]).to(model.dtype)
    next_index = packet['root_index'] + len(chosen)
    assert next_index + 2 < len(reference) and chosen[-1] == reference[next_index]
    truth = reference[next_index + 1:next_index + 3]
    rows, local = ([], {})
    expected_masks = packet['masks'][control][[0] + tree.depths].reshape(-1, packet['inputs'].shape[-1])
    assert torch.equal(packet['inputs'][0, 1 + tree.num_nodes:], expected_masks), 'Native control mask mismatch'
    for name in variants:
        inputs = packet['inputs'].clone()
        inputs[0, 1 + tree.num_nodes:] = packet['masks'][name][[0] + tree.depths].reshape(-1, inputs.shape[-1])
        kwargs = {**packet['kwargs'], 'past_key_values': deepcopy(packet['kwargs']['past_key_values'])}
        output = model(inputs, packet['positions'], **kwargs)
        assert torch.equal(output.logits[0, :1 + tree.num_nodes].argmax(-1), baseline.logits[0, :1 + tree.num_nodes].argmax(-1))
        current_accepted, current_chosen = walk(tree, output.logits[0])
        assert current_accepted == accepted and current_chosen == chosen
        logits = output.logits[0, seed_rows]
        probs = decoding.tree_probs(logits, 0.0)
        next_tree = decoding.build_tree(_convert(spec), probs, chosen[-1])
        cache = output.past_key_values
        length = int(packet['positions'][0, 0])
        kv_cache.keep_block_positions(cache, length, [0] + accepted)
        next_position = length + len(chosen)
        assert kv_cache.cache_len(cache) == next_position
        ids = torch.tensor([[chosen[-1]] + next_tree.tokens], device=model.device)
        embeds = model.embed(ids)[0]
        owner_masks = next_zero[[0] + next_tree.depths].reshape(-1, embeds.shape[-1])
        block = torch.cat([embeds, owner_masks])[None]
        attn, positions = DynamicTreeAttentionCache().get(next_tree, 2, next_position, next_position, model.dtype, model.device, model.min_mask_value())
        verified = model(block, positions, attention_mask_4d=attn, past_key_values=cache)
        next_accepted, next_chosen = walk(next_tree, verified.logits[0])
        matching = next((c for c in next_tree.children_of(0) if next_tree.tokens[c - 1] == truth[0]), None)
        ranks = [1 + int((logits[j] > logits[j, truth[j]]).sum()) for j in (0, 1)]
        row = {'variant': name, 'root_index': packet['root_index'], 'source_owner_depth': len(accepted), 'next_root_index': next_index, 'mask_logit_target_ranks': ranks, 'mask_logit_target_probs': [float(probs[j, truth[j]]) for j in (0, 1)], 'next_tree_widths': next_tree.branch_widths(), 'reference_d1_in_tree': matching is not None, 'reference_d2_under_d1': matching is not None and any((next_tree.tokens[c - 1] == truth[1] for c in next_tree.children_of(matching))), 'current_owner_argmax_unchanged': True, 'next_accept_depth': len(next_accepted), 'next_root_argmax_matches_reference': next_chosen[0] == truth[0], 'next_verified_chain_equals_reference': next_chosen == reference[next_index + 1:next_index + 1 + len(next_chosen)], 'argmax_passed': True, 'argmax_checked_tokens': len(chosen) + len(next_chosen)}
        rows.append(row)
        if name in packet.get('direction_states', {}):
            row['direction_state'] = packet['direction_states'][name]
        local[name] = {'seed_logits': logits.detach().cpu(), 'next_tree': vars(next_tree), 'next_chosen': next_chosen}
    for row in rows:
        row['d1_seed_logits_equal_zero'] = torch.equal(local[row['variant']]['seed_logits'][0], local['zero']['seed_logits'][0])
        if 'input_template' in variants:
            row['seed_logits_equal_input_first'] = torch.equal(local[row['variant']]['seed_logits'], local['input_first']['seed_logits'])
    assert next((r for r in rows if r['variant'] == control))['next_verified_chain_equals_reference'], 'Native control differs from old Run; stop, no numerical study'
    return (rows, local)
import sys
STUDY__audit_4b_norm_queue = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan'
READER = STUDY__audit_4b_norm_queue / 'analyze_results.py'

def sha__audit_4b_norm_queue(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit_once(data_root, output_dir, continuation_study=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = read(STUDY__audit_4b_norm_queue / 'index.json')['experiments']
    assert len(entries) == 8 and all((len(e['runs']) == 1 for e in entries))
    continuation = Path(continuation_study).resolve() if continuation_study else None
    selected = [(STUDY__audit_4b_norm_queue, e) for e in entries]
    if continuation:
        newer = read(continuation / 'index.json')['experiments']
        assert len(newer) == 7
        selected = [(STUDY__audit_4b_norm_queue, entries[0]), *[(continuation, e) for e in newer]]
    done, failed = ([], [])
    for cell_study, entry in selected:
        run = entry['runs'][0]['id']
        cfg = entry['factors']['algorithm.config']
        path = cell_study / 'runs' / run / 'result.json'
        result = read(path) if path.exists() else {}
        item = dict(run_id=run, case=f"{cfg['direction']}/{cfg['spec']['ema']['step_scale']:g}")
        if result.get('status') == 'succeeded':
            done.append(dict(item, result_sha256=sha__audit_4b_norm_queue(path)))
        elif result.get('status') == 'failed':
            failed.append(item)
    state = dict(complete=False, native_completed=len(done), total=8, completed=done, failed=failed, no_model_forward=True, no_timing=True, audited_cells=[])
    signature = hashlib.sha256(json.dumps([done, sha__audit_4b_norm_queue(READER), sha__audit_4b_norm_queue(STUDY__audit_4b_norm_queue / 'docs/source_refs.json'), sha__audit_4b_norm_queue(continuation / 'docs/source_refs.json') if continuation else None], sort_keys=True).encode()).hexdigest()
    latest = output / 'ledger.json'
    previous = read(latest) if latest.exists() else {}
    if previous.get('signature') == signature and previous.get('failed') == failed:
        return previous
    if done and (not failed):
        process = (continuation or STUDY__audit_4b_norm_queue) / 'process.json'
        if len(done) == 8 and (not process.exists() or not read(process).get('complete')):
            state['phase'] = 'waiting_for_native_study_process'
        else:
            target = output / f"{len(done):02d}-{done[-1]['run_id']}-{signature[:8]}"
            target.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
            with (target / 'audit.stdout.log').open('w') as stdout, (target / 'audit.stderr.log').open('w') as stderr:
                subprocess.run([sys.executable, str(READER), '--data-root', str(Path(data_root).resolve()), '--study-dir', str(STUDY__audit_4b_norm_queue), '--output-dir', str(target), *(['--continuation-study', str(continuation)] if continuation else [])], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
            body = read(target / ('norm_evidence_v1.json' if len(done) == 8 else 'norm_pending.json'))
            assert body['complete'] == (len(done) == 8)
            for item in done:
                cell = body['cells'][item['case']]
                assert cell['native_status'] == 'succeeded' and cell['run_id'] == item['run_id']
                assert len(cell['rows']) == 480
                totals = cell['totals']
                assert totals['argmax_checked_tokens'] == totals['new_tokens']
                state['audited_cells'].append(dict(item, totals=totals, files_sha256=cell['files_sha256']))
            state.update(complete=body['complete'], phase='accepted_completed_cells', signature=signature, readout_dir=str(target), readout_sha256=sha__audit_4b_norm_queue(target / ('norm_evidence_v1.json' if body['complete'] else 'norm_pending.json')))
    if not done or failed:
        state.update(phase='native_failure' if failed else 'waiting_for_first_complete_cell', signature=signature)
    latest.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return state
