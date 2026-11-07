"""Host cProfile attribution and timing statistics for an explicitly supplied CUDA workload."""

from __future__ import annotations

import contextlib
import cProfile
import io
import pstats
import time
from pathlib import Path
from typing import Any

COMPONENTS = [
    ('tree attention', ('tree_attention.py',)),
    ('tree build / prune', ('tree.py',)),
    ('kv cache', ('kv_cache.py',)),
    ('mask providers', ('algorithm/mask/',)),
    ('decoder loop', ('algorithm/decode/loop.py',)),
    ('model forward (torch nn / transformers)', ('torch/nn/', 'transformers/')),
    ('other torch', ('torch/',)),
]


def component(filename: str, name: str = '') -> str:
    path = filename.replace('\\', '/')
    if path.startswith('~') and ('torch' in name or 'Tensor' in name):
        return 'torch C calls (host self time)'
    for name, needles in COMPONENTS:
        if any(n in path for n in needles):
            return name
    if path.startswith('~') or path.startswith('<'):
        return 'other builtins'
    return 'other Python'


def profile_variant(variant: str, args: Any, model, prompts, device) -> dict:
    import torch

    import studies._shared.structure.evaluate as esp_mtp
    from studies._shared.structure.config import apply_variant
    from studies._shared.structure.config import _convert
    if torch.device(device).type != 'cuda':
        raise ValueError('profiling execution requires CUDA')
    spec = apply_variant({}, variant)
    eval_cfg = _convert({'max_new_tokens': args.new_tokens, 'temperature': 0.0, 'stop_on_eos': False,
                         'run_ar_baseline': True, 'log_per_prompt': False, 'warmup': True})
    profiler = cProfile.Profile()
    with contextlib.redirect_stdout(io.StringIO()):
        if args.timing_repeats:
            # Clean timing first (no profiler), best of N: spec / AR tok/s without cProfile overhead.
            best = None
            for _ in range(args.timing_repeats):
                timed = esp_mtp.evaluate(model, prompts, [], spec, eval_cfg, 123, device)
                spec_s = sum(r['spec_time'] for r in timed['rows'])
                best = spec_s if best is None else min(best, spec_s)
        start = time.perf_counter()
        profiler.enable()
        try:
            out = esp_mtp.evaluate(model, prompts, [], spec, eval_cfg, 123, device)
        finally:
            profiler.disable()
    wall = time.perf_counter() - start
    stats = pstats.Stats(profiler)
    total = sum(v[2] for v in stats.stats.values())  # type: ignore[attr-defined]
    groups: dict[str, float] = {}
    funcs = []
    for (filename, line, name), (_cc, ncalls, tottime, cumtime, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        key = component(filename, name)
        groups[key] = groups.get(key, 0.0) + tottime
        funcs.append({'function': f'{Path(filename).name}:{line}({name})', 'calls': ncalls,
                      'self_s': round(tottime, 4), 'cum_s': round(cumtime, 4)})
    funcs.sort(key=lambda f: -f['self_s'])
    if not out['evidence'] or any(item['stats']['verification_argmax_audit']['passed'] is not True for item in out['evidence']):
        raise RuntimeError('profile verification argmax audit failed')
    overall = out['summaries']['overall']
    rows = out['rows']
    spec_time = sum(r['spec_time'] for r in rows)
    ar_time = sum(r['ar_time'] for r in rows)
    tokens = sum(r['new_tokens'] for r in rows)
    clean = {}
    if args.timing_repeats:
        if best is None or best <= 0:
            raise RuntimeError('unprofiled timing did not produce a positive elapsed time')
        clean = {'spec_s_unprofiled_best': round(best, 4),
                 'spec_tok_per_s_unprofiled': round(sum(r['new_tokens'] for r in rows) / best, 1)}
    return {
        'variant': variant, 'wall_s_profiled': round(wall, 3), **clean,
        'spec_s': round(spec_time, 4), 'ar_s': round(ar_time, 4),
        'spec_tok_per_s': round(tokens / spec_time, 1) if spec_time else None,
        'ar_tok_per_s': round(tokens / ar_time, 1) if ar_time else None,
        'decode_calls': sum(r['decode_calls'] for r in rows), 'tau': round(float(overall['tau']), 4),
        'exact_match_rate': float(overall['exact_match_rate']),
        'self_time_share': {k: round(v / total, 4) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])},
        'top_functions': funcs[:args.top],
    }


def markdown(report: dict) -> str:
    scope = ('Host cProfile of CUDA workload' if report.get('identity', {}).get('protocol') == 'cuda-host-cprofile-v1'
             and str(report.get('device', '')).startswith('cuda:') else 'Saved cProfile (execution scope unknown)')
    lines = [f"# {scope} ({report['created']})", '',
             f"Device: {report.get('device', 'unknown')}; Machine: {report['machine']}; torch {report['torch']}, threads {report['threads']}. "
             f"Tiny LLaMA {report['model']}, {report['workload']}. Random weights: tau and speedup are not paper numbers; "
             'cProfile measures host call self time, not CUDA kernel time; profiling overhead inflates Python-heavy parts.', '',
             f"Rate scope: {report.get('rate_clock', 'unknown for this saved report')}. "
             f"Wall scope: {report.get('wall_clock', 'unknown for this saved report')}.", '',
             '| **variant** | **exact_match** | **tau** | **decode calls** | **spec tok/s, host clock** | **AR tok/s, host clock** '
             '| **spec tok/s, host clock, no profiler** |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in report['results']:
        lines.append(f"| `{r['variant']}` | {r['exact_match_rate']} | {r['tau']} | {r['decode_calls']} | "
                     f"{r['spec_tok_per_s']} | {r['ar_tok_per_s']} | {r.get('spec_tok_per_s_unprofiled', '-')} |")
    names = sorted({k for r in report['results'] for k in r['self_time_share']},
                   key=lambda k: -max(r['self_time_share'].get(k, 0) for r in report['results']))
    lines += ['', 'Host self-time share by component (profiled run, spec + AR + warm-up):', '',
              '| **component** | ' + ' | '.join(f"**{r['variant']}**" for r in report['results']) + ' |',
              '| --- |' + ' --- |' * len(report['results'])]
    for name in names:
        lines.append(f'| {name} | ' + ' | '.join(f"{100 * r['self_time_share'].get(name, 0):.1f}%"
                                                 for r in report['results']) + ' |')
    for r in report['results']:
        lines += ['', f"Top self time, `{r['variant']}`:", '', '| **function** | **calls** | **self s** | **cum s** |',
                  '| --- | --- | --- | --- |']
        lines += [f"| `{f['function']}` | {f['calls']} | {f['self_s']} | {f['cum_s']} |" for f in r['top_functions']]
    return '\n'.join(lines) + '\n'
