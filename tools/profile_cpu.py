"""CPU profile of the speculative decoder on a random tiny LLaMA (no weights, no network).

    python tools/profile_cpu.py                                  # smoke variants, default size
    python tools/profile_cpu.py --variant llama3/esp_bc60 --variant qwen3/esp_bc60
    python tools/profile_cpu.py --hidden 256 --layers 4 --new-tokens 64 --prompts 4 --out .tmp/profile

For each variant it calls ``esp_mtp.evaluate`` (the Algorithm's decode loop,
unchanged) under cProfile and reports

- wall time of the speculative pass and of the AR reference, tokens/s, ``tau``,
  ``speedup`` and ``exact_match_rate`` (must be 1);
- self time grouped by component: model forward (torch modules), tree
  attention mask, tree build / pruning, KV cache, mask providers, decoder loop,
  other Python, other torch;
- the top functions by self time.

Writes ``<out>/profile.json`` and ``<out>/profile.md``. Timings depend on the
machine and thread count (``--threads``); compare variants within one run, not
across machines. A random tiny model accepts few drafts, so ``speedup`` here is
not a paper number; the point is where CPU time goes per decode call.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import os
import platform
import pstats
import sys
import time
from pathlib import Path

COMPONENTS = [
    ('tree attention', ('tree_attention.py',)),
    ('tree build / prune', ('tree.py',)),
    ('kv cache', ('kv_cache.py',)),
    ('mask providers', ('masks.py',)),
    ('decoder loop', ('decode.py',)),
    ('model forward (torch nn / transformers)', ('torch/nn/', 'transformers/')),
    ('other torch', ('torch/',)),
]


def component(filename: str, name: str = '') -> str:
    path = filename.replace('\\', '/')
    if path.startswith('~') and ('torch' in name or 'Tensor' in name):
        return 'torch kernels (C calls)'
    for name, needles in COMPONENTS:
        if any(n in path for n in needles):
            return name
    if path.startswith('~') or path.startswith('<'):
        return 'other builtins'
    return 'other Python'


def profile_variant(variant: str, args: argparse.Namespace) -> dict:
    import torch

    import studies._shared.structure.evaluate as esp_mtp
    from studies._shared.structure.config import apply_variant
    from studies._shared.structure.config import _convert
    from spec_mtp.data import read_local_prompts as load_records
    from spec_mtp.model import build_model

    torch.manual_seed(0)
    model = build_model(_convert({'model': {'type': 'tiny_llama', 'attn_implementation': 'eager', 'tiny': {
        'vocab_size': args.vocab, 'hidden_size': args.hidden, 'intermediate_size': args.hidden * 2,
        'num_layers': args.layers, 'num_heads': max(args.hidden // 16, 1), 'num_kv_heads': max(args.hidden // 32, 1),
        'max_position_embeddings': 4096}}}))
    model.eval()
    prompts = load_records({'mode': 'synthetic', 'vocab_size': args.vocab, 'prompt_len': args.prompt_len,
                            'num_prompts': args.prompts, 'seed': 0})
    spec = apply_variant({}, variant)
    eval_cfg = _convert({'max_new_tokens': args.new_tokens, 'temperature': 0.0, 'stop_on_eos': False,
                         'run_ar_baseline': True, 'log_per_prompt': False, 'warmup': True})
    profiler = cProfile.Profile()
    with contextlib.redirect_stdout(io.StringIO()):
        if args.timing_repeats:
            # Clean timing first (no profiler), best of N: spec / AR tok/s without cProfile overhead.
            best = None
            for _ in range(args.timing_repeats):
                timed = esp_mtp.evaluate(model, prompts, [], spec, eval_cfg, 123, torch.device('cpu'))
                spec_s = sum(r['spec_time'] for r in timed['rows'])
                best = spec_s if best is None else min(best, spec_s)
        start = time.perf_counter()
        profiler.enable()
        out = esp_mtp.evaluate(model, prompts, [], spec, eval_cfg, 123, torch.device('cpu'))
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
    overall = out['summaries']['overall']
    rows = out['rows']
    spec_time = sum(r['spec_time'] for r in rows)
    ar_time = sum(r['ar_time'] for r in rows)
    tokens = sum(r['new_tokens'] for r in rows)
    clean = {}
    if args.timing_repeats:
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
    lines = [f"# CPU profile ({report['created']})", '',
             f"Machine: {report['machine']}; torch {report['torch']}, threads {report['threads']}. "
             f"Tiny LLaMA {report['model']}, {report['workload']}. Random weights: tau and speedup are not paper numbers; "
             'cProfile overhead inflates Python-heavy parts.', '',
             '| **variant** | **exact_match** | **tau** | **decode calls** | **spec tok/s** | **AR tok/s** '
             '| **spec tok/s, no profiler** |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in report['results']:
        lines.append(f"| `{r['variant']}` | {r['exact_match_rate']} | {r['tau']} | {r['decode_calls']} | "
                     f"{r['spec_tok_per_s']} | {r['ar_tok_per_s']} | {r.get('spec_tok_per_s_unprofiled', '-')} |")
    names = sorted({k for r in report['results'] for k in r['self_time_share']},
                   key=lambda k: -max(r['self_time_share'].get(k, 0) for r in report['results']))
    lines += ['', 'Self-time share by component (profiled run, spec + AR + warm-up):', '',
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--variant', action='append', help='smoke variant or paper preset (repeatable)')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--vocab', type=int, default=512)
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--prompts', type=int, default=4)
    parser.add_argument('--new-tokens', type=int, default=64)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--top', type=int, default=8)
    parser.add_argument('--timing-repeats', type=int, default=0,
                        help='also time N unprofiled runs and report the best spec tok/s')
    parser.add_argument('--out', default='.tmp/profile')
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    variants = args.variant or ['esp_static', 'esp_dynamic', 'ema_velocity']
    results = [profile_variant(v, args) for v in variants]
    report = {
        'created': time.strftime('%Y-%m-%d %H:%M %Z'),
        'machine': f'{platform.processor() or platform.machine()} ({os.cpu_count()} CPUs), {platform.system()}',
        'torch': torch.__version__, 'threads': torch.get_num_threads(),
        'model': f'hidden {args.hidden}, {args.layers} layers, vocab {args.vocab}',
        'workload': f'{args.prompts} prompts x {args.prompt_len} tokens, {args.new_tokens} new tokens, temperature 0',
        'results': results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'profile.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    (out / 'profile.md').write_text(markdown(report), encoding='utf-8')
    for r in results:
        share = ', '.join(f'{k} {100 * v:.0f}%' for k, v in list(r['self_time_share'].items())[:3])
        print(f"{r['variant']:<20} exact_match={r['exact_match_rate']} tau={r['tau']} "
              f"spec={r['spec_tok_per_s']} tok/s ar={r['ar_tok_per_s']} tok/s | {share}")
    print(f'wrote {out / "profile.md"}')
    bad = [r['variant'] for r in results if r['exact_match_rate'] != 1.0]
    if bad:
        print('exact_match_rate != 1:', bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
