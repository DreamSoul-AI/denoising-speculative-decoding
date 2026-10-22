"""Decode fingerprint: tokens and decode statistics for every shipped preset (no weights).

    python tools/decode_fingerprint.py .tmp/fp_before.json
    python tools/decode_fingerprint.py .tmp/fp_after.json --compare .tmp/fp_before.json

Runs ``SpecDecoder.generate`` and ``ar_generate`` on a random tiny LLaMA for
every smoke variant and every paper preset (alignment_probe excluded: no
decoder) x temperature {0, 0.7, 1} x 3 prompts, plus a context-limited model.
Records output tokens, ``new_tokens``, ``accepted``, ``committed``,
``decode_calls``, ``tau``, depth attempts / accepts, ``hit_context_limit`` and
whether spec == AR. Timing is excluded. ``--compare`` exits 1 on any
difference and prints the first few. Use it to prove a performance change is
output-identical.
"""

from __future__ import annotations

from spec_mtp.masks import build_mask_provider

import argparse
import hashlib
import json
import sys
from pathlib import Path

STAT_KEYS = ('new_tokens', 'accepted', 'committed', 'decode_calls', 'hit_context_limit')


def fingerprint() -> dict:
    import torch

    from studies._shared.structure.config import apply_variant
    from spec_mtp.decode import SpecDecoder, ar_generate
    from studies._shared.structure.paper_variants import PAPER_VARIANTS
    from studies._shared.structure.smoke_variants import VARIANTS
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM

    presets = sorted(k for k in {**VARIANTS, **PAPER_VARIANTS} if not k.endswith('/alignment_probe'))
    models = {}
    for max_pos in (1024, 64):
        torch.manual_seed(7)
        models[max_pos] = HFCausalLM(tiny_config={
            'vocab_size': 256, 'hidden_size': 64, 'intermediate_size': 128, 'num_layers': 2,
            'num_heads': 4, 'num_kv_heads': 2, 'max_position_embeddings': max_pos})
    gen = torch.Generator().manual_seed(11)
    prompts = [torch.randint(0, 256, (1, n), generator=gen) for n in (8, 21, 34)]
    cells = {}
    for preset in presets:
        spec = apply_variant({}, preset)
        for max_pos, max_new in ((1024, 32), (64, 48)):
            for temperature in (0.0, 0.7, 1.0):
                if max_pos == 64 and temperature == 0.7:
                    continue
                eval_cfg = _convert({'max_new_tokens': max_new, 'temperature': temperature, 'stop_on_eos': False})
                for i, prompt in enumerate(prompts):
                    seed = 100 + i
                    dec = SpecDecoder(models[max_pos], spec, eval_cfg, generator=torch.Generator().manual_seed(seed), mask_policy=build_mask_provider(spec, models[max_pos].embedding_table))
                    out, stats = dec.generate(prompt)
                    ar, _ = ar_generate(models[max_pos], prompt, eval_cfg, generator=torch.Generator().manual_seed(seed))
                    row = {k: int(stats[k]) for k in STAT_KEYS}
                    row['tau'] = round(stats['committed'] / max(stats['decode_calls'], 1), 12)
                    row['attempt_by_depth'] = [int(stats['attempt_by_depth'][d]) for d in (1, 2, 3)]
                    row['accept_by_depth'] = [int(stats['accept_by_depth'][d]) for d in (1, 2, 3)]
                    row['tokens'] = out[0].tolist()
                    row['equals_full_ar'] = bool(torch.equal(out, ar))
                    cells[f'{preset}|pos={max_pos}|T={temperature}|p={i}'] = row
    blob = json.dumps(cells, sort_keys=True).encode()
    return {'cells': cells, 'count': len(cells), 'sha256': hashlib.sha256(blob).hexdigest()}


def compare(old: dict, new: dict) -> list[str]:
    diffs = []
    for key in sorted(set(old['cells']) | set(new['cells'])):
        a, b = old['cells'].get(key), new['cells'].get(key)
        if a != b:
            fields = sorted(k for k in set(a or {}) | set(b or {}) if (a or {}).get(k) != (b or {}).get(k))
            diffs.append(f'{key}: {fields}')
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out')
    parser.add_argument('--compare', default=None)
    args = parser.parse_args(argv)
    report = fingerprint()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=0, sort_keys=True) + '\n', encoding='utf-8')
    print(f"{report['count']} cells, sha256 {report['sha256'][:16]}")
    if args.compare:
        diffs = compare(json.loads(Path(args.compare).read_text(encoding='utf-8')), report)
        print('identical' if not diffs else f'{len(diffs)} differences')
        for line in diffs[:10]:
            print('  ', line)
        return 1 if diffs else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
