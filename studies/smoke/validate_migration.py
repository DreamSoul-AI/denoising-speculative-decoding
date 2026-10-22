"""Fixed tiny decode probe for an archived or current source tree.

Run from outside the checkout with PYTHONPATH=<source>/src:
python <current>/studies/smoke/validate_migration.py --out <new-output.json>
Compare old/current JSON cells (not environment metadata). No weights downloaded.
"""

from __future__ import annotations

from spec_mtp.masks import build_mask_provider

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def probe():
    import torch

    try:
        decode = importlib.import_module('spec_mtp.decode')
        variants = importlib.import_module('studies._shared.structure.smoke_variants')
        named = importlib.import_module('spec_mtp.contracts')
        models = importlib.import_module('spec_mtp.model')
    except ModuleNotFoundError:
        decode = importlib.import_module('studies._shared.structure.algorithm.spec_mtp.decoding')
        variants = importlib.import_module('studies._shared.structure.algorithm.spec_mtp.smoke_variants')
        named = importlib.import_module('spec_mtp.structure.control.spec_mtp.named')
        models = importlib.import_module('spec_mtp.structure.model.spec_mtp.hf_causal')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = models.HFCausalLM(tiny_config={
        'vocab_size': 256, 'hidden_size': 64, 'intermediate_size': 128, 'num_layers': 2,
        'num_heads': 4, 'num_kv_heads': 2, 'max_position_embeddings': 1024})
    generator = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, 256, (1, 32), generator=generator) for _ in range(4)]
    lines = Path(decode.__file__).read_text(encoding='utf-8').splitlines()
    stop_line = next(i + 1 for i, line in enumerate(lines) if line == '            if stop:')
    cache_line = next(i + 1 for i, line in enumerate(lines) if 'n_star = accepted_nodes[-1]' in line)
    cells = {}
    for variant, spec in variants.VARIANTS.items():
        for temperature in (0.0, 0.7, 1.0):
            cfg = named._convert({'max_new_tokens': 32, 'temperature': temperature, 'stop_on_eos': False})
            for i, prompt in enumerate(prompts):
                trace = []

                def observe(frame, event, arg):
                    if frame.f_code.co_filename != decode.__file__ or frame.f_code.co_name != 'generate':
                        return None
                    if event == 'line':
                        local = frame.f_locals
                        if frame.f_lineno == stop_line:
                            trace.append({'root_position': local['root_position'], 'cache_before': local['cache_len'],
                                          'accepted_nodes': list(local['accepted_nodes']),
                                          'committed_tokens': list(local['committed_now']),
                                          'stopped': local['stop'], 'cache_after': None})
                        elif frame.f_lineno == cache_line:
                            trace[-1]['cache_after'] = decode.kv_cache.cache_len(local['cache'])
                    return observe

                decoder = decode.SpecDecoder(model, named._convert(spec), cfg,
                                             generator=torch.Generator().manual_seed(123 + i), mask_policy=build_mask_provider(named._convert(spec), model.embedding_table))
                previous = sys.gettrace()
                try:
                    sys.settrace(observe)
                    output, stats = decoder.generate(prompt)
                finally:
                    sys.settrace(previous)
                ar, _ = decode.ar_generate(model, prompt, cfg, generator=torch.Generator().manual_seed(123 + i))
                if 'trace' in stats:
                    assert stats['trace'] == trace
                cells[f'{variant}|T={temperature}|p={i}'] = {
                    'input_ids': prompt[0].tolist(), 'tokens': output[0].tolist(), 'ar_tokens': ar[0].tolist(),
                    'exact_match': torch.equal(output, ar), 'trace': trace,
                    **{key: stats[key] for key in ('accepted', 'new_tokens', 'committed', 'decode_calls',
                                                  'attempt_by_depth', 'accept_by_depth', 'hit_context_limit')}}
    assert len(cells) == 36 and all(row['exact_match'] for row in cells.values())
    return {'cells': cells, 'sha256': hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest(),
            'torch': torch.__version__, 'source': decode.__file__, 'seed': 123, 'device': 'cpu',
            'dtype': str(model.dtype), 'backend': model.hf.config._attn_implementation,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    report = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.compare:
        previous = json.loads(args.compare.read_text(encoding='utf-8'))
        assert previous['cells'] == report['cells'], 'decode migration changed fixed tiny observations'
    print(f"36 exact-match cells; {report['sha256']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
