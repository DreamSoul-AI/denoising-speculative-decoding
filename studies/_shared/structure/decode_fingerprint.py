"""Deterministic decode cells and field comparison; execution resources belong to flow."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from spec_mtp.structure.algorithm.mask import build_mask_provider

STAT_KEYS = ('new_tokens', 'accepted', 'committed', 'decode_calls', 'hit_context_limit')
MODEL_CONFIGS = {max_pos: {'vocab_size': 256, 'hidden_size': 64, 'intermediate_size': 128, 'num_layers': 2,
                         'num_heads': 4, 'num_kv_heads': 2, 'max_position_embeddings': max_pos}
                 for max_pos in (1024, 64)}


def default_presets():
    from studies._shared.structure.paper_variants import PAPER_VARIANTS
    from studies._shared.structure.smoke_variants import VARIANTS

    return sorted(k for k in {**VARIANTS, **PAPER_VARIANTS} if not k.endswith('/alignment_probe'))


def fingerprint_cell(model, prompt, spec, eval_cfg, seed):
    import torch
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, ar_generate
    from studies._shared.structure.inputs import make_generator

    if torch.device(model.device).type != 'cuda':
        raise ValueError('fingerprint execution requires CUDA')
    prompt = prompt.to(model.device)
    decoder = SpecDecoder(model, spec, eval_cfg, generator=make_generator(model.device, seed),
                          mask_policy=build_mask_provider(spec, model.embedding_table))
    out, stats = decoder.generate(prompt)
    if float(eval_cfg.get('temperature', 0.0)) <= 0 and stats.get('verification_argmax_audit', {}).get('passed') is not True:
        raise RuntimeError('fingerprint verification argmax audit failed')
    ar, _ = ar_generate(model, prompt, eval_cfg, generator=make_generator(model.device, seed))
    row: dict[str, Any] = {key: int(stats[key]) for key in STAT_KEYS}
    row['tau'] = round(stats['committed'] / max(stats['decode_calls'], 1), 12)
    row['attempt_by_depth'] = [int(stats['attempt_by_depth'][depth]) for depth in (1, 2, 3)]
    row['accept_by_depth'] = [int(stats['accept_by_depth'][depth]) for depth in (1, 2, 3)]
    row['tokens'] = out[0].tolist()
    row['equals_full_ar'] = bool(torch.equal(out, ar))
    return row


def fingerprint(models, prompts, device, *, presets=None) -> dict:
    import torch
    from studies._shared.structure.config import apply_variant, _convert

    if torch.device(device).type != 'cuda':
        raise ValueError('fingerprint execution requires CUDA')
    cells = {}
    for preset in default_presets() if presets is None else presets:
        spec = apply_variant({}, preset)
        for max_pos, max_new in ((1024, 32), (64, 48)):
            for temperature in (0.0, 0.7, 1.0):
                if max_pos == 64 and temperature == 0.7:
                    continue
                eval_cfg = _convert({'max_new_tokens': max_new, 'temperature': temperature, 'stop_on_eos': False})
                for index, prompt in enumerate(prompts):
                    row = fingerprint_cell(models[max_pos], prompt, spec, eval_cfg, 100 + index)
                    cells[f'{preset}|pos={max_pos}|T={temperature}|p={index}'] = row
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
