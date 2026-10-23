"""Evaluate ESP / EMA-velocity multi-token prediction (training-free).

Reports the paper's metrics per SpecBench category and overall:
  - accepted: draft tokens accepted by verification (the headline count).
  - tau: average acceptance length = committed tokens (accepted + bonus) per
    decode model call (Section 4 "Performance Metric"); tau ~ 1/model-calls.
  - call_reduction: 1 - 1/tau (Appendix G.1, Table 6).
  - speedup: AR wall time / spec wall time on the same prompts, when the AR
    reference pass is enabled (eval.run_ar_baseline).
  - exact_match: token-for-token equality with AR decoding under a shared
    seed, the losslessness audit.
"""

from spec_mtp.masks import build_mask_provider

import torch
from spec_mtp.analysis import _aggregate, totals as totals

from studies._shared.structure import csv_logger
from studies._shared.structure.inputs import legacy_inputs, make_generator
from spec_mtp.decode import SpecDecoder, ar_generate
from spec_mtp.model import sync_device






def _warmup(decoder, model, prompt_ids, eval_cfg, eos_ids, run_ar, device, seed):
    """Short untimed generation before the measured loop.

    The first CUDA call pays kernel selection and allocator growth,
    which may inflate the measurement.
    Disable with `eval.warmup: false`.
    """
    budget = int(eval_cfg.get('warmup_tokens', 8))
    original = decoder.max_new_tokens
    decoder.max_new_tokens = budget
    try:
        decoder.generate(prompt_ids)
    finally:
        decoder.max_new_tokens = original
    if run_ar:
        warm_cfg = dict(eval_cfg)
        warm_cfg['max_new_tokens'] = budget
        ar_generate(model, prompt_ids, warm_cfg, eos_ids=eos_ids,
                    generator=make_generator(device, seed))
    sync_device(device)


def evaluate(model, prompts, eos_ids, spec_cfg, eval_cfg, seed, device, save_dir=None, *, aggregate=True):
    """Decode every prompt; write spec_metrics.csv / summary.csv under save_dir.

    Returns {'rows': per-prompt rows, 'summaries': per-category summaries}.
    """
    run_ar = bool(eval_cfg.get('run_ar_baseline', True))
    seed = int(seed)

    decoder = SpecDecoder(
        model, spec_cfg, eval_cfg, eos_ids=eos_ids,
        generator=make_generator(device, seed),
     mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
    print(f'[esp_mtp] method={spec_cfg.get("method", "esp")} '
          f'num_masks={decoder.num_masks} '
          f'block_complexity={decoder.nominal_block_complexity} '
          f'impl={decoder.impl} prompts={len(prompts)}')

    per_prompt_logger = None
    if bool(eval_cfg.get('log_per_prompt', True)):
        per_prompt_logger = csv_logger.Logger(save_dir, filename='spec_metrics.csv')

    if bool(eval_cfg.get('warmup', True)) and prompts:
        _warmup(decoder, model, prompts[0]['input_ids'], eval_cfg, eos_ids,
                run_ar, device, seed)

    rows, evidence = [], []
    for idx, item in enumerate(prompts):
        prompt_ids = item['input_ids']
        # Fresh, identically seeded generators per prompt so spec and AR see
        # the same sample stream (exact-match audit at temperature > 0).
        decoder.generator = make_generator(device, seed + idx)

        sync_device(device)
        output, stats = decoder.generate(prompt_ids)
        sync_device(device)

        row = {
            'prompt': idx,
            'category': item['category'],
            'new_tokens': stats['new_tokens'],
            'committed': stats['committed'],
            'decode_calls': stats['decode_calls'],
            'accepted': stats['accepted'],
            'tau': stats['committed'] / max(stats['decode_calls'], 1),
            'spec_time': stats['prefill_time'] + stats['decode_time'],
            'hit_context_limit': stats['hit_context_limit'],
        }
        for depth in (1, 2, 3):
            row[f'attempt_d{depth}'] = int(stats['attempt_by_depth'][depth])
            row[f'accept_d{depth}'] = int(stats['accept_by_depth'][depth])
        if run_ar:
            # Context limit: the speculative block needs room for the tree and
            # k mask slots past the root, so SpecDecoder stops a few tokens
            # before ar_generate would. The audit then compares the same
            # budget: AR stops at the token count the speculative pass reached.
            ar_cfg = eval_cfg
            if stats['hit_context_limit']:
                ar_cfg = dict(eval_cfg)
                ar_cfg['max_new_tokens'] = stats['new_tokens']
            sync_device(device)
            ar_out, ar_stats = ar_generate(
                model, prompt_ids, ar_cfg, eos_ids=eos_ids,
                generator=make_generator(device, seed + idx),
            )
            sync_device(device)
            row['ar_time'] = ar_stats['ar_time']
            row['ar_calls'] = ar_stats['ar_calls']
            row['exact_match'] = int(torch.equal(output, ar_out))
        evidence.append({
            'prompt': idx, 'input_ids': prompt_ids[0].tolist(),
            'output_ids': output[0].tolist(), 'ar_ids': ar_out[0].tolist() if run_ar else None,
            'stats': stats,
        })
        rows.append(row)
        if per_prompt_logger is not None:
            per_prompt_logger.log(row)

    summaries = _aggregate(rows) if aggregate else {}
    summary_logger = csv_logger.Logger(save_dir, filename='summary.csv')
    for category, summary in summaries.items():
        summary_logger.log({'category': category, **summary})
    return {'rows': rows, 'summaries': summaries, 'evidence': evidence}


def test(cfg, model=None):
    model, prompts, eos_ids, device = legacy_inputs(cfg, model)
    return evaluate(model, prompts, eos_ids, cfg.spec, cfg.eval, cfg.run.seed,
                    device, cfg.run.get('save_dir', None))['summaries']
