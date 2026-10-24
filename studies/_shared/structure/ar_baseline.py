"""Plain autoregressive decoding baseline (timing / sanity reference)."""

from studies._shared.structure import csv_logger
from studies._shared.structure.inputs import legacy_inputs, make_generator
from spec_mtp.decode import ar_generate
from spec_mtp.model import sync_device


def evaluate(model, prompts, eos_ids, eval_cfg, seed, device, save_dir=None, *, aggregate=True):
    seed = int(seed)
    logger = csv_logger.Logger(save_dir, filename='spec_metrics.csv')
    total_tokens, total_time = 0, 0.0
    if bool(eval_cfg.get('warmup', True)) and prompts:
        # Untimed first pass: CUDA kernel selection / allocator growth
        # would otherwise inflate the measurement
        warm_cfg = dict(eval_cfg)
        warm_cfg['max_new_tokens'] = int(eval_cfg.get('warmup_tokens', 8))
        ar_generate(model, prompts[0]['input_ids'], warm_cfg, eos_ids=eos_ids)
        sync_device(device)
    rows, evidence = [], []
    for idx, item in enumerate(prompts):
        generator = make_generator(device, seed + idx)
        sync_device(device)
        output, stats = ar_generate(model, item['input_ids'], eval_cfg,
                               eos_ids=eos_ids, generator=generator)
        sync_device(device)
        row = {'prompt': idx, 'category': item['category'], **stats}
        logger.log(row)
        rows.append(row)
        evidence.append({'prompt': idx, 'input_ids': item['input_ids'][0].tolist(),
                         'output_ids': output[0].tolist(), 'stats': stats})
        total_tokens += stats['ar_new_tokens']
        total_time += stats['ar_time']

    summary = {
        'prompts': len(prompts),
        'tokens_per_s': total_tokens / total_time if total_time > 0 else 0.0,
    }
    if aggregate:
        csv_logger.Logger(save_dir, filename='summary.csv').log(summary)
    return {'rows': rows, 'summary': summary if aggregate else {}, 'evidence': evidence}


def test(cfg, model=None):
    model, prompts, eos_ids, device = legacy_inputs(cfg, model)
    return evaluate(model, prompts, eos_ids, cfg.eval, cfg.run.seed, device,
                    cfg.run.get('save_dir', None))['summary']
