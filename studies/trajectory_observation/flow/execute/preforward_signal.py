"""Run the fixed preforward signal protocol with explicitly bound local inputs."""

import argparse
import json
from pathlib import Path

from studies.trajectory_observation.flow.execute.fresh_replace import prepare_local_inputs


def run(args):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.data.prompts import load_questions
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.system.logger import Logger
    from studies.trajectory_observation.structure import preforward_signal as protocol

    output, weights, device, data, binding, verify = prepare_local_inputs(args, args.input)
    calibration_prompts = protocol.prompts_for('calibration')
    holdout_prompts = protocol.prompts_for('specbench', load_questions(data))
    verify()
    output.mkdir(parents=True, exist_ok=False)
    header = {**binding, 'protocol': 'preforward-signal-r2-v1',
        'config': {'features': protocol.FEATURES, 'primary': protocol.PRIMARY, 'ranks': protocol.RANKS,
                   'gate_r2': protocol.GATE_R2, 'gate_margin': protocol.GATE_MARGIN,
                   'folds': protocol.FOLDS, 'seed': protocol.SEED,
                   'calibration_tokens': 24, 'holdout_tokens': 64},
        'calibration_qids': [item['qid'] for item in calibration_prompts],
        'holdout_qids': [item['qid'] for item in holdout_prompts]}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, ensure_ascii=False, allow_nan=False, indent=2)
    logger = Logger(output / 'assets', managed=True)
    logger.info(f"preforward device={device} name={binding['device_name']}")
    model = HFCausalLM(str(weights), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    try:
        model.hf.to(device)
        if torch.device(model.device) != device:
            raise RuntimeError('preforward model did not move to its admitted device')
        logger.info(f'preforward model device={model.device} dtype={model.dtype}')
        original = model.forward
        def forward(*values, **options):
            options['last_hidden_only'] = True
            return original(*values, **options)
        setattr(model, 'forward', forward)
        setattr(model, '_preforward_tokenizer', AutoTokenizer.from_pretrained(str(weights), local_files_only=True))
        verify()
        calibration = protocol.run_prompts(model, calibration_prompts, 24, output / 'calibration.jsonl')
        holdout = protocol.run_prompts(model, holdout_prompts, 64, output / 'specbench.jsonl')
        if len(calibration['qid']) != 128 or len(holdout['qid']) != 128:
            raise RuntimeError('preforward collection requires both complete 128-prompt sets')
        summary = protocol.fit_and_summarize(calibration, holdout, output, device=device)
        verify()
    except Exception as exc:
        logger.exception('preforward protocol failed', exc)
        raise
    finally:
        model.close()
    with (output / 'protocol_complete.json').open('x', encoding='utf-8') as handle:
        json.dump({'protocol': header['protocol'], 'inputs_verified': True, 'device': str(device)}, handle)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('4b', '0.6b'), required=True)
    for name in ('model-root', 'model-sources', 'input', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
