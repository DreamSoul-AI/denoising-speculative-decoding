"""Explicit local-input entry for extra-branch-replace-v1; execution needs a bound plan."""

import argparse
import json
from pathlib import Path

from studies.trajectory_observation.flow.execute.fresh_replace import prepare_local_inputs


def run(args):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.system.logger import Logger
    from studies._shared.flow.write.run import publish_new_completion
    from studies.trajectory_observation.structure import fair_branch as protocol
    from studies.trajectory_observation.structure.fresh_replace import prompt_sha256

    output, weights, device, data, binding, verify_inputs = prepare_local_inputs(args, args.input)
    rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('protocol input must contain local JSONL objects')
    prompts = protocol.prompts_for(args.split, rows)
    verify_inputs()
    output.mkdir(parents=True, exist_ok=False)
    header = {**binding, 'schema': 1, 'protocol': protocol.PROTOCOL, 'split': args.split,
              'eval': protocol.EVAL, 'fit': protocol.FIT, 'tree': protocol.TREE,
              'prompt_sha256': prompt_sha256(prompts), 'qids': [item['qid'] for item in prompts]}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, ensure_ascii=False, indent=2, allow_nan=False)
    logger = Logger(output / 'assets', managed=True)
    logger.info(f"fair replacement device={device} name={binding['device_name']}")
    model_dir = output / args.model
    model_dir.mkdir()
    model = HFCausalLM(str(weights), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    primary = None
    try:
        model.hf.to(device)
        if torch.device(model.device) != device:
            raise RuntimeError(f'model did not move to the explicit device {device}')
        logger.info(f'fair replacement model device={model.device} dtype={model.dtype}')
        tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
        verify_inputs()
        prior = protocol.calibrate(model, tokenizer, model_dir)
        summary = protocol.run_split(model, tokenizer, prompts, prior, model_dir, args.split)
        if not summary.get('complete') or summary.get('argmax_all_passed') is not True:
            raise RuntimeError('fair replacement finished without both complete audited arms')
        verify_inputs()
    except BaseException as exc:
        primary = exc
        if isinstance(exc, Exception):
            logger.exception('fair replacement failed', exc)
        raise
    finally:
        try:
            model.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
    payload = json.dumps({'protocol': protocol.PROTOCOL, 'split': args.split,
                          'inputs_verified': True, 'device': str(device)}, allow_nan=False)
    verify_inputs()
    publish_new_completion(output / 'protocol_complete.json', payload)
    logger.info('fair replacement completed with verified inputs')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('4b', '0.6b', '8b'), required=True)
    parser.add_argument('--split', choices=('specbench', 'dolly'), required=True)
    for name in ('model-root', 'model-sources', 'input', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True, help='explicit CUDA device, for example cuda:0')
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
