"""Execute the two fixed hybrid gain protocols with bound local model and data."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from studies.trajectory_observation.flow.execute.fresh_replace import prepare_local_inputs
from studies.trajectory_observation.structure import bonus_aligned_gain as bonus
from studies.trajectory_observation.structure import output_hidden_kalman as hidden


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def run_prompts(model, tokenizer, prompts, spec, tokens, path, *, pair_keys=()):
    """Collect a fresh protocol arm, preserving partial rows and calibration tensors."""
    import torch
    from spec_mtp.structure.data.prompts import render
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    from spec_mtp.structure.algorithm.mask import build_mask_provider

    provider = build_mask_provider(spec, model.embedding_table)
    eval_cfg = {'max_new_tokens': tokens, 'temperature': 0.0, 'stop_on_eos': False}
    stored: dict[str, list[Any]] = {'qid': [], **{key: [] for key in pair_keys}}
    with path.open('x', encoding='utf-8') as stream:
        for prompt in prompts:
            ids = render(tokenizer, prompt['text']).to(model.device)
            started = time.perf_counter()
            _out, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=provider).generate(ids)
            if not stats['verification_argmax_audit']['passed']:
                raise RuntimeError(f"argmax audit failed for {prompt['qid']}")
            row = {'qid': prompt['qid'], 'accepted': int(stats['accepted']), 'committed': int(stats['committed']),
                   'decode_calls': int(stats['decode_calls']), 'new_tokens': int(stats['new_tokens']),
                   'seconds': time.perf_counter() - started, 'argmax_passed': True}
            if hasattr(provider, 'diagnostics'):
                row['diagnostics'] = provider.diagnostics()
            if pair_keys:
                harvested = provider.calibration_rows()
                stored['qid'].append(prompt['qid'])
                for key in pair_keys:
                    stored[key].append(harvested[key])
                torch.save(stored, path.with_suffix('.pt'))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            print(f"{path.name} {prompt['qid']} accepted={row['accepted']} calls={row['decode_calls']} "
                  f"{row['seconds']:.1f}s", flush=True)
    return stored


def run(args):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.data.prompts import load_questions
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.system.logger import Logger

    if args.protocol not in ('bonus-aligned', 'output-hidden'):
        raise ValueError('unknown fixed hybrid gain protocol')
    protocol = bonus if args.protocol == 'bonus-aligned' else hidden
    protocol_id = 'hybrid-gain-bonus-aligned-v1' if protocol is bonus else 'output-hidden-hybrid-kalman-v1'
    output, weights, device, data, binding, verify = prepare_local_inputs(args, args.input)
    splits = {'specbench': protocol.prompts_for('specbench', load_questions(data))}
    if protocol is hidden:
        splits['holdout'] = hidden.prompts_for('holdout')
    calibration = protocol.prompts_for('calibration')
    verify()
    output.mkdir(parents=True, exist_ok=False)
    header = {**binding, 'protocol': protocol_id, 'arms': protocol.ARMS,
              'calibration_qids': [item['qid'] for item in calibration],
              'split_qids': {name: [item['qid'] for item in prompts] for name, prompts in splits.items()},
              'prompt_sha256': hashlib.sha256(json.dumps([calibration, splits], sort_keys=True).encode()).hexdigest(),
              'config': {'calibration_tokens': 24, 'evaluation_tokens': 64,
                         'tree': protocol.tree_spec() if protocol is bonus else hidden.tree_spec('esp')}}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, ensure_ascii=False, allow_nan=False, indent=2)
    logger = Logger(output / 'assets')
    logger.info(f"{protocol_id} device={device} name={binding['device_name']}")
    model = HFCausalLM(str(weights), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    primary: BaseException | None = None
    try:
        model.hf.to(device)
        if torch.device(model.device) != device:
            raise RuntimeError('hybrid gain model did not move to its admitted CUDA device')
        logger.info(f'{protocol_id} model device={model.device} dtype={model.dtype}')
        if protocol is hidden:
            aliased = False
            if not model.tied_embeddings:
                left, right = model.embedding_table.detach(), model.hf.get_output_embeddings().weight.detach()
                if left.shape != right.shape or not torch.equal(left, right):
                    raise RuntimeError('output and input embeddings differ; not tying them')
                model.hf.get_output_embeddings().weight = model.embedding_table
                aliased = True
            tie = {'config_tie': bool(model.config.tie_word_embeddings),
                   'storage_tie': model.tied_embeddings, 'aliased_equal_weights': aliased}
            with (output / 'tie.json').open('x', encoding='utf-8') as handle:
                json.dump(tie, handle)
        tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
        verify()
        collect_spec = bonus.tree_spec({'mode': 'collect'}) if protocol is bonus else hidden.arm_spec('collect', None)
        pair_keys = ('m', 'z', 'first', 'last') if protocol is bonus else ('h', 'z', 'y')
        stored = run_prompts(model, tokenizer, calibration, collect_spec, 24, output / 'calibration.jsonl', pair_keys=pair_keys)
        prior = protocol.calibrate(stored, device)
        prior_name = 'priors.json' if protocol is bonus else 'prior.json'
        with (output / prior_name).open('x', encoding='utf-8') as handle:
            json.dump(prior, handle, allow_nan=False)
        verify()
        def check_rows(path, expected):
            rows = read_rows(path)
            if ([row['qid'] for row in rows] != [item['qid'] for item in expected]
                    or any(row.get('argmax_passed') is not True for row in rows)):
                raise RuntimeError('hybrid gain output lacks exact prompt coverage and passing argmax audits')
        check_rows(output / 'calibration.jsonl', calibration)
        usable = ({name: {key: prior[name][key] for key in ('k', 'p', 'r', 'q')} for name in ('first', 'last')}
                  if protocol is bonus else prior)
        all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for split, prompts in splits.items():
            all_rows[split] = {}
            for arm in protocol.ARMS:
                verify()
                path = output / f'{split}-{arm}.jsonl'
                run_prompts(model, tokenizer, prompts, protocol.arm_spec(arm, usable), 64, path)
                verify()
                check_rows(path, prompts)
                all_rows[split][arm] = read_rows(path)
        summary = bonus.summarize(all_rows['specbench']) if protocol is bonus else hidden.summarize(all_rows, args.model)
        verify()
        with (output / 'summary.json').open('x', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except BaseException as exc:
        primary = exc
        try:
            logger.exception('hybrid gain protocol failed', exc)
        except Exception as cleanup:
            setattr(exc, 'cleanup_errors', (*getattr(exc, 'cleanup_errors', ()), cleanup))
        raise
    finally:
        try:
            model.close()
        except Exception as cleanup:
            if primary is None:
                raise
            setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
    verify()
    with (output / 'protocol_complete.json').open('x', encoding='utf-8') as handle:
        json.dump({'protocol': protocol_id, 'inputs_verified': True, 'device': str(device)}, handle)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', choices=('bonus-aligned', 'output-hidden'))
    parser.add_argument('--model', choices=('4b', '0.6b'), required=True)
    for name in ('model-root', 'model-sources', 'input', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
