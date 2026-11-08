"""Bound decode fingerprints and host cProfile analysis of CUDA workloads.

Usage: python -m studies._shared.flow.diagnostics --help
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import math
from types import SimpleNamespace
from datetime import datetime, timezone

from spec_mtp.structure.artifact.layout import validate_study_root
from spec_mtp.structure.artifact.paths import validate_output_root
from spec_mtp.structure.artifact.provenance import file_sha256
from studies._shared.structure import decode_fingerprint as fingerprint, profile_cpu, sources

PROFILE_DEFAULTS = {'hidden': 128, 'layers': 4, 'vocab': 512, 'prompt_len': 32, 'prompts': 4,
                    'new_tokens': 64, 'threads': 1, 'top': 8, 'timing_repeats': 0, 'variants': None}


def _known_source(source):
    if (not isinstance(source, dict) or not isinstance(source.get('git'), dict)
            or not isinstance(source.get('checkout_git'), dict)):
        return False
    git = source.get('git', {})
    head = git.get('head')
    files = source.get('files')
    return (git.get('package_dirty') is False and isinstance(head, str) and len(head) == 40
            and all(char in '0123456789abcdef' for char in head) and isinstance(files, dict) and bool(files)
            and all(isinstance(path, str) and isinstance(sha, str) and len(sha) == 64
                    and all(char in '0123456789abcdef' for char in sha) for path, sha in files.items())
            and source.get('checkout_git', {}).get('head') == head
            and source.get('checkout_git', {}).get('package_dirty') is False)


def _source():
    return {**sources.manifest(), 'checkout_git': sources._git(Path(__file__).resolve().parents[3])}


def _valid_fingerprint(report, identity):
    if not isinstance(report, dict) or report.get('identity') != identity or not _known_source(report.get('source')):
        return False
    cells = report.get('cells')
    expected = {f'{variant}|pos={position}|T={temperature}|p={index}'
                for variant in identity['workload']['variants']
                for position, _tokens, temperatures in identity['workload']['matrix']
                for temperature in temperatures for index in range(3)}
    if not isinstance(cells, dict) or set(cells) != expected or report.get('count') != len(cells):
        return False
    for row in cells.values():
        if (not isinstance(row, dict) or any(type(row.get(key)) is not int or row[key] < 0 for key in fingerprint.STAT_KEYS)
                or type(row.get('equals_full_ar')) is not bool or not isinstance(row.get('tokens'), list)
                or any(type(token) is not int or not 0 <= token < 256 for token in row['tokens'])
                or type(row.get('tau')) not in (int, float) or not math.isfinite(row['tau'])
                or any(not isinstance(row.get(key), list) or len(row[key]) != 3
                       or any(type(value) is not int or value < 0 for value in row[key])
                       for key in ('attempt_by_depth', 'accept_by_depth'))):
            return False
    return report.get('sha256') == hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()


def _close_models(models, primary):
    failure = primary
    for model in reversed(models):
        try:
            model.close()
        except BaseException as cleanup:
            if failure is None:
                failure = cleanup
            else:
                setattr(failure, 'cleanup_errors', (*getattr(failure, 'cleanup_errors', ()), cleanup))
    if primary is None and failure is not None:
        raise failure


def run(args):
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.data.prompts import read_local_prompts
    from spec_mtp.structure.system.logger import Logger

    if args.command not in ('fingerprint', 'profile'):
        raise ValueError('unknown diagnostic command')
    config_path, plan = (Path(value).resolve(strict=True) for value in (args.config, args.plan))
    config_sha, plan_sha = file_sha256(config_path), file_sha256(plan)
    if plan_sha != args.plan_sha256:
        raise ValueError('research plan SHA256 differs from its explicit binding')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected = {'variants'} if args.command == 'fingerprint' else set(PROFILE_DEFAULTS)
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError('diagnostics require the complete known configuration fields')
    variants = config['variants']
    if variants is None:
        variants = fingerprint.default_presets() if args.command == 'fingerprint' else ['esp_static', 'esp_dynamic', 'ema_velocity']
    if (not isinstance(variants, list) or not variants or any(not isinstance(value, str) or value not in fingerprint.default_presets() for value in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError('diagnostic variants must be unique supported decode presets')
    if args.command == 'profile':
        for key in set(PROFILE_DEFAULTS) - {'variants'}:
            if type(config[key]) is not int or config[key] < (0 if key == 'timing_repeats' else 1):
                raise ValueError('profile budgets must be positive integers; timing_repeats may be zero')
        heads, kv = max(config['hidden'] // 16, 1), max(config['hidden'] // 32, 1)
        if config['hidden'] % heads or heads % kv or config['vocab'] < 8 or config['prompt_len'] > 4096:
            raise ValueError('invalid tiny profile model dimensions or prompt budget')
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ValueError('diagnostic execution requires an explicit available CUDA device')
    source = _source()
    if not _known_source(source):
        raise ValueError('diagnostic execution requires known clean source identity')
    baseline = Path(args.compare).resolve(strict=True) if args.compare else None
    if baseline is not None and args.command != 'fingerprint':
        raise ValueError('only decode fingerprints support comparison')
    inputs = {config_path: config_sha, plan: plan_sha}
    if baseline is not None:
        inputs[baseline] = file_sha256(baseline)
    output = validate_study_root(validate_output_root(args.output, *inputs))
    if output.exists() or output.is_symlink():
        raise FileExistsError('diagnostics require a new external output root')
    workload = ({'models': fingerprint.MODEL_CONFIGS, 'model_seed': 7, 'prompt_seed': 11,
                 'prompt_lengths': [8, 21, 34], 'variants': variants,
                 'matrix': [[1024, 32, [0.0, 0.7, 1.0]], [64, 48, [0.0, 1.0]]], 'decode_seeds': [100, 101, 102]}
                if args.command == 'fingerprint' else {**config, 'variants': variants, 'model_seed': 0, 'prompt_seed': 0, 'decode_seed': 123})
    # JSON canonicalization also normalizes integer model-map keys in persisted identity.
    workload = json.loads(json.dumps(workload, sort_keys=True, allow_nan=False))
    identity = {'protocol': 'decode-fingerprint-v1' if args.command == 'fingerprint' else 'cuda-host-cprofile-v1',
                'workload': workload, 'device': str(device), 'device_name': torch.cuda.get_device_name(device),
                'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
                'dtype': str(torch.get_default_dtype()), 'backend': 'eager'}
    old = None
    if baseline is not None:
        old = json.loads(baseline.read_text(encoding='utf-8'))
        if not _valid_fingerprint(old, identity):
            raise ValueError('fingerprint comparison requires matching workload/device identity and known source; old identity is unknown')
    def verify():
        if _source() != source or any(file_sha256(path) != sha for path, sha in inputs.items()):
            raise ValueError('diagnostic configuration, plan, comparison input or sources changed')
    verify()
    output.mkdir(parents=True, exist_ok=False)
    header = {'identity': identity, 'source': source,
              'input_sources': {str(path): sha for path, sha in inputs.items()},
              'plan': {'path': str(plan), 'sha256': plan_sha},
              'correctness_scope': 'T<=0 verification argmax audit; T>0 tokens/statistics/AR diagnostic only'}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, indent=2, allow_nan=False)
    logger = Logger(output / 'assets', managed=True)
    logger.info(f"{args.command} device={device} name={identity['device_name']} config={workload}")
    models, primary = [], None
    try:
        with torch.random.fork_rng(devices=[device.index]):
            if args.command == 'fingerprint':
                by_position = {}
                for position, model_config in fingerprint.MODEL_CONFIGS.items():
                    torch.manual_seed(7)
                    model = HFCausalLM(tiny_config=model_config)
                    models.append(model)
                    model.hf.to(device)
                    if torch.device(model.device) != device:
                        raise RuntimeError('fingerprint model did not move to its admitted CUDA device')
                    logger.info(f'fingerprint model position={position} device={model.device} dtype={model.dtype}')
                    by_position[position] = model
                generator = torch.Generator().manual_seed(11)
                prompts = [torch.randint(0, 256, (1, size), generator=generator) for size in (8, 21, 34)]
                report = {**fingerprint.fingerprint(by_position, prompts, device, presets=variants), **header}
                if not _valid_fingerprint(report, identity):
                    raise RuntimeError('fingerprint result does not cover its bound workload')
                verify()
                with (output / 'fingerprint.json').open('x', encoding='utf-8') as handle:
                    json.dump(report, handle, indent=0, sort_keys=True, allow_nan=False)
            else:
                torch.manual_seed(0)
                model = HFCausalLM(attn_implementation='eager', tiny_config={
                    'vocab_size': config['vocab'], 'hidden_size': config['hidden'], 'intermediate_size': config['hidden'] * 2,
                    'num_layers': config['layers'], 'num_heads': heads, 'num_kv_heads': kv, 'max_position_embeddings': 4096})
                models.append(model)
                model.hf.to(device)
                if torch.device(model.device) != device:
                    raise RuntimeError('profile model did not move to its admitted CUDA device')
                logger.info(f'profile model device={model.device} dtype={model.dtype}')
                prompts = read_local_prompts({'mode': 'synthetic', 'vocab_size': config['vocab'], 'prompt_len': config['prompt_len'],
                                              'num_prompts': config['prompts'], 'seed': 0})
                for prompt in prompts:
                    prompt['input_ids'] = prompt['input_ids'].to(device)
                previous_threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(config['threads'])
                    results = [profile_cpu.profile_variant(variant, SimpleNamespace(**config), model, prompts, device) for variant in variants]
                finally:
                    torch.set_num_threads(previous_threads)
                report = {**header, 'created': datetime.now(timezone.utc).isoformat(),
                    'machine': f'{platform.processor() or platform.machine()} ({os.cpu_count()} CPUs), {platform.system()}',
                    'torch': str(torch.__version__), 'threads': config['threads'], 'device': str(device),
                    'measurement': 'cProfile host self time of CUDA workload; not CUDA kernel time',
                    'rate_clock': 'decoder internal unsynchronized host-clock intervals; not end-to-end CUDA throughput',
                    'wall_clock': 'time.perf_counter around profiled evaluation; includes host execution and synchronization overhead',
                    'model': f"hidden {config['hidden']}, {config['layers']} layers, vocab {config['vocab']}",
                    'workload': f"{config['prompts']} prompts x {config['prompt_len']} tokens, {config['new_tokens']} new tokens, temperature 0",
                    'results': results}
                if any(result['exact_match_rate'] != 1.0 for result in results):
                    raise RuntimeError('profile AR diagnostic differs; report cannot be completed')
                verify()
                with (output / 'profile.json').open('x', encoding='utf-8') as handle:
                    json.dump(report, handle, indent=1, allow_nan=False)
                with (output / 'profile.md').open('x', encoding='utf-8') as handle:
                    handle.write(profile_cpu.markdown(report))
    except BaseException as exc:
        primary = exc
        try:
            logger.exception('diagnostic execution failed', exc)
        except Exception as cleanup:
            setattr(exc, 'cleanup_errors', (*getattr(exc, 'cleanup_errors', ()), cleanup))
        raise
    finally:
        _close_models(models, primary)
    verify()
    differences = fingerprint.compare(old, report) if old is not None else []
    if differences:
        with (output / 'differences.json').open('x', encoding='utf-8') as handle:
            json.dump(differences, handle)
        return 1
    with (output / 'protocol_complete.json').open('x', encoding='utf-8') as handle:
        json.dump({'protocol': identity['protocol'], 'inputs_verified': True, 'device': str(device)}, handle)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fingerprint', 'profile'))
    for name in ('config', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--compare', type=Path)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
