"""Run the existing synthetic pilot from an explicit configuration and bound plan."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from spec_mtp.structure.artifact.layout import validate_study_root
from spec_mtp.structure.artifact.paths import validate_output_root
from spec_mtp.structure.artifact.provenance import file_sha256


def run(args):
    config_path, plan = (Path(value).resolve(strict=True) for value in (args.config, args.plan))
    plan_sha = file_sha256(plan)
    if plan_sha != args.plan_sha256:
        raise ValueError('research plan SHA256 differs from the explicit binding')
    config_sha = file_sha256(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    from studies.trajectory_observation.structure import pilot
    import torch

    if not isinstance(config, dict) or not pilot.DEFAULT.keys() <= config.keys():
        raise ValueError('pilot requires a complete explicit configuration')
    flags = ('kalman_align', 'kalman_merge', 'kalman', 'correction', 'affine_ridge', 'affine', 'oracle_bound')
    if (set(config) - (set(pilot.DEFAULT) | set(flags) | {'bonus_slot', 'blend_gain'})
            or any(name in config and type(config[name]) is not bool for name in flags)):
        raise ValueError('unknown or invalid pilot protocol mode')
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not finite(config['sharpness']) or not isinstance(config['train'], dict) or not finite(config['train'].get('lr')) or config['train']['lr'] <= 0:
        raise ValueError('pilot sharpness and learning rate must be finite')
    positive = [(config, ('content_vocab', 'branching', 'max_new_tokens')),
                (config['train'], ('sequences', 'length', 'steps', 'batch')),
                (config['prompts'], ('fit', 'holdout', 'cross', 'length'))]
    if any(not isinstance(block, dict) or any(type(block.get(key)) is not int or block[key] < 1 for key in keys)
           for block, keys in positive):
        raise ValueError('pilot sizes and budgets must be positive integers')
    if (config['content_vocab'] < 2 or config['branching'] > config['content_vocab'] or config['train']['length'] < 4
            or config['prompts']['length'] < 3 or config['train']['sequences'] % 2
            or not isinstance(config['model'], dict) or not isinstance(config['trees'], list) or not config['trees']
            or not isinstance(config['seeds'], list) or not config['seeds']
            or any(type(seed) is not int for seed in config['seeds']) or len(set(config['seeds'])) != len(config['seeds'])):
        raise ValueError('invalid pilot language, model, trees or seed declaration')
    if (set(config['train']) != set(cast(dict[str, Any], pilot.DEFAULT['train']))
            or set(config['prompts']) != set(cast(dict[str, Any], pilot.DEFAULT['prompts']))
            or set(config['model']) != set(cast(dict[str, Any], pilot.DEFAULT['model']))
            or any(type(value) is not int or value < 1 for value in config['model'].values())):
        raise ValueError('pilot requires complete known model, training and prompt fields')
    model = config['model']
    if (model['hidden_size'] % model['num_attention_heads'] or model['num_attention_heads'] % model['num_key_value_heads']
            or model['max_position_embeddings'] < max(config['train']['length'], config['prompts']['length'] + config['max_new_tokens'])):
        raise ValueError('pilot model dimensions do not cover the declared token budget')
    for tree in config['trees']:
        if (not isinstance(tree, dict) or set(tree) != {'num_masks', 'branches'}
                or type(tree['num_masks']) is not int or not isinstance(tree['branches'], list)
                or tree['num_masks'] != len(tree['branches']) or tree['num_masks'] < 1
                or any(type(value) is not int or not 1 <= value <= config['content_vocab'] + 2 for value in tree['branches'])):
            raise ValueError('invalid pilot tree declaration')
    variants = config['variants']
    if (not isinstance(variants, list) or any(not isinstance(value, list) or len(value) != 2
            or any(not isinstance(part, str) for part in value) or tuple(value) not in pilot.SUFFIX for value in variants)
            or len(set(map(tuple, variants))) != len(variants)):
        raise ValueError('invalid pilot observation variants')
    modes = [name for name in ('bonus_slot', 'kalman_align', 'kalman_merge', 'kalman', 'correction', 'affine_ridge', 'affine')
             if config.get(name)]
    oracle = config.get('oracle_bound', False)
    if type(oracle) is not bool or len(modes) > 1 or (oracle and modes):
        raise ValueError('pilot protocol modes must remain separate')
    if (modes and variants) or (not modes and not oracle and not variants) or (oracle and variants != pilot.DEFAULT['variants']):
        raise ValueError('pilot protocol modes cannot combine with custom observation variants')
    if 'blend_gain' in config and (not config.get('kalman') or not finite(config['blend_gain'])):
        raise ValueError('blend gain belongs only to the existing Kalman comparison')
    if config.get('bonus_slot') not in (None, False, 'hard', 'topk'):
        raise ValueError('unknown bonus-slot protocol')
    output = validate_study_root(validate_output_root(args.output, config_path, plan))
    if output.exists() or output.is_symlink():
        raise FileExistsError('pilot requires a new external output directory')
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ValueError('pilot requires an explicit available CUDA device')
    identity = pilot.source_identity()
    if identity.get('dirty') is not False:
        raise ValueError('pilot requires known clean source identity')
    from studies._shared.structure.sources import package_files
    from spec_mtp.structure.system.logger import Logger

    package_sources = package_files()
    study_root = Path(pilot.__file__).resolve().parents[1]
    def sources():
        return {str(path): file_sha256(path) for pillar in ('structure', 'flow')
                for path in sorted((study_root / pillar).rglob('*.py'))}
    study_sources = sources()
    canonical = json.dumps(config, sort_keys=True, allow_nan=False)
    def verify():
        if (file_sha256(config_path) != config_sha or file_sha256(plan) != plan_sha
                or pilot.source_identity() != identity or sources() != study_sources
                or package_files() != package_sources or json.dumps(config, sort_keys=True, allow_nan=False) != canonical):
            raise ValueError('pilot configuration, plan or sources changed during execution')
    bonus_slot = config.get('bonus_slot')
    protocol = ({'hard': 'bonus-slot-hard-v1', 'topk': 'bonus-slot-topk-v1'}[bonus_slot]
                if isinstance(bonus_slot, str) else 'online-aligned-kalman-v1' if config.get('kalman_align')
                else 'online-trajectory-kalman-v1' if config.get('kalman_merge')
                else 'kalman-mask-scalar-v1' if config.get('kalman')
                else 'trajectory-verify-correction-v1' if config.get('correction')
                else 'trajectory-verify-affine-ridge-v1' if config.get('affine_ridge')
                else 'trajectory-verify-affine-v1' if config.get('affine')
                else pilot.PROTOCOL + ('-oracle-bound' if oracle else ''))
    verify()
    output.mkdir(parents=True, exist_ok=False)
    header = {'protocol': protocol, 'scope': 'synthetic engineering pilot', 'config': config,
        'config_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
        'config_input': {'path': str(config_path), 'sha256': config_sha},
        'plan': {'path': str(plan), 'sha256': plan_sha}, 'source': identity,
        'package_sources': package_sources, 'study_sources': study_sources,
        'torch': torch.__version__, 'device': str(device), 'device_name': torch.cuda.get_device_name(device), 'dtype': 'float32'}
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, ensure_ascii=False, allow_nan=False, indent=2)
    logger = Logger(output / 'assets', managed=True)
    logger.info(f"pilot device={device} dtype=float32 name={header['device_name']}")
    results = []
    for seed in config['seeds']:
        result = (pilot.run_oracle if oracle else pilot.run_seed)(config, seed, device=str(device))
        verify()
        audits: list[object] = []
        def audit(value):
            if isinstance(value, dict):
                audits.extend(item for key, item in value.items() if key in ('argmax_passed', 'argmax_all_passed'))
                for item in value.values():
                    audit(item)
            elif isinstance(value, list):
                for item in value:
                    audit(item)
        audit(result)
        if not audits or any(value is not True for value in audits):
            raise RuntimeError('pilot result lacks a complete passing argmax audit')
        with (output / f'seed-{seed}.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, allow_nan=False, indent=2)
        results.append(result)
    verify()
    with (output / 'protocol_complete.json').open('x', encoding='utf-8') as handle:
        json.dump({'protocol': protocol, 'inputs_verified': True, 'seeds': config['seeds']}, handle)
    return header, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
