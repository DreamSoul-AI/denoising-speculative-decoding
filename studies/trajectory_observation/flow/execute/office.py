"""Bound explicit local inputs before executing an existing office protocol.

    python -m studies.trajectory_observation.flow.execute.office --help
"""
import argparse
import json
from pathlib import Path

from spec_mtp.structure.artifact.layout import validate_study_root
from spec_mtp.structure.artifact.observation import verify_model_files
from spec_mtp.structure.artifact.paths import validate_output_root
from spec_mtp.structure.artifact.provenance import file_sha256
from studies.trajectory_observation.structure import office_qwen8b as q
from studies.trajectory_observation.structure import office_tree as t
from studies.trajectory_observation.structure import office_traj as tr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('qwen8b', 'tree', 'traj'))
    for name in ('dest', 'model-root', 'model-sources', 'dolly', 'plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', choices=('4b', '8b'), default='8b')
    parser.add_argument('--only', choices=('all', 'smoke', 'formal'), default='all')
    parser.add_argument('--protocol', choices=('ablate', 'xcat', 'xcat2', 'xcat3', 'xcat4',
        'xcat5', 'xcat6', 'xcat7', 'wall_by_length', 'gscan', 'first_fork', 'ar_determinism'), default='ablate')
    args = parser.parse_args(argv)
    if args.protocol in ('first_fork', 'ar_determinism'):
        raise RuntimeError('independent-AR numerical mechanism protocols are permanently stopped')
    if not args.device.startswith('cuda'):
        raise ValueError('office protocols require CUDA')
    if args.command != 'tree' and args.model != '8b':
        raise ValueError('this office protocol requires the 8b model label')
    weights, manifest, dolly, plan = (value.resolve(strict=True) for value in (
        args.model_root, args.model_sources, args.dolly, args.plan))
    if not weights.is_dir() or any(not value.is_file() for value in (manifest, dolly, plan)):
        raise ValueError('explicit local model, sources, Dolly and plan inputs are required')
    if manifest.is_relative_to(weights) or plan.is_relative_to(weights):
        raise ValueError('model sources and plan must stay outside the model root')
    dest = validate_output_root(args.dest, weights, manifest, dolly, plan)
    validate_study_root(dest)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('office protocols require a new output root; old runs remain read-only')
    plan_sha = file_sha256(plan)
    if plan_sha != args.plan_sha256:
        raise ValueError('plan SHA256 mismatch')
    identity_reader = {'qwen8b': q._identity, 'tree': t._identity, 'traj': tr._run_identity}[args.command]
    identity = identity_reader()
    if args.only != 'smoke' and identity['dirty']:
        raise RuntimeError('formal run requires a clean worktree before any model or output work')
    manifest_sha = file_sha256(manifest)
    sources = json.loads(manifest.read_text(encoding='utf-8'))
    verify_model_files(weights, sources)
    from studies._shared.structure.sources import package_files

    package_sources = package_files()
    study_root = Path(q.__file__).resolve().parents[1]
    input_files = {manifest: manifest_sha, dolly: file_sha256(dolly), plan: plan_sha}
    source_files = {value: file_sha256(value) for pillar in ('structure', 'flow')
                    for value in sorted((study_root / pillar).rglob('*.py'))}
    rows = [json.loads(line) for line in dolly.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Dolly input must contain local JSONL objects')
    selected = None
    if args.command == 'tree':
        selected = t._registered(args.model, rows)
    elif args.command == 'traj':
        cfg = tr.round_config(args.protocol)
        prompts, skipped = cfg['select'](rows)
        selected = (prompts, skipped, cfg['check'](prompts, skipped), cfg)
    elif args.only != 'smoke':
        prompts, skipped = q.prompts_after_used(rows)
        q.assert_registered_slice(prompts, skipped)
    if args.command in ('tree', 'traj') and selected is None:
        raise RuntimeError('office protocol selection is missing')
    import torch

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('office protocols require an available CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if device.index >= torch.cuda.device_count():
        raise ValueError('requested CUDA device is unavailable')

    def verify_inputs():
        verify_model_files(weights, sources)
        current = {value: file_sha256(value) for pillar in ('structure', 'flow')
                   for value in sorted((study_root / pillar).rglob('*.py'))}
        if (identity_reader() != identity or current != source_files or package_files() != package_sources
                or any(file_sha256(value) != sha for value, sha in input_files.items())):
            raise ValueError('office inputs or sources changed during execution')

    q._offline_caches(dest)
    verify_inputs()
    dest.mkdir(parents=True, exist_ok=False)
    protocol_name = q.PROTOCOL
    if selected is not None:
        protocol_name = selected[3] if args.command == 'tree' else selected[3]['protocol']
    binding = {'schema': 1, 'command': args.command, 'protocol': protocol_name,
        'stage': args.only, 'source': identity, 'device': str(device),
        'device_name': torch.cuda.get_device_name(device), 'model_root': str(weights),
        'model_sources': sources, 'inputs': {str(value): sha for value, sha in input_files.items()},
        'source_files': {str(value): sha for value, sha in source_files.items()},
        'package_sources': package_sources}
    with (dest / 'protocol_inputs.json').open('x', encoding='utf-8') as handle:
        json.dump(binding, handle, ensure_ascii=False, indent=2, allow_nan=False)
    from spec_mtp.structure.system.logger import Logger
    from studies._shared.flow.write.run import publish_new_completion

    logger = Logger(dest / 'assets', managed=True)
    logger.info(f"office device={device} name={binding['device_name']} stage={args.only}")
    model = None
    primary = None
    try:
        if args.command == 'qwen8b':
            model, tokenizer, _version = q._load_model(weights, device)
            logger.info(f'office model device={model.device} dtype={model.dtype}')
            verify_inputs()
            replace, kalman = q._fit(model, tokenizer, dest)
            if args.only in ('all', 'smoke'):
                q._smoke(model, tokenizer, replace, kalman, dest)
            if args.only in ('all', 'formal'):
                q._formal(model, tokenizer, replace, kalman, dest, rows)
        else:
            assert selected is not None
            model, tokenizer, _version = t._load_model(args.model, weights, device)
            logger.info(f'office model device={model.device} dtype={model.dtype}')
            verify_inputs()
            if args.command == 'tree':
                prompts, skipped, digest, protocol = selected
                prefix = t.PREFIXES[args.model]
                if args.only in ('all', 'smoke'):
                    t._smoke(model, tokenizer, prompts, dest, protocol, prefix)
                if args.only in ('all', 'formal'):
                    t._formal(model, tokenizer, prompts, skipped, digest, dest, protocol, prefix)
            else:
                prompts, skipped, digest, cfg = selected
                if args.only in ('all', 'smoke'):
                    (tr.gscan_smoke if args.protocol == 'gscan' else tr._smoke)(model, tokenizer, prompts, dest, cfg)
                if args.only in ('all', 'formal'):
                    if args.protocol == 'wall_by_length':
                        from studies.trajectory_observation.structure.office_traj_length import _formal_by_length
                        _formal_by_length(model, tokenizer, prompts, skipped, digest, dest, cfg)
                    else:
                        (tr.gscan_formal if args.protocol == 'gscan' else tr._formal)(model, tokenizer, prompts, skipped, digest, dest, cfg)
        verify_inputs()
    except BaseException as exc:
        primary = exc
        if isinstance(exc, Exception):
            logger.exception('office protocol failed', exc)
        raise
    finally:
        if model is not None:
            try:
                model.close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
    payload = json.dumps({'inputs_verified': True, 'device': str(device)})
    verify_inputs()
    publish_new_completion(dest / 'protocol_complete.json', payload)
    logger.info('office protocol completed with verified inputs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
