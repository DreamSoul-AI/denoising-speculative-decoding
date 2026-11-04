"""Fresh replication entry and shared local-input admission for replacement flows."""

import argparse
import hashlib
import json
from pathlib import Path

from spec_mtp.structure.artifact.observation import verify_model_files
from spec_mtp.structure.artifact.paths import validate_output_root
from spec_mtp.structure.artifact.layout import validate_study_root
from spec_mtp.structure.artifact.provenance import file_sha256
from studies.trajectory_observation.structure import fresh_replace as protocol


def prepare_local_inputs(args, input_path):
    """Bind local model/data/plan and both source pillars before either flow writes."""
    plan = Path(args.plan).resolve(strict=True)
    if not plan.is_file():
        raise ValueError('research plan must be a local file')
    plan_sha = file_sha256(plan)
    if plan_sha != args.plan_sha256:
        raise ValueError('research plan SHA256 differs from the explicit binding')
    import torch
    from studies._shared.structure.sources import package_files
    from studies.trajectory_observation.structure import fair_branch

    weights = Path(args.model_root).resolve(strict=True)
    manifest = Path(args.model_sources).resolve(strict=True)
    data = Path(input_path).resolve(strict=True)
    if not weights.is_dir() or not manifest.is_file() or not data.is_file():
        raise ValueError('model root must be a directory; model sources and data must be files')
    if manifest.is_relative_to(weights):
        raise ValueError('model sources must be outside the model directory')
    if plan.is_relative_to(weights):
        raise ValueError('research plan must be outside the model directory')
    output = validate_study_root(validate_output_root(args.output, weights, manifest, data, plan))
    if output.exists() or output.is_symlink():
        raise FileExistsError('replacement protocols require a new output directory')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('replacement protocols require an available explicit CUDA device')
    if device.index is None or device.index >= torch.cuda.device_count():
        raise ValueError('device must name an available CUDA index, for example cuda:0')
    manifest_sha = file_sha256(manifest)
    sources = json.loads(manifest.read_text(encoding='utf-8'))
    verify_model_files(weights, sources)
    identity = fair_branch.identity()
    if identity.get('dirty') is not False:
        raise RuntimeError('replacement protocols require a known clean source worktree')
    package_sources = package_files()
    study_root = Path(fair_branch.__file__).resolve().parents[1]

    def study_sources():
        return {path: file_sha256(path) for pillar in ('structure', 'flow')
                for path in sorted((study_root / pillar).rglob('*.py'))}

    source_files = study_sources()
    inputs = {data: file_sha256(data), **source_files}
    inputs[plan] = plan_sha
    inputs[manifest] = manifest_sha
    def verify_inputs():
        verify_model_files(weights, sources)
        if (fair_branch.identity() != identity or study_sources() != source_files
                or package_files() != package_sources or any(file_sha256(path) != sha for path, sha in inputs.items())):
            raise ValueError('replacement inputs or sources changed during execution')

    binding = {
        'source': identity, 'torch': torch.__version__,
        'device': str(device), 'device_name': torch.cuda.get_device_name(device),
        'model': args.model, 'model_root': str(weights), 'model_sources': sources,
        'plan': {'path': str(plan), 'sha256': plan_sha},
        'package_sources': package_sources,
        'input_sources': {str(path): sha for path, sha in inputs.items()},
    }
    return output, weights, device, data, binding, verify_inputs


def run(args):
    import torch
    from transformers import AutoTokenizer
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from studies.trajectory_observation.structure import fair_branch

    output, weights, device, data, binding, verify_inputs = prepare_local_inputs(args, args.dolly)
    rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Dolly input must contain local JSONL objects')
    prompts = protocol.fresh_prompts(rows)
    digest = protocol.assert_known_slice(prompts)
    if set(item['text'] for item in prompts) & set(fair_branch.templates()[:128]):
        raise RuntimeError('fresh prompts overlap calibration text')
    verify_inputs()
    output.mkdir(parents=True, exist_ok=False)
    header = {
        **binding, 'protocol': protocol.PROTOCOL,
        'eval': fair_branch.EVAL, 'tree': fair_branch.TREE,
        'esp': {'method': fair_branch.ESP['method'], 'num_masks': fair_branch.ESP['num_masks']},
        'split': protocol.SPLIT, 'skip': protocol.SKIP, 'take': protocol.TAKE, 'prompt_sha256': digest,
        'bootstrap': {'seed': protocol.BOOT_SEED, 'resamples': protocol.BOOT_RESAMPLES},
    }
    with (output / 'header.json').open('x', encoding='utf-8') as handle:
        json.dump(header, handle, ensure_ascii=False, indent=2, allow_nan=False)
    from spec_mtp.structure.system.logger import Logger
    logger = Logger(output / 'assets')
    logger.info(f'fresh replication device={device} name={header["device_name"]}')
    model_dir = output / args.model
    model_dir.mkdir()
    with (model_dir / 'prompt-index.json').open('x', encoding='utf-8') as handle:
        json.dump([{'qid': prompt['qid'], 'sha256': hashlib.sha256(prompt['text'].encode('utf-8')).hexdigest()}
                   for prompt in prompts], handle, ensure_ascii=False)
    model = HFCausalLM(str(weights), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    try:
        model.hf.to(device)
        if torch.device(model.device) != device:
            raise RuntimeError(f'model did not move to the explicit device {device}')
        tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
        verify_inputs()
        prior = fair_branch.calibrate(model, tokenizer, model_dir)
        fair_branch.run_split(model, tokenizer, prompts, prior, model_dir, protocol.SPLIT)
        esp = fair_branch.read_rows(model_dir / f'{protocol.SPLIT}-esp.jsonl')
        replaced = fair_branch.read_rows(model_dir / f'{protocol.SPLIT}-replace.jsonl')
        if len(esp) != protocol.TAKE or len(replaced) != protocol.TAKE:
            raise RuntimeError('replication finished without both arms complete')
        summary = protocol.replication_summary(esp, replaced)
        summary.update(model=args.model, weights=str(weights), prompt_sha256=digest, k_mean=prior['k_mean'])
        verify_inputs()
        with (model_dir / 'replication-summary.json').open('x', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    except Exception as exc:
        logger.exception('fresh protocol failed', exc)
        raise
    finally:
        model.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('4b', '0.6b'), required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--model-sources', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True, help='unique current research plan')
    parser.add_argument('--plan-sha256', required=True, help='expected SHA256 of the plan file bytes')
    parser.add_argument('--device', required=True, help='explicit CUDA device, for example cuda:0')
    parser.add_argument('--dolly', type=Path, required=True, help='local Dolly JSONL; no dataset download')
    parser.add_argument('--output', type=Path, required=True, help='new external output directory')
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
