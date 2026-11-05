"""Explicit local-input capacity decode; importing this module never loads a model."""
import argparse
import json
from pathlib import Path
from typing import Any

from spec_mtp.structure.artifact.provenance import file_sha256


def measure_decode(model, prompt, setting, *, max_new_tokens=256):
    """Measure the original template/step-scale decode, with its actual mask policy."""
    if not __debug__:
        raise RuntimeError('capacity requires assertion checks; Python -O is unsupported')
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from studies._shared.structure.config import _convert
    from studies._shared.structure.validation import _check_argmax
    from studies.qwen3_4b_bf16_bias_norm_scan.norm_control import observe_initial_bias

    if model.device.type != 'cuda' or model.dtype != torch.bfloat16:
        raise ValueError('capacity decode requires an actual CUDA BF16 model')
    info: dict[str, Any] = {}
    provider = observe_initial_bias(build_mask_provider(_convert(setting['spec']), model.embedding_table), 'template', info)
    ids = torch.tensor([prompt['input_ids']], device=model.device)
    with torch.inference_mode(), patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
        output, stats = decoding.SpecDecoder(model, _convert(setting['spec']),
            _convert({'temperature': 0., 'max_new_tokens': max_new_tokens, 'stop_on_eos': False}),
            eos_ids=(), mask_policy=provider).generate(ids)
    checked = _check_argmax({'input_ids': prompt['input_ids'], 'output_ids': output[0].tolist(), 'stats': stats})
    if stats['new_tokens'] != max_new_tokens or checked != max_new_tokens:
        raise ValueError('capacity decode did not complete its full argmax-checked token budget')
    if stats['prefill_time'] != 0 or stats['decode_time'] != 0:
        raise ValueError('capacity decode unexpectedly recorded timing')
    torch.cuda.synchronize(model.device)
    return {'complete': True, 'capacity_only': True, 'no_timing': True, 'dtype': str(model.dtype),
        'question_id': prompt['question_id'], 'input_tokens': len(prompt['input_ids']),
        'generated_tokens': max_new_tokens, 'argmax_checked_tokens': checked,
        'eos_disabled_for_capacity_bound_only': True, 'formal_tau_reported': False,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(model.device),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(model.device), 'bias': info}


def run(args):
    if not __debug__:
        raise RuntimeError('capacity requires assertion checks; Python -O is unsupported')
    import torch
    import yaml
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.observation import verify_model_files
    from studies._shared.structure.sources import package_files
    from studies._shared.structure import gpu_admission
    from studies._shared.flow.write.run import publish_new_completion

    aliases = {name: Path(getattr(args, name)).absolute() for name in ('model_root', 'model_sources', 'prompts', 'study', 'plan')}
    paths = {name: path.resolve(strict=True) for name, path in aliases.items()}
    weights = paths['model_root']
    if not weights.is_dir() or any(not paths[name].is_file() for name in paths if name != 'model_root'):
        raise ValueError('capacity inputs require a local model directory and explicit files')
    if any(paths[name].is_relative_to(weights) for name in paths if name != 'model_root'):
        raise ValueError('capacity input manifests and plan must be outside the model directory')
    output = validate_study_root(validate_output_root(args.output_root, *paths.values()))
    if output.exists() or output.is_symlink():
        raise FileExistsError('capacity requires a new external output root')
    if not isinstance(args.version, str) or not args.version.strip():
        raise ValueError('capacity requires an explicit new version identity')
    device = torch.device(args.device)
    if device != torch.device('cuda:0') or not torch.cuda.is_available():
        raise ValueError('single-GPU capacity calibration requires available explicit cuda:0; no CPU fallback')
    hashes = {name: file_sha256(path) for name, path in paths.items() if name != 'model_root'}
    for name in ('prompts', 'study', 'plan'):
        if hashes[name] != getattr(args, name + '_sha256'):
            raise ValueError(f'capacity {name} differs from the explicit SHA256 binding')
    sources = json.loads(paths['model_sources'].read_text(encoding='utf-8'))
    verify_model_files(weights, sources)
    prompts = json.loads(paths['prompts'].read_text(encoding='utf-8'))['rendered_prompts']
    prompt = max(prompts, key=lambda value: len(value['input_ids']))
    if prompt['question_id'] != 295 or len(prompt['input_ids']) != 1500:
        raise ValueError('capacity requires the original q295/1500-token longest prompt')
    setting = yaml.safe_load(paths['study'].read_text(encoding='utf-8'))['axes']['algorithm.config'][1]
    if setting['direction'] != 'template' or setting['spec']['ema']['step_scale'] != 2:
        raise ValueError('capacity requires the original template/step-scale-2 setting')
    package = package_files()
    from studies.qwen3_4b_bf16_bias_norm_scan import norm_control
    from studies._shared.structure import geometry_4b as direction_control
    local_sources = {}
    for module in (norm_control, direction_control):
        assert module.__file__ is not None
        local_sources[Path(module.__file__)] = file_sha256(module.__file__)
    local_sources[Path(__file__)] = file_sha256(__file__)
    local_sources[Path(publish_new_completion.__code__.co_filename)] = file_sha256(publish_new_completion.__code__.co_filename)

    def verify():
        if any(aliases[name].resolve(strict=True) != path for name, path in paths.items()):
            raise ValueError('capacity input path was redirected')
        verify_model_files(weights, sources)
        if (package_files() != package or any(file_sha256(paths[name]) != sha for name, sha in hashes.items())
                or any(file_sha256(path) != sha for path, sha in local_sources.items())):
            raise ValueError('capacity inputs or sources changed during execution')

    admission = gpu_admission.observe(initial=True)
    if (admission['foreign_gpu_processes']
            or admission['host_available_bytes'] < 8060927256 + admission['current_worker_rss_bytes'] + 2147483648):
        raise RuntimeError('capacity requires the original host budget and actual nonpreempted CUDA device')
    verify()
    output.mkdir(parents=True, exist_ok=False)
    header = {'version': args.version,
        'scope': 'New capacity decode only; historical calibration and host process-tree peak are not resigned',
        'device': str(device), 'device_name': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'inputs': {name: {'path': str(paths[name]), 'sha256': sha} for name, sha in hashes.items()},
        'model_root': str(weights), 'model_sources': sources, 'package_sources': package,
        'readout_sources': {str(path): sha for path, sha in local_sources.items()}, 'resource_admission': admission}
    with (output / 'capacity_inputs.json').open('x', encoding='utf-8') as stream:
        json.dump(header, stream, indent=2, allow_nan=False)
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.system.runtime import apply_runtime
    from spec_mtp.structure.system.config import SystemConfig
    torch.set_num_threads(1)
    apply_runtime(123, SystemConfig.from_mapping({'config': {'deterministic': False}}))
    print(f'capacity device={device} name={header["device_name"]}; q295/1500/256; no timing', flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    from unittest.mock import patch
    from transformers import AutoModelForCausalLM
    original_load = AutoModelForCausalLM.from_pretrained
    def direct_cuda(*values, **kwargs):
        if 'device_map' in kwargs:
            raise ValueError('capacity owns its explicit CUDA placement')
        return original_load(*values, **dict(kwargs, device_map=str(device)))
    # Preserve the original direct-CUDA load, rather than a resident CPU copy
    # whose later transfer would change the memory-capacity measurement.
    with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=direct_cuda):
        model = HFCausalLM(str(weights), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
    try:
        model.hf.to(device)
        if model.device != device or not model.tied_embeddings:
            raise ValueError('capacity requires the actual tied CUDA model')
        verify()
        result = measure_decode(model, prompt, setting)
    except BaseException as error:
        try:
            model.close()
        except Exception as close_error:
            add_note = getattr(error, 'add_note', None)
            if callable(add_note):
                add_note(f'capacity model close also failed: {close_error}')
        raise
    else:
        model.close()
    verify()
    with (output / 'capacity_decode.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    verify()
    publish_new_completion(output / 'capacity_complete.json', json.dumps({'complete': True, 'capacity_only': True,
        'decode_sha256': file_sha256(output / 'capacity_decode.json'), 'device': str(device),
        'historical_calibration_revalidated': False, 'host_process_tree_peak_measured': False}, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-root', 'model-sources', 'prompts', 'prompts-sha256', 'study', 'study-sha256',
                 'plan', 'plan-sha256', 'device', 'output-root', 'version'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
