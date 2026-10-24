"""Fixed provider slot-rounding/norm-backend controls; scientific src is untouched.

    python tools/diagnose_q178_operators.py --out .tmp/q178-operators-<new-id>
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from diagnose_output_windows import file_hash
from diagnose_q178_numerics import synchronized_transfer, tensor_equal


def map_masks(provider, proxies, mode):
    import torch
    assert mode in ('native', 'slots64', 'norm64', 'norm32cpu')
    if mode == 'native':
        return NATIVE(provider, proxies)
    anchor, velocity = provider.anchor.clone(), provider.velocity.clone()
    if mode == 'slots64':
        anchor, velocity, proxies = anchor.cpu().double(), velocity.cpu().double(), proxies.cpu().double()
    owners = []
    for owner in range(provider.num_masks + 1):
        if owner < provider.num_masks:
            y = proxies[owner].detach()
            if provider.order:
                velocity = provider.beta_prime * velocity + (1 - provider.beta_prime) * (y - anchor)
            anchor = y
        else:
            anchor = anchor + velocity
        slots = torch.stack([anchor + slot * velocity for slot in range(provider.num_masks)])
        slots = slots.to(device=provider.scale.device, dtype=torch.float32)
        if mode in ('norm64', 'norm32cpu'):
            values = slots.cpu().double() if mode == 'norm64' else slots.cpu()
            norms = values.norm(dim=-1, keepdim=True).to(device=slots.device, dtype=slots.dtype)
        else:
            norms = slots.norm(dim=-1, keepdim=True)
        assert torch.isfinite(slots).all() and torch.isfinite(norms).all() and (norms > 0).all()
        owners.append(provider.scale * slots / norms)
    result = torch.stack(owners)
    assert torch.isfinite(result).all()
    return result


# Kept outside patches so the native control remains the scientific implementation.
from spec_mtp.masks import OutputHiddenMaskProvider
NATIVE = OutputHiddenMaskProvider.masks_per_owner


def replay(model, states, mode, intervention_call, **kwargs):
    import torch
    calls, differences = 0, []
    def mapped(provider, proxies):
        nonlocal calls
        calls += 1
        original = NATIVE(provider, proxies)
        if calls <= intervention_call or mode == 'native':
            return original
        saved = [provider.anchor.clone(), provider.velocity.clone(), proxies.clone()]
        result = map_masks(provider, proxies, mode)
        assert all(torch.equal(a, b) for a, b in zip(saved, [provider.anchor, provider.velocity, proxies]))
        differences.append(float((result - original).abs().max()))
        return result
    with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', mapped):
        samples, evidence, intervention, records = synchronized_transfer(model, states, **kwargs)
    assert OutputHiddenMaskProvider.masks_per_owner is NATIVE
    return samples, evidence, records, differences


def profile_norm(provider, proxies, out):
    import torch
    result = {'available': False}
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            NATIVE(provider, proxies)
            torch.cuda.synchronize()
        trace = out / 'provider.chrome.json'
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text(encoding='utf-8'))['traceEvents']
        norms = [e for e in events if e.get('name') == 'aten::linalg_vector_norm' and e.get('cat') == 'cpu_op']
        ids = {e.get('args', {}).get('External id') for e in norms}
        kernels = [e for e in events if e.get('cat') == 'kernel']
        matched = [e for e in kernels if e.get('args', {}).get('External id') in ids]
        result.update(available=bool(kernels), trace_sha256=file_hash(trace), cuda_kernel_count=len(kernels),
                      norm_cpu_calls=len(norms), norm_correlated_kernel_names=sorted({e['name'] for e in matched}),
                      all_kernel_names=sorted({e['name'] for e in kernels}))
    except (RuntimeError, OSError) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('new output required; existing evidence is preserved')
    args.out.mkdir(parents=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import yaml
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM
    root = Path(__file__).resolve().parents[1]
    study = root / 'studies/output_space_extension'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    paths = {'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'), 'native': study / 'docs/execution_v1.json',
        'old_report': root / '.tmp/q178-numerical-20261014-v1/diagnosis.json',
        'old_tensor': root / '.tmp/q178-numerical-20261014-v1/selected_tensors.pt',
        'snapshot': study / 'docs/modelscope_snapshot.json', 'tool': Path(__file__),
        'canonical_helper': Path(__file__).with_name('diagnose_q178_numerics.py'),
        'transfer_helper': Path(__file__).with_name('diagnose_proxy_transfer.py'),
        'observer_helper': Path(__file__).with_name('diagnose_output_windows.py')}
    native = read(paths['native'])
    rows = {r['case']: r for r in native['runs']}
    for filename in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
        paths[filename] = study / 'runs' / rows['output_order1']['run_id'] / filename
    binding = {k: file_hash(p) for k, p in paths.items()}
    previous = read(paths['old_report'])
    assert previous['quality_passed']
    assert binding['old_tensor'] == previous['raw_tensor_sha256'] == '2f71c78210b1609a889b0d3cfec223fd818dd74c625d6eed32540d91008d86a0'
    report = {'version': 'q178-provider-operators-20261014-v1', 'source_binding': binding,
              'source_digest': native['source_digest'], 'cases': [], 'frozen_mapping_comparisons': [],
              'quality_passed': False, 'generation_budget': 800, 'auxiliary_model_forwards': 0}
    selected = {}
    try:
        old = torch.load(paths['old_tensor'], map_location='cpu', weights_only=True)
        assert manifest()['digest'] == native['source_digest'] == '670c744e0535dcad'
        snapshot = read(paths['snapshot'])
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
        torch.set_num_threads(1)
        apply_runtime(126, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': True,
            'cudnn_deterministic': True, 'cudnn_benchmark': False, 'matmul_allow_tf32': False,
            'cudnn_allow_tf32': False}))
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True, warn_only=False)
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == native['model_digest']
        strict = runtime()
        assert strict == previous['runtime'] and not torch.is_deterministic_algorithms_warn_only_enabled()
        report['runtime'] = strict
        report['model_digest'] = native['model_digest']
        report['cublas_workspace_config'] = os.environ['CUBLAS_WORKSPACE_CONFIG']
        provider = OutputHiddenMaskProvider(2, model.embedding_table, order=1)
        for case in ('canonical_one_self', 'canonical_one_from_zero'):
            for position in (86, 89, 108):
                state = old[f'{case}:{position}']['current_state']
                provider.anchor, provider.velocity = state['anchor'].cuda(), state['velocity'].cuda()
                proxies = state['proxies'].cuda()
                actual = NATIVE(provider, proxies)
                assert torch.equal(actual.cpu(), old[f'{case}:{position}']['current_masks'])
                for mode in ('slots64', 'norm64', 'norm32cpu'):
                    mapped = map_masks(provider, proxies, mode)
                    report['frozen_mapping_comparisons'].append({'case': case, 'root': position, 'mode': mode,
                        'bit_equal': torch.equal(actual, mapped), 'max_abs': float((actual - mapped).abs().max())})
                if case == 'canonical_one_self' and position == 86:
                    report['profiler'] = profile_norm(provider, proxies, args.out)
        ar = rows['AR']['raw']['evidence'][3]
        reference = rows['output_order1']['raw']['evidence'][3]
        assert reference['output_ids'] == ar['output_ids']
        prompt = torch.tensor([ar['input_ids']], device=model.device)
        cfg = yaml.safe_load(paths['config.yaml'].read_text(encoding='utf-8'))
        inputs = read(paths['assets/decode_inputs.json'])
        assert inputs['prompts'][3]['input_ids'] == ar['input_ids']
        targets = sorted(int(k.rsplit(':', 1)[1]) for k in old if k.startswith('baseline:output_order1:'))
        # Use the old union, including roots visited by graft but absent in baseline.
        targets = sorted(set(targets) | {int(k.split(':')[1]) for k in old
                          if k.startswith('canonical_one_from_zero:') and k.split(':')[1].isdigit()})
        call = next(i + 1 for i, t in enumerate(reference['stats']['trace']) if t['root_position'] == 85)
        for mode in ('native', 'slots64', 'norm64', 'norm32cpu'):
            for graft in (False, True):
                label = mode + ('_graft' if graft else '_self')
                donor = 'output_order0' if graft else 'output_order1'
                samples, replayed, changes, differences = replay(model, old['canonical_states']['output_order1'], mode, call,
                    prompt=prompt, spec=_convert(rows['output_order1']['spec']),
                    eval_cfg=_convert(cfg['algorithm']['config']['eval']), eos=inputs['eos_ids'],
                    reference=reference, donor=old[f'baseline:{donor}:85']['current_masks'][0, 1],
                    original_masks=old['baseline:output_order1:85']['current_masks'],
                    root=85, targets=targets, placebo=False, seed=126)
                old_label = 'canonical_one_from_zero' if graft else 'canonical_one_self'
                if mode == 'native':
                    prior = next(c['replay'] for c in previous['cases'] if c['label'] == old_label)
                    assert replayed['trace'] == prior['trace'] and replayed['stats'] == prior['stats']
                    assert all(tensor_equal(s, old[f'{old_label}:{p}']) for p, s in samples.items())
                report['cases'].append({'label': label, 'mode': mode, 'graft': graft, 'replay': replayed,
                    'canonical_replacements': changes, 'changed_mapping_calls': len(differences),
                    'mapping_max_abs': max(differences, default=0),
                    'native_selected_exact_repeat': True if mode == 'native' else None})
                selected.update({f'{label}:{p}': s for p, s in samples.items()})
        assert model_digest(model) == native['model_digest'] and runtime() == strict
        assert manifest()['digest'] == native['source_digest']
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if selected:
            torch.save(selected, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [
        {'label': c['label'], 'accepted': c['replay']['stats']['accepted']} for c in report['cases']]}))


if __name__ == '__main__':
    main()
