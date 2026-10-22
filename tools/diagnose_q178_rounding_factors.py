"""Arithmetic factorization of saved q178 Phi/RMSNorm differences, zero model forwards.

    python tools/diagnose_q178_rounding_factors.py --out .tmp/q178-rounding-factors-<new-id>
"""

import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import file_hash
from studies.output_space_extension.kernel_common import profile


def delta(value, reference):
    import torch
    value = value.detach().cpu().float()
    reference = reference.detach().cpu().float()
    assert value.shape == reference.shape and torch.isfinite(value).all()
    return {'bit_equal': torch.equal(value, reference),
            'changed_coordinates': int((value != reference).sum()),
            'max_abs': float((value - reference).abs().max()),
            'L2': float((value - reference).norm())}


def phi_factors(slots, scale):
    import torch
    assert slots.dtype == scale.dtype == torch.float32 and slots.shape[0] == 2
    native_norm = slots.norm(dim=-1, keepdim=True)[1]
    exact_norm = slots[1].cpu().double().norm().to(slots.device)
    values = {}
    for norm64, multiply64, divide64 in itertools.product((0, 1), repeat=3):
        denominator = exact_norm if norm64 else native_norm
        numerator = scale.double() * slots[1].double() if multiply64 else scale * slots[1]
        value = numerator.double() / denominator.double() if divide64 else numerator.float() / denominator.float()
        values[f'{norm64}{multiply64}{divide64}'] = value.float()
    return values, {'norm32': float(native_norm), 'norm64': float(exact_norm),
                    'norm64_rounded32_equal': bool(torch.equal(native_norm, exact_norm.float().reshape_as(native_norm)))}


def rms_factors(inputs, weight, epsilon, bits):
    import torch
    assert len(bits) == 5 and all(x in (0, 1) for x in bits)
    dtype = lambda high: torch.float64 if high else torch.float32
    row = inputs[:, 4:5]
    value = row.to(dtype(bits[0])).pow(2)
    value = value.to(dtype(bits[1])).mean(-1, keepdim=True) + epsilon.to(dtype(bits[1]))
    inverse = torch.rsqrt(value.to(dtype(bits[2])))
    normalized = row.to(dtype(bits[3])) * inverse.to(dtype(bits[3]))
    result = weight.to(dtype(bits[4])) * normalized.to(dtype(bits[4]))
    return result.float(), {'variance_plus_epsilon': float(value), 'inverse': float(inverse)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    from safetensors import safe_open
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.evidence import runtime
    root = Path(__file__).resolve().parents[1]
    study = root / 'studies/output_space_extension'
    paths = {'tool': Path(__file__), 'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
        'phi_tool': root / 'tools/diagnose_q178_phi_rms.py',
        'common': study / 'kernel_common.py',
        'report': root / '.tmp/q178-phi-rms-20261014-v1/diagnosis.json',
        'tensor': root / '.tmp/q178-phi-rms-20261014-v1/selected_tensors.pt',
        'previous_tensor': root / '.tmp/q178-numerical-20261014-v1/selected_tensors.pt',
        'weight': study / 'shared/model/modelscope/Qwen/Qwen3-0.6B/model.safetensors'}
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    prior = read(paths['report'])
    import transformers.models.qwen3.modeling_qwen3 as qwen
    paths['norm_implementation'] = Path(qwen.__file__)
    binding = {k: file_hash(p) for k, p in paths.items()}
    assert prior['quality_passed'] and binding['tensor'] == prior['raw_tensor_sha256']
    assert binding['previous_tensor'] == prior['source_binding']['previous_tensor']
    assert binding['phi_tool'] == prior['source_binding']['tool']
    assert binding['norm_implementation'] == prior['source_binding']['norm_implementation']
    assert binding['weight'] == 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
    assert manifest()['digest'] == '670c744e0535dcad'
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    saved = torch.load(paths['tensor'], weights_only=True, map_location='cpu', mmap=True)
    old = torch.load(paths['previous_tensor'], weights_only=True, map_location='cpu', mmap=True)
    with safe_open(paths['weight'], framework='pt', device='cpu') as f:
        weight = f.get_tensor('model.layers.0.post_attention_layernorm.weight').float().cuda()
    epsilon = torch.tensor(prior['native_epsilon'], dtype=torch.float32, device='cuda')
    report = {'version': 'q178-rounding-factors-20261015-v1', 'source_binding': binding,
        'source_digest': manifest()['digest'], 'runtime': runtime(), 'phi': [], 'rms': [],
        'model_forwards': 0, 'new_tokens': 0, 'quality_passed': False}
    tensors = {}
    try:
        for case in ('output_order0', 'output_order1'):
            state = old[f'baseline:{case}:85']['current_state']
            y, velocity, anchor = (state[k].cuda() for k in ('proxies', 'velocity', 'anchor'))
            y = y[0]
            if case.endswith('1'):
                velocity = .99 * velocity + (1 - .99) * (y - anchor)
            slots = torch.stack([y + j * velocity for j in range(2)])
            scale = torch.tensor(next(v['native_scale_held'] for v in prior['entry_vectors'] if v['case'] == case),
                                 dtype=torch.float32, device='cuda')
            values, scalars = phi_factors(slots, scale)
            native, high = saved[case + ':native_entry'], saved[case + ':phi64_entry']
            assert torch.equal(values['000'].cpu(), native)
            cpu64 = (scale.cpu().double() * slots[1].cpu().double() / slots[1].cpu().double().norm()).float()
            assert torch.equal(cpu64, high)
            report['phi'].append({'case': case, 'scalars': scalars,
                'factors': {k: {'to_native': delta(v, native), 'to_saved_cpu64': delta(v, high)} for k, v in values.items()}})
            tensors.update({case + ':' + k: v.cpu() for k, v in values.items()})
            if case.endswith('1'):
                report['phi_profile'] = profile(lambda: phi_factors(slots, scale),
                    args.out / 'phi-factors.chrome.json', {'aten::linalg_vector_norm', 'aten::mul', 'aten::div'})
        for case in ('native_self', 'native_graft'):
            capture = saved[case + ':norm_capture']
            inputs, native = capture['inputs'].cuda(), capture['output'][:, 4:5]
            row64 = inputs[:, 4:5].cpu().double()
            high = (weight.cpu().double() * (row64 * torch.rsqrt(row64.pow(2).mean(-1, keepdim=True) + epsilon.cpu().double()))).float()
            expected = next(v['max_abs'] for v in prior['norm_comparisons'] if v['case'] == case and v['mode'] == 'reference64')
            assert delta(high, native)['max_abs'] == expected
            rows = {}
            for bits in itertools.product((0, 1), repeat=5):
                value, scalars = rms_factors(inputs, weight, epsilon, bits)
                name = ''.join(map(str, bits))
                if name == '00000':
                    assert torch.equal(value.cpu(), native)
                rows[name] = {'to_native': delta(value, native), 'to_saved_cpu64': delta(value, high), 'scalars': scalars}
                tensors[case + ':' + name] = value.cpu()
            report['rms'].append({'case': case, 'input_shape': list(inputs.shape), 'factors': rows})
            if case == 'native_self':
                report['rms_profile'] = profile(lambda: rms_factors(inputs, weight, epsilon, (0, 0, 0, 0, 0)),
                    args.out / 'rms-factors.chrome.json', {'aten::pow', 'aten::mean', 'aten::rsqrt', 'aten::mul'})
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert manifest()['digest'] == report['source_digest']
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'factor_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'factor_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'phi_cells': 16, 'rms_cells': 64, 'model_forwards': 0}))


if __name__ == '__main__':
    main()
