"""Entry Phi mapping and fixed layer0 post-attention RMSNorm controls.

    python tools/diagnose_q178_phi_rms.py --out .tmp/q178-phi-rms-<new-id>
"""

import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies.output_space_extension.kernel_common import load, verify, compare_control, profile
from diagnose_output_windows import file_hash
from diagnose_q178_numerics import synchronized_transfer
from spec_mtp.masks import OutputHiddenMaskProvider


def phi64(provider, proxies):
    import torch
    y = proxies[0].detach().float()
    velocity = provider.velocity
    if provider.order:
        velocity = provider.beta_prime * velocity + (1 - provider.beta_prime) * (y - provider.anchor)
    slot = y + velocity
    value = slot.cpu().double()
    norm = value.norm()
    assert torch.isfinite(value).all() and norm > 0
    result = (provider.scale.cpu().double() * value / norm).float()
    assert torch.isfinite(result).all()
    return result


def norm_row(module, inputs, mode):
    import torch
    row = inputs[:, 4:5]
    if mode == 'singleton':
        return module.forward(row)
    assert mode == 'reference64'
    value = row.cpu().double()
    epsilon = torch.tensor(module.variance_epsilon, dtype=torch.float32).double()
    variance = value.pow(2).mean(-1, keepdim=True)
    result = module.weight.cpu().double() * (value * torch.rsqrt(variance + epsilon))
    return result.to(inputs.device, inputs.dtype)


def run(c, donor, mode):
    import torch
    norm = c.model.hf.model.layers[0].post_attention_layernorm
    active, captured, changes = {}, {}, []
    def before(_module, args, kwargs):
        positions = args[1] if len(args) > 1 else kwargs['position_ids']
        active['root'] = None if kwargs.get('attention_mask_4d') is None else int(positions[0, 0])
    def replace(module, args, output):
        if active['root'] != 85:
            return output
        inputs = args[0]
        assert inputs.shape == output.shape == (1, 9, 1024)
        captured.update(inputs=inputs.detach().clone(), output=output.detach().clone())
        if mode == 'native':
            return output
        replacement = norm_row(module, inputs, mode)
        assert replacement.shape == output[:, 4:5].shape and torch.isfinite(replacement).all()
        changed = output.clone()
        changed[:, 4:5] = replacement
        assert torch.equal(changed[:, :4], output[:, :4]) and torch.equal(changed[:, 5:], output[:, 5:])
        changes.append({'root': 85, 'row': 4, 'other_rows_bit_equal': True,
            'replacement_bit_equal': torch.equal(replacement, output[:, 4:5]),
            'norm_output_max_abs': float((replacement - output[:, 4:5]).abs().max())})
        return changed
    handles = []
    try:
        handles = [c.model.register_forward_pre_hook(before, with_kwargs=True), norm.register_forward_hook(replace)]
        samples, replay, _, records = synchronized_transfer(c.model, c.states, donor=donor, **c.keywords)
    finally:
        for handle in handles:
            handle.remove()
    assert captured and len(changes) == (0 if mode == 'native' else 1)
    assert not norm._forward_hooks and not c.model._forward_pre_hooks and not c.model._forward_hooks
    return samples, replay, records, captured, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    root = Path(__file__).resolve().parents[1]
    report, tensors = {'quality_passed': False, 'cases': [], 'entry_vectors': [], 'norm_comparisons': []}, {}
    try:
        c = load(args.out, (root / 'docs/研究计划 Denoising Speculative Decoding.md'), Path(__file__))
        norm = c.model.hf.model.layers[0].post_attention_layernorm
        c.paths['norm_implementation'] = Path(inspect.getsourcefile(type(norm)))
        c.binding['norm_implementation'] = file_hash(c.paths['norm_implementation'])
        report.update(version='q178-entry-phi-rms-20261014-v1', source_binding=c.binding,
            source_digest=c.native['source_digest'], model_digest=c.native['model_digest'],
            runtime=c.previous['runtime'], generation_budget=800, auxiliary_model_forwards=0,
            norm_class=type(norm).__name__, native_epsilon=norm.variance_epsilon)
        donors = {}
        for case in ('output_order0', 'output_order1'):
            state = c.old[f'baseline:{case}:85']['current_state']
            provider = OutputHiddenMaskProvider(2, c.model.embedding_table, order=0 if case.endswith('0') else 1)
            provider.anchor, provider.velocity = state['anchor'].cuda(), state['velocity'].cuda()
            proxies = state['proxies'].cuda()
            native = provider.masks_per_owner(proxies).cpu()
            assert torch.equal(native, c.old[f'baseline:{case}:85']['current_masks'])
            high = phi64(provider, proxies)
            donors[case] = {'native': native[0, 1], 'phi64': high}
            report['entry_vectors'].append({'case': case, 'bit_equal': torch.equal(high, native[0, 1]),
                'max_abs': float((high - native[0, 1]).abs().max()), 'L2': float((high - native[0, 1]).norm()),
                'native_scale_held': float(provider.scale)})
            tensors[case + ':native_entry'] = native[0, 1]
            tensors[case + ':phi64_entry'] = high
        for stage, mode in [('native', 'native'), ('phi64', 'native'),
                            ('norm_singleton', 'singleton'), ('norm64', 'reference64')]:
            for graft in (False, True):
                label = stage + ('_graft' if graft else '_self')
                donor_case = 'output_order0' if graft else 'output_order1'
                donor = donors[donor_case]['phi64' if stage == 'phi64' else 'native']
                samples, replay, records, captured, changes = run(c, donor, mode)
                if stage == 'native':
                    compare_control(c, samples, replay, graft)
                    with torch.no_grad():
                        for alternate in ('singleton', 'reference64'):
                            result = norm_row(norm, captured['inputs'], alternate)
                            report['norm_comparisons'].append({'case': label, 'mode': alternate,
                                'input_shape': list(captured['inputs'].shape), 'input_stride': list(captured['inputs'].stride()),
                                'bit_equal': torch.equal(result, captured['output'][:, 4:5]),
                                'max_abs': float((result - captured['output'][:, 4:5]).abs().max())})
                        if not graft:
                            x = captured['inputs']
                            report['packed_norm_profile'] = profile(lambda: norm.forward(x),
                                args.out / 'rms-packed.chrome.json', {'aten::pow', 'aten::mean', 'aten::rsqrt', 'aten::mul'})
                            report['singleton_norm_profile'] = profile(lambda: norm.forward(x[:, 4:5]),
                                args.out / 'rms-singleton.chrome.json', {'aten::pow', 'aten::mean', 'aten::rsqrt', 'aten::mul'})
                report['cases'].append({'label': label, 'stage': stage, 'graft': graft, 'replay': replay,
                    'canonical_replacements': records, 'norm_changes': changes,
                    'entry_vector_changed': not torch.equal(donor, donors[donor_case]['native'])})
                tensors.update({f'{label}:{p}': s for p, s in samples.items()})
                tensors[label + ':norm_capture'] = {k: v.cpu() for k, v in captured.items()}
        verify(c)
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [(c['label'], c['replay']['stats']['accepted']) for c in report['cases']]}))


if __name__ == '__main__':
    main()
