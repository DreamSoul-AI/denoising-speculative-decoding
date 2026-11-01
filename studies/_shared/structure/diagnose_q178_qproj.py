"""Fixed layer0 q_proj row packing/rounding experiment with cloned-cache replays.

    python studies/_shared/structure/diagnose_q178_qproj.py --out .tmp/q178-qproj-<new-id>
"""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from studies.output_space_extension.kernel_common import load, verify, compare_control, profile
from diagnose_output_windows import file_hash
from diagnose_q178_numerics import synchronized_transfer


def linear_row(module, inputs, row, mode):
    import torch.nn.functional as F
    x = inputs[:, row:row + 1]
    if mode == 'singleton':
        return F.linear(x, module.weight, module.bias)
    assert mode == 'reference64'
    bias = None if module.bias is None else module.bias.cpu().double()
    return F.linear(x.cpu().double(), module.weight.cpu().double(), bias).to(inputs.device, inputs.dtype)


def run(c, graft, mode, capture=False):
    import torch
    active, frames, changes = {}, {}, []
    projection = c.model.hf.model.layers[0].self_attn.q_proj

    def before(_module, args, kwargs):
        positions = args[1] if len(args) > 1 else kwargs['position_ids']
        current = None if kwargs.get('attention_mask_4d') is None else int(positions[0, 0])
        active['root'] = current
        if capture and current in (85, 86):
            cache = kwargs['past_key_values']
            cloned = copy.deepcopy(cache)
            assert cloned.get_seq_length() == cache.get_seq_length() == current
            for original, saved in zip(cache.layers, cloned.layers):
                for key in ('keys', 'values'):
                    a, b = getattr(original, key), getattr(saved, key)
                    assert a.data_ptr() != b.data_ptr() and torch.equal(a, b) and a.stride() == b.stride()
            frames[current] = {'inputs': (args[0] if args else kwargs['inputs_embeds']).detach().clone(),
                'positions': positions.detach().clone(), 'attention': kwargs['attention_mask_4d'].detach().clone(),
                'cache': cloned}

    def projection_after(module, args, output):
        current = active['root']
        inputs = args[0]
        if capture and current in frames:
            frames[current].update(q_input=inputs.detach().clone(), q_output=output.detach().clone())
        if current != 85 or mode == 'native':
            return output
        assert inputs.shape[1] == output.shape[1] == 9
        replacement = linear_row(module, inputs, 4, mode)
        assert replacement.shape == output[:, 4:5].shape and torch.isfinite(replacement).all()
        changed = output.clone()
        changed[:, 4:5] = replacement
        assert torch.equal(changed[:, :4], output[:, :4]) and torch.equal(changed[:, 5:], output[:, 5:])
        changes.append({'root': current, 'row': 4, 'other_rows_bit_equal': True,
                        'q_projection_max_abs': float((replacement - output[:, 4:5]).abs().max()),
                        'replacement_bit_equal': torch.equal(replacement, output[:, 4:5])})
        return changed

    def after(_module, _args, _kwargs, output):
        if capture and active['root'] in frames:
            frames[active['root']].update(logits=output.logits.detach().clone(),
                                         hidden=output.last_hidden_state.detach().clone())

    handles = []
    try:
        handles = [c.model.register_forward_pre_hook(before, with_kwargs=True),
            projection.register_forward_hook(projection_after),
            c.model.register_forward_hook(after, with_kwargs=True)]
        donor_case = 'output_order0' if graft else 'output_order1'
        donor = c.old[f'baseline:{donor_case}:85']['current_masks'][0, 1]
        samples, replay, _, records = synchronized_transfer(c.model, c.states, donor=donor, **c.keywords)
    finally:
        for handle in handles:
            handle.remove()
    assert not c.model._forward_hooks and not c.model._forward_pre_hooks and not projection._forward_hooks
    assert len(changes) == (0 if mode == 'native' else 1)
    if capture:
        assert set(frames) == {85, 86}
    return samples, replay, records, frames, changes


def exact_block(model, frame):
    import torch
    cache = copy.deepcopy(frame['cache'])
    before = [(layer.keys.clone(), layer.values.clone()) for layer in frame['cache'].layers]
    output = model(frame['inputs'], frame['positions'], attention_mask_4d=frame['attention'],
                   past_key_values=cache, last_hidden_only=True)
    assert torch.equal(output.logits, frame['logits']) and torch.equal(output.last_hidden_state, frame['hidden'])
    assert all(torch.equal(layer.keys, k) and torch.equal(layer.values, v)
               for layer, (k, v) in zip(frame['cache'].layers, before))
    return {'logits_bit_equal': True, 'final_hidden_bit_equal': True, 'saved_cache_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    root = Path(__file__).resolve().parents[3]
    report, tensors = {'quality_passed': False, 'cases': [], 'block_replays': [], 'linear_comparisons': []}, {}
    try:
        c = load(args.out, (root / 'docs/研究计划 Denoising Speculative Decoding.md'), Path(__file__))
        # Bind the original fixed target declaration as well as its execution addendum.
        original_plan = (root / 'docs/研究计划 Denoising Speculative Decoding.md')
        c.paths['original_kernel_plan'] = original_plan
        c.binding['original_kernel_plan'] = file_hash(original_plan)
        report.update(version='q178-fixed-qproj-20261014-v1', source_binding=c.binding,
            source_digest=c.native['source_digest'], model_digest=c.native['model_digest'],
            runtime=c.previous['runtime'], generation_budget=600, auxiliary_model_forwards=4)
        projection = c.model.hf.model.layers[0].self_attn.q_proj
        frames_by_case = {}
        for graft in (False, True):
            label = 'native_graft' if graft else 'native_self'
            samples, replay, records, frames, changes = run(c, graft, 'native', capture=True)
            compare_control(c, samples, replay, graft)
            report['cases'].append({'label': label, 'mode': 'native', 'graft': graft,
                'replay': replay, 'canonical_replacements': records, 'q_projection_changes': changes})
            tensors.update({f'{label}:{p}': s for p, s in samples.items()})
            frames_by_case[label] = frames
            for position, frame in frames.items():
                check = exact_block(c.model, frame)
                report['block_replays'].append({'case': label, 'root': position, **check})
                native_row = frame['q_output'][:, 4:5]
                singleton = linear_row(projection, frame['q_input'], 4, 'singleton')
                reference = linear_row(projection, frame['q_input'], 4, 'reference64')
                report['linear_comparisons'].append({'case': label, 'root': position,
                    'input_shape': list(frame['q_input'].shape), 'input_stride': list(frame['q_input'].stride()),
                    'output_shape': list(frame['q_output'].shape), 'dtype': str(frame['q_input'].dtype),
                    'singleton_native_max_abs': float((singleton - native_row).abs().max()),
                    'reference_native_max_abs': float((reference - native_row).abs().max()),
                    'singleton_reference_max_abs': float((singleton - reference).abs().max()),
                    'singleton_native_bit_equal': torch.equal(singleton, native_row)})
                tensors[f'{label}:block:{position}'] = {
                    **{k: v.cpu() for k, v in frame.items() if isinstance(v, torch.Tensor)},
                    'cache_layers': [{'keys': layer.keys.cpu(), 'values': layer.values.cpu()}
                                     for layer in frame['cache'].layers]}
                if position == 85:
                    x = frame['q_input']
                    report.setdefault('profiles', []).append({'case': label,
                        'packed': profile(lambda: projection(x), args.out / (label + '-packed.chrome.json'),
                                          {'aten::mm', 'aten::addmm'}),
                        'singleton': profile(lambda: linear_row(projection, x, 4, 'singleton'),
                            args.out / (label + '-singleton.chrome.json'), {'aten::mm', 'aten::addmm'})})
        for mode in ('singleton', 'reference64'):
            for graft in (False, True):
                label = mode + ('_graft' if graft else '_self')
                samples, replay, records, _, changes = run(c, graft, mode)
                report['cases'].append({'label': label, 'mode': mode, 'graft': graft,
                    'replay': replay, 'canonical_replacements': records, 'q_projection_changes': changes})
                tensors.update({f'{label}:{p}': s for p, s in samples.items()})
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
