"""Strict-repeat and canonical committed-state controls for the fixed q178 graft.

    python tools/diagnose_q178_numerics.py --out .tmp/q178-numerical-<new-id>
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from diagnose_output_windows import cosine, file_hash, observe
from diagnose_proxy_transfer import transfer


def canonical_states(hidden, embedding_table, order):
    from spec_mtp.masks import OutputHiddenMaskProvider
    provider = OutputHiddenMaskProvider(2, embedding_table, order=order, beta=.9, beta_prime=.99)
    states = {}
    for position, value in enumerate(hidden):
        provider.observe_committed(value)
        states[position + 1] = {'anchor': provider.anchor.clone(), 'velocity': provider.velocity.clone()}
    return states


def synchronized_transfer(model, states, *args, **kwargs):
    import torch
    from spec_mtp import model as kv_cache
    from spec_mtp.masks import OutputHiddenMaskProvider
    original = OutputHiddenMaskProvider.masks_per_owner
    latest, records = {}, []

    def after(_module, _args, _kwargs, output):
        latest['cache'] = output.past_key_values

    def before(_module, args, kwargs):
        if kwargs.get('attention_mask_4d') is not None:
            positions = args[1] if len(args) > 1 else kwargs['position_ids']
            assert int(positions[0, 0]) == latest['provider_root']

    def synchronize(provider, proxies):
        root = kv_cache.cache_len(latest['cache'])
        latest['provider_root'] = root
        if root > kwargs['root']:
            proxy_copy = proxies.clone()
            target = states[root]
            previous = {'anchor': provider.anchor.clone(), 'velocity': provider.velocity.clone()}
            provider.anchor = target['anchor'].to(provider.anchor.device).clone()
            provider.velocity = target['velocity'].to(provider.velocity.device).clone()
            assert all(torch.equal(getattr(provider, k).cpu(), target[k]) for k in ('anchor', 'velocity'))
            result = original(provider, proxies)
            assert torch.equal(proxies, proxy_copy)
            records.append({'root_position': root, 'anchor_max_abs_replacement': float((previous['anchor'] - provider.anchor).abs().max()),
                            'velocity_max_abs_replacement': float((previous['velocity'] - provider.velocity).abs().max()),
                            'incoming_proxies_unchanged': True, 'canonical_state_bit_equal': True})
            return result
        return original(provider, proxies)

    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', synchronize):
            samples, replay, intervention = transfer(model, *args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert len(records) == sum(t['root_position'] > kwargs['root'] for t in replay['trace'])
    return samples, replay, intervention, records


def tensor_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tensor_equal(v, b[k]) for k, v in a.items())
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new; prior evidence is preserved')
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
    data = read(study / 'docs/execution_v1.json')
    selection = read(study / 'docs/window_selection_v1.json')['prompts'][3]
    assert selection['question_id'] == 178 and selection['intervention_root'] == 85
    paths = {'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'), 'matrix': study / 'docs/execution_v1.json',
        'selection': study / 'docs/window_selection_v1.json', 'snapshot': study / 'docs/modelscope_snapshot.json',
        'tool': Path(__file__), 'observer_helper': Path(__file__).with_name('diagnose_output_windows.py'),
        'transfer_helper': Path(__file__).with_name('diagnose_proxy_transfer.py'),
        'old_transfer_report': root / '.tmp/extension-transfer-20261014-v1/diagnosis.json',
        'old_transfer_tensor': root / '.tmp/extension-transfer-20261014-v1/selected_tensors.pt'}
    native = {r['case']: r for r in data['runs']}
    for case in ('output_order0', 'output_order1', 'AR'):
        for filename in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
            paths[case + ':' + filename] = study / 'runs' / native[case]['run_id'] / filename
        assert file_hash(paths[case + ':assets/decode_observations.json']) == native[case]['raw_sha256']
    old = read(paths['old_transfer_report'])
    assert old['quality_passed'] and file_hash(paths['old_transfer_tensor']) == old['raw_tensor_sha256']
    binding = {k + '_sha256': file_hash(p) for k, p in paths.items()}
    report = {'version': 'q178-numerical-history-20261014-v1', 'source_digest': data['source_digest'],
        'source_binding': binding, 'strict_runtime_only_override': True, 'generation_budget': 1100,
        'teacher_forward_budget': 1, 'teacher_positions': 112, 'baselines': [], 'cases': [],
        'repeat_checks': [], 'quality_passed': False}
    tensors = {}
    try:
        assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        snapshot = read(paths['snapshot'])
        for filename, metadata in snapshot['files'].items():
            assert file_hash(cache / filename) == metadata['sha256']
        torch.set_num_threads(1)
        apply_runtime(126, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': True,
            'cudnn_deterministic': True, 'cudnn_benchmark': False, 'matmul_allow_tf32': False, 'cudnn_allow_tf32': False}))
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True, warn_only=False)
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == data['model_digest']
        strict = runtime()
        report['runtime'] = strict
        report['deterministic_warn_only'] = torch.is_deterministic_algorithms_warn_only_enabled()
        report['cublas_workspace_config'] = os.environ['CUBLAS_WORKSPACE_CONFIG']
        assert strict['deterministic'] and not report['deterministic_warn_only']
        assert not strict['tf32_matmul'] and not strict['tf32_cudnn'] and strict['matmul_precision'] == 'highest'
        ar = native['AR']['raw']['evidence'][3]
        assert len(ar['output_ids']) == 112
        prompt = torch.tensor([ar['input_ids']], device=model.device)
        baseline, configs, inputs, refs = {}, {}, {}, {}
        targets = selection['targets']
        for case in ('output_order0', 'output_order1'):
            refs[case] = native[case]['raw']['evidence'][3]
            assert refs[case]['output_ids'] == ar['output_ids']
            cfg = yaml.safe_load(paths[case + ':config.yaml'].read_text(encoding='utf-8'))
            configs[case] = _convert(cfg['algorithm']['config']['eval'])
            inputs[case] = read(paths[case + ':assets/decode_inputs.json'])
            assert inputs[case]['prompts'][3]['input_ids'] == ar['input_ids']
            visited = {t['root_position'] for t in refs[case]['stats']['trace']}
            samples, replay = observe(model, prompt, _convert(native[case]['spec']), configs[case],
                inputs[case]['eos_ids'], refs[case], sorted(set(targets) & visited), seed=126)
            baseline[case] = samples
            tensors.update({f'baseline:{case}:{r}': s for r, s in samples.items()})
            report['baselines'].append({'case': case, 'replay': replay})
        tokens = torch.tensor([ar['output_ids']], device=model.device)
        out = model(model.embed(tokens), torch.arange(tokens.shape[1], device=model.device)[None], last_hidden_only=True)
        hidden = out.last_hidden_state[0].detach().cpu().clone()
        del out
        assert all(cosine(s['exact_root_hidden'], hidden[r]) >= .99999 for b in baseline.values() for r, s in b.items())
        states = {case: canonical_states(hidden, model.embedding_table, native[case]['spec']['output']['order'])
                  for case in ('output_order0', 'output_order1')}
        tensors['canonical_hidden'] = hidden
        tensors['canonical_states'] = states
        runs = [('strict_one_from_zero_1', 'output_order1', 'output_order0', False),
                ('strict_one_from_zero_2', 'output_order1', 'output_order0', False),
                ('strict_zero_from_one_1', 'output_order0', 'output_order1', False),
                ('strict_zero_from_one_2', 'output_order0', 'output_order1', False),
                ('strict_one_placebo', 'output_order1', 'output_order1', False),
                ('canonical_zero_self', 'output_order0', 'output_order0', True),
                ('canonical_one_self', 'output_order1', 'output_order1', True),
                ('canonical_one_from_zero', 'output_order1', 'output_order0', True),
                ('canonical_zero_from_one', 'output_order0', 'output_order1', True)]
        completed = {}
        for label, recipient, donor, canonical in runs:
            keyword = {'spec': _convert(native[recipient]['spec']), 'eval_cfg': configs[recipient],
                'eos': inputs[recipient]['eos_ids'], 'reference': refs[recipient],
                'donor': baseline[donor][85]['current_masks'][0, 1],
                'original_masks': baseline[recipient][85]['current_masks'], 'root': 85, 'targets': targets,
                'placebo': recipient == donor and not canonical, 'seed': 126}
            if canonical:
                samples, replay, intervention, changes = synchronized_transfer(model, states[recipient], prompt=prompt, **keyword)
            else:
                samples, replay, intervention = transfer(model, prompt, **keyword)
                changes = []
            completed[label] = samples
            tensors.update({f'{label}:{r}': s for r, s in samples.items()})
            tensors[label + ':intervention'] = intervention
            report['cases'].append({'label': label, 'recipient': recipient, 'donor': donor,
                'canonical_committed_state': canonical, 'replay': replay, 'canonical_replacements': changes})
        for first, second in [('strict_one_from_zero_1', 'strict_one_from_zero_2'), ('strict_zero_from_one_1', 'strict_zero_from_one_2')]:
            a = next(c for c in report['cases'] if c['label'] == first)
            b = next(c for c in report['cases'] if c['label'] == second)
            equal = tensor_equal(completed[first], completed[second])
            assert a['replay'] == b['replay'] and equal
            report['repeat_checks'].append({'first': first, 'second': second, 'all_selected_tensors_bit_equal': equal,
                                            'full_actual_semantic_trace_equal': True})
        assert all(file_hash(p) == binding[k + '_sha256'] for k, p in paths.items())
        assert model_digest(model) == data['model_digest'] and runtime() == strict
        assert manifest()['digest'] == data['source_digest']
        for filename, metadata in snapshot['files'].items():
            assert file_hash(cache / filename) == metadata['sha256']
        report.update(teacher_exact_hidden_consistency_passed=True, model_digest=data['model_digest'], quality_passed=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [{'label': c['label'], 'accepted': c['replay']['stats']['accepted']} for c in report['cases']]}))


if __name__ == '__main__':
    main()
