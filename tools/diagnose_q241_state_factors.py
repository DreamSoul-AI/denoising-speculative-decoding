"""Selective committed anchor/velocity controls on the fixed q241 counterexample.

    python tools/diagnose_q241_state_factors.py --out .tmp/q241-state-factors-<new-id>
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import file_hash
from diagnose_q178_numerics import tensor_equal
from diagnose_proxy_transfer import transfer


def selective_transfer(model, states, coordinate, **kwargs):
    import torch
    from unittest.mock import patch
    from spec_mtp import model as kv_cache
    from spec_mtp.masks import OutputHiddenMaskProvider
    assert coordinate in ('anchor', 'velocity')
    original = OutputHiddenMaskProvider.masks_per_owner
    latest, records = {}, []
    other = 'velocity' if coordinate == 'anchor' else 'anchor'

    def after(_module, _args, _kwargs, output):
        latest['cache'] = output.past_key_values

    def before(_module, args, kw):
        if kw.get('attention_mask_4d') is not None:
            positions = args[1] if len(args) > 1 else kw['position_ids']
            assert int(positions[0, 0]) == latest['root']

    def synchronize(provider, proxies):
        root = kv_cache.cache_len(latest['cache'])
        latest['root'] = root
        if root > kwargs['root']:
            previous = getattr(provider, coordinate).clone()
            untouched, incoming = getattr(provider, other).clone(), proxies.clone()
            target = states[root][coordinate]
            setattr(provider, coordinate, target.to(previous.device).clone())
            result = original(provider, proxies)
            assert torch.equal(getattr(provider, coordinate).cpu(), target)
            assert torch.equal(getattr(provider, other), untouched) and torch.equal(proxies, incoming)
            records.append({'root_position': root, 'coordinate': coordinate,
                'max_abs_replacement': float((previous - getattr(provider, coordinate)).abs().max()),
                'other_state_bit_equal': True, 'incoming_proxies_unchanged': True,
                'selected_state_bit_equal': True})
            return result
        return original(provider, proxies)

    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                   model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', synchronize):
            samples, replay, intervention = transfer(model, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert len(records) == sum(t['root_position'] > kwargs['root'] for t in replay['trace'])
    return samples, replay, intervention, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    import yaml
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM
    root = Path(__file__).resolve().parents[1]
    study = root / 'studies/output_space_cross_task'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    paths = {'tool': Path(__file__), 'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
        'native': study / 'docs/execution_v1.json', 'snapshot': study / 'docs/modelscope_snapshot.json',
        'batch_hypotheses': study / 'docs/hypotheses_v1.json',
        'selection': study / 'docs/window_selection_v1.json',
        'old_report': root / '.tmp/cross-task-transfer-20261014-v1/diagnosis.json',
        'old_tensor': root / '.tmp/cross-task-transfer-20261014-v1/selected_tensors.pt',
        'canonical_report': root / '.tmp/q241-canonical-20261014-v1/diagnosis.json',
        'canonical_tensor': root / '.tmp/q241-canonical-20261014-v1/selected_tensors.pt'}
    for name in ('diagnose_output_windows.py', 'diagnose_proxy_transfer.py', 'diagnose_q178_numerics.py'):
        paths[name] = root / 'tools' / name
    data, old_report = read(paths['native']), read(paths['old_report'])
    assert data['study_quality_passed'] and old_report['quality_passed'] and data['question_ids'][0] == 241
    selection = read(paths['selection'])['prompts'][0]
    assert selection['intervention_root'] == 716 and old_report['prompts'][0]['question_id'] == 241
    rows = {r['case']: r for r in data['runs']}
    folder = study / 'runs' / rows['output_order1']['run_id']
    for name in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
        paths[name] = folder / name
    binding = {k: file_hash(p) for k, p in paths.items()}
    assert binding['old_tensor'] == old_report['raw_tensor_sha256']
    report, saved = {'version': 'q241-state-factors-20261015-v1', 'source_binding': binding,
        'cases': [], 'quality_passed': False, 'generation_budget': 600, 'teacher_forward_budget': 0}, {}
    try:
        old = torch.load(paths['old_tensor'], weights_only=True, map_location='cpu', mmap=True)
        assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
        snapshot = read(paths['snapshot'])
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
        torch.set_num_threads(1)
        apply_runtime(123, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
            torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
        ar = rows['AR']['raw']['evidence'][0]
        reference = rows['output_order1']['raw']['evidence'][0]
        assert len(ar['output_ids']) == 792 and reference['output_ids'] == ar['output_ids']
        prompt = torch.tensor([ar['input_ids']], device=model.device)
        prior = read(paths['canonical_report'])
        assert prior['quality_passed'] and binding['canonical_tensor'] == prior['raw_tensor_sha256']
        canonical = torch.load(paths['canonical_tensor'], weights_only=True, map_location='cpu', mmap=True)
        states = canonical['canonical_states']
        cfg = yaml.safe_load(paths['config.yaml'].read_text(encoding='utf-8'))
        inputs = read(paths['assets/decode_inputs.json'])
        assert inputs['prompts'][0]['input_ids'] == ar['input_ids']
        keywords = dict(prompt=prompt, spec=_convert(rows['output_order1']['spec']),
            eval_cfg=_convert(cfg['algorithm']['config']['eval']), eos=inputs['eos_ids'],
            reference=reference, original_masks=old['q241:baseline:output_order1:716']['current_masks'],
            root=716, targets=selection['targets'], placebo=False, seed=123)
        for coordinate in ('native', 'anchor', 'velocity'):
            for graft in (False, True):
                label = coordinate + ('_graft' if graft else '_self')
                donor_case = 'output_order0' if graft else 'output_order1'
                donor = old[f'q241:baseline:{donor_case}:716']['current_masks'][0, 1]
                if coordinate != 'native':
                    samples, replay, _, records = selective_transfer(model, states, coordinate, donor=donor, **keywords)
                else:
                    samples, replay, _ = transfer(model, donor=donor, **keywords)
                    records = []
                    old_label = 'output_order1_from_' + donor_case
                    previous = next(c['replay'] for c in old_report['prompts'][0]['cases'] if c['label'] == old_label)
                    assert replay['trace'] == previous['trace'] and replay['stats'] == previous['stats']
                    assert all(tensor_equal(s, old[f'q241:{old_label}:{p}']) for p, s in samples.items())
                saved.update({f'{label}:{p}': s for p, s in samples.items()})
                report['cases'].append({'label': label, 'coordinate': coordinate, 'graft': graft,
                    'replay': replay, 'canonical_replacements': records})
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
        assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
        assert manifest()['digest'] == data['source_digest']
        report.update(source_digest=data['source_digest'], model_digest=data['model_digest'],
            runtime=runtime(), quality_passed=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if saved:
            torch.save(saved, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [(c['label'], c['replay']['stats']['accepted']) for c in report['cases']]}))


if __name__ == '__main__':
    main()
