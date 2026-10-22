"""Actual source717 owner1 state-proxy controls and conditional stochastic acceptance.

    python tools/diagnose_q241_owner1.py --out .tmp/q241-owner1-<new-id>
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import file_hash, margin
from diagnose_q178_numerics import synchronized_transfer, tensor_equal
from diagnose_q241_local_factors import fixed_masks, replay_fixed_block
from diagnose_sampling_paths import conditional_acceptance



def capture_frames(model, states, kwargs, capture):
    import copy
    frames, active = {}, {}
    def before(_module, args, kw):
        positions = args[1] if len(args) > 1 else kw['position_ids']
        root = None if kw.get('attention_mask_4d') is None else int(positions[0, 0])
        active['root'] = root
        if capture and root == 717:
            cache = kw['past_key_values']
            assert cache.get_seq_length() == root
            frames[root] = {'inputs': args[0].detach().clone(), 'positions': positions.detach().clone(),
                'attention': kw['attention_mask_4d'].detach().clone(), 'cache': copy.deepcopy(cache)}
    def after(_module, _args, _kw, output):
        if capture and active['root'] in frames:
            frames[active['root']].update(logits=output.logits.detach().clone(),
                hidden=output.last_hidden_state.detach().clone())
    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                   model.register_forward_hook(after, with_kwargs=True)]
        samples, replay, _, records = synchronized_transfer(model, states, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert not model._forward_pre_hooks and not model._forward_hooks and not model.hf.model.norm._forward_hooks
    assert set(frames) == ({717} if capture else set())
    return samples, replay, records, frames


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
        'canonical_tensor': root / '.tmp/q241-canonical-20261014-v1/selected_tensors.pt',
        'local_report': root / '.tmp/q241-local-factors-20261015-v1/diagnosis.json',
        'local_tensor': root / '.tmp/q241-local-factors-20261015-v1/selected_tensors.pt',
        'local_helper': root / 'tools/diagnose_q241_local_factors.py',
        'sampling_helper': root / 'tools/diagnose_sampling_paths.py'}
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
    report, saved = {'version': 'q241-owner1-20261015-v1', 'source_binding': binding,
        'cases': [], 'quality_passed': False, 'generation_budget': 200, 'block_forward_budget': 12, 'teacher_forward_budget': 0}, {}
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
        previous = torch.load(paths['canonical_tensor'], weights_only=True, map_location='cpu', mmap=True)
        states = previous['canonical_states']
        cfg = yaml.safe_load(paths['config.yaml'].read_text(encoding='utf-8'))
        inputs = read(paths['assets/decode_inputs.json'])
        assert inputs['prompts'][0]['input_ids'] == ar['input_ids']
        keywords = dict(prompt=prompt, spec=_convert(rows['output_order1']['spec']),
            eval_cfg=_convert(cfg['algorithm']['config']['eval']), eos=inputs['eos_ids'],
            reference=reference, original_masks=old['q241:baseline:output_order1:716']['current_masks'],
            root=716, targets=selection['targets'], placebo=False, seed=123)
        frames = {}
        for graft in (False, True):
            label = 'canonical_graft' if graft else 'canonical_self'
            donor_case = 'output_order0' if graft else 'output_order1'
            kwargs = dict(keywords, donor=old[f'q241:baseline:{donor_case}:716']['current_masks'][0, 1])
            samples, replay, records, captured = capture_frames(model, states, kwargs, capture=not graft)
            expected = next(c['replay'] for c in prior['cases'] if c['label'] == label)
            assert replay == expected
            assert all(tensor_equal(s, previous[f'{label}:{p}']) for p, s in samples.items())
            report['cases'].append({'label': label, 'replay': replay, 'canonical_replacements': records})
            if not graft:
                frames = captured
        from spec_mtp.masks import OutputHiddenMaskProvider
        provider = OutputHiddenMaskProvider(2, model.embedding_table, order=1)
        report['blocks'], report['steps'] = [], []
        for position in (716, 717, 719, 720):
            steps = []
            for label in ('canonical_self', 'canonical_graft'):
                sample = previous[f'{label}:{position}']
                trace = next(t for t in next(c['replay'] for c in prior['cases'] if c['label'] == label)['trace']
                             if t['root_position'] == position)
                steps.append({'label': label, 'trace': trace, 'source_root': sample['source_root'],
                    'source_owner': sample['source_owner'], 'seed_positions': sample['seed_positions'],
                    'candidate_ids': sample['seed_logits'].argmax(-1).tolist(),
                    'first': margin(sample['seed_logits'][0], ar['output_ids'][position + 1])})
            report['steps'].append({'root': position, 'cases': steps})
        for position, frame in frames.items():
            self_sample, graft_sample = (previous[f'{label}:{position}'] for label in ('canonical_self', 'canonical_graft'))
            assert all(torch.equal(self_sample['current_state'][k], graft_sample['current_state'][k])
                       for k in ('anchor', 'velocity'))
            fixed_state = {k: self_sample['current_state'][k] for k in ('anchor', 'velocity')}
            native_state = {k: previous[f'native_self:{position}']['current_state'][k] for k in ('anchor', 'velocity')}
            control_masks = fixed_masks(provider, fixed_state, self_sample['current_state']['proxies'])
            assert torch.equal(control_masks.cpu(), self_sample['current_masks'])
            assert torch.equal(control_masks.reshape(1, 6, -1), frame['inputs'][:, 3:])
            expected_cache = [(layer.keys.clone(), layer.values.clone()) for layer in frame['cache'].layers]
            assert position == 717
            assert previous['canonical_self:719']['source_owner'] == 1
            assert previous['canonical_self:719']['source_root'] == position
            target = ar['output_ids'][position + 3]
            for state_changed in (False, True):
                for proxy_changed in (False, True):
                    state = native_state if state_changed else fixed_state
                    proxies = graft_sample['current_state']['proxies'] if proxy_changed else self_sample['current_state']['proxies']
                    masks = fixed_masks(provider, state, proxies)
                    name = f'r{position}:S{int(state_changed)}P{int(proxy_changed)}'
                    values = []
                    for repeat in range(3):
                        logits, hidden = replay_fixed_block(model, frame, masks)
                        if not state_changed and not proxy_changed:
                            assert torch.equal(logits, frame['logits'].cpu()) and torch.equal(hidden, frame['hidden'].cpu())
                        if values:
                            assert torch.equal(logits, values[0][0]) and torch.equal(hidden, values[0][1])
                        values.append((logits, hidden))
                        report['blocks'].append({'label': name, 'root': position, 'repeat': repeat,
                            'state_changed': state_changed, 'proxy_changed': proxy_changed,
                            'state_anchor_max_abs': float((state['anchor'] - fixed_state['anchor']).abs().max()),
                            'state_velocity_max_abs': float((state['velocity'] - fixed_state['velocity']).abs().max()),
                            'proxy_max_abs': float((proxies - self_sample['current_state']['proxies']).abs().max()),
                            'mask_max_abs': float((masks - control_masks).abs().max()),
                            'verification_logits_bit_equal': True, 'verification_hidden_bit_equal': True,
                            'first_seed': margin(logits[0, 5], target),
                            'second_seed': margin(logits[0, 6], ar['output_ids'][position + 4]),
                            'fixed_root_node_embeddings': True, 'fresh_identical_KV': True})
                    saved[name] = {'masks': masks.cpu(), 'logits': values[0][0], 'hidden': values[0][1]}
            assert all(torch.equal(layer.keys, k) and torch.equal(layer.values, v)
                       for layer, (k, v) in zip(frame['cache'].layers, expected_cache))
            saved[f'frame:{position}'] = {k: v.cpu() for k, v in frame.items() if k != 'cache'}
            saved[f'frame:{position}']['KV'] = [(k.cpu(), v.cpu()) for k, v in expected_cache]
        local_report = read(paths['local_report'])
        assert local_report['quality_passed'] and binding['local_tensor'] == local_report['raw_tensor_sha256']
        local = torch.load(paths['local_tensor'], weights_only=True, map_location='cpu', mmap=True)
        candidates = {f'S{s}P{p}': int(saved[f'r717:S{s}P{p}']['logits'][0, 5].argmax())
                      for s in (0, 1) for p in (0, 1)}
        report['conditional'] = [conditional_acceptance(local['frame:719']['logits'][0, 0].cuda(),
            candidates, temperature, range(200, 264)) for temperature in (.8, 1.)]
        report['owner'] = 1
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
