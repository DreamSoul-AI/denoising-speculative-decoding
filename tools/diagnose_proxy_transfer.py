"""One predeclared q173 root83 owner0 slot1 transfer; unchanged SpecDecoder.

    python tools/diagnose_proxy_transfer.py --study studies/output_space_small_matrix --diagnostic .tmp/output-windows-20261014-v1 --out .tmp/output-transfer-<new-id>
"""

from spec_mtp.masks import build_mask_provider

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from diagnose_output_windows import SEMANTIC_KEYS, TRACE_KEYS, cosine, file_hash, margin


def transfer(model, prompt, spec, eval_cfg, eos, reference, donor, original_masks,
             root, targets, placebo=False, seed=124):
    import torch
    from spec_mtp.decode import SpecDecoder
    from spec_mtp.masks import OutputHiddenMaskProvider

    assert spec['num_masks'] == 2 and spec['tree']['branches'] == [1, 1] and not spec['tree']['pruning']
    expected = reference['stats']['trace']
    intervention_call = next(i + 1 for i, t in enumerate(expected) if t['root_position'] == root)
    active, state, previous, samples, intervention = {}, {}, {}, {}, {}
    calls = provider_calls = 0
    original = OutputHiddenMaskProvider.masks_per_owner

    def provider_patch(provider, proxies):
        nonlocal provider_calls
        provider_calls += 1
        masks = original(provider, proxies)
        state.update(anchor=provider.anchor.detach().clone(), velocity=provider.velocity.detach().clone(),
                     proxies=proxies.detach().clone())
        if provider_calls == intervention_call:
            assert torch.equal(masks.cpu(), original_masks), 'original intervention masks differ'
            replacement = donor.to(device=masks.device, dtype=masks.dtype)
            assert torch.isfinite(replacement).all() and replacement.norm() > 0
            changed = masks.clone()
            changed[0, 1] = replacement
            untouched = torch.ones(masks.shape[:2], dtype=torch.bool, device=masks.device)
            untouched[0, 1] = False
            assert torch.equal(changed[untouched], masks[untouched])
            assert torch.equal(changed[0, 1], replacement)
            intervention.update(provider_call=provider_calls, unchanged_coordinates=True,
                                original_vector=masks[0, 1].detach().cpu().clone(),
                                donor_vector=changed[0, 1].detach().cpu().clone())
            return changed
        return masks

    def before(_module, args, kwargs):
        nonlocal calls
        calls += 1
        embeds = args[0] if args else kwargs['inputs_embeds']
        positions = args[1] if len(args) > 1 else kwargs['position_ids']
        prefill = kwargs.get('attention_mask_4d') is None
        active.update(embeds=embeds[0], positions=positions[0], prefill=prefill)
        if prefill:
            return
        current = int(positions[0, 0])
        if provider_calls == intervention_call:
            assert current == root, 'intervention call reached a different root'
            intervention['actual_root'] = current
        if current not in targets:
            return
        source_root = previous['root']
        owner = None if source_root is None else current - source_root - 1
        assert owner is None or owner in (0, 1, 2)
        rows = [0, 1] if owner is None else [owner * 2, owner * 2 + 1]
        seed_positions = previous['positions'][rows]
        assert seed_positions.tolist() == [current, current + 1]
        seed_logits = previous['logits'][rows]
        assert torch.equal(model.embed(seed_logits.argmax(dim=-1)[None])[0], embeds[0, 1:3])
        samples[current] = {'source_root': source_root, 'source_owner': owner,
                            'seed_positions': seed_positions.cpu().tolist(),
                            'seed_logits': seed_logits.cpu().clone(),
                            'seed_hidden': previous['hidden'][rows].cpu().clone(),
                            'seed_mask_inputs': previous['embeds'][rows].cpu().clone(),
                            'current_masks': embeds[0, 3:].reshape(3, 2, -1).detach().cpu().clone(),
                            'current_state': {k: v.cpu().clone() for k, v in state.items()}}

    def norm_hook(_module, _args, out):
        active['hidden'] = out[0]

    def after(_module, _args, _kwargs, out):
        embeds, positions, hidden = active['embeds'], active['positions'], active['hidden']
        current = None if active['prefill'] else int(positions[0])
        if current in samples:
            samples[current]['current_mask_hidden'] = hidden[3:].detach().cpu().reshape(3, 2, -1).clone()
            samples[current]['exact_root_hidden'] = hidden[0].detach().cpu().clone()
        # Keep only the immediately preceding frame; actual root progress selects its owner.
        rows = list(range(prompt.shape[1], prompt.shape[1] + 2)) if active['prefill'] else list(range(3, 9))
        previous.update(root=current, logits=out.logits[0, rows].detach().clone(),
                        hidden=hidden[rows].detach().clone(), embeds=embeds[rows].detach().clone(),
                        positions=positions[rows].detach().clone())

    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                   model.hf.model.norm.register_forward_hook(norm_hook),
                   model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', provider_patch):
            output, stats = SpecDecoder(model, spec, eval_cfg, eos_ids=eos,
                                       generator=torch.Generator(device=model.device).manual_seed(seed), mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert intervention['actual_root'] == root and calls == 1 + stats['decode_calls']
    assert output[0].tolist() == reference['output_ids'] and not stats['hit_context_limit']
    trace = [{k: t[k] for k in TRACE_KEYS} for t in stats['trace']]
    baseline = [{k: t[k] for k in TRACE_KEYS} for t in expected]
    assert [t for t in trace if t['root_position'] <= root] == [t for t in baseline if t['root_position'] <= root]
    if placebo:
        assert trace == baseline and all(stats[k] == reference['stats'][k] for k in SEMANTIC_KEYS)
    by_root = {t['root_position']: i for i, t in enumerate(trace)}
    for position, sample in samples.items():
        index = by_root[position]
        if index:
            prior = trace[index - 1]
            assert sample['source_root'] == prior['root_position']
            assert sample['source_owner'] == (prior['accepted_nodes'][-1] if prior['accepted_nodes'] else 0)
    assert set(samples) == set(targets) & set(by_root)
    return samples, {'output_equal_independent_AR': True, 'pre_intervention_trace_equal': True,
                     'placebo_full_semantic_equal': trace == baseline if placebo else None,
                     'observer_forward_calls': calls, 'modified_provider_calls': 1,
                     'intervention_root': root, 'only_selected_vector_replaced': True,
                     'replacement_equal_donor': True, 'visited_targets': sorted(samples),
                     'unvisited_targets': sorted(set(targets) - set(samples)),
                     'stats': {k: stats[k] for k in SEMANTIC_KEYS}, 'trace': trace}, intervention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new; prior evidence is preserved')
    args.out.mkdir(parents=True)
    import torch
    import yaml
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM

    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    study = args.study.resolve()
    matrix = study / 'docs/execution_v1.json'
    data = read(matrix)
    observer = read(args.diagnostic / 'diagnosis.json')
    tensor_path = args.diagnostic / 'selected_tensors.pt'
    assert observer['quality_passed'] and data['study_quality_passed']
    assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
    assert file_hash(matrix) == observer['source_binding']['matrix_sha256']
    assert file_hash(tensor_path) == observer['raw_tensor_sha256'] == '47166a3b41e0c00ed39110593fe7f83b20323c2e3714fe0355f2a43aaa9dc14c'
    paths = {'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'), 'matrix': matrix, 'tool': Path(__file__),
             'helper': Path(__file__).with_name('diagnose_output_windows.py'),
             'window_plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
             'observer': args.diagnostic / 'diagnosis.json', 'tensor': tensor_path,
             'snapshot': study / 'docs/modelscope_snapshot.json'}
    rows = {r['case']: r for r in data['runs']}
    for case in ('output_order0', 'output_order1'):
        folder = study / 'runs' / rows[case]['run_id']
        for filename in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
            paths[case + ':' + filename] = folder / filename
        assert file_hash(folder / 'assets/decode_observations.json') == rows[case]['raw_sha256']
    binding = {k + '_sha256': file_hash(p) for k, p in paths.items()}
    report = {'version': 'output-space-q173-transfer-20261014-v1', 'question_id': 173,
              'kind': 'single earlier slot vector transfer, unchanged decoder',
              'source_digest': data['source_digest'], 'source_binding': binding,
              'intervention': {'root': 83, 'owner': 0, 'slot': 1}, 'targets': [83, 84, 86, 94, 97],
              'generation_budget': 300, 'auxiliary_forwards': 0, 'cases': [], 'quality_passed': False}
    tensors = {}
    try:
        raw = torch.load(tensor_path, weights_only=True)
        masks = {c: raw[c + ':83']['current_masks'] for c in ('output_order0', 'output_order1')}
        assert torch.equal(masks['output_order0'][0, 0], masks['output_order1'][0, 0])
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        snapshot = read(paths['snapshot'])
        for name, metadata in snapshot['files'].items():
            assert file_hash(cache / name) == metadata['sha256']
        torch.set_num_threads(1)
        apply_runtime(124, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
        ar = rows['AR']['raw']['evidence'][1]
        prompt = torch.tensor([ar['input_ids']], device=model.device)
        for recipient, donor in [('output_order1', 'output_order0'), ('output_order0', 'output_order1'),
                                 ('output_order1', 'output_order1')]:
            label = recipient + '_from_' + donor
            row = rows[recipient]
            inputs = read(paths[recipient + ':assets/decode_inputs.json'])
            cfg = yaml.safe_load(paths[recipient + ':config.yaml'].read_text(encoding='utf-8'))
            assert inputs['prompts'][1]['input_ids'] == ar['input_ids']
            reference = row['raw']['evidence'][1]
            assert reference['output_ids'] == ar['output_ids']
            samples, replay, intervention = transfer(model, prompt, _convert(row['spec']),
                _convert(cfg['algorithm']['config']['eval']), inputs['eos_ids'], reference,
                masks[donor][0, 1], masks[recipient], 83, report['targets'], placebo=recipient == donor)
            windows = []
            entry = {'label': label, 'recipient': recipient, 'donor': donor, 'replay': replay, 'windows': windows}
            report['cases'].append(entry)
            tensors[label + ':intervention'] = intervention
            for root, sample in sorted(samples.items()):
                tensors[label + ':' + str(root)] = sample
                windows.append({'root_position': root, 'source_block_root': sample['source_root'],
                    'source_owner': sample['source_owner'], 'seed_positions': sample['seed_positions'],
                    'first_slot': margin(sample['seed_logits'][0], ar['output_ids'][root + 1]),
                    'proxy_norms': sample['current_state']['proxies'].norm(dim=-1).tolist(),
                    'velocity_norm': float(sample['current_state']['velocity'].norm()),
                    'seed_mask_norm': float(sample['seed_mask_inputs'][0].norm()),
                    'seed_vs_original_cosine': (cosine(sample['seed_mask_inputs'][0], raw[recipient + ':' + str(root)]['seed_mask_inputs'][0])
                                                if root in (83, 94, 97) else None)})
        assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
        assert manifest()['digest'] == data['source_digest']
        assert all(file_hash(p) == binding[k + '_sha256'] for k, p in paths.items())
        for name, metadata in snapshot['files'].items():
            assert file_hash(cache / name) == metadata['sha256']
        report.update(model_digest=data['model_digest'], runtime=runtime(), quality_passed=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [{ 'label': c['label'], 'stats': c['replay']['stats']} for c in report['cases']]}))


if __name__ == '__main__':
    main()
