"""Read-only BF16 4B mask/hidden capture after non-preemptive admission.

    python tools/capture_4b_geometry.py --study studies/qwen3_4b_bf16_five_methods --out .tmp/4b-geometry-capture-v1
"""

from spec_mtp.masks import build_mask_provider
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch


def admission(initial=True):
    """Observe available capacity; this is not an atomic cross-agent reservation."""
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert len(gpu) == 3, 'Only the declared single GPU is supported'
    command = ("Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | "
               'Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress')
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True)
    processes = json.loads(raw) if raw.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    foreign = [p for p in processes if p['ProcessId'] != os.getpid()]
    cpu_ci = [p for p in foreign if any(name in (p['CommandLine'] or '') for name in
                                      ('watch-cleanup.py', 'watch-third.py', 'release-ci.py'))
              or ('RPipe' in (p['CommandLine'] or '') and 'and not gpu and not slow' in (p['CommandLine'] or ''))]
    # Exact inspected scripts only: downloader and container build without --gpus/devices.
    cpu_scripts = {'download_layers.py': '10689bac808d0e5fa1958d306508f459d3c4637dc92a51bf5b6c1bb915545026',
                   'build_official_container_v2.py': 'b22cbe9814bcfcf85e9e6caade2a60a2258d2bb83f5e1d856124753bf8976e44'}
    verified_cpu = []
    for p in foreign:
        command = p['CommandLine'] or ''
        for name, digest in cpu_scripts.items():
            script = Path('D:/GitHub/DreamSoul/quantization/.tmp/paroquant-reference') / name
            if ('quantization\\.tmp\\paroquant-paper-env\\python.exe' in command
                    and f'.tmp/paroquant-reference/{name}' in command and script.is_file()
                    and hashlib.sha256(script.read_bytes()).hexdigest() == digest):
                verified_cpu.append(p)
    # Windows venv launchers can briefly expose a child with no command line.
    for _ in foreign:
        known_ids = {p['ProcessId'] for p in cpu_ci + verified_cpu}
        children = [p for p in foreign if p not in cpu_ci and p not in verified_cpu
                    and p.get('ParentProcessId') in known_ids]
        if not children:
            break
        cpu_ci.extend(children)
    unknown = [p for p in foreign if p not in cpu_ci and p not in verified_cpu]
    passed = not unknown and (not initial or int(gpu[1]) >= 18 * 1024)
    return {'passed': passed, 'gpu_uuid': gpu[0], 'free_mib': int(gpu[1]), 'used_mib': int(gpu[2]),
            'foreign_python_count': len(foreign), 'recognized_cpu_ci_count': len(cpu_ci),
            'verified_cpu_preparation_count': len(verified_cpu),
            'unclassified_python_count': len(unknown), 'initial_capacity_check': initial}


def raw_output_slots(anchor, velocity, proxies, order, beta_prime):
    """Reconstruct unscaled predictions for observation, without changing the provider."""
    import torch
    a, v = anchor.float().clone(), velocity.float().clone()
    result, ratios = [], []
    for depth in range(3):
        if depth < 2:
            y = proxies[depth].float()
            if order:
                v = beta_prime * v + (1 - beta_prime) * (y - a)
            a = y
        else:
            a = a + v
        result.append(torch.stack([a, a + v]))
        ratios.append(float(v.norm() / a.norm()))
    return torch.stack(result), ratios


def observe(model, prompt, spec, eval_cfg, eos, seed):
    import torch
    import spec_mtp.decode as decoding
    from spec_mtp.masks import OutputHiddenMaskProvider
    from studies._shared.structure.config import _convert
    original_provider, original_tree = build_mask_provider, decoding.build_tree
    original_masks = OutputHiddenMaskProvider.masks_per_owner
    active, frames = {}, []
    calls = 0

    def provider_builder(*args, **kwargs):
        provider = original_provider(*args, **kwargs)
        active['provider'] = provider
        return provider

    def tree_builder(*args, **kwargs):
        tree = original_tree(*args, **kwargs)
        active['tree'] = tree
        return tree

    def masks_spy(provider, proxies):
        active['state'] = {k: getattr(provider, k).detach().cpu().clone() for k in ('anchor', 'velocity')}
        active['state']['proxies'] = proxies.detach().cpu().clone()
        return original_masks(provider, proxies)

    def before(_module, args, kwargs):
        nonlocal calls
        calls += 1
        active['prefill'] = kwargs.get('attention_mask_4d') is None
        if not active['prefill']:
            embeds, positions = args[:2]
            tree = active['tree']
            active['masks'] = embeds[0, 1 + tree.num_nodes:].reshape(1 + tree.num_nodes, 2, -1)
            active['root_index'] = int(positions[0, 0]) - prompt.shape[1]

    def norm_hook(_module, _args, output):
        active['hidden'] = output

    def after(_module, _args, _kwargs, output):
        if active['prefill']:
            return
        tree, owner, accepted = active['tree'], 0, []
        while True:
            target = int(output.logits[0, owner].argmax())
            match = next((child for child in tree.children_of(owner) if tree.tokens[child - 1] == target), None)
            if match is None:
                break
            accepted.append(match)
            owner = match
        owners = [0] + accepted
        depths = [0] + [tree.depths[owner - 1] for owner in accepted]
        provider = active['provider']
        frame = {'root_index': active['root_index'], 'accepted_nodes': accepted, 'depths': depths,
                 'masks': active['masks'][owners].detach().cpu().clone(),
                 'real_hidden': active['hidden'][0, owners].detach().cpu().clone()}
        if spec['method'] == 'output_hidden':
            state = active['state']
            raw, ratios = raw_output_slots(state['anchor'], state['velocity'], state['proxies'],
                                           spec['output']['order'], spec['output']['beta_prime'])
            frame.update(raw_predictions=raw[depths], velocity_ratios=[ratios[d] for d in depths],
                         target_scale=float(provider.scale), provider_state=state)
        else:
            frame.update(raw_predictions=frame['masks'].float().clone(),
                         velocity_ratios=[float(provider._velocity.float().norm() / provider._level.float().norm())] * len(owners),
                         input_velocity=provider._velocity.detach().cpu().clone(),
                         input_level=provider._level.detach().cpu().clone())
        frames.append(frame)

    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
               model.hf.model.norm.register_forward_hook(norm_hook),
               model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch.object(decoding, 'build_tree', tree_builder), \
                patch.object(OutputHiddenMaskProvider, 'masks_per_owner', masks_spy), \
                patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
            output, stats = decoding.SpecDecoder(model, _convert(spec), _convert(eval_cfg), eos_ids=eos,
                generator=torch.Generator(device=model.device).manual_seed(seed), mask_policy=provider_builder(_convert(spec), model.embedding_table)).generate(prompt)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == stats['prefill_calls'] + stats['decode_calls'] == 1 + len(frames)
    assert stats['verification_argmax_audit']['passed']
    for frame, trace in zip(frames, stats['trace']):
        assert frame['root_index'] == trace['root_position'] - prompt.shape[1]
        assert frame['accepted_nodes'] == trace['accepted_nodes']
    return output[0].detach().cpu(), stats, frames, calls


def cosine(a, b):
    import torch
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1))


def point_rows(frames, generated, embedding_table, case, qid, common_prefix):
    import torch
    points, skipped = [], 0
    for frame in frames:
        for i, depth in enumerate(frame['depths']):
            owner_index = frame['root_index'] + depth
            for slot in (0, 1):
                target_index = owner_index + slot + 1
                if target_index >= len(generated):
                    skipped += 1
                    continue
                target = embedding_table[int(generated[target_index])].detach().cpu().float()
                mask, raw = frame['masks'][i, slot].float(), frame['raw_predictions'][i, slot].float()
                h = frame['real_hidden'][i].float()
                row = {'case': case, 'question_id': qid, 'root_index': frame['root_index'], 'depth': depth,
                       'slot': slot, 'target_index': target_index, 'shared_prefix': target_index < common_prefix,
                       'mask_norm': float(mask.norm()), 'raw_norm': float(raw.norm()), 'target_norm': float(target.norm()),
                       'mask_target_cos': cosine(mask, target), 'raw_target_cos': cosine(raw, target),
                       'mask_target_norm_ratio': float(mask.norm() / target.norm()),
                       'slot_pair_cos': cosine(frame['masks'][i, 0], frame['masks'][i, 1]),
                       'velocity_anchor_ratio': frame['velocity_ratios'][i]}
                if slot == 0:
                    row.update(real_hidden_norm=float(h.norm()), real_hidden_target_cos=cosine(h, target),
                               predicted_hidden_real_cos=cosine(raw, h))
                if 'target_scale' in frame:
                    row['mask_scale_relative_error'] = abs(float(mask.norm()) / frame['target_scale'] - 1)
                    row['reconstruction_direction_cos'] = cosine(mask, raw)
                    assert row['mask_scale_relative_error'] < .005 and row['reconstruction_direction_cos'] > .99999
                assert all(torch.isfinite(v).all() for v in (target, mask, raw, h))
                points.append(row)
    return points, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a new output directory'
    args.out.mkdir(parents=True)
    from analyze_4b_geometry import sha
    report = {'version': 'qwen3-4b-mask-geometry-20261015-v1', 'quality_passed': False, 'status': 'preflight',
              'admission': [], 'cases': [], 'points': [], 'new_token_limit': 768, 'model_forward_limit': 768,
              'no_8b': True, 'no_timing': True, 'no_independent_ar_forward': True}
    def write():
        (args.out / 'geometry.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        for _ in range(2):
            observation = admission()
            report['admission'].append(observation)
            if not observation['passed']:
                report['status'] = 'deferred_nonpreemptive_admission'
                write()
                print(report['status'], flush=True)
                return
            time.sleep(5)
        import torch
        from transformers import AutoModelForCausalLM
        from studies._shared.structure.sources import manifest
        from studies._shared.structure.evidence import model_digest, runtime
        from spec_mtp.model import HFCausalLM
        from rpipe.structure.system.runtime import apply_runtime
        from rpipe.structure.system.config import SystemConfig
        torch.set_num_threads(1)
        study = args.study
        paired = json.loads((study / 'docs/paired_evidence_v1.json').read_text())
        prompts = json.loads((study / 'docs/prompts_manifest.json').read_text())
        snapshot = json.loads((study / 'docs/modelscope_snapshot.json').read_text())
        assert manifest()['digest'] == '538e52c929a542a8'
        report['source_binding'] = {'source_digest': manifest()['digest'], 'tool_sha256': sha(__file__),
            'plan_sha256': sha((Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md')), 'paired_local_sha256': sha(study / 'docs/paired_evidence_v1.json')}
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-4B'
        for file in snapshot['files']:
            assert sha(cache / file['path']) == file['sha256']
        observation = admission()
        report['admission'].append(observation)
        if not observation['passed']:
            report['status'] = 'deferred_after_model_hash_check'
            write()
            return
        apply_runtime(123, SystemConfig.from_mapping({'config': {'deterministic': False}}))
        load = AutoModelForCausalLM.from_pretrained
        with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=lambda *a, **kw: load(*a, **dict(kw, device_map='cuda'))):
            model = HFCausalLM(str(cache), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
        assert model.device.type == 'cuda' and model.dtype == torch.bfloat16
        assert model.embedding_table.data_ptr() == model.hf.get_output_embeddings().weight.data_ptr()
        actual_digest = model_digest(model)
        assert actual_digest == paired['cells']['30/input_first']['model_digest']
        report.update(model_digest=actual_digest, runtime=runtime(), dtype=str(model.dtype), tied_storage=True)
        selected = [next(p for p in prompts['rendered_prompts'] if p['category'] == category)
                    for category in ('coding', 'math_reasoning', 'summarization', 'translation')]
        raw_cases, tensors = {}, {}
        for method in ('input_first', 'output_zero', 'output_first'):
            cell = paired['cells'][f'30/{method}']
            path = study / 'runs' / cell['run_id'] / 'assets/decode_observations.json'
            assert sha(path) == cell['files_sha256']['assets/decode_observations.json']
            raw_cases[method] = json.loads(path.read_text())['out']['evidence']
        ar_cell = paired['cells']['AR']
        ar_path = study / 'runs' / ar_cell['run_id'] / 'assets/decode_observations.json'
        assert sha(ar_path) == ar_cell['files_sha256']['assets/decode_observations.json']
        raw_ar = json.loads(ar_path.read_text())['out']['evidence']
        for prompt in selected:
            qid = prompt['question_id']
            index = next(i for i, p in enumerate(prompts['rendered_prompts']) if p['question_id'] == qid)
            observed = {}
            for method in ('input_first', 'output_zero', 'output_first'):
                current = admission(initial=False)
                report['admission'].append(current)
                assert current['passed'], 'New foreign Python work detected; stop this diagnostic, do not preempt it'
                cell = paired['cells'][f'30/{method}']
                seed = cell['rows'][index]['generator_seed']
                apply_runtime(seed, SystemConfig.from_mapping({'config': {'deterministic': False}}))
                eval_cfg = dict(cell['actual_settings']['eval'], max_new_tokens=64)
                ids = torch.tensor([prompt['input_ids']], device='cuda')
                output, stats, frames, calls = observe(model, ids, cell['actual_settings']['spec'], eval_cfg,
                                                      prompts['eos_token_ids'], seed)
                reference = raw_cases[method][index]['output_ids']
                assert output.tolist() == reference[:len(output)], 'Observed prefix must match the completed full Run'
                generated = output[len(prompt['input_ids']):]
                ar_ids = raw_ar[index]['output_ids']
                quality = {'case': method, 'question_id': qid, 'category': prompt['category'], 'generator_seed': seed,
                           'new_tokens': len(generated), 'model_calls': calls, 'prefix_equals_full_run': True,
                           'argmax_passed': stats['verification_argmax_audit']['passed'],
                           'argmax_checked_tokens': stats['verification_argmax_audit']['checked_committed_tokens'],
                           'AR_prefix_exact_diagnostic': output.tolist() == ar_ids[:len(output)]}
                report['cases'].append(quality)
                observed[method] = (generated, frames)
                tensors[f'{qid}/{method}'] = {'output_ids': output, 'frames': frames, 'stats': stats}
                write()
                print(qid, method, quality['new_tokens'], 'argmax/prefix passed', flush=True)
            sequences = [value[0].tolist() for value in observed.values()]
            common = 0
            for tokens in zip(*sequences):
                if len(set(tokens)) != 1:
                    break
                common += 1
            for method, (generated, frames) in observed.items():
                points, skipped = point_rows(frames, generated, model.embedding_table, method, qid, common)
                report['points'].extend(points)
                next(c for c in report['cases'] if c['case'] == method and c['question_id'] == qid).update(
                    common_prefix_tokens=common, missing_tail_targets=skipped, geometry_points=len(points))
        report['total_new_tokens'] = sum(c['new_tokens'] for c in report['cases'])
        report['total_model_calls'] = sum(c['model_calls'] for c in report['cases'])
        assert report['total_new_tokens'] <= 768 and report['total_model_calls'] <= 768 and len(report['cases']) == 12
        assert manifest()['digest'] == report['source_binding']['source_digest']
        torch.save(tensors, args.out / 'vectors.pt')
        report.update(raw_vectors_sha256=sha(args.out / 'vectors.pt'), quality_passed=True, status='complete')
    except Exception as error:
        report.update(status='failed_preserved', error=repr(error))
        write()
        raise
    write()
    print('Complete: 12 bounded 4B captures; no teacher, timing or scientific-source change', flush=True)


if __name__ == '__main__':
    main()
