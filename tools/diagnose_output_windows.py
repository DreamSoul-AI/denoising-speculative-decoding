"""Read-only matched-window observation around the existing SpecDecoder.

    python tools/diagnose_output_windows.py --study studies/output_space_small_matrix --out .tmp/output-windows-<new-id>

Replays saved q173 input/order0/order1 once, requires identical token/semantic
traces, and captures preceding seed masks/logits plus exact normalized hiddens.
No alternative decoder/Run scheduler, source change, download or tuning.
"""

from spec_mtp.masks import build_mask_provider

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch


SEMANTIC_KEYS = ('accepted', 'new_tokens', 'committed', 'decode_calls', 'hit_context_limit',
                 'attempt_by_depth', 'accept_by_depth')
TRACE_KEYS = ('root_position', 'cache_before', 'cache_after', 'accepted_nodes', 'committed_tokens', 'stopped')


def margin(logits, target):
    import torch

    row = logits.detach().float().cpu()
    top = row.topk(2)
    probs = torch.softmax(row, dim=-1)
    best_other = top.values[1] if int(top.indices[0]) == target else top.values[0]
    return {'top1_id': int(top.indices[0]), 'top2_id': int(top.indices[1]),
            'top1_probability': float(probs[top.indices[0]]),
            'top2_probability': float(probs[top.indices[1]]), 'target_id': int(target),
            'target_probability': float(probs[target]),
            'target_margin_vs_best_other': float(row[target] - best_other)}


def cosine(a, b):
    import torch

    return float(torch.nn.functional.cosine_similarity(a.float().cpu(), b.float().cpu(), dim=0))


def observe(model, prompt, spec, eval_cfg, eos, reference, targets, seed=124):
    import torch
    from spec_mtp.decode import SpecDecoder
    from spec_mtp.masks import OutputHiddenMaskProvider

    k = int(spec['num_masks'])
    assert k == 2 and spec['tree']['branches'] == [1, 1] and not spec['tree']['pruning']
    expected = {t['root_position']: t for t in reference['stats']['trace']}
    assert set(targets) <= set(expected)
    active, latest_norm, latest_state, previous_seed, samples = {}, {}, {}, {}, {}
    calls = 0
    original = OutputHiddenMaskProvider.masks_per_owner

    def provider_spy(provider, proxies):
        result = original(provider, proxies)
        latest_state['value'] = {'anchor': provider.anchor.detach().clone(),
                                 'velocity': provider.velocity.detach().clone(),
                                 'proxies': proxies.detach().clone()}
        return result

    def before(_module, args, kwargs):
        nonlocal calls
        calls += 1
        embeds = args[0] if args else kwargs['inputs_embeds']
        positions = args[1] if len(args) > 1 else kwargs['position_ids']
        active.update(embeds=embeds, positions=positions,
                      prefill=kwargs.get('attention_mask_4d') is None,
                      state=latest_state.get('value'))

    def norm_hook(_module, _args, out):
        latest_norm['value'] = out

    def after(_module, _args, _kwargs, out):
        h = latest_norm['value'][0]
        embeds, positions = active['embeds'][0], active['positions'][0]
        if active['prefill']:
            rows = list(range(prompt.shape[1], prompt.shape[1] + k))
            source_root, owner = None, None
        else:
            root = int(positions[0])
            nodes = len(embeds) // (k + 1) - 1
            assert nodes == k
            if root in targets:
                assert int(previous_seed['positions'][0]) == root
                assert int(previous_seed['positions'][1]) == root + 1
                seed_logits = previous_seed['logits']
                draft_ids = seed_logits.argmax(dim=-1)
                # Unique top1s are required for these observations: verify
                # the observed draft embeddings equal the actual tree rows.
                assert torch.equal(model.embed(draft_ids[None])[0], embeds[1:1 + k])
                samples[root] = {'seed_logits': seed_logits.detach().cpu().clone(),
                                 'seed_hidden': previous_seed['hidden'].detach().cpu().clone(),
                                 'seed_mask_inputs': previous_seed['mask_inputs'].detach().cpu().clone(),
                                 'exact_root_hidden': h[0].detach().cpu().clone(),
                                 'current_masks': embeds[1 + nodes:].detach().cpu().reshape(k + 1, k, -1).clone(),
                                 'source_root': previous_seed['source_root'],
                                 'source_owner': previous_seed['source_owner'],
                                 'seed_positions': previous_seed['positions'].cpu().tolist(),
                                 'source_state': {key: val.cpu().clone() for key, val in
                                                  (previous_seed['state'] or {}).items()},
                                 'current_state': {key: val.cpu().clone() for key, val in
                                                   (active['state'] or {}).items()}}
            accepted = expected[root]['accepted_nodes']
            owner = accepted[-1] if accepted else 0
            rows = [1 + nodes + owner * k + j for j in range(k)]
            source_root = root
        previous_seed.update(logits=out.logits[0, rows].detach().clone(),
                             hidden=h[rows].detach().clone(), mask_inputs=embeds[rows].detach().clone(),
                             positions=positions[rows].detach().clone(), source_root=source_root,
                             source_owner=owner, state=active['state'])

    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                   model.hf.model.norm.register_forward_hook(norm_hook),
                   model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', provider_spy):
            output, stats = SpecDecoder(model, spec, eval_cfg, eos_ids=eos,
                                       generator=torch.Generator(device=model.device).manual_seed(seed), mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert calls == 1 + stats['decode_calls']
    assert output[0].tolist() == reference['output_ids']
    assert all(stats[key] == reference['stats'][key] for key in SEMANTIC_KEYS)
    actual_trace = [{key: t[key] for key in TRACE_KEYS} for t in stats['trace']]
    expected_trace = [{key: t[key] for key in TRACE_KEYS} for t in reference['stats']['trace']]
    assert actual_trace == expected_trace and sorted(samples) == sorted(targets)
    return samples, {'replayed_output_equal': True, 'semantic_trace_equal': True,
                     'decode_calls': stats['decode_calls'], 'observer_target_forward_calls': calls,
                     'expected_target_forward_calls': 1 + stats['decode_calls']}


def file_hash(path):
    path = Path(path)
    if path.exists():
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    # Consolidated historical plan identities refer to fixed Git bytes, never today's plan.
    root = Path(__file__).resolve().parents[1]
    relative = path.resolve().relative_to(root).as_posix()
    manifest = json.loads((root / 'docs/research/plan-consolidation-manifest.json').read_text(encoding='utf-8'))
    item = next((x for x in manifest['plans'] if x['source_path'] == relative), None)
    if item is None:
        raise FileNotFoundError(path)
    blob = subprocess.check_output(['git', 'show', item['source_commit'] + ':' + relative], cwd=root)
    assert hashlib.sha256(blob).hexdigest() == item['git_blob_sha256']
    candidates = (blob, blob.replace(b'\n', b'\r\n'))
    assert item['source_sha256'] in {hashlib.sha256(data).hexdigest() for data in candidates}
    return item['source_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new; existing evidence is preserved')
    args.out.mkdir(parents=True)
    import torch
    import yaml
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM

    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    study = args.study.resolve()
    data = read(study / 'docs/execution_v1.json')
    assert data['study_quality_passed'] and data['question_ids'] == [172, 173, 174]
    assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
    plan = (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md')
    tool = Path(__file__).resolve()
    inputs = {'plan_sha256': file_hash(plan), 'tool_sha256': file_hash(tool),
              'matrix_sha256': file_hash(study / 'docs/execution_v1.json'),
              'source_digest': manifest()['digest']}
    report = {'version': 'output-space-q173-window-20261014-v1', 'question_id': 173,
              'windows': [83, 94, 97], 'kind': 'instrumented unchanged-decoder replay',
              'source_binding': inputs, 'cases': [], 'quality_passed': False}
    tensors = {}
    try:
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        for name, metadata in read(study / 'docs/modelscope_snapshot.json')['files'].items():
            assert file_hash(cache / name) == metadata['sha256']
        torch.set_num_threads(1)
        apply_runtime(124, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == data['model_digest']
        rows = {r['case']: r for r in data['runs']}
        assert runtime() == rows['output_order1']['runtime']
        ar = rows['AR']['raw']['evidence'][1]
        tokens = torch.tensor([ar['output_ids']], device=model.device)
        prompt = torch.tensor([ar['input_ids']], device=model.device)
        positions = torch.arange(tokens.shape[1], device=model.device)[None]
        teacher = model(model.embed(tokens), positions, last_hidden_only=True)
        exact = teacher.last_hidden_state[0].detach().cpu().clone()
        del teacher
        for case in ['frozen_mean_chain', 'output_order0', 'output_order1']:
            row = rows[case]
            folder = study / 'runs' / row['run_id']
            native_inputs = read(folder / 'assets/decode_inputs.json')
            assert native_inputs['prompts'][1]['input_ids'] == ar['input_ids']
            assert file_hash(folder / 'assets/decode_observations.json') == row['raw_sha256']
            cfg = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
            ev = row['raw']['evidence'][1]
            samples, replay = observe(model, prompt, _convert(row['spec']),
                                      _convert(cfg['algorithm']['config']['eval']),
                                      native_inputs['eos_ids'], ev, report['windows'])
            windows = []
            report['cases'].append({'case': case, 'native_reference_run': row['run_id'],
                                    'replay': replay, 'windows': windows})
            tensors.update({f'{case}:{root}': sample for root, sample in samples.items()})
            for root, sample in sorted(samples.items()):
                target = int(tokens[0, root + 1])
                first = margin(sample['seed_logits'][0], target)
                proxy, true = sample['seed_hidden'][0], sample['exact_root_hidden']
                assert cosine(true, exact[root]) >= .99999, f'{case} root{root}: exact hidden consistency failed'
                prefix = torch.cat([model.embed(tokens[:, :root]),
                                    sample['seed_mask_inputs'][0].to(model.device)[None, None]], dim=1)
                isolated = model(prefix, torch.arange(root + 1, device=model.device)[None], last_hidden_only=True)
                isolated_logits = isolated.logits[0, -1]
                isolated_hidden = isolated.last_hidden_state[0, -1]
                separate = margin(isolated_logits, target)
                assert first['top1_id'] == separate['top1_id'], f'{case} root{root}: isolated top1 mismatch'
                assert cosine(proxy, isolated_hidden) >= .99999, f'{case} root{root}: isolated hidden consistency failed'
                second_valid = first['top1_id'] == target
                second = margin(sample['seed_logits'][1], int(tokens[0, root + 2]))
                windows.append({'root_position': root, 'source_block_root': sample['source_root'],
                                'source_owner': sample['source_owner'], 'seed_positions': sample['seed_positions'],
                                'accepted_nodes': next(t['accepted_nodes'] for t in ev['stats']['trace'] if t['root_position'] == root),
                                'first_slot': first, 'second_slot': second,
                                'second_AR_prefix_valid': second_valid,
                                'seed_mask_norm': float(sample['seed_mask_inputs'][0].norm()),
                                'proxy_exact_root_cosine': cosine(proxy, true),
                                'proxy_exact_relative_L2': float((proxy - true).norm() / true.norm()),
                                'teacher_exact_root_cosine': cosine(true, exact[root]),
                                'isolated_prefix_seed_cosine': cosine(proxy, isolated_hidden),
                                'isolated_first_slot': separate,
                                'isolated_target_margin_delta': separate['target_margin_vs_best_other'] - first['target_margin_vs_best_other']})
                tensors[f'{case}:{root}'] = sample
                tensors[f'{case}:{root}']['teacher_exact_hidden'] = exact[root]
        assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
        assert inputs['plan_sha256'] == file_hash(plan) and inputs['tool_sha256'] == file_hash(tool)
        assert inputs['matrix_sha256'] == file_hash(study / 'docs/execution_v1.json')
        assert manifest()['digest'] == inputs['source_digest']
        for name, metadata in read(study / 'docs/modelscope_snapshot.json')['files'].items():
            assert file_hash(cache / name) == metadata['sha256']
        report['auxiliary_forwards'] = 10
        report['runtime'] = runtime()
        report['model_digest'] = data['model_digest']
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False) + '\n',encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [c['case'] for c in report['cases']], 'windows': report['windows']}))


if __name__ == '__main__':
    main()
