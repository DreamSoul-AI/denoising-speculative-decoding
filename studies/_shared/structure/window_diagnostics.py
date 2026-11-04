"""Library helpers folded from one-shot structure scripts."""

from spec_mtp.structure.artifact.provenance import file_sha256
from spec_mtp.structure.algorithm.mask import build_mask_provider
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch
SEMANTIC_KEYS = ('accepted', 'new_tokens', 'committed', 'decode_calls', 'hit_context_limit', 'attempt_by_depth', 'accept_by_depth')
TRACE_KEYS = ('root_position', 'cache_before', 'cache_after', 'accepted_nodes', 'committed_tokens', 'stopped')

def margin(logits, target):
    import torch
    row = logits.detach().float().cpu()
    top = row.topk(2)
    probs = torch.softmax(row, dim=-1)
    best_other = top.values[1] if int(top.indices[0]) == target else top.values[0]
    return {'top1_id': int(top.indices[0]), 'top2_id': int(top.indices[1]), 'top1_probability': float(probs[top.indices[0]]), 'top2_probability': float(probs[top.indices[1]]), 'target_id': int(target), 'target_probability': float(probs[target]), 'target_margin_vs_best_other': float(row[target] - best_other)}

def cosine(a, b):
    import torch
    return float(torch.nn.functional.cosine_similarity(a.float().cpu(), b.float().cpu(), dim=0))

def observe(model, prompt, spec, eval_cfg, eos, reference, targets, seed=124):
    import torch
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    k = int(spec['num_masks'])
    assert k == 2 and spec['tree']['branches'] == [1, 1] and (not spec['tree']['pruning'])
    expected = {t['root_position']: t for t in reference['stats']['trace']}
    assert set(targets) <= set(expected)
    active, latest_norm, latest_state, previous_seed, samples = ({}, {}, {}, {}, {})
    calls = 0
    original = OutputHiddenMaskProvider.masks_per_owner

    def provider_spy(provider, proxies):
        result = original(provider, proxies)
        latest_state['value'] = {'anchor': provider.anchor.detach().clone(), 'velocity': provider.velocity.detach().clone(), 'proxies': proxies.detach().clone()}
        return result

    def before(_module, args, kwargs):
        nonlocal calls
        calls += 1
        embeds = args[0] if args else kwargs['inputs_embeds']
        positions = args[1] if len(args) > 1 else kwargs['position_ids']
        active.update(embeds=embeds, positions=positions, prefill=kwargs.get('attention_mask_4d') is None, state=latest_state.get('value'))

    def norm_hook(_module, _args, out):
        latest_norm['value'] = out

    def after(_module, _args, _kwargs, out):
        h = latest_norm['value'][0]
        embeds, positions = (active['embeds'][0], active['positions'][0])
        if active['prefill']:
            rows = list(range(prompt.shape[1], prompt.shape[1] + k))
            source_root, owner = (None, None)
        else:
            root = int(positions[0])
            nodes = len(embeds) // (k + 1) - 1
            assert nodes == k
            if root in targets:
                assert int(previous_seed['positions'][0]) == root
                assert int(previous_seed['positions'][1]) == root + 1
                seed_logits = previous_seed['logits']
                draft_ids = seed_logits.argmax(dim=-1)
                assert torch.equal(model.embed(draft_ids[None])[0], embeds[1:1 + k])
                samples[root] = {'seed_logits': seed_logits.detach().cpu().clone(), 'seed_hidden': previous_seed['hidden'].detach().cpu().clone(), 'seed_mask_inputs': previous_seed['mask_inputs'].detach().cpu().clone(), 'exact_root_hidden': h[0].detach().cpu().clone(), 'current_masks': embeds[1 + nodes:].detach().cpu().reshape(k + 1, k, -1).clone(), 'source_root': previous_seed['source_root'], 'source_owner': previous_seed['source_owner'], 'seed_positions': previous_seed['positions'].cpu().tolist(), 'source_state': {key: val.cpu().clone() for key, val in (previous_seed['state'] or {}).items()}, 'current_state': {key: val.cpu().clone() for key, val in (active['state'] or {}).items()}}
            accepted = expected[root]['accepted_nodes']
            owner = accepted[-1] if accepted else 0
            rows = [1 + nodes + owner * k + j for j in range(k)]
            source_root = root
        previous_seed.update(logits=out.logits[0, rows].detach().clone(), hidden=h[rows].detach().clone(), mask_inputs=embeds[rows].detach().clone(), positions=positions[rows].detach().clone(), source_root=source_root, source_owner=owner, state=active['state'])
    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.hf.model.norm.register_forward_hook(norm_hook), model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', provider_spy):
            output, stats = SpecDecoder(model, spec, eval_cfg, eos_ids=eos, generator=torch.Generator(device=model.device).manual_seed(seed), mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert calls == 1 + stats['decode_calls']
    assert output[0].tolist() == reference['output_ids']
    assert all((stats[key] == reference['stats'][key] for key in SEMANTIC_KEYS))
    actual_trace = [{key: t[key] for key in TRACE_KEYS} for t in stats['trace']]
    expected_trace = [{key: t[key] for key in TRACE_KEYS} for t in reference['stats']['trace']]
    assert actual_trace == expected_trace and sorted(samples) == sorted(targets)
    return (samples, {'replayed_output_equal': True, 'semantic_trace_equal': True, 'decode_calls': stats['decode_calls'], 'observer_target_forward_calls': calls, 'expected_target_forward_calls': 1 + stats['decode_calls']})

def file_hash(path):
    path = Path(path)
    if path.exists():
        return file_sha256(path)
    root = Path(__file__).resolve().parents[3]
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

def transfer(model, prompt, spec, eval_cfg, eos, reference, donor, original_masks, root, targets, placebo=False, seed=124):
    import torch
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    assert spec['num_masks'] == 2 and spec['tree']['branches'] == [1, 1] and (not spec['tree']['pruning'])
    expected = reference['stats']['trace']
    intervention_call = next((i + 1 for i, t in enumerate(expected) if t['root_position'] == root))
    active, state, previous, samples, intervention = ({}, {}, {}, {}, {})
    calls = provider_calls = 0
    original = OutputHiddenMaskProvider.masks_per_owner

    def provider_patch(provider, proxies):
        nonlocal provider_calls
        provider_calls += 1
        masks = original(provider, proxies)
        state.update(anchor=provider.anchor.detach().clone(), velocity=provider.velocity.detach().clone(), proxies=proxies.detach().clone())
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
            intervention.update(provider_call=provider_calls, unchanged_coordinates=True, original_vector=masks[0, 1].detach().cpu().clone(), donor_vector=changed[0, 1].detach().cpu().clone())
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
        samples[current] = {'source_root': source_root, 'source_owner': owner, 'seed_positions': seed_positions.cpu().tolist(), 'seed_logits': seed_logits.cpu().clone(), 'seed_hidden': previous['hidden'][rows].cpu().clone(), 'seed_mask_inputs': previous['embeds'][rows].cpu().clone(), 'current_masks': embeds[0, 3:].reshape(3, 2, -1).detach().cpu().clone(), 'current_state': {k: v.cpu().clone() for k, v in state.items()}}

    def norm_hook(_module, _args, out):
        active['hidden'] = out[0]

    def after(_module, _args, _kwargs, out):
        embeds, positions, hidden = (active['embeds'], active['positions'], active['hidden'])
        current = None if active['prefill'] else int(positions[0])
        if current in samples:
            samples[current]['current_mask_hidden'] = hidden[3:].detach().cpu().reshape(3, 2, -1).clone()
            samples[current]['exact_root_hidden'] = hidden[0].detach().cpu().clone()
        rows = list(range(prompt.shape[1], prompt.shape[1] + 2)) if active['prefill'] else list(range(3, 9))
        previous.update(root=current, logits=out.logits[0, rows].detach().clone(), hidden=hidden[rows].detach().clone(), embeds=embeds[rows].detach().clone(), positions=positions[rows].detach().clone())
    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.hf.model.norm.register_forward_hook(norm_hook), model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', provider_patch):
            output, stats = SpecDecoder(model, spec, eval_cfg, eos_ids=eos, generator=torch.Generator(device=model.device).manual_seed(seed), mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert intervention['actual_root'] == root and calls == 1 + stats['decode_calls']
    assert output[0].tolist() == reference['output_ids'] and (not stats['hit_context_limit'])
    trace = [{k: t[k] for k in TRACE_KEYS} for t in stats['trace']]
    baseline = [{k: t[k] for k in TRACE_KEYS} for t in expected]
    assert [t for t in trace if t['root_position'] <= root] == [t for t in baseline if t['root_position'] <= root]
    if placebo:
        assert trace == baseline and all((stats[k] == reference['stats'][k] for k in SEMANTIC_KEYS))
    by_root = {t['root_position']: i for i, t in enumerate(trace)}
    for position, sample in samples.items():
        index = by_root[position]
        if index:
            prior = trace[index - 1]
            assert sample['source_root'] == prior['root_position']
            assert sample['source_owner'] == (prior['accepted_nodes'][-1] if prior['accepted_nodes'] else 0)
    assert set(samples) == set(targets) & set(by_root)
    return (samples, {'output_equal_independent_AR': True, 'pre_intervention_trace_equal': True, 'placebo_full_semantic_equal': trace == baseline if placebo else None, 'observer_forward_calls': calls, 'modified_provider_calls': 1, 'intervention_root': root, 'only_selected_vector_replaced': True, 'replacement_equal_donor': True, 'visited_targets': sorted(samples), 'unvisited_targets': sorted(set(targets) - set(samples)), 'stats': {k: stats[k] for k in SEMANTIC_KEYS}, 'trace': trace}, intervention)

def crossed_masks(input_vector, output_vector):
    import torch
    vectors = {'input': input_vector.detach().float().cpu(), 'output': output_vector.detach().float().cpu()}
    norms = {name: vector.norm() for name, vector in vectors.items()}
    assert all((torch.isfinite(v).all() and norms[name] > 0 for name, v in vectors.items()))
    return [{'direction': direction, 'norm_source': scale, 'vector': vector.clone() if direction == scale else vector * (norms[scale] / norms[direction]), 'norm': float(norms[scale]), 'observed_combination': direction == scale} for direction, vector in vectors.items() for scale in norms]

def select_window(zero, one):
    by_zero = {t['root_position']: t for t in zero}
    by_one = {t['root_position']: t for t in one}
    common = sorted(set(by_zero) & set(by_one))
    hit = next((r for r in common if by_zero[r]['accepted_nodes'] and by_one[r]['accepted_nodes']))
    index = common.index(hit)
    assert index > 0
    root = common[index - 1]
    assert not by_zero[root]['accepted_nodes'] and (not by_one[root]['accepted_nodes'])
    return (root, hit, sorted((set(by_zero) | set(by_one)) & set(range(root, max(common) + 1))))

def compare_window(sample, baseline, target):
    import torch
    states = {}
    for key in ('anchor', 'velocity', 'proxies'):
        a, b = (sample['current_state'][key], baseline['current_state'][key])
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        states[key] = {'bit_equal': torch.equal(a, b), 'max_abs_delta': float((a - b).abs().max()), 'L2_delta': float((a - b).norm())}
    return {'root_position': sample['seed_positions'][0], 'source_block_root': sample['source_root'], 'source_owner': sample['source_owner'], 'first_slot': margin(sample['seed_logits'][0], target), 'baseline_first_slot': margin(baseline['seed_logits'][0], target), 'first_seed_bit_equal': torch.equal(sample['seed_mask_inputs'][0], baseline['seed_mask_inputs'][0]), 'first_seed_cosine': cosine(sample['seed_mask_inputs'][0], baseline['seed_mask_inputs'][0]), 'committed_velocity_norm': float(sample['current_state']['velocity'].norm()), 'states': states}
