"""Probe a fixed 4B prefix/tree with output interventions and an input control, without timing.

    python tools/diagnose_4b_mask_windows.py --study studies/qwen3_4b_bf16_five_methods --out .tmp/4b-mask-windows-v1
"""

from spec_mtp.masks import build_mask_provider
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch


VARIANTS = ('zero', 'input_first', 'first', 'shifted', 'radial', 'both')
TEMPLATE_VARIANTS = ('zero', 'input_first', 'input_template', 'input_mean')
RECIPROCAL_VARIANTS = ('zero', 'input_first', 'first')
DIRECTION_VARIANTS = ('zero', 'input_first', 'input_template', 'input_mean',
                      'input_random_0', 'input_random_1', 'input_random_2', 'input_content')
EARLY_ROOTS = {121: 1, 401: 0, 241: 1, 161: 2}


def admission(initial=True):
    import csv
    gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    raw = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader,nounits'], text=True)
    foreign = []
    for row in csv.reader(raw.splitlines()):
        if not row:
            continue
        pid, name = row
        if int(pid) != os.getpid() and Path(name.strip().replace('\\', '/')).name.lower().startswith('python'):
            foreign.append(int(pid))
    required = 14 * 1024
    return {'passed': not foreign and (not initial or int(gpu[0]) >= required),
            'free_mib': int(gpu[0]), 'used_mib': int(gpu[1]), 'required_free_mib': required,
            'foreign_gpu_python_pids': foreign, 'gpu_index': 0, 'initial': initial}


def intervention_masks(shadow, proxies, zero):
    """Native state simulation; only the slot offset and radial mapping vary."""
    a, v = shadow.anchor.clone(), shadow.velocity.clone()
    anchors, velocities = [], []
    for depth in range(3):
        if depth < 2:
            y = proxies[depth].detach().float()
            v = shadow.beta_prime * v + (1 - shadow.beta_prime) * (y - a)
            a = y
        else:
            a = a + v
        anchors.append(a)
        velocities.append(v)
    a, v = torch.stack(anchors)[:, None], torch.stack(velocities)[:, None]
    masks = {'zero': zero}
    for name in ('first', 'shifted', 'radial', 'both'):
        offset = int(name in ('shifted', 'both'))
        slots = a + (torch.arange(2, device=a.device) + offset)[None, :, None] * v
        norms = a.norm(dim=-1, keepdim=True) if name in ('radial', 'both') else slots.norm(dim=-1, keepdim=True)
        assert torch.isfinite(slots).all() and (norms > 0).all()
        masks[name] = (shadow.scale * slots / norms).to(zero.dtype)
    masks['first'] = shadow.masks_per_owner(proxies).to(zero.dtype)
    return masks


def walk(tree, logits):
    from spec_mtp.decode import select_token
    owner, accepted, chosen = 0, [], []
    while True:
        token = select_token(logits[owner], 0., None)
        assert token == int(logits[owner].argmax())
        chosen.append(token)
        child = next((c for c in tree.children_of(owner) if tree.tokens[c - 1] == token), None)
        if child is None:
            return accepted, chosen
        accepted.append(child)
        owner = child


def active_predecessor(trace, prompt_length):
    for previous, current in zip(trace, trace[1:]):
        if current['accepted_nodes'] == [1]:
            return previous['root_position'] - prompt_length
    raise ValueError('No predefined top1-only control opportunity; do not replace the prompt')


@torch.no_grad()
def input_control(model, prompt, generated_prefix, spec, velocity_mode='native', random_seed=None,
                  content_start=None, state_report=None):
    from spec_mtp.masks import build_mask_provider
    from studies._shared.structure.config import _convert
    provider = build_mask_provider(_convert(spec), model.embedding_table)
    embeds = model.embed(prompt)[0]
    provider.init_from_prompt(embeds)
    if velocity_mode == 'content':
        assert content_start is not None and 0 <= content_start < len(embeds) - 1
        level = provider._level
        provider.init_from_prompt(embeds[content_start:])
        provider._level = level
    previous = embeds[-1]
    for token in generated_prefix:
        current = model.embedding_table[token]
        provider.on_token_committed(current, previous)
        previous = current
    if velocity_mode == 'template':
        provider._velocity = (embeds[1] - embeds[0]).clone()
    elif velocity_mode == 'zero':
        provider._velocity = torch.zeros_like(provider._velocity)
    elif velocity_mode == 'random':
        assert random_seed is not None
        direction = torch.randn(embeds.shape[-1], generator=torch.Generator().manual_seed(random_seed), dtype=torch.float32)
        direction *= (embeds[1] - embeds[0]).float().norm().cpu() / direction.norm()
        provider._velocity = direction.to(device=embeds.device, dtype=embeds.dtype)
        assert abs(float(provider._velocity.float().norm() / (embeds[1] - embeds[0]).float().norm()) - 1) < .01
    else:
        assert velocity_mode in ('native', 'content')
    if state_report is not None:
        v, fixed = provider._velocity.float(), (embeds[1] - embeds[0]).float()
        state_report.update(random_seed=random_seed, content_start=content_start,
            velocity_norm=float(v.norm()), template_norm=float(fixed.norm()),
            template_norm_ratio=float(v.norm() / fixed.norm()),
            level_norm=float(provider._level.float().norm()),
            template_cosine=float(torch.nn.functional.cosine_similarity(v, fixed, dim=0)))
    return provider.masks().to(model.dtype)[None].expand(3, -1, -1).clone()


def content_start_of(tokenizer, body, expected_ids):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == expected_ids and rendered.count(body) == 1
    start, end = rendered.index(body), rendered.index(body) + len(body)
    indices = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if start <= a < b <= end]
    assert len(indices) >= 2 and indices[1] == indices[0] + 1
    return indices[0]


class WindowCaptured(Exception):
    pass


@torch.no_grad()
def capture_window(model, prompt, spec, eval_cfg, eos, seed, minimum=8):
    import spec_mtp.decode as decoding
    from spec_mtp.masks import OutputHiddenMaskProvider
    from studies._shared.structure.config import _convert
    original_build, original_tree = build_mask_provider, decoding.build_tree
    original_observe, original_masks = OutputHiddenMaskProvider.observe_committed, OutputHiddenMaskProvider.masks_per_owner
    active, packet, generated = {}, {}, []

    def build(*args, **kwargs):
        provider = original_build(*args, **kwargs)
        active.update(provider=provider, shadow=OutputHiddenMaskProvider(2, model.embedding_table, order=1, beta=.9, beta_prime=.99))
        return provider

    def observe(provider, hidden):
        original_observe(provider, hidden)
        if provider is active['provider']:
            original_observe(active['shadow'], hidden)

    def masks(provider, proxies):
        zero = original_masks(provider, proxies)
        if provider is active['provider']:
            active.update(proxies=proxies, zero=zero)
        return zero

    def tree_builder(*args, **kwargs):
        tree = original_tree(*args, **kwargs)
        active['tree'] = tree
        return tree

    def before(_module, args, kwargs):
        active['prefill'] = kwargs.get('attention_mask_4d') is None
        if not active['prefill']:
            root = int(args[1][0, 0]) - prompt.shape[1]
            if root >= minimum and not packet:
                packet.update(root_index=root, inputs=args[0].clone(), positions=args[1].clone(),
                    kwargs={**kwargs, 'past_key_values': deepcopy(kwargs['past_key_values'])}, tree=deepcopy(active['tree']),
                    masks=intervention_masks(active['shadow'], active['proxies'], active['zero'].to(model.dtype)))

    def after(_module, _args, _kwargs, output):
        if active['prefill']:
            generated.append(int(output.logits[0, prompt.shape[1] - 1].argmax()))
            return
        accepted, chosen = walk(active['tree'], output.logits[0])
        generated.extend(chosen)
        if packet:
            packet.update(baseline=output, prefix_generated=list(generated), accepted=accepted, chosen=chosen,
                          zero_provider=active['provider'])
            raise WindowCaptured()

    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch.object(decoding, 'build_tree', tree_builder), \
                patch.object(OutputHiddenMaskProvider, 'observe_committed', observe), \
                patch.object(OutputHiddenMaskProvider, 'masks_per_owner', masks), \
                patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
            try:
                decoding.SpecDecoder(model, _convert(spec), _convert(dict(eval_cfg, max_new_tokens=32)), eos,
                    torch.Generator(device=model.device).manual_seed(seed), mask_policy=build(_convert(spec), model.embedding_table)).generate(prompt)
            except WindowCaptured:
                pass
    finally:
        for handle in handles:
            handle.remove()
    assert packet and packet['root_index'] < 32, 'No admissible fixed window; do not choose a result-dependent replacement'
    return packet


@torch.no_grad()
def capture_input_window(model, prompt, spec, eval_cfg, eos, seed, minimum):
    import spec_mtp.decode as decoding
    from spec_mtp.masks import OutputHiddenMaskProvider
    from spec_mtp.tree import mask_indices_of
    from studies._shared.structure.config import _convert
    original_tree = decoding.build_tree
    shadows = [OutputHiddenMaskProvider(2, model.embedding_table, order=i, beta=.9, beta_prime=.99) for i in (0, 1)]
    active, packet, generated = {}, {}, []

    def tree_builder(*args, **kwargs):
        tree = original_tree(*args, **kwargs)
        active['tree'] = tree
        return tree

    def before(_module, args, kwargs):
        kwargs['last_hidden_only'] = True
        active['prefill'] = kwargs.get('attention_mask_4d') is None
        if not active['prefill'] and int(args[1][0, 0]) - prompt.shape[1] >= minimum:
            zero = shadows[0].masks_per_owner(active['proxies']).to(model.dtype)
            packet.update(root_index=int(args[1][0, 0]) - prompt.shape[1], inputs=args[0].clone(),
                positions=args[1].clone(), tree=deepcopy(active['tree']), zero_provider=shadows[0],
                kwargs={**kwargs, 'past_key_values': deepcopy(kwargs['past_key_values'])},
                masks=intervention_masks(shadows[1], active['proxies'], zero))
        return args, kwargs

    def after(_module, _args, _kwargs, output):
        if active['prefill']:
            for shadow in shadows:
                shadow.init_hidden(output.last_hidden_state[0, :prompt.shape[1]])
            active['proxies'] = output.last_hidden_state[0, prompt.shape[1]:].clone()
            generated.append(int(output.logits[0, prompt.shape[1] - 1].argmax()))
            return
        accepted, chosen = walk(active['tree'], output.logits[0])
        generated.extend(chosen)
        if packet:
            packet.update(baseline=output, prefix_generated=list(generated), accepted=accepted, chosen=chosen)
            raise WindowCaptured()
        for row in [0] + accepted:
            for shadow in shadows:
                shadow.observe_committed(output.last_hidden_state[0, row])
        owner = accepted[-1] if accepted else 0
        active['proxies'] = output.last_hidden_state[0, mask_indices_of(active['tree'], 2, owner)].clone()

    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch.object(decoding, 'build_tree', tree_builder), \
                patch.object(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.)):
            try:
                decoding.SpecDecoder(model, _convert(spec), _convert(dict(eval_cfg, max_new_tokens=32)), eos,
                    torch.Generator(device=model.device).manual_seed(seed), mask_policy=build_mask_provider(_convert(spec), model.embedding_table)).generate(prompt)
            except WindowCaptured:
                pass
    finally:
        for handle in handles:
            handle.remove()
    assert packet and packet['root_index'] < 32
    return packet


@torch.no_grad()
def probe_window(model, packet, spec, reference, variants=VARIANTS, control='zero'):
    from spec_mtp import decode as decoding, model as kv_cache
    from spec_mtp.tree import DynamicTreeAttentionCache, mask_indices_of
    from studies._shared.structure.config import _convert
    tree, baseline = packet['tree'], packet['baseline']
    accepted, chosen = packet['accepted'], packet['chosen']
    assert packet['prefix_generated'] == reference[:len(packet['prefix_generated'])]
    owner = accepted[-1] if accepted else 0
    seed_rows = mask_indices_of(tree, 2, owner)
    next_zero = packet.get('next_control_masks')
    if next_zero is None:
        next_zero = packet['zero_provider'].masks_per_owner(baseline.last_hidden_state[0, seed_rows]).to(model.dtype)
    next_index = packet['root_index'] + len(chosen)
    assert next_index + 2 < len(reference) and chosen[-1] == reference[next_index]
    truth = reference[next_index + 1:next_index + 3]
    rows, local = [], {}
    expected_masks = packet['masks'][control][[0] + tree.depths].reshape(-1, packet['inputs'].shape[-1])
    assert torch.equal(packet['inputs'][0, 1 + tree.num_nodes:], expected_masks), 'Native control mask mismatch'
    for name in variants:
        inputs = packet['inputs'].clone()
        inputs[0, 1 + tree.num_nodes:] = packet['masks'][name][[0] + tree.depths].reshape(-1, inputs.shape[-1])
        kwargs = {**packet['kwargs'], 'past_key_values': deepcopy(packet['kwargs']['past_key_values'])}
        output = model(inputs, packet['positions'], **kwargs)
        assert torch.equal(output.logits[0, :1 + tree.num_nodes].argmax(-1), baseline.logits[0, :1 + tree.num_nodes].argmax(-1))
        current_accepted, current_chosen = walk(tree, output.logits[0])
        assert current_accepted == accepted and current_chosen == chosen
        logits = output.logits[0, seed_rows]
        probs = decoding.tree_probs(logits, 0.)
        next_tree = decoding.build_tree(_convert(spec), probs, chosen[-1])
        cache = output.past_key_values
        length = int(packet['positions'][0, 0])
        kv_cache.keep_block_positions(cache, length, [0] + accepted)
        next_position = length + len(chosen)
        assert kv_cache.cache_len(cache) == next_position
        ids = torch.tensor([[chosen[-1]] + next_tree.tokens], device=model.device)
        embeds = model.embed(ids)[0]
        owner_masks = next_zero[[0] + next_tree.depths].reshape(-1, embeds.shape[-1])
        block = torch.cat([embeds, owner_masks])[None]
        attn, positions = DynamicTreeAttentionCache().get(next_tree, 2, next_position, next_position,
                                                         model.dtype, model.device, model.min_mask_value())
        verified = model(block, positions, attention_mask_4d=attn, past_key_values=cache)
        next_accepted, next_chosen = walk(next_tree, verified.logits[0])
        matching = next((c for c in next_tree.children_of(0) if next_tree.tokens[c - 1] == truth[0]), None)
        ranks = [1 + int((logits[j] > logits[j, truth[j]]).sum()) for j in (0, 1)]
        row = {'variant': name, 'root_index': packet['root_index'], 'source_owner_depth': len(accepted),
               'next_root_index': next_index, 'mask_logit_target_ranks': ranks,
               'mask_logit_target_probs': [float(probs[j, truth[j]]) for j in (0, 1)],
               'next_tree_widths': next_tree.branch_widths(), 'reference_d1_in_tree': matching is not None,
               'reference_d2_under_d1': matching is not None and any(next_tree.tokens[c - 1] == truth[1] for c in next_tree.children_of(matching)),
               'current_owner_argmax_unchanged': True, 'next_accept_depth': len(next_accepted),
               'next_root_argmax_matches_reference': next_chosen[0] == truth[0],
               'next_verified_chain_equals_reference': next_chosen == reference[next_index + 1:next_index + 1 + len(next_chosen)],
               'argmax_passed': True, 'argmax_checked_tokens': len(chosen) + len(next_chosen)}
        rows.append(row)
        if name in packet.get('direction_states', {}):
            row['direction_state'] = packet['direction_states'][name]
        local[name] = {'seed_logits': logits.detach().cpu(), 'next_tree': vars(next_tree), 'next_chosen': next_chosen}
    for row in rows:
        row['d1_seed_logits_equal_zero'] = torch.equal(local[row['variant']]['seed_logits'][0], local['zero']['seed_logits'][0])
        if 'input_template' in variants:
            row['seed_logits_equal_input_first'] = torch.equal(local[row['variant']]['seed_logits'], local['input_first']['seed_logits'])
    assert next(r for r in rows if r['variant'] == control)['next_verified_chain_equals_reference'], 'Native control differs from old Run; stop, no numerical study'
    return rows, local


def main():
    from analyze_4b_geometry import sha
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.evidence import model_digest, runtime
    from spec_mtp.model import HFCausalLM
    from transformers import AutoModelForCausalLM
    from rpipe.structure.system.runtime import apply_runtime
    from rpipe.structure.system.config import SystemConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--active-depth', action='store_true', help='Separate conditioned q121/q401 protocol; not a random-sample extension')
    selection.add_argument('--early-template', action='store_true', help='Fixed earlier output-control roots before selected input-path differences')
    parser.add_argument('--template-control', action='store_true', help='Frozen template velocity and zero velocity at the same native input mean')
    parser.add_argument('--reciprocal', action='store_true', help='Swap input first and output zero/first at the fixed early roots')
    parser.add_argument('--input-origin', action='store_true', help='Reconstruct the original input-first path for the reverse swap')
    parser.add_argument('--direction-controls', action='store_true', help='Fixed template, three norm-matched random seeds, zero and content initialization')
    args = parser.parse_args()
    assert not args.early_template or args.template_control
    assert not args.reciprocal or args.early_template
    assert not args.input_origin or args.reciprocal
    assert not args.direction_controls or (args.early_template and not args.reciprocal)
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    qids, limit = ((121, 401), 64) if args.active_depth else ((121, 401, 241, 161), 128)
    plan = 'MASK_ACTIVE_PLAN.md' if args.active_depth else 'MASK_WINDOW_PLAN.md'
    variants = TEMPLATE_VARIANTS if args.template_control else VARIANTS
    if args.template_control:
        plan = 'MASK_TEMPLATE_PLAN.md'
        limit = 48 if args.active_depth else 96
    if args.early_template:
        plan, limit = 'MASK_EARLY_TEMPLATE_PLAN.md', 64
    if args.reciprocal:
        plan, variants = 'MASK_RECIPROCAL_PLAN.md', RECIPROCAL_VARIANTS
    if args.direction_controls:
        plan, variants, limit = 'MASK_DIRECTION_PLAN.md', DIRECTION_VARIANTS, 96
    report = {'status': 'preflight', 'quality_passed': False, 'admission': [], 'rows': [], 'model_calls': 0,
              'forward_limit': limit, 'no_timing': True, 'no_8b': True, 'tool_sha256': sha(__file__),
              'variants': variants, 'template_control': args.template_control,
              'reference_method': 'input_first' if args.input_origin else 'output_zero',
              'sample_mode': 'selected_early_predecessors' if args.early_template else 'conditioned_top1_only' if args.active_depth else 'unconditioned_first_root_ge8',
              'plan_sha256': sha(Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'), 'plan_section': 'legacy-studies-qwen3-4b-bf16-five-methods-docs-' + plan[:-3].lower().replace('_', '-')}
    if args.reciprocal:
        report['sample_mode'] = 'reciprocal_input_origin' if args.input_origin else 'reciprocal_output_origin'
    if args.direction_controls:
        report['sample_mode'] = 'selected_early_direction_controls'
    def save():
        (args.out / 'windows.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    local = {}
    try:
        for _ in range(2):
            check = admission()
            report['admission'].append(check)
            if not check['passed']:
                report['status'] = 'deferred_nonpreemptive_admission'
                save()
                return
            time.sleep(5)
        study = args.study
        paired = json.loads((study / 'docs/paired_evidence_v1.json').read_text())
        prompts = json.loads((study / 'docs/prompts_manifest.json').read_text())
        cell = paired['cells']['30/' + report['reference_method']]
        raw_path = study / 'runs' / cell['run_id'] / 'assets/decode_observations.json'
        assert sha(raw_path) == cell['files_sha256']['assets/decode_observations.json']
        raw = json.loads(raw_path.read_text())['out']['evidence']
        snapshot = json.loads((study / 'docs/modelscope_snapshot.json').read_text())
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-4B'
        for f in snapshot['files']:
            assert sha(cache / f['path']) == f['sha256']
        content_starts = {}
        if args.direction_controls:
            from transformers import AutoTokenizer
            data_path = study / 'shared/data/specbench/question.jsonl'
            assert sha(data_path) == prompts['data_sha256']
            records = {x['question_id']: x for x in map(json.loads, data_path.read_text(encoding='utf-8').splitlines())}
            tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True)
            for p in prompts['rendered_prompts']:
                if p['question_id'] in qids:
                    content_starts[p['question_id']] = content_start_of(tokenizer, records[p['question_id']]['turns'][0], p['input_ids'])
            report.update(data_sha256=sha(data_path), prompts_manifest_sha256=sha(study / 'docs/prompts_manifest.json'),
                          content_start_indices=content_starts, random_direction_seeds=[0, 1, 2])
        check = admission()
        report['admission'].append(check)
        if not check['passed']:
            report['status'] = 'deferred_after_model_hash_check'
            save()
            return
        assert manifest()['digest'] == '538e52c929a542a8'
        apply_runtime(123, SystemConfig.from_mapping({'config': {'deterministic': False}}))
        torch.set_num_threads(1)
        load = AutoModelForCausalLM.from_pretrained
        with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=lambda *a, **kw: load(*a, **dict(kw, device_map='cuda'))):
            model = HFCausalLM(str(cache), local_files_only=True, torch_dtype='bfloat16', attn_implementation='eager')
        assert model.device.type == 'cuda' and model.dtype == torch.bfloat16
        assert model.embedding_table.data_ptr() == model.hf.get_output_embeddings().weight.data_ptr()
        assert model_digest(model) == cell['model_digest']
        report.update(source_digest=manifest()['digest'], model_digest=cell['model_digest'], runtime=runtime(),
                      reference_run=cell['run_id'], reference_raw_sha256=sha(raw_path))
        def count(_module, _args):
            report['model_calls'] += 1
            assert report['model_calls'] <= limit
        handle = model.register_forward_pre_hook(count)
        try:
            for qid in qids:
                check = admission(initial=False)
                report['admission'].append(check)
                assert check['passed'], 'New foreign work; stop this diagnostic'
                index = next(i for i, p in enumerate(prompts['rendered_prompts']) if p['question_id'] == qid)
                p = prompts['rendered_prompts'][index]
                assert p['input_ids'] == raw[index]['input_ids']
                seed = cell['rows'][index]['generator_seed']
                apply_runtime(seed, SystemConfig.from_mapping({'config': {'deterministic': False}}))
                ids = torch.tensor([p['input_ids']], device=model.device)
                minimum = active_predecessor(raw[index]['stats']['trace'], len(p['input_ids'])) if args.active_depth else 8
                if args.early_template:
                    minimum = EARLY_ROOTS[qid]
                if args.active_depth:
                    assert minimum == {121: 3, 401: 19}[qid]
                capture = capture_input_window if args.input_origin else capture_window
                packet = capture(model, ids, cell['actual_settings']['spec'], cell['actual_settings']['eval'], prompts['eos_token_ids'], seed, minimum=minimum)
                if args.early_template:
                    assert packet['root_index'] == minimum
                packet['masks']['input_first'] = input_control(model, ids,
                    raw[index]['output_ids'][len(p['input_ids']):len(p['input_ids']) + packet['root_index'] + 1],
                    paired['cells']['30/input_first']['actual_settings']['spec'])
                if args.input_origin:
                    packet['next_control_masks'] = input_control(model, ids, packet['prefix_generated'], cell['actual_settings']['spec'])
                if args.template_control and not args.reciprocal:
                    for variant, mode in (('input_template', 'template'), ('input_mean', 'zero')):
                        packet['masks'][variant] = input_control(model, ids,
                            raw[index]['output_ids'][len(p['input_ids']):len(p['input_ids']) + packet['root_index'] + 1],
                            paired['cells']['30/input_first']['actual_settings']['spec'], velocity_mode=mode)
                if args.direction_controls:
                    packet['direction_states'] = {}
                    modes = [('input_first', 'native', None), ('input_template', 'template', None), ('input_mean', 'zero', None)]
                    modes += [(f'input_random_{seed}', 'random', seed) for seed in (0, 1, 2)]
                    modes += [('input_content', 'content', None)]
                    for variant, mode, direction_seed in modes:
                        info = {}
                        packet['masks'][variant] = input_control(model, ids,
                            raw[index]['output_ids'][len(p['input_ids']):len(p['input_ids']) + packet['root_index'] + 1],
                            paired['cells']['30/input_first']['actual_settings']['spec'], velocity_mode=mode,
                            random_seed=direction_seed, content_start=content_starts[qid] if mode == 'content' else None, state_report=info)
                        packet['direction_states'][variant] = info
                rows, tensors = probe_window(model, packet, cell['actual_settings']['spec'], raw[index]['output_ids'][len(p['input_ids']):], variants,
                    control='input_first' if args.input_origin else 'zero')
                if args.active_depth:
                    control = tensors['zero']
                    assert packet['root_index'] == minimum and rows[0]['next_accept_depth'] == 1
                    assert len(rows[0]['next_tree_widths']) == 2 and rows[0]['next_tree_widths'][1] > 0
                    assert control['next_tree']['tokens'][0] == control['next_chosen'][0]
                for row in rows:
                    row['question_id'] = qid
                report['rows'].extend(rows)
                local[str(qid)] = tensors
                save()
                print(qid, len(variants), 'fixed-window variants passed', flush=True)
                del packet
        finally:
            handle.remove()
        assert len(report['rows']) == len(qids) * len(variants) and manifest()['digest'] == report['source_digest']
        torch.save(local, args.out / 'private_logits.pt')
        report.update(status='complete', quality_passed=True, private_logits_sha256=sha(args.out / 'private_logits.pt'))
    except Exception as error:
        report.update(status='failed_preserved', error=repr(error))
        save()
        raise
    save()
    print('Complete:', len(report['rows']), 'local interventions; no full generation or timing', flush=True)


if __name__ == '__main__':
    main()
