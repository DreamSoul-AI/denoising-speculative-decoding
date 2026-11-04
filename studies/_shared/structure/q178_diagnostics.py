"""Library helpers folded from one-shot structure scripts."""

def _load_theme(name):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        return importlib.import_module("studies._shared.structure." + name)
from unittest.mock import patch
_folded_window_diagnostics = _load_theme('window_diagnostics')
transfer = _folded_window_diagnostics.transfer

def canonical_states(hidden, embedding_table, order):
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    provider = OutputHiddenMaskProvider(2, embedding_table, order=order, beta=0.9, beta_prime=0.99)
    states = {}
    for position, value in enumerate(hidden):
        provider.observe_committed(value)
        states[position + 1] = {'anchor': provider.anchor.clone(), 'velocity': provider.velocity.clone()}
    return states

def synchronized_transfer(model, states, *args, **kwargs):
    import torch
    from spec_mtp.structure.model import hf_causal as kv_cache
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    original = OutputHiddenMaskProvider.masks_per_owner
    latest, records = ({}, [])

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
            assert all((torch.equal(getattr(provider, k).cpu(), target[k]) for k in ('anchor', 'velocity')))
            result = original(provider, proxies)
            assert torch.equal(proxies, proxy_copy)
            records.append({'root_position': root, 'anchor_max_abs_replacement': float((previous['anchor'] - provider.anchor).abs().max()), 'velocity_max_abs_replacement': float((previous['velocity'] - provider.velocity).abs().max()), 'incoming_proxies_unchanged': True, 'canonical_state_bit_equal': True})
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
    assert len(records) == sum((t['root_position'] > kwargs['root'] for t in replay['trace']))
    return (samples, replay, intervention, records)

def tensor_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all((tensor_equal(v, b[k]) for k, v in a.items()))
    return a == b
import json
_folded_window_diagnostics = _load_theme('window_diagnostics')
file_hash = _folded_window_diagnostics.file_hash

def map_masks(provider, proxies, mode):
    import torch
    assert mode in ('native', 'slots64', 'norm64', 'norm32cpu')
    if mode == 'native':
        return NATIVE(provider, proxies)
    anchor, velocity = (provider.anchor.clone(), provider.velocity.clone())
    if mode == 'slots64':
        anchor, velocity, proxies = (anchor.cpu().double(), velocity.cpu().double(), proxies.cpu().double())
    owners = []
    for owner in range(provider.num_masks + 1):
        if owner < provider.num_masks:
            y = proxies[owner].detach()
            if provider.order:
                velocity = provider.beta_prime * velocity + (1 - provider.beta_prime) * (y - anchor)
            anchor = y
        else:
            anchor = anchor + velocity
        slots = torch.stack([anchor + slot * velocity for slot in range(provider.num_masks)])
        slots = slots.to(device=provider.scale.device, dtype=torch.float32)
        if mode in ('norm64', 'norm32cpu'):
            values = slots.cpu().double() if mode == 'norm64' else slots.cpu()
            norms = values.norm(dim=-1, keepdim=True).to(device=slots.device, dtype=slots.dtype)
        else:
            norms = slots.norm(dim=-1, keepdim=True)
        assert torch.isfinite(slots).all() and torch.isfinite(norms).all() and (norms > 0).all()
        owners.append(provider.scale * slots / norms)
    result = torch.stack(owners)
    assert torch.isfinite(result).all()
    return result
from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
NATIVE = OutputHiddenMaskProvider.masks_per_owner

def replay(model, states, mode, intervention_call, **kwargs):
    import torch
    calls, differences = (0, [])

    def mapped(provider, proxies):
        nonlocal calls
        calls += 1
        original = NATIVE(provider, proxies)
        if calls <= intervention_call or mode == 'native':
            return original
        saved = [provider.anchor.clone(), provider.velocity.clone(), proxies.clone()]
        result = map_masks(provider, proxies, mode)
        assert all((torch.equal(a, b) for a, b in zip(saved, [provider.anchor, provider.velocity, proxies])))
        differences.append(float((result - original).abs().max()))
        return result
    with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', mapped):
        samples, evidence, intervention, records = synchronized_transfer(model, states, **kwargs)
    assert OutputHiddenMaskProvider.masks_per_owner is NATIVE
    return (samples, evidence, records, differences)

def profile_norm(provider, proxies, out):
    import torch
    result = {'available': False}
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            NATIVE(provider, proxies)
            torch.cuda.synchronize()
        trace = out / 'provider.chrome.json'
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text(encoding='utf-8'))['traceEvents']
        norms = [e for e in events if e.get('name') == 'aten::linalg_vector_norm' and e.get('cat') == 'cpu_op']
        ids = {e.get('args', {}).get('External id') for e in norms}
        kernels = [e for e in events if e.get('cat') == 'kernel']
        matched = [e for e in kernels if e.get('args', {}).get('External id') in ids]
        result.update(available=bool(kernels), trace_sha256=file_hash(trace), cuda_kernel_count=len(kernels), norm_cpu_calls=len(norms), norm_correlated_kernel_names=sorted({e['name'] for e in matched}), all_kernel_names=sorted({e['name'] for e in kernels}))
    except (RuntimeError, OSError) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result

def mapped(provider, proxies, alternative=True):
    import torch
    anchor, velocity = (provider.anchor.clone(), provider.velocity.clone())
    owners, details = ([], [])
    for owner in range(provider.num_masks + 1):
        if owner < provider.num_masks:
            y = proxies[owner].detach().float()
            if provider.order:
                velocity = provider.beta_prime * velocity + (1 - provider.beta_prime) * (y - anchor)
            anchor = y
        else:
            anchor = anchor + velocity
        slots = torch.stack([anchor + i * velocity for i in range(provider.num_masks)])
        native = slots.norm(dim=-1, keepdim=True)
        alt = (slots * slots).sum(dim=-1, keepdim=True).sqrt()
        reference = slots.cpu().double().norm(dim=-1, keepdim=True).float().to(slots.device)
        norms = alt if alternative else native
        assert torch.isfinite(norms).all() and (norms > 0).all()
        owners.append(provider.scale * slots / norms)
        details.append({'owner': owner, 'native_norm': native.cpu().flatten().tolist(), 'alternative_norm': alt.cpu().flatten().tolist(), 'reference64_rounded_norm': reference.cpu().flatten().tolist(), 'native_reference_max_abs': float((native - reference).abs().max()), 'alternative_reference_max_abs': float((alt - reference).abs().max())})
    result = torch.stack(owners)
    assert torch.isfinite(result).all()
    return (result, details)

def run(c, donor, post):
    import torch
    calls, deltas = (0, [])

    def substitute(provider, proxies):
        nonlocal calls
        calls += 1
        original = NATIVE(provider, proxies)
        if not post or calls <= c.call:
            return original
        state = [provider.anchor.clone(), provider.velocity.clone(), proxies.clone()]
        result, _ = mapped(provider, proxies)
        assert all((torch.equal(a, b) for a, b in zip(state, [provider.anchor, provider.velocity, proxies])))
        deltas.append(float((result - original).abs().max()))
        return result
    with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', substitute):
        samples, evidence, _, records = synchronized_transfer(c.model, c.states, donor=donor, **c.keywords)
    assert OutputHiddenMaskProvider.masks_per_owner is NATIVE
    return (samples, evidence, records, deltas)

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

def run__diagnose_q178_phi_rms(c, donor, mode):
    import torch
    norm = c.model.hf.model.layers[0].post_attention_layernorm
    active, captured, changes = ({}, {}, [])

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
        changes.append({'root': 85, 'row': 4, 'other_rows_bit_equal': True, 'replacement_bit_equal': torch.equal(replacement, output[:, 4:5]), 'norm_output_max_abs': float((replacement - output[:, 4:5]).abs().max())})
        return changed
    handles = []
    try:
        handles = [c.model.register_forward_pre_hook(before, with_kwargs=True), norm.register_forward_hook(replace)]
        samples, replay, _, records = synchronized_transfer(c.model, c.states, donor=donor, **c.keywords)
    finally:
        for handle in handles:
            handle.remove()
    assert captured and len(changes) == (0 if mode == 'native' else 1)
    assert not norm._forward_hooks and (not c.model._forward_pre_hooks) and (not c.model._forward_hooks)
    return (samples, replay, records, captured, changes)
import copy

def linear_row(module, inputs, row, mode):
    import torch.nn.functional as F
    x = inputs[:, row:row + 1]
    if mode == 'singleton':
        return F.linear(x, module.weight, module.bias)
    assert mode == 'reference64'
    bias = None if module.bias is None else module.bias.cpu().double()
    return F.linear(x.cpu().double(), module.weight.cpu().double(), bias).to(inputs.device, inputs.dtype)

def run__diagnose_q178_qproj(c, graft, mode, capture=False):
    import torch
    active, frames, changes = ({}, {}, [])
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
                    a, b = (getattr(original, key), getattr(saved, key))
                    assert a.data_ptr() != b.data_ptr() and torch.equal(a, b) and (a.stride() == b.stride())
            frames[current] = {'inputs': (args[0] if args else kwargs['inputs_embeds']).detach().clone(), 'positions': positions.detach().clone(), 'attention': kwargs['attention_mask_4d'].detach().clone(), 'cache': cloned}

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
        changes.append({'root': current, 'row': 4, 'other_rows_bit_equal': True, 'q_projection_max_abs': float((replacement - output[:, 4:5]).abs().max()), 'replacement_bit_equal': torch.equal(replacement, output[:, 4:5])})
        return changed

    def after(_module, _args, _kwargs, output):
        if capture and active['root'] in frames:
            frames[active['root']].update(logits=output.logits.detach().clone(), hidden=output.last_hidden_state.detach().clone())
    handles = []
    try:
        handles = [c.model.register_forward_pre_hook(before, with_kwargs=True), projection.register_forward_hook(projection_after), c.model.register_forward_hook(after, with_kwargs=True)]
        donor_case = 'output_order0' if graft else 'output_order1'
        donor = c.old[f'baseline:{donor_case}:85']['current_masks'][0, 1]
        samples, replay, _, records = synchronized_transfer(c.model, c.states, donor=donor, **c.keywords)
    finally:
        for handle in handles:
            handle.remove()
    assert not c.model._forward_hooks and (not c.model._forward_pre_hooks) and (not projection._forward_hooks)
    assert len(changes) == (0 if mode == 'native' else 1)
    if capture:
        assert set(frames) == {85, 86}
    return (samples, replay, records, frames, changes)

def exact_block(model, frame):
    import torch
    cache = copy.deepcopy(frame['cache'])
    before = [(layer.keys.clone(), layer.values.clone()) for layer in frame['cache'].layers]
    output = model(frame['inputs'], frame['positions'], attention_mask_4d=frame['attention'], past_key_values=cache, last_hidden_only=True)
    assert torch.equal(output.logits, frame['logits']) and torch.equal(output.last_hidden_state, frame['hidden'])
    assert all((torch.equal(layer.keys, k) and torch.equal(layer.values, v) for layer, (k, v) in zip(frame['cache'].layers, before)))
    return {'logits_bit_equal': True, 'final_hidden_bit_equal': True, 'saved_cache_unchanged': True}
import itertools

def delta(value, reference):
    import torch
    value = value.detach().cpu().float()
    reference = reference.detach().cpu().float()
    assert value.shape == reference.shape and torch.isfinite(value).all()
    return {'bit_equal': torch.equal(value, reference), 'changed_coordinates': int((value != reference).sum()), 'max_abs': float((value - reference).abs().max()), 'L2': float((value - reference).norm())}

def phi_factors(slots, scale):
    import torch
    assert slots.dtype == scale.dtype == torch.float32 and slots.shape[0] == 2
    native_norm = slots.norm(dim=-1, keepdim=True)[1]
    exact_norm = slots[1].cpu().double().norm().to(slots.device)
    values = {}
    for norm64, multiply64, divide64 in itertools.product((0, 1), repeat=3):
        denominator = exact_norm if norm64 else native_norm
        numerator = scale.double() * slots[1].double() if multiply64 else scale * slots[1]
        value = numerator.double() / denominator.double() if divide64 else numerator.float() / denominator.float()
        values[f'{norm64}{multiply64}{divide64}'] = value.float()
    return (values, {'norm32': float(native_norm), 'norm64': float(exact_norm), 'norm64_rounded32_equal': bool(torch.equal(native_norm, exact_norm.float().reshape_as(native_norm)))})

def rms_factors(inputs, weight, epsilon, bits):
    import torch
    assert len(bits) == 5 and all((x in (0, 1) for x in bits))
    dtype = lambda high: torch.float64 if high else torch.float32
    row = inputs[:, 4:5]
    value = row.to(dtype(bits[0])).pow(2)
    value = value.to(dtype(bits[1])).mean(-1, keepdim=True) + epsilon.to(dtype(bits[1]))
    inverse = torch.rsqrt(value.to(dtype(bits[2])))
    normalized = row.to(dtype(bits[3])) * inverse.to(dtype(bits[3]))
    result = weight.to(dtype(bits[4])) * normalized.to(dtype(bits[4]))
    return (result.float(), {'variance_plus_epsilon': float(value), 'inverse': float(inverse)})
