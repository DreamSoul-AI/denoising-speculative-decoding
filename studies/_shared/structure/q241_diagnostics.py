"""Library helpers folded from one-shot structure scripts."""

def _load_theme(name):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        return importlib.import_module("studies._shared.structure." + name)
_folded_q178_diagnostics = _load_theme('q178_diagnostics')
synchronized_transfer = _folded_q178_diagnostics.synchronized_transfer
tensor_equal = _folded_q178_diagnostics.tensor_equal

def capture_frames(model, states, kwargs, capture):
    import copy
    frames, active = ({}, {})

    def before(_module, args, kw):
        positions = args[1] if len(args) > 1 else kw['position_ids']
        root = None if kw.get('attention_mask_4d') is None else int(positions[0, 0])
        active['root'] = root
        if capture and root in (719, 720):
            cache = kw['past_key_values']
            assert cache.get_seq_length() == root
            frames[root] = {'inputs': args[0].detach().clone(), 'positions': positions.detach().clone(), 'attention': kw['attention_mask_4d'].detach().clone(), 'cache': copy.deepcopy(cache)}

    def after(_module, _args, _kw, output):
        if capture and active['root'] in frames:
            frames[active['root']].update(logits=output.logits.detach().clone(), hidden=output.last_hidden_state.detach().clone())
    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
        samples, replay, _, records = synchronized_transfer(model, states, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert not model._forward_pre_hooks and (not model._forward_hooks) and (not model.hf.model.norm._forward_hooks)
    assert set(frames) == ({719, 720} if capture else set())
    return (samples, replay, records, frames)

def fixed_masks(provider, state, proxies):
    import torch
    originals = {k: v.clone() for k, v in state.items()}
    incoming = proxies.clone()
    provider.anchor = state['anchor'].to(provider.scale.device).clone()
    provider.velocity = state['velocity'].to(provider.scale.device).clone()
    result = provider.masks_per_owner(proxies.to(provider.scale.device))
    assert all((torch.equal(v, state[k]) for k, v in originals.items())) and torch.equal(proxies, incoming)
    assert torch.equal(provider.anchor.cpu(), state['anchor'].cpu())
    assert torch.equal(provider.velocity.cpu(), state['velocity'].cpu())
    return result

def replay_fixed_block(model, frame, masks):
    import copy
    import torch
    inputs = frame['inputs'].clone()
    inputs[:, 3:] = masks.reshape(1, 6, -1)
    assert torch.equal(inputs[:, :3], frame['inputs'][:, :3])
    cache = copy.deepcopy(frame['cache'])
    assert cache.get_seq_length() == int(frame['positions'][0, 0])
    assert all((torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values) and (a.keys.data_ptr() != b.keys.data_ptr()) for a, b in zip(cache.layers, frame['cache'].layers)))
    output = model(inputs, frame['positions'], attention_mask_4d=frame['attention'], past_key_values=cache, last_hidden_only=True)
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.last_hidden_state).all()
    assert torch.equal(output.logits[:, :3], frame['logits'][:, :3])
    assert torch.equal(output.last_hidden_state[:, :3], frame['hidden'][:, :3])
    assert torch.equal(frame['inputs'][:, :3], inputs[:, :3])
    return (output.logits.detach().cpu().clone(), output.last_hidden_state.detach().cpu().clone())
_folded_q178_diagnostics = _load_theme('q178_diagnostics')
synchronized_transfer = _folded_q178_diagnostics.synchronized_transfer
tensor_equal = _folded_q178_diagnostics.tensor_equal

def capture_frames__diagnose_q241_owner1(model, states, kwargs, capture):
    import copy
    frames, active = ({}, {})

    def before(_module, args, kw):
        positions = args[1] if len(args) > 1 else kw['position_ids']
        root = None if kw.get('attention_mask_4d') is None else int(positions[0, 0])
        active['root'] = root
        if capture and root == 717:
            cache = kw['past_key_values']
            assert cache.get_seq_length() == root
            frames[root] = {'inputs': args[0].detach().clone(), 'positions': positions.detach().clone(), 'attention': kw['attention_mask_4d'].detach().clone(), 'cache': copy.deepcopy(cache)}

    def after(_module, _args, _kw, output):
        if capture and active['root'] in frames:
            frames[active['root']].update(logits=output.logits.detach().clone(), hidden=output.last_hidden_state.detach().clone())
    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
        samples, replay, _, records = synchronized_transfer(model, states, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert not model._forward_pre_hooks and (not model._forward_hooks) and (not model.hf.model.norm._forward_hooks)
    assert set(frames) == ({717} if capture else set())
    return (samples, replay, records, frames)
_folded_window_diagnostics = _load_theme('window_diagnostics')
transfer = _folded_window_diagnostics.transfer

def selective_transfer(model, states, coordinate, **kwargs):
    import torch
    from unittest.mock import patch
    from spec_mtp.structure.model import hf_causal as kv_cache
    from spec_mtp.structure.algorithm.mask import OutputHiddenMaskProvider
    assert coordinate in ('anchor', 'velocity')
    original = OutputHiddenMaskProvider.masks_per_owner
    latest, records = ({}, [])
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
            untouched, incoming = (getattr(provider, other).clone(), proxies.clone())
            target = states[root][coordinate]
            setattr(provider, coordinate, target.to(previous.device).clone())
            result = original(provider, proxies)
            assert torch.equal(getattr(provider, coordinate).cpu(), target)
            assert torch.equal(getattr(provider, other), untouched) and torch.equal(proxies, incoming)
            records.append({'root_position': root, 'coordinate': coordinate, 'max_abs_replacement': float((previous - getattr(provider, coordinate)).abs().max()), 'other_state_bit_equal': True, 'incoming_proxies_unchanged': True, 'selected_state_bit_equal': True})
            return result
        return original(provider, proxies)
    handles = []
    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after, with_kwargs=True)]
        with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', synchronize):
            samples, replay, intervention = transfer(model, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert OutputHiddenMaskProvider.masks_per_owner is original
    assert len(records) == sum((t['root_position'] > kwargs['root'] for t in replay['trace']))
    return (samples, replay, intervention, records)
