"""Fixed-frame causal timing and state/proxy interventions preserve untouched inputs."""

import copy
import torch

from test_window_observer import ROOT, model as model
from spec_mtp import model as kv_cache
from spec_mtp.masks import OutputHiddenMaskProvider


def test_fixed_masks_only_use_selected_state_and_proxy(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q241_local_factors import fixed_masks
    provider = OutputHiddenMaskProvider(2, torch.arange(35, dtype=torch.float32).reshape(5, 7) + 1)
    state = {'anchor': torch.arange(7, dtype=torch.float32) + 1, 'velocity': torch.ones(7) * .1}
    proxies = torch.arange(14, dtype=torch.float32).reshape(2, 7) + 1
    old = {k: v.clone() for k, v in state.items()}
    value = fixed_masks(provider, state, proxies)
    assert value.shape == (3, 2, 7) and torch.isfinite(value).all()
    assert all(torch.equal(v, old[k]) for k, v in state.items())
    changed = fixed_masks(provider, state, proxies + 2)
    assert not torch.equal(value, changed)
    assert all(torch.equal(v, old[k]) for k, v in state.items())


def test_future_mask_change_preserves_current_verification(model, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_q241_local_factors import replay_fixed_block
    cache = kv_cache.new_cache()
    model(model.embed(torch.tensor([[1, 2]])), torch.tensor([[0, 1]]), past_key_values=cache)
    inputs = model.embed(torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]])).detach()
    allowed = torch.cat([torch.ones(9, 2), torch.tril(torch.ones(9, 9))], dim=-1).bool()
    attention = torch.where(allowed, 0., model.min_mask_value()).reshape(1, 1, 9, 11)
    positions = torch.arange(2, 11)[None]
    output = model(inputs, positions, attention_mask_4d=attention,
        past_key_values=copy.deepcopy(cache), last_hidden_only=True)
    frame = {'inputs': inputs.clone(), 'positions': positions, 'attention': attention,
        'cache': copy.deepcopy(cache), 'logits': output.logits.clone(), 'hidden': output.last_hidden_state.clone()}
    original_masks = inputs[:, 3:].reshape(3, 2, -1)
    a, h = replay_fixed_block(model, frame, original_masks)
    assert torch.equal(a, output.logits) and torch.equal(h, output.last_hidden_state)
    b, _ = replay_fixed_block(model, frame, original_masks + .1)
    assert torch.equal(a[:, :3], b[:, :3]) and not torch.equal(a[:, 3:], b[:, 3:])
    assert torch.equal(frame['inputs'], inputs) and frame['cache'].get_seq_length() == 2


def test_frozen_targets_retain_entry_and_bound_windows(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_state_extension import frozen_targets
    selected = frozen_targets({'intervention_root': 10, 'targets': [10, 11, 13, 14, 18, 20, 22, 23]})
    assert selected == [10, 11, 13, 14, 22, 23]
    assert frozen_targets({'intervention_root': 10, 'targets': [10, 11]}) == [10, 11]


def test_greedy_seed_is_not_an_independent_sampling_axis():
    from spec_mtp.decode import select_token
    logits = torch.tensor([.1, .9, .2])
    assert select_token(logits, 0., torch.Generator().manual_seed(124)) == 1
    assert select_token(logits, 0., torch.Generator().manual_seed(127)) == 1


def test_recorded_first_divergence_uses_previous_owner_seeds():
    import json
    report = json.loads((ROOT / 'studies/output_space_cross_task/docs/local_factor_diagnosis_v1.json').read_text(encoding='utf-8'))
    assert [s['root'] for s in report['steps']] == [716, 717, 719, 720]
    for prior, current in zip(report['steps'], report['steps'][1:]):
        for left, right in zip(prior['cases'], current['cases']):
            assert right['source_root'] == prior['root']
            assert right['source_owner'] == (left['trace']['accepted_nodes'][-1] if left['trace']['accepted_nodes'] else 0)
            assert right['trace']['root_position'] == left['trace']['cache_after']
    last = report['steps'][-1]['cases']
    assert last[0]['first']['top1_id'] != last[0]['first']['target_id']
    assert last[1]['first']['top1_id'] == last[1]['first']['target_id']
    source = next(b for b in report['blocks'] if b['label'] == 'r719:S0P1' and b['repeat'] == 0)
    assert source['first_seed'] == last[1]['first']
    assert all(b['verification_logits_bit_equal'] for b in report['blocks'] if b['root'] == 720)
