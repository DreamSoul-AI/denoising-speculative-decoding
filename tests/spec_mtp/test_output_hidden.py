"""Output-space state math, final-norm observation and real tiny lossless loop."""

from spec_mtp.masks import build_mask_provider

import pytest
import torch

from spec_mtp.masks import OutputHiddenMaskProvider
from spec_mtp.decode import SpecDecoder, ar_generate
from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(7)
    m = HFCausalLM(tiny_config={'vocab_size': 64, 'hidden_size': 32, 'num_layers': 2,
                              'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 128})
    m.hf.lm_head.weight = m.hf.model.embed_tokens.weight
    m.hf.config.tie_word_embeddings = True
    return m


def spec(order=1, impl='efficient'):
    return _convert({'method': 'output_hidden', 'num_masks': 2, 'impl': impl,
                     'tree': {'mode': 'static', 'branches': [1, 1], 'pruning': False},
                     'output': {'order': order, 'beta': .9, 'beta_prime': .99}})


def test_only_final_norm_matches_all_layer_reference_and_preserves_logits(model):
    embeds = model.embed(torch.tensor([[1, 2, 3, 4]]))
    pos = torch.arange(4).unsqueeze(0)
    full = model(embeds, pos, output_hidden_states=True)
    last = model(embeds, pos, last_hidden_only=True)
    assert torch.equal(last.last_hidden_state, full.hidden_states[-1])
    assert torch.equal(last.logits, full.logits)
    assert last.hidden_states is None
    assert not model.hf.model.norm._forward_hooks
    with pytest.raises(ValueError, match='no all-layer'):
        model(embeds, pos, last_hidden_only=True, output_hidden_states=True)


def test_hook_removed_when_forward_fails(model, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('injected norm failure')
    monkeypatch.setattr(model.hf.model.norm, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        model(model.embed(torch.tensor([[1]])), torch.tensor([[0]]), last_hidden_only=True)
    assert not model.hf.model.norm._forward_hooks


def test_state_observation_delay_proxy_updates_and_norm_mapping():
    provider = OutputHiddenMaskProvider(2, torch.tensor([[3., 4.], [3., 4.]]), beta=.5, beta_prime=.75)
    provider.init_hidden(torch.tensor([[1., 0.], [3., 2.]]))
    assert torch.equal(provider.anchor, torch.tensor([3., 2.]))
    assert torch.equal(provider.velocity, torch.tensor([1., 1.]))
    anchor, velocity = provider.anchor.clone(), provider.velocity.clone()
    proxies = torch.tensor([[5., 2.], [7., 4.]])
    masks = provider.masks_per_owner(proxies)
    first_v = .75 * velocity + .25 * (proxies[0] - anchor)
    second_v = .75 * first_v + .25 * (proxies[1] - proxies[0])
    slots = torch.stack([torch.stack([proxies[0], proxies[0] + first_v]),
                         torch.stack([proxies[1], proxies[1] + second_v]),
                         torch.stack([proxies[1] + second_v, proxies[1] + 2 * second_v])])
    assert torch.allclose(masks, 5 * slots / slots.norm(dim=-1, keepdim=True))
    assert torch.equal(provider.anchor, anchor) and torch.equal(provider.velocity, velocity)
    assert torch.allclose(masks.norm(dim=-1), torch.full((3, 2), 5.))
    assert not torch.equal(masks[0], masks[1])
    # The current generated root has no observation yet. Only after its next
    # block forward does the exact root hidden advance committed state.
    provider.observe_committed(torch.tensor([6., 2.]))
    assert torch.equal(provider.velocity, torch.tensor([2., .5]))


@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('temperature', [0., .7])
@pytest.mark.parametrize('impl', ['naive', 'efficient'])
def test_output_chain_lossless_sampling_budget_and_exact_state_positions(model, order, temperature, impl):
    cfg = _convert({'max_new_tokens': 12, 'temperature': temperature, 'stop_on_eos': False})
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    calls = []
    handle = model.hf.model.norm.register_forward_hook(lambda _m, _a, out: calls.append(out.shape[1]))
    try:
        out, stats = SpecDecoder(model, spec(order, impl), cfg,
                                 generator=torch.Generator().manual_seed(123), mask_policy=build_mask_provider(spec(order, impl), model.embedding_table)).generate(prompt)
    finally:
        handle.remove()
    ar, _ = ar_generate(model, prompt, cfg, generator=torch.Generator().manual_seed(123))
    assert torch.equal(out, ar)
    assert len(calls) == 1 + stats['decode_calls']
    assert out.shape[1] == prompt.shape[1] + 12
    for trace in stats['trace']:
        observation = trace['output_hidden']
        assert observation['owner_count'] == 3 and observation['slot_count'] == 2
        assert observation['observed_block_rows'] == ([] if trace['stopped'] else [0] + trace['accepted_nodes'])
        if not trace['stopped']:
            assert trace['cache_after'] == trace['root_position'] + 1 + len(trace['accepted_nodes'])
    assert not model.hf.model.norm._forward_hooks


def test_output_rejects_untied_and_invalid_states(model):
    untied = HFCausalLM(tiny_config={'vocab_size': 64, 'hidden_size': 32, 'num_layers': 1})
    with pytest.raises(ValueError, match='actually tied'):
        SpecDecoder(untied, spec(), _convert({}), mask_policy=build_mask_provider(spec(), untied.embedding_table))
    with pytest.raises(ValueError, match='order0/1'):
        OutputHiddenMaskProvider(2, model.embedding_table, order=2)
    provider = OutputHiddenMaskProvider(2, model.embedding_table, order=0)
    provider.init_hidden(torch.ones(1, 32))
    with pytest.raises(ValueError, match='nonzero norms'):
        provider.masks_per_owner(torch.zeros(2, 32))
    with pytest.raises(ValueError, match='nonfinite'):
        provider.observe_committed(torch.full((32,), float('nan')))


def test_observed_exact_hiddens_equal_full_causal_prefix(model, monkeypatch):
    observed = []
    original = OutputHiddenMaskProvider.observe_committed
    def capture(self, hidden):
        observed.append(hidden.clone())
        return original(self, hidden)
    monkeypatch.setattr(OutputHiddenMaskProvider, 'observe_committed', capture)
    prompt = torch.tensor([[2, 4, 6, 8]])
    cfg = _convert({'max_new_tokens': 10, 'temperature': 0., 'stop_on_eos': False})
    out, _ = SpecDecoder(model, spec(), cfg, mask_policy=build_mask_provider(spec(), model.embedding_table)).generate(prompt)
    prefix = out[:, :len(observed)]
    reference = model(model.embed(prefix), torch.arange(prefix.shape[1]).unsqueeze(0),
                      last_hidden_only=True).last_hidden_state[0]
    assert len(observed) > prompt.shape[1]
    assert torch.allclose(torch.stack(observed), reference, rtol=1e-5, atol=1e-6)


def test_output_sampling_eos_truncation_matches_ar(model):
    prompt = torch.tensor([[2, 4, 6, 8]])
    cfg = _convert({'max_new_tokens': 12, 'temperature': 1., 'stop_on_eos': False})
    ar, _ = ar_generate(model, prompt, cfg, generator=torch.Generator().manual_seed(12))
    eos = int(ar[0, prompt.shape[1] + 3])
    cfg.stop_on_eos = True
    out, _ = SpecDecoder(model, spec(), cfg, eos_ids=[eos],
                         generator=torch.Generator().manual_seed(12), mask_policy=build_mask_provider(spec(), model.embedding_table)).generate(prompt)
    reference, _ = ar_generate(model, prompt, cfg, eos_ids=[eos], generator=torch.Generator().manual_seed(12))
    assert torch.equal(out, reference) and int(out[0, -1]) == eos


@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('mode,budget', [('dynamic',30), ('dynamic',60), ('static',15)])
def test_bf16_branching_siblings_share_depth_masks_without_mutating_state(model, monkeypatch, order, mode, budget):
    from copy import deepcopy
    import spec_mtp.decode as decoding

    m = deepcopy(model).to(torch.bfloat16)
    trees, blocks = [], []
    build = decoding.build_tree
    predict = OutputHiddenMaskProvider.masks_per_owner
    def capture_tree(*args):
        tree = build(*args)
        trees.append(tree)
        return tree
    def pure_prediction(self, proxies):
        anchor, velocity = self.anchor.clone(), self.velocity.clone()
        result = predict(self, proxies)
        assert torch.equal(self.anchor, anchor) and torch.equal(self.velocity, velocity)
        return result
    monkeypatch.setattr(decoding, 'build_tree', capture_tree)
    monkeypatch.setattr(OutputHiddenMaskProvider, 'masks_per_owner', pure_prediction)
    handle = m.register_forward_pre_hook(lambda _m, args: blocks.append(args[0][0].clone()))
    cfg = spec(order, 'naive')
    cfg.tree = _convert({'mode': mode, 'pruning': True,
                         **({'branches': [2,2]} if mode == 'static' else {})})
    cfg.block_complexity = budget
    try:
        out, stats = SpecDecoder(m, cfg, _convert({'max_new_tokens': 12, 'temperature': 0.,
                                                  'stop_on_eos': False}), mask_policy=build_mask_provider(cfg, m.embedding_table)).generate(torch.tensor([[1,2,3,4]]))
    finally:
        handle.remove()
    assert out.shape[1] == 16 and len(blocks) == 1 + stats['decode_calls']
    assert stats['verification_argmax_audit']['passed']
    assert stats['verification_argmax_audit']['checked_committed_tokens'] == 12
    for tree, block, trace in zip(trees, blocks[1:], stats['trace']):
        assert block.shape[0] == budget and tree.num_nodes == budget // 3 - 1
        if mode == 'static':
            assert tree.depths == [1,1,2,2]
        masks = block[1 + tree.num_nodes:].reshape(1 + tree.num_nodes, 2, -1)
        for depth in set(tree.depths):
            siblings = [i+1 for i,d in enumerate(tree.depths) if d == depth]
            assert all(torch.equal(masks[i], masks[siblings[0]]) for i in siblings)
        assert trace['output_hidden']['owner_count'] == 1 + tree.num_nodes
        assert trace['output_hidden']['observed_block_rows'] == ([] if trace['stopped'] else [0]+trace['accepted_nodes'])


def test_nonfirst_sibling_updates_actual_hidden_path_and_cache(model, monkeypatch):
    import spec_mtp.decode as decoding
    from spec_mtp.tree import DraftTree

    prompt = torch.tensor([[2,4,6,8]])
    cfg = _convert({'max_new_tokens': 9, 'temperature': 0., 'stop_on_eos': False})
    reference, _ = ar_generate(model, prompt, cfg)
    cursor = prompt.shape[1] + 1
    def candidates(_cfg, _probs, root):
        nonlocal cursor
        expected = int(reference[0, cursor])
        cursor += 2
        return DraftTree(root_token=root, tokens=[(expected+1) % model.vocab_size, expected],
                         depths=[1,1], parents=[0,0], probs=[.6,.4])
    observed, forwards = [], []
    observe = OutputHiddenMaskProvider.observe_committed
    def capture(self, hidden):
        observed.append(hidden.clone())
        return observe(self, hidden)
    monkeypatch.setattr(decoding, 'build_tree', candidates)
    monkeypatch.setattr(OutputHiddenMaskProvider, 'observe_committed', capture)
    handle = model.hf.model.norm.register_forward_hook(lambda _m, _a, out: forwards.append(out[0].clone()))
    tree_cfg = spec(); tree_cfg.tree.branches = [2]
    try:
        out, stats = SpecDecoder(model, tree_cfg, cfg, mask_policy=build_mask_provider(tree_cfg, model.embedding_table)).generate(prompt)
    finally:
        handle.remove()
    assert torch.equal(out, reference) and len(forwards) == 1 + stats['decode_calls']
    assert all(t['accepted_nodes'] == [2] for t in stats['trace'])
    expected = list(forwards[0][:prompt.shape[1]])
    for matrix, trace in zip(forwards[1:], stats['trace']):
        if not trace['stopped']:
            assert trace['output_hidden']['observed_block_rows'] == [0,2]
            expected.extend([matrix[0], matrix[2]])
    assert torch.equal(torch.stack(observed), torch.stack(expected))
    prefix = out[:, :len(observed)]
    causal = model(model.embed(prefix), torch.arange(prefix.shape[1]).unsqueeze(0),
                   last_hidden_only=True).last_hidden_state[0]
    assert torch.allclose(torch.stack(observed), causal, rtol=1e-5, atol=1e-6)
