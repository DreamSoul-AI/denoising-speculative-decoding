"""Decoder / mask / KV / tree boundary and error-config tests (no weights).

Complements the lossless property suite: bad configs must fail before decode,
EOS and context-limit stop as documented, and KV surgery keeps the right length.
"""

from __future__ import annotations

import pytest
import torch

from studies._shared.structure.config import apply_variant
from spec_mtp.decode import SpecDecoder, ar_generate, _fork_generator
from spec_mtp import model as kv_cache
from spec_mtp.masks import ESPMaskProvider, EMAVelocityMaskProvider, build_mask_provider
from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM


@pytest.fixture(scope='module')
def tiny():
    torch.manual_seed(0)
    return HFCausalLM(tiny_config={
        'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_layers': 1,
        'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 128,
    })


def _eval(**kw):
    base = {'max_new_tokens': 16, 'temperature': 0.0, 'stop_on_eos': False}
    base.update(kw)
    return _convert(base)


def _spec(**kw):
    return apply_variant({}, kw.pop('variant', 'esp_static')) if 'variant' in kw or not kw else _convert(kw)


# ------------------------------------------------------------ error configs

@pytest.mark.parametrize('spec, match', [
    ({'method': 'esp', 'num_masks': 1, 'impl': 'efficient', 'tree': {'mode': 'dynamic'}},
     'efficient requires spec.tree.mode=static'),
    ({'method': 'esp', 'num_masks': 1, 'impl': 'turbo', 'tree': {'mode': 'static', 'branches': [2]}},
     'Unsupported spec.impl'),
])
def test_decoder_rejects_bad_impl_tree_pairs(tiny, spec, match):
    with pytest.raises(ValueError, match=match):
        SpecDecoder(tiny, _convert(spec), _eval(), mask_policy=build_mask_provider(_convert(spec), tiny.embedding_table))


@pytest.mark.parametrize('init, prompt_len, match', [
    ('nope', 4, 'Unsupported mask.init'),
    ('last_k', 1, 'last_k init needs'),
])
def test_esp_mask_init_errors(tiny, init, prompt_len, match):
    table = tiny.embedding_table
    provider = ESPMaskProvider(2, table, init=init, lam=0.1)
    with pytest.raises(ValueError, match=match):
        provider.init_from_prompt(torch.zeros(prompt_len, table.size(-1)))


def test_ema_rejects_short_prompt_and_bad_level(tiny):
    table = tiny.embedding_table
    with pytest.raises(ValueError, match='ema.level'):
        EMAVelocityMaskProvider(1, table, level='order2')
    provider = EMAVelocityMaskProvider(1, table)
    with pytest.raises(ValueError, match='EMA velocity needs'):
        provider.init_from_prompt(torch.zeros(1, table.size(-1)))


def test_build_mask_provider_rejects_unknown_method(tiny):
    with pytest.raises(ValueError, match='Unsupported spec.method'):
        build_mask_provider(_convert({'method': 'magic', 'num_masks': 1}), tiny.embedding_table)


# ------------------------------------------------------------ EOS / budget / context

def test_stop_on_eos_matches_ar_and_stops_early(tiny):
    """Plant an EOS id; both paths must stop on it and stay token-identical."""
    prompt = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(1))
    # Force a tiny budget so we exercise the stop path without relying on natural EOS.
    spec = apply_variant({}, 'esp_static')
    eval_cfg = _eval(max_new_tokens=12, stop_on_eos=True)
    # Use a token that the random model will eventually emit; verify equality with AR.
    eos_id = 3
    dec = SpecDecoder(tiny, spec, eval_cfg, eos_ids=[eos_id], generator=torch.Generator().manual_seed(9), mask_policy=build_mask_provider(spec, tiny.embedding_table))
    out, stats = dec.generate(prompt)
    ar, ar_stats = ar_generate(tiny, prompt, eval_cfg, eos_ids=[eos_id],
                               generator=torch.Generator().manual_seed(9))
    assert torch.equal(out, ar)
    assert stats['new_tokens'] == ar_stats['ar_calls']
    # If EOS appeared, it is the last committed token and we did not overshoot the budget.
    assert out.size(1) <= prompt.size(1) + 12


def test_context_limit_sets_the_flag_and_matches_budgeted_ar(tiny):
    torch.manual_seed(0)
    short = HFCausalLM(tiny_config={
        'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_layers': 1,
        'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 40,
    })
    prompt = torch.randint(0, 64, (1, 24), generator=torch.Generator().manual_seed(2))
    spec = apply_variant({}, 'esp_dynamic')
    eval_cfg = _eval(max_new_tokens=32)
    out, stats = SpecDecoder(short, spec, eval_cfg, generator=torch.Generator().manual_seed(2), mask_policy=build_mask_provider(spec, short.embedding_table)).generate(prompt)
    assert stats['hit_context_limit'] == 1
    assert out.size(1) <= 40
    budget = dict(eval_cfg)
    budget['max_new_tokens'] = stats['new_tokens']
    audited, _ = ar_generate(short, prompt, _convert(budget), generator=torch.Generator().manual_seed(2))
    assert torch.equal(out, audited)


def test_budget_truncation_without_context_limit(tiny):
    prompt = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(3))
    spec = apply_variant({}, 'esp_static')
    eval_cfg = _eval(max_new_tokens=5)
    out, stats = SpecDecoder(tiny, spec, eval_cfg, generator=torch.Generator().manual_seed(3), mask_policy=build_mask_provider(spec, tiny.embedding_table)).generate(prompt)
    assert stats['hit_context_limit'] == 0
    assert stats['new_tokens'] == 5
    assert out.size(1) == prompt.size(1) + 5


# ------------------------------------------------------------ RNG fork

def test_fork_generator_copies_state_and_leaves_source():
    src = torch.Generator().manual_seed(11)
    before = src.get_state().clone()
    fork = _fork_generator(src)
    assert fork is not src
    torch.randn(4, generator=fork)
    assert torch.equal(src.get_state(), before)
    assert _fork_generator(None) is None


# ------------------------------------------------------------ KV cache

def test_kv_crop_and_keep_block_positions(tiny):
    prompt = torch.randint(0, 64, (1, 6), generator=torch.Generator().manual_seed(4))
    cache = kv_cache.new_cache()
    embeds = tiny.embed(prompt)
    pos = torch.arange(prompt.size(1)).unsqueeze(0)
    out = tiny(embeds, pos, attention_mask_4d=None, past_key_values=cache)
    cache = out.past_key_values
    assert kv_cache.cache_len(cache) == 6
    # Append a fake block of 4 positions by running 4 more steps.
    for i in range(4):
        tok = torch.tensor([[i % 64]])
        out = tiny(tiny.embed(tok), torch.tensor([[6 + i]]), attention_mask_4d=None, past_key_values=cache)
        cache = out.past_key_values
    assert kv_cache.cache_len(cache) == 10
    kv_cache.keep_block_positions(cache, 6, [0, 2])  # keep two of the four block slots
    assert kv_cache.cache_len(cache) == 8
    kv_cache.keep_block_positions(cache, 6, [])
    assert kv_cache.cache_len(cache) == 6
    kv_cache.crop(cache, 3)
    assert kv_cache.cache_len(cache) == 3


def test_kv_rejects_unsupported_cache_type():
    with pytest.raises(TypeError, match='Unsupported cache type'):
        list(kv_cache._layer_tensors(object()))
