"""Hypothesis: losslessness over every shipped spec preset (smoke + every paper esp_mtp preset),
random prompts, budgets, temperatures and seeds, plus the context-limit audit invariant.

Tests only; the decoder is unchanged. Vocab 256 covers every preset's token range."""

from spec_mtp.masks import build_mask_provider

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from studies._shared.structure.config import apply_variant
from spec_mtp.decode import SpecDecoder, ar_generate
from studies._shared.structure.paper_variants import PAPER_VARIANTS
from studies._shared.structure.smoke_variants import VARIANTS
from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM

# The alignment_probe preset only configures masks for the probe (no tree, no SpecDecoder).
PRESETS = sorted(k for k in {**VARIANTS, **PAPER_VARIANTS} if not k.endswith('/alignment_probe'))
_MODELS = {}


def _model(max_positions):
    if max_positions not in _MODELS:
        torch.manual_seed(7)
        _MODELS[max_positions] = HFCausalLM(tiny_config={
            'vocab_size': 256, 'hidden_size': 64, 'intermediate_size': 128, 'num_layers': 2,
            'num_heads': 4, 'num_kv_heads': 2, 'max_position_embeddings': max_positions})
    return _MODELS[max_positions]


def _decode(model, preset, prompt, max_new, temperature, seed):
    eval_cfg = _convert({'max_new_tokens': max_new, 'temperature': temperature, 'stop_on_eos': False})
    spec_cfg = apply_variant({}, preset)
    dec = SpecDecoder(model, spec_cfg, eval_cfg, generator=torch.Generator().manual_seed(seed), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
    out, stats = dec.generate(prompt)
    return out, stats, eval_cfg


cases = dict(
    preset=st.sampled_from(PRESETS),
    prompt_len=st.integers(2, 40),
    max_new=st.integers(1, 20),
    temperature=st.sampled_from([0.0, 0.7, 1.0]),
    seed=st.integers(0, 2**16),
)


@settings(max_examples=60, deadline=None)
@given(**cases)
def test_spec_equals_ar_for_any_preset(preset, prompt_len, max_new, temperature, seed):
    model = _model(1024)
    prompt = torch.randint(0, 256, (1, prompt_len), generator=torch.Generator().manual_seed(seed))
    out, stats, eval_cfg = _decode(model, preset, prompt, max_new, temperature, seed)
    ar_out, ar_stats = ar_generate(model, prompt, eval_cfg, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(out, ar_out), (preset, prompt_len, max_new, temperature, seed)
    assert stats['new_tokens'] == max_new and not stats['hit_context_limit']
    assert stats['committed'] >= stats['decode_calls']  # tau >= 1 (max_new 1 needs no decode call)
    assert stats['accepted'] >= stats['accept_by_depth'][1] >= 0


@settings(max_examples=40, deadline=None)
@given(**cases)
def test_context_limit_output_is_the_budgeted_ar_output(preset, prompt_len, max_new, temperature, seed):
    """Option A invariant: when the speculative pass stops at the position table, its output
    equals AR run to the same token count, and is a prefix of AR's own (longer) output."""
    max_positions = 48
    model = _model(max_positions)
    prompt_len = min(prompt_len, 30)
    max_new = max_new + 20  # push past the 48-position table
    prompt = torch.randint(0, 256, (1, prompt_len), generator=torch.Generator().manual_seed(seed))
    out, stats, eval_cfg = _decode(model, preset, prompt, max_new, temperature, seed)
    full, _ = ar_generate(model, prompt, eval_cfg, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(out, full[:, :out.size(1)])
    if stats['hit_context_limit']:
        budget = dict(eval_cfg)
        budget['max_new_tokens'] = stats['new_tokens']
        audited, _ = ar_generate(model, prompt, _convert(budget), generator=torch.Generator().manual_seed(seed))
        assert torch.equal(out, audited)
        assert out.size(1) <= max_positions
    else:
        assert torch.equal(out, full)


def test_presets_cover_every_paper_cell():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / 'configs' / 'hf'
    family = {'llama3_1_8b': 'llama3', 'llama3_2_3b': 'llama3', 'qwen3_8b': 'qwen3', 'qwen3_32b': 'qwen3'}
    cells = {f'{family[p.parts[-3]]}/{p.stem}' for p in root.glob('*/*/*.yaml')
             if yaml.safe_load(p.read_text(encoding='utf-8'))['run']['algorithm'] == 'esp_mtp'}
    assert cells and cells <= set(PRESETS)


@pytest.mark.parametrize('preset', PRESETS)
def test_every_preset_builds_a_decoder(preset):
    eval_cfg = _convert({'max_new_tokens': 2, 'temperature': 0.0, 'stop_on_eos': False})
    SpecDecoder(_model(1024), apply_variant({}, preset), eval_cfg, generator=torch.Generator().manual_seed(0), mask_policy=build_mask_provider(apply_variant({}, preset), _model(1024).embedding_table))


def test_sample_init_is_token_identical_at_positive_temperature():
    """fix/mask-init-rng: mask init draws from a fork of the sampling generator."""
    model = _model(1024)
    prompt = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(1))
    out, _, eval_cfg = _decode(model, 'llama3/init_sample_bc30', prompt, 16, 1.0, 1)
    ar_out, _ = ar_generate(model, prompt, eval_cfg, generator=torch.Generator().manual_seed(1))
    assert torch.equal(out, ar_out)


def test_sample_init_is_token_identical_at_temperature_zero():
    model = _model(1024)
    prompt = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(1))
    out, _, eval_cfg = _decode(model, 'llama3/init_sample_bc30', prompt, 16, 0.0, 1)
    assert torch.equal(out, ar_generate(model, prompt, eval_cfg, generator=torch.Generator().manual_seed(1))[0])


def test_sample_init_noise_is_unchanged_by_the_fork():
    """The fork starts from the sampling generator's state, so the mask noise equals the old
    shared-generator draw; only the later token sampling stream differs (now equal to AR)."""
    from spec_mtp.decode import _fork_generator
    from spec_mtp.masks import build_mask_provider

    table = torch.randn(64, 8, generator=torch.Generator().manual_seed(3))
    spec = apply_variant({}, 'llama3/init_sample_bc30')
    shared = torch.Generator().manual_seed(5)
    old = build_mask_provider(spec, table, generator=shared)
    old.init_from_prompt(torch.zeros(4, 8))
    sampling = torch.Generator().manual_seed(5)
    new = build_mask_provider(spec, table, generator=_fork_generator(sampling))
    new.init_from_prompt(torch.zeros(4, 8))
    assert torch.equal(old.masks(), new.masks())
    assert torch.equal(sampling.get_state(), torch.Generator().manual_seed(5).get_state())  # not consumed
    assert _fork_generator(None) is None
