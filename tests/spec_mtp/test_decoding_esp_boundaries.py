"""Boundary / error-path contracts for decoding.py and esp_mtp.py (no algorithm change).

Covers helpers that lossless suites skip, EOS mid-pass truncation, AR EOS stop,
budget=1 (prefill-only), esp_mtp aggregation / totals / warmup / CSV / empty prompts,
and run_ar_baseline=false.
"""

from __future__ import annotations

from spec_mtp.masks import build_mask_provider

import pytest
import torch

from spec_mtp import decode as decoding
from studies._shared.structure import evaluate as esp_mtp
from studies._shared.structure.config import apply_variant
from spec_mtp.decode import SpecDecoder, ar_generate, select_token, tree_probs
from studies._shared.structure.smoke_variants import VARIANTS
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


def _prompt(seed=1, length=8):
    return torch.randint(0, 64, (1, length), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------ select_token / tree_probs

def test_select_token_greedy_and_sample():
    logits = torch.tensor([0.1, 3.0, 0.2])
    assert select_token(logits, 0.0, None) == 1
    g = torch.Generator().manual_seed(0)
    drawn = {select_token(logits, 1.0, g) for _ in range(40)}
    assert drawn <= {0, 1, 2} and 1 in drawn


def test_tree_probs_temperature_scales_and_greedy_matches_softmax():
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    greedy = tree_probs(logits, 0.0)
    assert torch.allclose(greedy, torch.softmax(logits.float(), dim=-1))
    hot = tree_probs(logits, 2.0)
    assert torch.allclose(hot, torch.softmax(logits.float() / 2.0, dim=-1))
    assert not torch.allclose(hot, greedy)


# ------------------------------------------------------------ SpecDecoder / ar_generate edges

def test_max_new_tokens_one_is_prefill_only(tiny):
    """Budget 1: only the prefill root is committed; the decode loop never runs."""
    out, stats = SpecDecoder(
        tiny, apply_variant({}, 'esp_static'), _eval(max_new_tokens=1),
        generator=torch.Generator().manual_seed(0),
     mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), tiny.embedding_table)).generate(_prompt())
    assert out.size(1) == _prompt().size(1) + 1
    assert stats['new_tokens'] == 1 and stats['decode_calls'] == 0 and stats['committed'] == 0
    assert stats['accepted'] == 0 and stats['hit_context_limit'] == 0
    assert stats['block_complexity'] == SpecDecoder(
        tiny, apply_variant({}, 'esp_static'), _eval(), mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), tiny.embedding_table)).nominal_block_complexity


def test_naive_static_uses_literal_block_attention(tiny):
    """Table 4 ablation: static + naive keeps the uncached builder (line 180)."""
    spec = apply_variant({}, 'llama3/impl_naive_bc30')
    dec = SpecDecoder(tiny, spec, _eval(max_new_tokens=3), generator=torch.Generator().manual_seed(0), mask_policy=build_mask_provider(spec, tiny.embedding_table))
    assert dec.impl == 'naive' and dec._attn_cache is None and dec._dynamic_attn is None
    out, stats = dec.generate(_prompt())
    assert stats['decode_calls'] >= 1 and out.size(1) == _prompt().size(1) + 3


def test_nominal_block_complexity_static_vs_dynamic(tiny):
    static = SpecDecoder(tiny, apply_variant({}, 'esp_static'), _eval(), mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), tiny.embedding_table))
    dynamic = SpecDecoder(tiny, apply_variant({}, 'esp_dynamic'), _eval(), mask_policy=build_mask_provider(apply_variant({}, 'esp_dynamic'), tiny.embedding_table))
    assert static.impl == 'efficient' and static._attn_cache is not None
    assert dynamic.impl == 'naive' and dynamic._dynamic_attn is not None
    assert static._dynamic_attn is None and dynamic._attn_cache is None
    # Smoke presets both target block_complexity 30; the formula still runs either way.
    assert static.nominal_block_complexity == 30
    assert dynamic.nominal_block_complexity == 30


def test_eos_truncates_committed_now_mid_pass(tiny, monkeypatch):
    """Plant non-EOS root then EOS on the first verify sample → mid-pass truncation."""
    eos = 7
    seq = iter([10, eos])

    def fake(*_a, **_k):
        try:
            return next(seq)
        except StopIteration:
            return eos

    monkeypatch.setattr(decoding, 'select_token', fake)
    out, stats = SpecDecoder(
        tiny, apply_variant({}, 'esp_static'),
        _eval(max_new_tokens=10, stop_on_eos=True),
        eos_ids=[eos],
     mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), tiny.embedding_table)).generate(_prompt())
    assert out[0, -1].item() == eos
    assert stats['new_tokens'] == 2  # root + EOS
    assert stats['decode_calls'] == 1
    assert stats['committed'] == 1  # only the EOS from the decode pass


def test_ar_generate_stops_on_eos(tiny, monkeypatch):
    eos = 7
    monkeypatch.setattr(decoding, 'select_token', lambda *_a, **_k: eos)
    out, stats = ar_generate(
        tiny, _prompt(),
        _eval(max_new_tokens=10, stop_on_eos=True),
        eos_ids=[eos],
    )
    assert out[0, -1].item() == eos
    assert stats['ar_new_tokens'] == 1 and stats['ar_calls'] == 1


def test_ar_generate_respects_context_limit(tiny):
    """AR stops when the next position would exceed max_positions (pos+1 check)."""
    short = HFCausalLM(tiny_config={
        'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_layers': 1,
        'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 12,
    })
    prompt = _prompt(length=10)
    out, stats = ar_generate(short, prompt, _eval(max_new_tokens=20))
    # Prefill uses 0..9; loop emits while pos+1 <= max_positions, so length <= max_positions+1.
    assert stats['ar_new_tokens'] < 20
    assert out.size(1) <= short.max_positions + 1
    assert out.size(1) == prompt.size(1) + stats['ar_new_tokens']


def test_stop_on_eos_false_keeps_going_past_eos_id(tiny, monkeypatch):
    """With stop_on_eos=False the EOS id is just another token."""
    eos = 7
    n = {'i': 0}

    def fake(*_a, **_k):
        n['i'] += 1
        return eos if n['i'] <= 2 else 11

    monkeypatch.setattr(decoding, 'select_token', fake)
    out, stats = ar_generate(
        tiny, _prompt(),
        _eval(max_new_tokens=4, stop_on_eos=False),
        eos_ids=[eos],
    )
    assert stats['ar_new_tokens'] == 4
    assert (out[0, -4:] == torch.tensor([eos, eos, 11, 11])).all()


# ------------------------------------------------------------ esp_mtp aggregation / evaluate

def test_totals_sums_pooled_counts():
    rows = [
        {'accepted': 2, 'committed': 5, 'decode_calls': 3, 'new_tokens': 6},
        {'accepted': 1, 'committed': 4, 'decode_calls': 2, 'new_tokens': 4},
    ]
    assert esp_mtp.totals(rows) == {
        'accepted': 3, 'committed': 9, 'decode_calls': 5, 'new_tokens': 10,
    }
    assert esp_mtp.totals([]) == {
        'accepted': 0, 'committed': 0, 'decode_calls': 0, 'new_tokens': 0,
    }


def test_aggregate_categories_and_zero_decode_calls_guard():
    rows = [
        {'category': 'a', 'committed': 0, 'decode_calls': 0, 'new_tokens': 1,
         'spec_time': 0.0, 'attempt_d1': 0, 'accept_d1': 0,
         'attempt_d2': 0, 'accept_d2': 0, 'attempt_d3': 0, 'accept_d3': 0},
        {'category': 'b', 'committed': 4, 'decode_calls': 2, 'new_tokens': 5,
         'spec_time': 2.0, 'ar_time': 4.0, 'exact_match': 1,
         'attempt_d1': 2, 'accept_d1': 1,
         'attempt_d2': 1, 'accept_d2': 0, 'attempt_d3': 0, 'accept_d3': 0},
    ]
    summaries = esp_mtp._aggregate(rows)
    assert set(summaries) == {'a', 'b', 'overall'}
    # decode_calls=0 → max(..., 1) so tau = committed/1 = 0 for category a
    assert summaries['a']['tau'] is None
    assert summaries['a']['call_reduction'] is None
    assert summaries['a']['tokens_per_s'] == 0.0
    assert 'exact_match_rate' not in summaries['a']  # category a group has no ar_time
    assert summaries['overall']['exact_match_rate'] is None
    assert summaries['overall']['AR_complete'] is False
    assert summaries['overall']['AR_observed_prompts'] == 1
    assert summaries['b']['speedup'] == 2.0
    assert summaries['b']['accept_rate_d1'] == 0.5
    assert summaries['overall']['prompts'] == 2


def test_evaluate_empty_prompts_skips_warmup(tiny, tmp_path):
    """No prompts: warmup short-circuits; _aggregate is {}; CSV loggers never write a row."""
    result = esp_mtp.evaluate(
        tiny, [], [], apply_variant({}, 'esp_static'),
        _eval(warmup=True, log_per_prompt=True, run_ar_baseline=False),
        seed=0, device='cpu', save_dir=str(tmp_path),
    )
    assert result['rows'] == []
    assert result['summaries'] == {}


def test_evaluate_warmup_restores_budget_and_writes_csvs(tiny, tmp_path):
    prompts = [{'category': 'synthetic', 'input_ids': _prompt(seed=2)}]
    eval_cfg = _eval(
        max_new_tokens=6, warmup=True, warmup_tokens=2,
        log_per_prompt=True, run_ar_baseline=True,
    )
    result = esp_mtp.evaluate(
        tiny, prompts, [], apply_variant({}, 'esp_static'),
        eval_cfg, seed=3, device='cpu', save_dir=str(tmp_path),
    )
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['new_tokens'] == 6  # warmup must not leak its shorter budget
    assert row['exact_match'] == 1
    assert (tmp_path / 'spec_metrics.csv').is_file()
    assert (tmp_path / 'summary.csv').is_file()
    text = (tmp_path / 'spec_metrics.csv').read_text(encoding='utf-8')
    assert 'exact_match' in text and 'synthetic' in text
    assert result['summaries']['overall']['exact_match_rate'] == 1.0
    assert result['summaries']['overall']['prompts'] == 1


def test_evaluate_without_ar_baseline_omits_match_fields(tiny, tmp_path):
    prompts = [{'category': 'x', 'input_ids': _prompt(seed=4, length=6)}]
    result = esp_mtp.evaluate(
        tiny, prompts, [], _convert(dict(VARIANTS['esp_dynamic'])),
        _eval(max_new_tokens=4, warmup=False, log_per_prompt=False, run_ar_baseline=False),
        seed=5, device='cpu', save_dir=str(tmp_path),
    )
    row = result['rows'][0]
    assert 'exact_match' not in row and 'ar_time' not in row
    assert 'exact_match_rate' not in result['summaries']['overall']
    assert 'speedup' not in result['summaries']['overall']
    assert row['decode_calls'] >= 1


def test_warmup_restores_max_new_tokens_on_generate_failure(tiny, monkeypatch):
    dec = SpecDecoder(tiny, apply_variant({}, 'esp_static'), _eval(max_new_tokens=9), mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), tiny.embedding_table))

    def boom(_prompt_ids):
        raise RuntimeError('warmup boom')

    monkeypatch.setattr(dec, 'generate', boom)
    with pytest.raises(RuntimeError, match='warmup boom'):
        esp_mtp._warmup(dec, tiny, _prompt(), _eval(), [], False, 'cpu', 0)
    assert dec.max_new_tokens == 9


def test_esp_mtp_test_entry_wires_legacy_inputs(tiny, monkeypatch, tmp_path):
    """esp_mtp.test is the single-config CLI shim; stub legacy_inputs only."""
    prompts = [{'category': 'legacy', 'input_ids': _prompt(seed=6, length=5)}]

    def fake_legacy(cfg, model=None):
        return tiny, prompts, [], 'cpu'

    monkeypatch.setattr(esp_mtp, 'legacy_inputs', fake_legacy)
    cfg = _convert({
        'spec': dict(VARIANTS['esp_static']),
        'eval': {
            'max_new_tokens': 3, 'temperature': 0.0, 'stop_on_eos': False,
            'warmup': False, 'log_per_prompt': False, 'run_ar_baseline': True,
        },
        'run': {'seed': 7, 'save_dir': str(tmp_path)},
    })
    # NamedDict nesting: esp_mtp.test expects cfg.spec / cfg.eval / cfg.run
    summaries = esp_mtp.test(cfg)
    assert summaries['overall']['exact_match_rate'] == 1.0
    assert summaries['overall']['prompts'] == 1
