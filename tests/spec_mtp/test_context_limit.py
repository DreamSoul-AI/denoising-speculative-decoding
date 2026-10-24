"""Context limit audit (option A): AR stops at the token count the speculative pass reached."""

from spec_mtp.masks import build_mask_provider

import torch

import studies._shared.structure.evaluate as esp_mtp
from spec_mtp.decode import SpecDecoder, ar_generate
from studies._shared.structure.smoke_variants import VARIANTS
from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM


def _model(max_pos):
    torch.manual_seed(123)
    return HFCausalLM(tiny_config={'vocab_size': 256, 'hidden_size': 64, 'intermediate_size': 128,
                                   'num_layers': 2, 'num_heads': 4, 'num_kv_heads': 2,
                                   'max_position_embeddings': max_pos}).eval()


def _prompts():
    g = torch.Generator().manual_seed(0)
    return [{'category': 'synthetic', 'input_ids': torch.randint(0, 256, (1, 32), generator=g)} for _ in range(3)]


def _eval(variant, max_pos, temperature):
    cfg = {'max_new_tokens': 32, 'temperature': temperature, 'stop_on_eos': False,
           'run_ar_baseline': True, 'log_per_prompt': False, 'warmup': False}
    return esp_mtp.evaluate(_model(max_pos), _prompts(), [], _convert(dict(VARIANTS[variant])),
                            _convert(cfg), 123, 'cpu')


def test_spec_output_is_prefix_of_unbounded_ar():
    model = _model(56)
    cfg = _convert({'max_new_tokens': 32, 'temperature': 0.0, 'stop_on_eos': False})
    for item in _prompts():
        out, stats = SpecDecoder(model, _convert(dict(VARIANTS['esp_static'])), cfg, mask_policy=build_mask_provider(_convert(dict(VARIANTS['esp_static'])), model.embedding_table)).generate(item['input_ids'])
        ar, _ = ar_generate(model, item['input_ids'], cfg)
        assert stats['hit_context_limit'] == 1
        assert out.size(1) < ar.size(1) and torch.equal(ar[:, :out.size(1)], out)


def test_context_limit_cells_audit_on_the_reached_budget():
    for variant in ('esp_static', 'esp_dynamic', 'ema_velocity'):
        for temperature in (0.0, 0.8):
            out = _eval(variant, 56, temperature)
            assert all(r['hit_context_limit'] == 1 for r in out['rows'])
            assert out['summaries']['overall']['exact_match_rate'] == 1.0
            assert all(r['ar_calls'] == r['new_tokens'] for r in out['rows'])


def test_no_change_without_context_limit():
    out = _eval('esp_static', 1024, 0.0)
    assert all(r['hit_context_limit'] == 0 and r['ar_calls'] == 32 for r in out['rows'])
    assert out['summaries']['overall']['exact_match_rate'] == 1.0
