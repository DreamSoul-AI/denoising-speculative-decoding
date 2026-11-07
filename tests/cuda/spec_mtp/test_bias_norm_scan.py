"""Native magnitude controls keep the shared mean, fixed direction and argmax gate."""

from tests.conftest import norm_control_module as module

from spec_mtp.structure.algorithm.mask import build_mask_provider



from types import SimpleNamespace

import pytest

import torch


from spec_mtp.structure.algorithm.mask import EMAVelocityMaskProvider, ESPMaskProvider

@pytest.mark.parametrize('mode', ['template', 'random0', 'random1', 'random2'])
def test_native_scale_retains_mean_and_seed_direction(monkeypatch, mode):
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    print("device", torch.cuda.get_device_name(0), "cuda:0", flush=True)
    m = module(monkeypatch)
    embeds = torch.randn(10, 16, generator=torch.Generator().manual_seed(7)).to(device="cuda:0", dtype=torch.bfloat16)
    directions = []
    for scale in (0., .5, 1., 2.):
        provider = EMAVelocityMaskProvider(2, embeds, beta=.9999, level='order0', lam=.1, step_scale=scale)
        info = {}
        m.observe_initial_bias(provider, mode, info).init_from_prompt(embeds[:8])
        assert info['rounded_slot_displacement_norms'] == (provider.masks().float() - provider._level.float()).norm(dim=-1).tolist()
        zero = ESPMaskProvider(2, embeds, lam=.1)
        zero.init_from_prompt(embeds[:8])
        directions.append(provider._velocity.clone())
        for previous, new in zip(embeds[7:9], embeds[8:10]):
            assert torch.equal(provider._level, zero.masks()[0])
            if scale == 0:
                assert torch.equal(provider.masks(), zero.masks())
            provider.on_token_committed(new, previous)
            zero.on_token_committed(new, previous)
            assert torch.equal(provider._velocity, directions[-1])
        assert info['step_scale'] == scale and info['full_prompt_mean_preserved']
    assert all(torch.equal(v, directions[0]) for v in directions)

@pytest.mark.parametrize('scale', [.5, 2.])
def test_native_tiny_bf16_argmax_audit_for_scaled_bias(monkeypatch, scale):
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from studies._shared.structure.config import _convert
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    m = module(monkeypatch)
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    print("device", torch.cuda.get_device_name(0), "cuda:0", flush=True)
    torch.manual_seed(17)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2,
        'max_position_embeddings': 128}, torch_dtype='bfloat16', attn_implementation='eager').to('cuda:0')
    spec = _convert({'method': 'ema_velocity', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
        'tree': {'mode': 'dynamic', 'pruning': True},
        'ema': {'level': 'order0', 'beta': .9999, 'lam': .1, 'step_scale': scale}})
    original = build_mask_provider
    monkeypatch.setattr(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.))
    policy = m.observe_initial_bias(original(spec, model.embedding_table), 'random1', {})
    output, stats = decoding.SpecDecoder(model, spec, _convert({'temperature': 0., 'max_new_tokens': 12}), mask_policy=policy).generate(torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0"))
    assert output.shape[1] == 17
    assert stats['verification_argmax_audit']['passed'] and stats['verification_argmax_audit']['checked_committed_tokens'] == 12
