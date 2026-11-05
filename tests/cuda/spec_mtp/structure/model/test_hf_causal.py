"""Model resource ownership and concrete KV selection."""

import pytest

import torch

from spec_mtp.structure.model import hf_causal as kv_cache

def test_borrowed_model_survives_close_and_closed_session_rejects_forward(cuda_decode, record_property):
    from transformers import LlamaConfig, LlamaForCausalLM

    device = cuda_decode.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[device.index]), torch.device(device):
        native = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert torch.get_default_device() == default_device
    before_weights = {name: value.detach().clone() for name, value in native.state_dict().items()}
    pointers = {name: (value.data_ptr(), value.untyped_storage().data_ptr())
                for name, value in native.state_dict().items()}
    embedding = native.get_input_embeddings().weight
    assert before_weights and all(value.device == device for value in before_weights.values())
    assert embedding.device == device
    model = kv_cache.HFCausalLM(module=native)
    native_forward_calls = []
    hook = native.register_forward_pre_hook(lambda *_: native_forward_calls.append(True))
    try:
        assert model.tied_embeddings
        model.close()
        model.close()
        assert native.get_input_embeddings().weight.shape == (16, 8)
        assert model.hf is native
        assert native.get_input_embeddings().weight is embedding
        after_weights = native.state_dict()
        assert after_weights.keys() == before_weights.keys()
        for name, value in after_weights.items():
            assert value.device == device
            assert value.data_ptr() == pointers[name][0]
            assert value.untyped_storage().data_ptr() == pointers[name][1]
            assert torch.equal(value, before_weights[name])
        with pytest.raises(RuntimeError, match='closed'):
            model(model.embed(torch.tensor([[1]], device=device)), torch.tensor([[0]], device=device))
        assert native_forward_calls == []
        record_property('native_forward_calls', len(native_forward_calls))
        record_property('borrowed_tensor_devices', sorted({str(value.device) for value in after_weights.values()}))
        record_property('borrowed_model_oracle', {'native_identity_preserved': True,
                        'storage_identity_preserved': True, 'weights_preserved': True,
                        'tensor_count': len(before_weights), 'role': 'borrowed_closed_guard'})
        record_property('rng_default_device_preserved', True)
    finally:
        hook.remove()
        model.close()

def test_cache_retains_each_layers_own_values_and_requested_path(cuda_decode, record_property):
    device = cuda_decode.device
    cache = kv_cache.new_cache()
    originals = []
    for layer in range(2):
        keys = torch.arange(24., device=device).reshape(1, 1, 6, 4) + layer * 100
        values = keys + 50
        assert keys.device == values.device == device
        saved_keys, saved_values = keys.clone(), values.clone()
        assert saved_keys.data_ptr() != keys.data_ptr()
        assert saved_values.data_ptr() != values.data_ptr()
        originals.append((saved_keys, saved_values))
        cache.update(keys, values, layer)
    kv_cache.keep_block_positions(cache, 2, [0, 2])
    assert kv_cache.cache_len(cache) == 4
    for index, (keys, values) in enumerate(originals):
        assert cache.layers[index].keys.device == cache.layers[index].values.device == device
        assert torch.equal(cache.layers[index].keys, keys[:, :, [0, 1, 2, 4]])
        assert torch.equal(cache.layers[index].values, values[:, :, [0, 1, 2, 4]])
    record_property('kv_snapshot_independent', True)
    record_property('kv_tensor_devices', sorted({str(value.device)
                    for layer in cache.layers for value in (layer.keys, layer.values)}))
    record_property('kv_selection', {'layers': 2, 'prefix_length': 2,
                    'block_indices': [0, 2], 'retained_positions': [0, 1, 2, 4]})

def test_kv_crop_and_keep_block_positions(cuda_tiny_model, cuda_decode, record_property):
    tiny = cuda_tiny_model()
    prompt = torch.randint(0, 64, (1, 6), generator=torch.Generator(device=cuda_decode.device).manual_seed(4), device=cuda_decode.device)
    cache = kv_cache.new_cache()
    embeds = tiny.embed(prompt)
    pos = torch.arange(prompt.size(1), device=cuda_decode.device).unsqueeze(0)
    out = tiny(embeds, pos, attention_mask_4d=None, past_key_values=cache)
    cache = out.past_key_values
    assert kv_cache.cache_len(cache) == 6
    # Append a fake block of 4 positions by running 4 more steps.
    for i in range(4):
        tok = torch.tensor([[i % 64]], device=cuda_decode.device)
        out = tiny(tiny.embed(tok), torch.tensor([[6 + i]], device=cuda_decode.device), attention_mask_4d=None, past_key_values=cache)
        cache = out.past_key_values
    assert kv_cache.cache_len(cache) == 10
    kv_cache.keep_block_positions(cache, 6, [0, 2])  # keep two of the four block slots
    assert kv_cache.cache_len(cache) == 8
    kv_cache.keep_block_positions(cache, 6, [])
    assert kv_cache.cache_len(cache) == 6
    kv_cache.crop(cache, 3)
    assert kv_cache.cache_len(cache) == 3
