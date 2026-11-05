"""Model resource ownership and concrete KV selection."""
import pytest
import torch

from spec_mtp.structure.model.hf_causal import HFCausalLM, new_cache, keep_block_positions, cache_len


def test_borrowed_model_survives_close_and_closed_session_rejects_forward():
    from transformers import LlamaConfig, LlamaForCausalLM
    native = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True))
    model = HFCausalLM(module=native)
    assert model.tied_embeddings
    model.close()
    model.close()
    assert native.get_input_embeddings().weight.shape == (16, 8)
    with pytest.raises(RuntimeError, match='closed'):
        model(model.embed(torch.tensor([[1]])), torch.tensor([[0]]))


def test_cache_retains_each_layers_own_values_and_requested_path():
    cache = new_cache()
    originals = []
    for layer in range(2):
        keys = torch.arange(24.).reshape(1, 1, 6, 4) + layer * 100
        values = keys + 50
        cache.update(keys, values, layer)
        originals.append((keys, values))
    keep_block_positions(cache, 2, [0, 2])
    assert cache_len(cache) == 4
    for index, (keys, values) in enumerate(originals):
        assert torch.equal(cache.layers[index].keys, keys[:, :, [0, 1, 2, 4]])
        assert torch.equal(cache.layers[index].values, values[:, :, [0, 1, 2, 4]])



from spec_mtp.structure.model import hf_causal as kv_cache

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


def test_kv_rejects_unsupported_cache_type():
    with pytest.raises(TypeError, match='Unsupported cache type'):
        list(kv_cache._layer_tensors(object()))
