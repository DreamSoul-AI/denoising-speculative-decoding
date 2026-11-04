"""Model resource ownership and concrete KV selection."""
import pytest
import torch

from spec_mtp.model import HFCausalLM, new_cache, keep_block_positions, cache_len


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
