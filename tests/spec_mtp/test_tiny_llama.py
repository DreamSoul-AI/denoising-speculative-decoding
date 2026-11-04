"""The CPU smoke model. Does not download weights."""

import torch

from studies._shared.structure.config import _convert
from spec_mtp.model import build_model


def test_tiny_llama_embeds_token_ids():
    cfg = _convert({'model': {'type': 'tiny_llama', 'tiny': {
        'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_layers': 1, 'num_heads': 4, 'num_kv_heads': 2,
        'max_position_embeddings': 64,
    }}})
    model = build_model(cfg)
    ids = torch.randint(0, 32, (1, 5))
    embeds = model.embed(ids)
    assert embeds.shape == (1, 5, 16)
    out = model(embeds, position_ids=torch.arange(5).view(1, -1))
    assert out.logits.shape == (1, 5, 32)
