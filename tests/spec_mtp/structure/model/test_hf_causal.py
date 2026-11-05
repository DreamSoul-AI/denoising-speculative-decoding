"""Model resource ownership and concrete KV selection."""

import pytest


from spec_mtp.structure.model import hf_causal as kv_cache

def test_kv_rejects_unsupported_cache_type():
    with pytest.raises(TypeError, match='Unsupported cache type'):
        list(kv_cache._layer_tensors(object()))
