"""Factory method admission before any mask execution."""

import pytest



from studies._shared.structure.config import _convert

from spec_mtp.structure.algorithm.mask import build_mask_provider

def test_build_mask_provider_rejects_unknown_method():
    with pytest.raises(ValueError, match='Unsupported spec.method'):
        build_mask_provider(_convert({'method': 'magic', 'num_masks': 1}), object())
