"""NamedDict used by the decoder config."""

from studies._shared.structure.config import NamedDict, _convert, apply_override


def test_convert_nests_attribute_access():
    cfg = _convert({'model': {'torch_dtype': 'float32'}, 'flags': [True]})
    assert isinstance(cfg, NamedDict)
    assert cfg.model.torch_dtype == 'float32'
    assert cfg.flags == [True]


def test_override_types_the_value():
    cfg = _convert({'algorithm': {'config': {'beta': 0.9}}})
    apply_override(cfg, 'algorithm.config.beta', '0.9999')
    assert cfg.algorithm.config.beta == 0.9999
