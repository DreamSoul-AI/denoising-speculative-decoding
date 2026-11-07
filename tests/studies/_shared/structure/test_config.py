"""Pure legacy decoder config stays independent between calls."""
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from studies._shared.structure.config import NamedDict, _convert, apply_variant, spec_for
from studies._shared.structure.paper_variants import PAPER_VARIANTS
from studies._shared.structure.smoke_variants import VARIANTS

_VARIANT_PROPERTY_SETTINGS = settings(max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture])
_VARIANT_PROPERTY_PRESETS = {**VARIANTS, **PAPER_VARIANTS}
_VARIANT_PROPERTY_NAMES = st.sampled_from(sorted(_VARIANT_PROPERTY_PRESETS))
_VARIANT_UNKNOWN_NAMES = st.text(st.characters(codec='ascii', categories=['L', 'N']), min_size=1, max_size=8)


@_VARIANT_PROPERTY_SETTINGS
@given(_VARIANT_PROPERTY_NAMES,
       st.dictionaries(st.sampled_from(['impl', 'method', 'extra']), st.text(max_size=4), max_size=3))
def test_apply_variant_overrides_preset_keys_and_is_idempotent(name, base):
    preset = _VARIANT_PROPERTY_PRESETS[name]
    original = dict(base)
    once = apply_variant(base, name)
    assert apply_variant(dict(once), name) == once
    assert base == original
    for key in preset:
        assert once[key] == preset[key]
    for key in set(base) - set(preset):
        assert once[key] == base[key]


@_VARIANT_PROPERTY_SETTINGS
@given(_VARIANT_PROPERTY_NAMES)
def test_apply_variant_is_case_insensitive(name):
    assert apply_variant({}, name.upper()) == apply_variant({}, name)


@_VARIANT_PROPERTY_SETTINGS
@given(_VARIANT_UNKNOWN_NAMES.filter(lambda name: name.lower() not in _VARIANT_PROPERTY_PRESETS))
def test_unknown_variant_raises(name):
    with pytest.raises(ValueError, match='unknown spec variant'):
        apply_variant({}, name)


def test_empty_variant_returns_input():
    base = {'impl': 'naive'}
    assert apply_variant(base, None) is base and apply_variant(base, '') is base


def test_convert_nests_attribute_access():
    cfg = _convert({'model': {'torch_dtype': 'float32'}, 'flags': [True]})
    assert isinstance(cfg, NamedDict)
    assert cfg.model.torch_dtype == 'float32'
    assert cfg.flags == [True]


@pytest.mark.parametrize('name', ['frozen_mean', 'last_token', 'updating_mean',
                                  'gamma_last', 'gamma_last_unit', 'mean_plus_gamma'])
def test_branch_override_and_returned_mutation_do_not_change_later_defaults(name):
    baseline = spec_for(name)
    assert baseline.tree.branches == [14]
    overridden = spec_for(name, branches=[2, 3])
    assert overridden.tree.branches == [2, 3]
    overridden.tree.branches.append(9)
    assert spec_for(name).tree.branches == [14]
    assert spec_for('frozen_mean').tree.branches == [14]
    assert baseline.tree.branches == [14]


def test_legacy_cfg_takes_device_from_system_and_seed_from_run():
    from types import SimpleNamespace

    from spec_mtp.structure.algorithm.config import AlgorithmConfig

    from studies._shared.structure.config import legacy_cfg

    algo = AlgorithmConfig.from_mapping({'mode': 'eval', 'source': 'spec_mtp',
                                         'config': {'eval': {'device': 'cuda', 'max_new_tokens': 8},
                                                    'spec': {'impl': 'naive'}}})
    cfg = legacy_cfg(SimpleNamespace(meta={'data_cfg': {'name': 'x'}}), SimpleNamespace(meta={}),
                     SimpleNamespace(device='cpu'), algo, '7')
    assert cfg.eval.device == 'cpu' and cfg.eval.max_new_tokens == 8
    assert cfg.run.seed == 7 and cfg.spec.impl == 'naive' and cfg.data.name == 'x' and dict(cfg.model) == {}


def test_frozen_mean_does_not_update():
    spec = spec_for('frozen_mean')
    assert spec.method == 'esp'
    assert spec.mask['update'] is False
    assert spec.mask.init == 'mean_prompt'
    assert spec.tree.branches == [14]
    assert spec.num_masks == 1


def test_mean_plus_gamma_uses_study_overrides():
    spec = spec_for('mean_plus_gamma', beta=0.999, lam=0.1, gamma=1, branches=[7, 2], num_masks=2)
    assert spec.method == 'ema_velocity'
    assert spec.ema.level == 'order0'
    assert spec.ema.beta == 0.999
    assert spec.ema.lam == 0.1
    assert spec.ema.step_scale == 1
    assert spec.tree.branches == [7, 2]
    assert spec.num_masks == 2


def test_unknown_mask_is_rejected():
    with pytest.raises(ValueError, match='unknown mask setting'):
        spec_for('not_a_mask')
