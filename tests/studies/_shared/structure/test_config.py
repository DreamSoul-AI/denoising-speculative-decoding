"""Pure legacy decoder config stays independent between calls."""
import pytest

from studies._shared.structure.config import NamedDict, _convert, spec_for


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
