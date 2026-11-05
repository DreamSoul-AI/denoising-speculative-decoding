"""Tiny real CUDA calibration for both fixed hybrid gain protocols."""

import pytest
import torch

from studies.trajectory_observation.structure import bonus_aligned_gain as bonus
from studies.trajectory_observation.structure import output_hidden_kalman as hidden


@pytest.mark.parametrize('name', ['bonus', 'hidden'])
def test_hybrid_calibration_computes_gain_and_diagnostics_on_cuda(tmp_path, monkeypatch, record_property, name):
    assert torch.cuda.is_available(), 'hybrid calibration requires real CUDA; no CPU fallback'
    module = bonus if name == 'bonus' else hidden
    device = torch.device('cuda:0')
    delta = torch.tensor([[1., 2., 3.], [2., 3., 4.]], device=device)
    zero = torch.zeros_like(delta)
    stored = {'qid': [item['qid'] for item in module.prompts_for('calibration')]}
    values = {'m': zero, 'z': delta, 'first': 2 * delta, 'last': 3 * delta} if name == 'bonus' else {'h': zero, 'z': delta, 'y': 3 * delta}
    stored.update({key: [value] * 128 for key, value in values.items()})
    original = module.calibrate_diagonal_gain
    def gain(*values):
        assert all(value.device == device for value in values)
        return original(*values)
    monkeypatch.setattr(module, 'calibrate_diagonal_gain', gain)
    result = module.calibrate(stored, device)
    assert not list(tmp_path.iterdir()), 'structure calibration must not write derived files'
    if name == 'bonus':
        assert result['first']['k'] == [2., 2., 2.]
        assert result['last']['k'] == [3., 3., 3.]
        assert result['pairs']['rounds_first_ne_last'] == 256
    else:
        assert result['k'] == [3., 3., 3.]
    record_property('cuda_device', torch.cuda.get_device_name(device))
