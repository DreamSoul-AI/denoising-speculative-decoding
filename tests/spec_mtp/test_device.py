"""Device and dtype helpers."""

import pytest
import torch

from spec_mtp.model import get_device, get_dtype, seed_everything


def test_cpu_device_stays_cpu():
    assert get_device('cpu') == 'cpu'


def test_dtype_names():
    assert get_dtype('float32') is torch.float32
    assert get_dtype('bf16') is torch.bfloat16
    assert get_dtype('auto') == 'auto'


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match='Unsupported torch_dtype'):
        get_dtype('float64')


def test_seed_is_repeatable():
    seed_everything(123)
    a = torch.rand(3)
    seed_everything(123)
    b = torch.rand(3)
    assert torch.equal(a, b)
