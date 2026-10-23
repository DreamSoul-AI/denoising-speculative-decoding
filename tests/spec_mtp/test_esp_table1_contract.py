"""Accounting regressions for fixed-input, CPU-only Table1 reconciliation."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('esp_contract', Path(__file__).resolve().parents[2] / 'tools/reconcile_esp_table1_contract.py')
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def test_weighting_decomposes_within_and_between_categories():
    rows = [dict(category='a', committed=3, decode_calls=2),
            dict(category='a', committed=8, decode_calls=4),
            dict(category='b', committed=4, decode_calls=4)]
    result = contract.accounting(rows)
    assert result['pooled_tau'] == 1.5
    assert result['prompt_mean_tau'] == 1.5
    assert result['within_category_call_weight_term'] == pytest.approx(1 / 18)
    assert result['between_category_call_weight_term'] == pytest.approx(-1 / 18)
    assert result['category_mean_tau'] == pytest.approx(17 / 12)


def test_prefix_counts_partial_boundary_and_preserves_early_stop():
    assert contract.prefix_accounting([2, 3], limit=4) == dict(
        committed=3, decode_calls=2, new_tokens=4, boundary_partial_blocks=1)
    assert contract.prefix_accounting([2, 3], limit=20) == dict(
        committed=5, decode_calls=2, new_tokens=6, boundary_partial_blocks=0)
    assert contract.prefix_accounting([2, 3], limit=1) == dict(
        committed=0, decode_calls=0, new_tokens=1, boundary_partial_blocks=0)


def test_invalid_denominators_or_block_sizes_are_rejected():
    with pytest.raises(AssertionError):
        contract.accounting([dict(category='a', committed=1, decode_calls=0)])
    with pytest.raises(AssertionError):
        contract.prefix_accounting([2, 0])
    with pytest.raises(AssertionError):
        contract.prefix_accounting([2], limit=0)
