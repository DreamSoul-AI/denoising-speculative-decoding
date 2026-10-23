"""Request values remain typed at the user-input boundary."""
from dataclasses import replace

import pytest

from spec_mtp.contracts import Prompt
from test_decode import request


@pytest.mark.parametrize('qid', [True, 1, None, ''])
def test_prompt_identity_is_not_coerced(qid):
    with pytest.raises(ValueError):
        Prompt(qid, (1, 2))


@pytest.mark.parametrize('stop', ['false', 0, None, 1])
def test_eos_flag_is_a_boolean(stop):
    with pytest.raises(ValueError):
        replace(request(), stop_on_eos=stop)


@pytest.mark.parametrize('budget', [True, -1, 1.5])
def test_budget_is_not_coerced(budget):
    with pytest.raises(ValueError):
        replace(request(), max_new_tokens=budget)
