"""Rare-event zero observations do not erase mass or imply sufficient statistical power."""

import math
import pytest

from test_window_observer import ROOT


def test_detectability_preserves_tiny_nonzero_mass(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_rare_events import detectability
    result = detectability(1e-20, 4096)
    assert result['ideal_Bernoulli_probability_at_least_one'] == pytest.approx(4096e-20, rel=1e-12, abs=0.)
    assert result['ideal_trials_for_95pct_detection'] > 10**20
    assert result['one_sided_95pct_upper_after_zero'] > 1e-20


def test_zero_and_certain_mass_endpoints(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_rare_events import detectability
    zero, one = detectability(0, 4096), detectability(1, 4096)
    assert zero['ideal_Bernoulli_probability_at_least_one'] == 0 and zero['ideal_trials_for_95pct_detection'] is None
    assert one['ideal_Bernoulli_probability_at_least_one'] == 1 and one['ideal_trials_for_95pct_detection'] == 1


def test_zero_observation_upper_bound_and_detection_threshold(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_rare_events import detectability
    result = detectability(.001, 4096)
    n = result['ideal_trials_for_95pct_detection']
    assert 1 - .999**n >= .95 and 1 - .999**(n - 1) < .95
    assert (1 - result['one_sided_95pct_upper_after_zero'])**4096 == pytest.approx(.05)
    assert result['ideal_Bernoulli_probability_at_least_one'] == pytest.approx(-math.expm1(4096 * math.log(.999)))


@pytest.mark.parametrize('p,n', [(-.1, 64), (1.1, 64), (.1, 0)])
def test_detectability_rejects_invalid_inputs(monkeypatch, p, n):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_rare_events import detectability
    with pytest.raises(ValueError):
        detectability(p, n)


def frame(root, count):
    return {'root_position': root, 'cache_before': root, 'cache_after': root + count + 1,
            'accepted_nodes': list(range(1, count + 1)),
            'committed_tokens': list(range(root + 1, root + count + 2)), 'stopped': False}


def accounting(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from studies.output_space_cross_task.analyze_first_delta import path_accounting
    return path_accounting


def test_first_gain_can_end_in_loss_with_real_root_accounting(monkeypatch):
    compare = accounting(monkeypatch)
    left = [frame(10, 1), frame(12, 1), frame(14, 1)]
    right = [frame(10, 2), frame(13, 0), frame(14, 0), frame(15, 0)]
    result = compare(left, right, 3, 2)
    assert result['first_accepted_delta'] == 1 and result['final_accepted_delta'] == -1
    assert not result['direction_prediction_preserved']
    assert result['common_root_delta'] + result['exclusive_root_delta'] == -1
    assert result['self_only_roots'] == [12] and result['graft_only_roots'] == [13, 15]


def test_first_gain_can_cancel_to_zero(monkeypatch):
    compare = accounting(monkeypatch)
    result = compare([frame(10, 0), frame(11, 1)], [frame(10, 1), frame(12, 0)], 1, 1)
    assert result['first_accepted_delta'] == 1 and result['final_accepted_delta'] == 0
    assert not result['direction_prediction_preserved']
    assert result['common_root_delta'] == 1 and result['exclusive_root_delta'] == -1


def test_preserved_direction_and_diagnostic_metadata_scope(monkeypatch):
    compare = accounting(monkeypatch)
    result = compare([frame(10, 0), frame(11, 0)], [frame(10, 1)], 0, 1)
    assert result['direction_prediction_preserved']
    left, right = frame(10, 0), frame(10, 0)
    left['mask_norm'], right['mask_norm'] = .9, 1.1
    same = compare([left], [right], 0, 0)
    assert same['first_divergence_root'] is None and same['final_accepted_delta'] == 0


@pytest.mark.parametrize('damaged', ['count', 'prefix'])
def test_bad_counts_or_unshared_prefix_are_rejected(monkeypatch, damaged):
    compare = accounting(monkeypatch)
    with pytest.raises(AssertionError):
        if damaged == 'count':
            compare([frame(10, 0)], [frame(10, 0)], 1, 0)
        else:
            compare([frame(10, 0)], [frame(11, 0)], 0, 0)
